# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets import custom_build_dataset





# In[12]:
# 1. 钩子函数方法（推荐）
def print_hook(module, input, output):
    """打印每层的输入、权重和输出前10个数"""
    print(f"\n{module.__class__.__name__} ({module.__class__.__name__}):")
    print("=" * 60)

    # 1. 打印输入信息
    if input is not None and len(input) > 0:
        if isinstance(input[0], torch.Tensor):
            input_tensor = input[0]
            flat_input = input_tensor.flatten()
            input_values = flat_input[:10].detach().cpu().numpy()
            print(f"  Input Shape: {input_tensor.shape}")
            print(f"  First 10 input values: ", end="")
            for i, val in enumerate(input_values):
                print(f"{val:.5f}", end=" ")
            print()
        else:
            print(f"  Input type: {type(input[0])}")
    else:
        print("  No input")

    # 2. 打印权重信息（如果模块有权重）
    if hasattr(module, 'weight') and module.weight is not None:
        weight_tensor = module.weight
        flat_weight = weight_tensor.flatten()
        weight_values = flat_weight[:10].detach().cpu().numpy()
        print(f"  Weight Shape: {weight_tensor.shape}")
        print(f"  First 10 weight values: ", end="")
        for i, val in enumerate(weight_values):
            print(f"{val:.5f}", end=" ")
        print()

        # 如果有偏置项，也打印
        if hasattr(module, 'bias') and module.bias is not None:
            bias_tensor = module.bias
            flat_bias = bias_tensor.flatten()
            bias_values = flat_bias[:min(10, len(flat_bias))].detach().cpu().numpy()
            print(f"  Bias Shape: {bias_tensor.shape}")
            print(f"  First {len(bias_values)} bias values: ", end="")
            for i, val in enumerate(bias_values):
                print(f"{val:.5f}", end=" ")
            print()

    # 3. 打印输出信息
    if isinstance(output, torch.Tensor):
        flat_output = output.flatten()
        output_values = flat_output[:10].detach().cpu().numpy()
        print(f"  Output Shape: {output.shape}")
        print(f"  First 10 output values: ", end="")
        for i, val in enumerate(output_values):
            print(f"{val:.5f}", end=" ")
        print()

    # 4. 打印计算量信息
    if isinstance(module, nn.Conv2d):
        # 对于卷积层，计算FLOPs
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            in_shape = input[0].shape
            out_shape = output.shape
            kernel_size = module.kernel_size
            groups = module.groups

            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            # 计算FLOPs
            flops = (2 * out_shape[1] * out_shape[2] * out_shape[3] *
                    in_shape[1] // groups * kernel_size[0] * kernel_size[1])
            print(f"  FLOPs: {flops:,}")
            print(f"  Groups: {groups}")

    return output


# 注册钩子
def register_hooks(model):
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只注册叶子模块
            handle = module.register_forward_hook(print_hook)
            handles.append(handle)
    return handles





def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   eval_model=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)
    flash_backend = cfg.get('flash_attention_backend', None)
    if flash_backend not in (None, 'fp32-math'):
        raise ValueError(f'unsupported flash_attention_backend: {flash_backend!r}')
    if flash_backend is not None and eval_model is not None:
        raise ValueError('flash_attention_backend with a separate eval_model is not supported')


    # prepare data loaders
   
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    #assert len(dataset)==1s
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
            runner_type=cfg.runner,
            pin_memory=cfg.data.get('pin_memory', False),
        ) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        # if dist.is_initialized():
            # if dist.get_rank() == 0:
                # handles = register_hooks(model)

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if eval_model is not None:
            eval_model = MMDistributedDataParallel(
                eval_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        print(f"cfg.gpu_ids: {cfg.gpu_ids[0]}")
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids, find_unused_parameters=False)  # 加上这个
        if eval_model is not None:
            eval_model = MMDataParallel(
                eval_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids, find_unused_parameters=False)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if eval_model is not None:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                eval_model=eval_model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))
    else:
        runner = build_runner(
            cfg.runner,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    
    # register profiler hook
    #trace_config = dict(type='tb_trace', dir_name='work_dir')
    #profiler_config = dict(on_trace_ready=trace_config)
    #runner.register_profiler_hook(profiler_config)
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type='DistributedGroupSampler'),
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type='DistributedSampler'),
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join('val', cfg.work_dir, time.ctime().replace(' ','_').replace(':','_'))
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    image_precision_hook = None
    image_precision = cfg.get('image_precision', None)
    if image_precision is not None:
        if image_precision not in ('image-fp32', 'feature-amp-consumer-fp32',
                                   'feature-amp-det2d-mixed'):
            raise ValueError(f'unsupported image_precision: {image_precision!r}')
        if image_precision.startswith('feature-amp-') and flash_backend != 'fp32-math':
            raise ValueError('feature-AMP image precision requires fp32-math point attention')
        from projects.mmdet3d_plugin.core.hook.image_precision import ImageFP32Hook
        image_precision_hook = ImageFP32Hook(mode=image_precision)
        runner.register_hook(image_precision_hook, priority='LOW')

    attention_precision_hook = None
    if flash_backend == 'fp32-math':
        from projects.mmdet3d_plugin.core.hook.attention_precision import FlashFP32Hook
        attention_precision_hook = FlashFP32Hook()
        runner.register_hook(attention_precision_hook, priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if flash_backend is None:
        # Record the current choice, not a resumed checkpoint's former policy.
        if runner.meta is None:
            runner.meta = {}
        runner.meta['flash_backend'] = 'native-half'
        logger.info('Attention precision backend: native-half (unchanged)')
    close_point_worker = getattr(model.module, 'close_point_worker', None)
    try:
        runner.run(data_loaders, cfg.workflow)
    except BaseException:
        try:
            if close_point_worker is not None:
                close_point_worker()
        except BaseException:
            pass  # Preserve the original training exception.
        raise
    else:
        if close_point_worker is not None:
            close_point_worker()
    finally:
        try:
            if attention_precision_hook is not None:
                attention_precision_hook.close()
        finally:
            if image_precision_hook is not None:
                image_precision_hook.close()

