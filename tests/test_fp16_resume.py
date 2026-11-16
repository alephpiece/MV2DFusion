"""CPU checks for scaler restoration through the training API and MMCV."""
from contextlib import ExitStack
from copy import deepcopy
from functools import partial
import importlib
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from mmcv import ConfigDict
from mmcv.runner import Fp16OptimizerHook, IterBasedRunner, OptimizerHook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = importlib.import_module('projects.mmdet3d_plugin.core.apis.mmdet_train')

SCALER_STATE = dict(scale=1024.0, growth_factor=2.0, backoff_factor=0.5,
                    growth_interval=2000, _growth_tracker=7)


class FakeRunner:
    def __init__(self, payload, meta=None, loads=1, failure=None):
        self.payload = payload
        self.meta = meta
        self.loads = loads
        self.failure = failure
        self.load_calls = 0
        self.resume_calls = 0

    def load_checkpoint(self, checkpoint):
        self.load_calls += 1
        if self.failure == 'load':
            raise RuntimeError('load failed')
        return self.payload

    def resume(self, checkpoint):
        self.resume_calls += 1
        for _ in range(self.loads):
            self.load_checkpoint(checkpoint)
        if self.failure == 'resume':
            raise RuntimeError('resume failed')


class FP16ResumeTest(unittest.TestCase):
    def test_single_load_preserves_metadata_and_loader(self):
        for with_meta in (False, True):
            for instance_override in (False, True):
                with self.subTest(meta=with_meta, override=instance_override):
                    payload = dict(meta=dict(fp16=dict(loss_scaler=deepcopy(SCALER_STATE))))
                    meta = dict(current_run=True, fp16=dict(other='keep')) if with_meta else None
                    runner = FakeRunner(payload, meta)
                    if instance_override:
                        previous = runner.load_checkpoint
                        runner.load_checkpoint = previous
                    api._resume_with_fp16_scaler_state(runner, 'checkpoint.pth')
                    self.assertEqual((runner.load_calls, runner.resume_calls), (1, 1))
                    self.assertEqual(runner.meta['fp16']['loss_scaler'], SCALER_STATE)
                    if with_meta:
                        self.assertIs(runner.meta, meta)
                        self.assertTrue(meta['current_run'])
                        self.assertEqual(meta['fp16']['other'], 'keep')
                    payload['meta']['fp16']['loss_scaler']['scale'] = 1.0
                    self.assertEqual(runner.meta['fp16']['loss_scaler']['scale'], 1024.0)
                    if instance_override:
                        self.assertIs(runner.load_checkpoint, previous)
                    else:
                        self.assertNotIn('load_checkpoint', vars(runner))

    def test_failures_restore_loader_without_injecting_metadata(self):
        valid = dict(meta=dict(fp16=dict(loss_scaler=SCALER_STATE)))
        cases = [
            ({}, 1, None, RuntimeError, 'missing meta.fp16.loss_scaler'),
            (dict(meta=dict(fp16=dict(loss_scaler=None))), 1, None, TypeError, 'must be a dict'),
            (valid, 0, None, RuntimeError, 'exactly one checkpoint, got 0'),
            (valid, 2, None, RuntimeError, 'exactly one checkpoint, got 2'),
            (valid, 1, 'load', RuntimeError, 'load failed'),
            (valid, 1, 'resume', RuntimeError, 'resume failed'),
        ]
        for payload, loads, failure, error_type, message in cases:
            for instance_override in (False, True):
                with self.subTest(message=message, override=instance_override):
                    meta = dict(current_run=True, fp16=dict(other='keep'))
                    runner = FakeRunner(payload, meta, loads, failure)
                    if instance_override:
                        previous = runner.load_checkpoint
                        runner.load_checkpoint = previous
                    with self.assertRaisesRegex(error_type, message):
                        api._resume_with_fp16_scaler_state(runner, 'checkpoint.pth')
                    self.assertEqual(meta, dict(current_run=True, fp16=dict(other='keep')))
                    if instance_override:
                        self.assertIs(runner.load_checkpoint, previous)
                    else:
                        self.assertNotIn('load_checkpoint', vars(runner))

    def test_actual_mmcv_resume_and_fp16_hook(self):
        class Model(torch.nn.Linear):
            def train_step(self, data, optimizer):
                return dict(loss=self(data).sum(), num_samples=len(data))

        def make_runner(directory, meta):
            model = Model(3, 2)
            return IterBasedRunner(model=model,
                optimizer=torch.optim.AdamW(model.parameters(), lr=4e-4),
                work_dir=directory, meta=meta, max_iters=20,
                logger=logging.getLogger('fp16-resume-test'))

        with tempfile.TemporaryDirectory() as directory:
            source = make_runner(directory, dict(fp16=dict(loss_scaler=deepcopy(SCALER_STATE))))
            for _ in range(5):
                source.optimizer.zero_grad()
                source.model(torch.ones(1, 3)).sum().backward()
                source.optimizer.step()
            source._iter, source._epoch = 4, 2
            source.save_checkpoint(directory, create_symlink=False)
            path = str(Path(directory) / 'iter_5.pth')
            disk = torch.load(path, map_location='cpu', weights_only=False)
            self.assertTrue(disk['optimizer']['state'])

            native = make_runner(directory, dict(current_run=True))
            native.resume(path, map_location='cpu')
            repaired = make_runner(directory, dict(current_run=True))
            repaired.resume = partial(repaired.resume, map_location='cpu')
            with mock.patch.object(repaired, 'load_checkpoint', wraps=repaired.load_checkpoint) as loader:
                api._resume_with_fp16_scaler_state(repaired, path)
                loader.assert_called_once()
                self.assertIs(repaired.load_checkpoint, loader)

            for runner in (native, repaired):
                for key, tensor in disk['state_dict'].items():
                    torch.testing.assert_close(runner.model.state_dict()[key], tensor, rtol=0, atol=0)
                restored = runner.optimizer.state_dict()
                self.assertEqual(restored['param_groups'], disk['optimizer']['param_groups'])
                for index, state in disk['optimizer']['state'].items():
                    for key, tensor in state.items():
                        torch.testing.assert_close(restored['state'][index][key], tensor, rtol=0, atol=0)
                self.assertEqual((runner.iter, runner.epoch),
                                 (disk['meta']['iter'], disk['meta']['epoch']))
                self.assertTrue(runner.meta['current_run'])

            self.assertNotIn('fp16', native.meta)
            self.assertEqual(repaired.meta['fp16']['loss_scaler'], SCALER_STATE)
            # Exercise the real before_run reader without allocating a GPU scaler.
            for runner, expected in ((native, False), (repaired, True)):
                hook = Fp16OptimizerHook(loss_scale='dynamic')
                hook.loss_scaler = torch.amp.GradScaler('cpu')
                hook.before_run(runner)
                self.assertEqual(hook.loss_scaler.state_dict() == SCALER_STATE, expected)

    def test_training_api_uses_registered_hook_for_resume(self):
        fp16_hook = Fp16OptimizerHook(loss_scale='dynamic')
        for hook, resume_from, load_from, selected in (
            (fp16_hook, 'resume.pth', 'weights.pth', 'scaler'),
            (OptimizerHook(), 'resume.pth', None, 'resume'),
            (fp16_hook, None, 'weights.pth', 'load'),
        ):
            with self.subTest(path=selected):
                runner = SimpleNamespace(meta={}, hooks=[hook],
                    register_training_hooks=mock.Mock(), register_hook=mock.Mock(),
                    resume=mock.Mock(), load_checkpoint=mock.Mock(), run=mock.Mock())
                model = SimpleNamespace()
                model.cuda = lambda *args: model
                # FP16 is configured through optimizer_config; there is no cfg.fp16.
                cfg = ConfigDict(log_level='INFO', gpu_ids=[0], seed=42,
                    data=dict(samples_per_gpu=1, workers_per_gpu=0,
                        shuffler_sampler={}, nonshuffler_sampler={}),
                    runner=dict(type='IterBasedRunner', max_iters=1), optimizer={},
                    optimizer_config=dict(type='Fp16OptimizerHook'),
                    lr_config={}, checkpoint_config={}, log_config={},
                    work_dir='/tmp', workflow=[('train', 1)],
                    resume_from=resume_from, load_from=load_from)
                with ExitStack() as patches:
                    replacements = dict(get_root_logger=mock.Mock(),
                        build_dataloader=mock.Mock(return_value=[]), build_optimizer=mock.Mock(),
                        build_runner=mock.Mock(return_value=runner),
                        MMDataParallel=lambda value, **kwargs: SimpleNamespace(module=value))
                    for name, replacement in replacements.items():
                        patches.enter_context(mock.patch.object(api, name, replacement))
                    repair = patches.enter_context(mock.patch.object(api, '_resume_with_fp16_scaler_state'))
                    api.custom_train_detector(model, object(), cfg)
                if selected == 'scaler':
                    repair.assert_called_once_with(runner, resume_from)
                    runner.resume.assert_not_called()
                    runner.load_checkpoint.assert_not_called()
                elif selected == 'resume':
                    repair.assert_not_called()
                    runner.resume.assert_called_once_with(resume_from)
                    runner.load_checkpoint.assert_not_called()
                else:
                    repair.assert_not_called()
                    runner.resume.assert_not_called()
                    runner.load_checkpoint.assert_called_once_with(load_from)
                runner.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()
