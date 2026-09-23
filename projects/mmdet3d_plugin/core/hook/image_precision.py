"""Opt-in FP32 scopes for image modules.

No mode is a whole-model FP32 or convergence guarantee. Point modules,
FlashAttention, loss methods, parameter objects and fp16 flags are left alone.
Install after Fp16OptimizerHook has configured the model.
"""
import functools
from types import MethodType

import torch
from mmcv.runner import Fp16OptimizerHook, Hook


ROOTS = ('img_backbone', 'img_neck', 'img_roi_head',
         'img_roi_extractor', 'img_query_generator')
METHODS = ('extract_img_feat', 'forward_roi_head_train', 'forward_roi_head')
FEATURE_AMP = 'feature-amp-consumer-fp32'
MODE_SCOPES = {
    'image-fp32': (METHODS, ROOTS),
    FEATURE_AMP: (
        ('forward_roi_head_train', 'forward_roi_head'),
        ('img_roi_head', 'img_roi_extractor', 'img_query_generator')),
}


def _promote(value):
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    if isinstance(value, tuple):
        return tuple(_promote(item) for item in value)
    if isinstance(value, list):
        return [_promote(item) for item in value]
    if isinstance(value, dict):
        return {key: _promote(item) for key, item in value.items()}
    return value


def _mmcv_decorated(function):
    return (function is not None and hasattr(function, '__code__') and
            function.__code__.co_filename.endswith('fp16_utils.py') and
            hasattr(function, '__wrapped__'))


class ImageFP32Policy:
    def __init__(self, model, mode='image-fp32'):
        if mode not in MODE_SCOPES:
            raise ValueError('unsupported image precision mode: ' + repr(mode))
        self.model = model.module if hasattr(model, 'module') else model
        self.mode = mode
        self.bindings = []
        self.numerics = None

    def _bind(self, module, name, function):
        had_local = name in vars(module)
        previous = vars(module).get(name)

        @functools.wraps(function)
        def controlled(current, *args, **kwargs):
            with torch.autocast('cuda', enabled=False):
                return function(current, *_promote(args), **_promote(kwargs))

        installed = MethodType(controlled, module)
        setattr(module, name, installed)
        self.bindings.append((module, name, installed, had_local, previous))

    def install(self):
        if self.bindings:
            raise RuntimeError('image precision policy is already installed')
        plan = []
        methods, roots = MODE_SCOPES[self.mode]
        for name in methods:
            function = getattr(getattr(self.model, name), '__func__', None)
            if not _mmcv_decorated(function):
                raise RuntimeError('expected an unmodified MMCV precision wrapper: ' + name)
            plan.append((self.model, name, function.__wrapped__))
        for name in roots:
            root = getattr(self.model, name)
            for child_name, module in root.named_modules():
                function = getattr(module.forward, '__func__', None)
                decorated = _mmcv_decorated(function)
                if decorated or not child_name:
                    if function is None:
                        raise RuntimeError('unsupported image forward: ' + name + '.' + child_name)
                    plan.append((module, 'forward', function.__wrapped__ if decorated else function))
        seen = set()
        try:
            for module, name, function in plan:
                key = (id(module), name)
                if key in seen:
                    continue
                seen.add(key)
                self._bind(module, name, function)
            if self.mode == FEATURE_AMP:
                from projects.mmdet3d_plugin.core.hook.feature_amp_numerics import FeatureAMPNumericsPolicy
                self.numerics = FeatureAMPNumericsPolicy(self.model)
                self.numerics.install()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.numerics is not None:
            self.numerics.close()
            self.numerics = None
        # Do not silently overwrite a wrapper installed by another owner.
        for module, name, installed, _, _ in self.bindings:
            if getattr(module, name) is not installed:
                raise RuntimeError('remove outer image precision wrappers before closing: ' + name)
        for module, name, _, had_local, previous in reversed(self.bindings):
            if had_local:
                setattr(module, name, previous)
            else:
                delattr(module, name)
        self.bindings.clear()


class ImageFP32Hook(Hook):
    def __init__(self, mode='image-fp32'):
        if mode not in MODE_SCOPES:
            raise ValueError('unsupported image precision mode: ' + repr(mode))
        self.mode = mode
        self.policy = None

    def before_run(self, runner):
        if self.policy is not None:
            raise RuntimeError('image precision hook is already active')
        optimizers = [index for index, hook in enumerate(runner.hooks)
                      if isinstance(hook, Fp16OptimizerHook)]
        if len(optimizers) != 1 or optimizers[0] >= runner.hooks.index(self):
            raise RuntimeError('image precision hook must run after one Fp16OptimizerHook')
        self.policy = ImageFP32Policy(runner.model, mode=self.mode)
        self.policy.install()

    def after_run(self, runner):
        self.close()

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None
