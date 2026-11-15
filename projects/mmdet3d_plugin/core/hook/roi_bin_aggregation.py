"""Reversible, model-local ROIAlign backward aggregation for training."""
import functools
import math
from types import MethodType

import torch
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from mmcv.ops.roi_align import RoIAlign, RoIAlignFunction
from mmcv.runner import HOOKS, Hook


class _AggregatedRoIAlign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, rois, output_size, spatial_scale, sampling_ratio,
                pool_mode, aligned, limit):
        ctx.aggregate_limit = limit
        if rois.shape[0] == 0:
            # Explicit mathematical empty-output contract, without a zero-grid
            # extension launch. Nonempty forward remains the installed oracle.
            ctx.output_size = _pair(output_size)
            ctx.spatial_scale = spatial_scale
            ctx.sampling_ratio = sampling_ratio
            ctx.pool_mode = 1
            ctx.aligned = aligned
            ctx.input_shape = inputs.size()
            empty = inputs.new_empty(0)
            ctx.save_for_backward(rois, empty, empty)
            return inputs.new_empty((0, inputs.shape[1], *ctx.output_size))
        return RoIAlignFunction.forward(ctx, inputs, rois, output_size,
                                        spatial_scale, sampling_ratio, pool_mode, aligned)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, _, _ = ctx.saved_tensors
        if (grad_output.dtype == torch.float32 and rois.dtype == torch.float32
                and grad_output.is_cuda and rois.device == grad_output.device
                and ctx.pool_mode == 1 and ctx.sampling_ratio <= 0):
            from ...ops.roi_align_bin_aggregate import backward
            dx = backward(grad_output, rois, ctx.input_shape, ctx.output_size,
                          ctx.spatial_scale, ctx.sampling_ratio, ctx.aligned,
                          limit=ctx.aggregate_limit)
            return dx, None, None, None, None, None, None, None
        return (*RoIAlignFunction.backward(ctx, grad_output), None)


def _eligible(module, inputs, rois):
    # Metadata/control-flow checks only; never synchronize tensor contents.
    return (torch.is_grad_enabled() and inputs.requires_grad
            and not torch.jit.is_scripting() and not torch.jit.is_tracing()
            and not torch.onnx.is_in_onnx_export()
            and not module.use_torchvision and module.pool_mode == 'avg'
            and module.sampling_ratio <= 0
            and inputs.dtype == torch.float32 and rois.dtype == torch.float32
            and inputs.is_cuda and rois.device == inputs.device
            and inputs.ndim == 4 and rois.ndim == 2 and rois.shape[1] == 5
            and inputs.is_contiguous() and rois.is_contiguous()
            and min(inputs.shape[1:]) > 0)


class RoIAlignBinAggregationPolicy:
    """Bind selected RoIAlign instances; leave MMCV globals and other models alone.

    ``module_prefixes`` restricts ownership to named subtrees. Unsupported
    dtype, pooling, sampling, layout or export paths call the saved method.
    ``scale_limits`` configures a limit by exact spatial_scale; no GPU name
    dispatch or data-dependent host synchronization is used.
    """
    def __init__(self, model, limit=4, scale_limits=None, module_prefixes=None):
        if type(limit) is not int or limit not in (2, 4):
            raise ValueError('limit must be 2 or 4')
        self.limit = limit
        self.scale_limits = {}
        for scale, value in (scale_limits or {}).items():
            if type(value) is not int or value not in (2, 4):
                raise ValueError('scale limits must be 2 or 4')
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0 or scale in self.scale_limits:
                raise ValueError('scale_limits requires distinct positive finite scales')
            self.scale_limits[scale] = value
        if module_prefixes is not None and (isinstance(module_prefixes, str)
                or not all(isinstance(value, str) and value for value in module_prefixes)):
            raise ValueError('module_prefixes must be a sequence of nonempty names')
        self.module_prefixes = None if module_prefixes is None else tuple(module_prefixes)
        self.model = model.module if hasattr(model, 'module') else model
        self.bindings = []
        self.module_names = []

    def install(self):
        if self.bindings:
            raise RuntimeError('ROI aggregation policy is already installed')
        selected = [(name, module) for name, module in self.model.named_modules()
                    if isinstance(module, RoIAlign) and (self.module_prefixes is None
                       or any(name == prefix or name.startswith(prefix + '.')
                              for prefix in self.module_prefixes))]
        if not selected:
            raise RuntimeError('no RoIAlign modules match this policy')
        # Validate all owners before changing any method.
        for name, module in selected:
            if getattr(module.forward, '__func__', None) is not RoIAlign.forward:
                raise RuntimeError('ROI forward is already wrapped: ' + name)
        try:
            for name, module in selected:
                previous = module.forward
                had_local = 'forward' in vars(module)
                previous_local = vars(module).get('forward')
                policy = self

                def bind(original):
                    @functools.wraps(original.__func__)
                    def controlled(current, input, rois):
                        if not _eligible(current, input, rois):
                            return original(input, rois)
                        selected_limit = policy.scale_limits.get(current.spatial_scale, policy.limit)
                        return _AggregatedRoIAlign.apply(
                            input, rois, current.output_size, current.spatial_scale,
                            current.sampling_ratio, current.pool_mode, current.aligned, selected_limit)
                    return controlled

                installed = MethodType(bind(previous), module)
                module.forward = installed
                self.bindings.append((module, installed, had_local, previous_local))
                self.module_names.append(name)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for module, installed, _, _ in self.bindings:
            if module.forward is not installed:
                raise RuntimeError('remove outer ROI wrappers before closing this policy')
        for module, _, had_local, previous in reversed(self.bindings):
            if had_local:
                module.forward = previous
            else:
                del module.forward
        self.bindings.clear()
        self.module_names.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exception):
        self.close()


@HOOKS.register_module()
class RoIAlignBinAggregationHook(Hook):
    """Install before training; ``close`` also supports an external finally."""
    def __init__(self, limit=4, scale_limits=None, module_prefixes=None):
        self.options = dict(limit=limit, scale_limits=scale_limits, module_prefixes=module_prefixes)
        self.policy = None

    def before_run(self, runner):
        if self.policy is not None:
            raise RuntimeError('ROI aggregation hook is already active')
        self.policy = RoIAlignBinAggregationPolicy(runner.model, **self.options).install()
        runner.logger.info('ROI backward aggregation bound to: %s', ', '.join(self.policy.module_names))

    def after_run(self, runner):
        self.close()

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None
