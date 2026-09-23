"""Backward precision handling for the feature-AMP image modes.

Only the layer1 downsample weight gradient and eight FPN SyncBN reductions change.
Both bindings are opt-in, reversible, and leave parameters and optimizer intact.
"""
from types import MethodType

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


_TARGET_CONV = 'img_backbone.layer1.0.downsample.0'
_TARGET_WEIGHT_SHAPE = (256, 64, 1, 1)


class _FP32WeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        if (inputs.ndim != 4 or inputs.dtype != torch.float16 or
                weight.dtype != torch.float32 or not weight.is_leaf or
                tuple(weight.shape) != _TARGET_WEIGHT_SHAPE or
                inputs.device != weight.device or inputs.shape[1] != weight.shape[1]):
            raise RuntimeError('unsupported feature-AMP 1x1 dW contract')
        with torch.autocast(inputs.device.type, enabled=False):
            half_weight = weight.to(dtype=torch.float16)
            output = F.conv2d(inputs, half_weight, None, 1, 0, 1, 1)
        ctx.save_for_backward(inputs, half_weight)
        ctx.weight_shape = tuple(weight.shape)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        inputs, half_weight = ctx.saved_tensors
        if grad_output.dtype != torch.float16 or grad_output.device != inputs.device:
            raise RuntimeError('unsupported feature-AMP 1x1 dW gradient')
        dx = dw = None
        with torch.autocast(inputs.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                dx = torch.nn.grad.conv2d_input(
                    inputs.shape, half_weight, grad_output, stride=1, padding=0,
                    dilation=1, groups=1)
            if ctx.needs_input_grad[1]:
                dw = torch.nn.grad.conv2d_weight(
                    inputs.float(), ctx.weight_shape, grad_output.float(),
                    stride=1, padding=0, dilation=1, groups=1)
        return dx, dw


class FeatureAMPNumericsPolicy:
    def __init__(self, model):
        self.model = model.module if hasattr(model, 'module') else model
        self.conv_binding = None
        self.reduce_binding = None

    def install(self):
        if self.conv_binding is not None or self.reduce_binding is not None:
            raise RuntimeError('feature-AMP numerics already installed')
        modules = dict(self.model.named_modules())
        conv = modules.get(_TARGET_CONV)
        if (type(conv) is not nn.Conv2d or
                tuple(conv.weight.shape) != _TARGET_WEIGHT_SHAPE or
                (conv.stride, conv.padding, conv.dilation, conv.groups,
                 conv.bias, conv.padding_mode) !=
                ((1, 1), (0, 0), (1, 1), 1, None, 'zeros') or
                conv.weight.dtype != torch.float32 or not conv.weight.is_leaf or
                not conv.weight.requires_grad or 'forward' in vars(conv) or
                conv.forward.__func__ is not nn.Conv2d.forward):
            raise RuntimeError('unexpected feature-AMP target convolution')
        norms = []
        for family in ('fpn_convs', 'lateral_convs'):
            group = getattr(self.model.img_neck, family)
            norms.extend(module.norm for module in group)
        pointers = {int(norm.weight.data_ptr()) for norm in norms}
        if len(norms) != 8 or len(pointers) != 8 or any(
                not isinstance(norm, nn.SyncBatchNorm) or
                norm.weight.dtype != torch.float32 for norm in norms):
            raise RuntimeError('unexpected feature-AMP FPN SyncBN contract')

        original_forward = conv.forward
        weight = conv.weight

        def forward(owner, inputs):
            if owner.weight is not weight:
                raise RuntimeError('feature-AMP target parameter changed')
            active_half = (inputs.device.type in ('cpu', 'cuda') and
                           torch.is_autocast_enabled(inputs.device.type) and
                           torch.get_autocast_dtype(inputs.device.type) == torch.float16)
            if torch.is_grad_enabled() and weight.requires_grad and (
                    inputs.dtype == torch.float16 or
                    (inputs.dtype == torch.float32 and active_half)):
                if inputs.dtype != torch.float16:
                    inputs = inputs.to(dtype=torch.float16)
                return _FP32WeightGradient.apply(inputs, weight)
            return original_forward(inputs)

        original_reduce = torch.batch_norm_backward_reduce

        def reduce(*args, **kwargs):
            keys = ('grad_out', 'input', 'mean', 'invstd', 'weight',
                    'input_g', 'weight_g', 'bias_g')
            values = [args[index] if index < len(args) else kwargs.get(key)
                      for index, key in enumerate(keys)]
            norm_weight, dy, x = values[4], values[0], values[1]
            if (isinstance(norm_weight, torch.Tensor) and
                    int(norm_weight.data_ptr()) in pointers and
                    isinstance(x, torch.Tensor) and isinstance(dy, torch.Tensor) and
                    x.dtype == dy.dtype == torch.float16):
                args, kwargs = list(args), dict(kwargs)
                for index in (0, 1):
                    if index < len(args):
                        args[index] = values[index].float()
                    else:
                        kwargs[keys[index]] = values[index].float()
            return original_reduce(*args, **kwargs)

        installed_forward = MethodType(forward, conv)
        try:
            conv.forward = installed_forward
            self.conv_binding = (conv, original_forward, installed_forward)
            torch.batch_norm_backward_reduce = reduce
            self.reduce_binding = (original_reduce, reduce)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.reduce_binding is not None:
            original, installed = self.reduce_binding
            if torch.batch_norm_backward_reduce is not installed:
                raise RuntimeError('feature-AMP FPN reduction binding changed')
            torch.batch_norm_backward_reduce = original
            self.reduce_binding = None
        if self.conv_binding is not None:
            conv, original, installed = self.conv_binding
            if vars(conv).get('forward') is not installed:
                raise RuntimeError('feature-AMP convolution binding changed')
            del conv.forward
            if conv.forward != original:
                raise RuntimeError('feature-AMP convolution did not restore')
            self.conv_binding = None
