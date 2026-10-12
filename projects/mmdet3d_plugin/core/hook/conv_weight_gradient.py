"""Reversible first-order weight-gradient optimization for selected image convolutions."""
from types import MethodType

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from projects.mmdet3d_plugin.ops import conv_weight_gradient as kernels


def _specifications():
    specs = {}

    def add(name, ci, co, h, w, operator, *, kernel=3, input_dtype='float16', bias=False,
            dynamic_batch=False, stride=1, padding=None, input_gradient=True):
        specs.setdefault(name, []).append({
            'input_tail': [ci, h, w], 'batches': [9, 10, 11, 12] if dynamic_batch else [12],
            'weight_shape': [co, ci, kernel, kernel], 'input_dtype': input_dtype,
            'bias': bias, 'stride': stride, 'padding': kernel // 2 if padding is None else padding,
            'operator': operator, 'input_gradient': input_gradient})

    add('img_neck.fpn_convs.0.conv', 256, 256, 160, 400, 'coalesced', input_dtype='float32')
    add('img_roi_head.rpn_head.rpn_conv', 256, 256, 160, 400, 'coalesced',
        bias=True, dynamic_batch=True)
    for index in range(3):
        add(f'img_backbone.layer1.{index}.conv2', 64, 64, 160, 400, 'batch-tap')
        add(f'img_backbone.layer1.{index}.conv3', 64, 256, 160, 400, 'pointwise', kernel=1)
    for index in (1, 2):
        add(f'img_backbone.layer1.{index}.conv1', 256, 64, 160, 400, 'pointwise', kernel=1)
    for index in range(4):
        add(f'img_backbone.layer2.{index}.conv3', 128, 512, 80, 200, 'pointwise', kernel=1)
    for index in (1, 2, 3):
        add(f'img_backbone.layer2.{index}.conv1', 512, 128, 80, 200, 'pointwise', kernel=1)
    add('img_backbone.layer3.0.conv1', 512, 256, 80, 200, 'pointwise', kernel=1)
    add('img_neck.lateral_convs.0.conv', 256, 256, 160, 400, 'pointwise', kernel=1)
    add('img_neck.lateral_convs.1.conv', 512, 256, 80, 200, 'pointwise', kernel=1)
    for index in range(6):
        add(f'img_backbone.layer3.{index}.conv3', 256, 1024, 40, 100, 'pointwise-gemm', kernel=1)
    for name, co in (('rpn_cls', 3), ('rpn_reg', 12)):
        add('img_roi_head.rpn_head.' + name, 256, co, 160, 400, 'pointwise',
            kernel=1, bias=True, dynamic_batch=True)
    add('img_neck.fpn_convs.1.conv', 256, 256, 80, 200, 'coalesced', input_dtype='float32')
    add('img_neck.fpn_convs.2.conv', 256, 256, 40, 100, 'batch-tap', input_dtype='float32')
    for index in (1, 2, 3):
        add(f'img_backbone.layer2.{index}.conv2', 128, 128, 80, 200, 'batch-tap')
    for index in (1, 2, 3, 4, 5):
        add(f'img_backbone.layer3.{index}.conv2', 256, 256, 40, 100, 'batch-tap')
    for h, w, operator in ((80, 200, 'coalesced'), (40, 100, 'batch-tap')):
        add('img_roi_head.rpn_head.rpn_conv', 256, 256, h, w, operator,
            bias=True, dynamic_batch=True)
    add('img_backbone.layer1.0.downsample.0', 64, 256, 160, 400, 'protected', kernel=1)
    add('img_backbone.conv1', 3, 64, 640, 1600, 'stem', kernel=7,
        input_dtype='float32', stride=2, padding=3, input_gradient=False)
    return specs


def _weight_gradient(inputs, gradient, spec):
    operator = spec['operator']
    if operator == 'protected':
        return kernels.pointwise_weight_gradient(inputs, gradient, 16, output_dtype=torch.float32)
    if operator == 'pointwise':
        return kernels.pointwise_weight_gradient(inputs, gradient,
            16 if inputs.shape[2:] == (160, 400) else 8)
    if operator == 'pointwise-gemm':
        return kernels.pointwise_gemm_weight_gradient(inputs, gradient)
    if operator == 'coalesced':
        return kernels.unfold_weight_gradient(inputs, gradient)
    if operator == 'batch-tap':
        return kernels.conv3x3_weight_gradient(inputs, gradient)
    if operator == 'stem':
        return kernels.stem_weight_gradient(inputs, gradient)
    raise RuntimeError('unknown convolution weight-gradient implementation')


class _ConvWeightGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, spec):
        with torch.autocast(inputs.device.type, enabled=False):
            x_half, w_half = inputs.half(), weight.half()
            b_half = bias.half() if bias is not None else None
            output = F.conv2d(x_half, w_half, b_half, stride=spec['stride'], padding=spec['padding'])
        ctx.save_for_backward(x_half, w_half)
        ctx.input_dtype, ctx.bias_present = inputs.dtype, bias is not None
        ctx.spec = spec
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        x_half, w_half = ctx.saved_tensors
        spec, operator = ctx.spec, ctx.spec['operator']
        dx = dw = db = None
        with torch.autocast(x_half.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                dx = nn.grad.conv2d_input(x_half.shape, w_half, gradient,
                    stride=spec['stride'], padding=spec['padding']).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                copy_dy = operator == 'pointwise' and w_half.shape[0] < 32
                if gradient.dtype == torch.float16 and (gradient.is_contiguous() or copy_dy):
                    dw = _weight_gradient(x_half, gradient.contiguous() if copy_dy else gradient, spec)
                    if operator == 'protected' and dw.dtype != torch.float32:
                        raise RuntimeError('protected weight gradient narrowed before FP32 return')
                else:
                    if operator == 'protected' and gradient.dtype != torch.float16:
                        raise RuntimeError('unsupported protected weight-gradient dtype')
                    dw = nn.grad.conv2d_weight(
                        x_half.float() if operator == 'protected' else x_half, w_half.shape,
                        gradient.float() if operator == 'protected' else gradient,
                        stride=spec['stride'], padding=spec['padding'])
                dw = dw.float()
            if ctx.bias_present and ctx.needs_input_grad[2]:
                db = gradient.sum((0, 2, 3)).float()
        return dx, dw, db, None


class ImageConvWeightGradientPolicy:
    """Install after image numerics and close before it; parameters stay registered."""
    def __init__(self, model, protected_binding):
        self.model = model.module if hasattr(model, 'module') else model
        self.protected_binding = protected_binding
        self.bindings = []

    def install(self):
        if self.bindings:
            raise RuntimeError('image convolution weight-gradient policy is already installed')
        modules = dict(self.model.named_modules())
        plan = []
        for name, specs in _specifications().items():
            conv, first = modules.get(name), specs[0]
            if (not isinstance(conv, nn.Conv2d) or list(conv.weight.shape) != first['weight_shape']
                    or conv.weight.dtype != torch.float32 or not conv.weight.is_leaf
                    or (conv.bias is not None) != first['bias']
                    or conv.stride != (first['stride'],) * 2
                    or conv.padding != (first['padding'],) * 2
                    or conv.dilation != (1, 1) or conv.groups != 1 or conv.padding_mode != 'zeros'):
                raise ValueError('unsupported image convolution specification: ' + name)
            if conv.bias is not None and (conv.bias.dtype != torch.float32 or not conv.bias.is_leaf
                    or tuple(conv.bias.shape) != (conv.out_channels,) or conv.bias.device != conv.weight.device):
                raise ValueError('unsupported image convolution bias: ' + name)
            if first['operator'] == 'protected':
                binding = self.protected_binding
                if binding is None or binding[0] is not conv or conv.forward is not binding[2]:
                    raise ValueError('FP32 weight-gradient protection is not installed: ' + name)
            elif (getattr(conv.forward, '__self__', None) is not conv
                    or getattr(conv.forward, '__func__', None) is not nn.Conv2d.forward
                    or getattr(conv._conv_forward, '__self__', None) is not conv
                    or getattr(conv._conv_forward, '__func__', None) is not nn.Conv2d._conv_forward):
                raise ValueError('custom image convolution forward is not supported: ' + name)
            plan.append((name, conv, specs))
        try:
            for name, conv, specs in plan:
                self._bind(name, conv, specs)
        except BaseException:
            self.close()
            raise
        return self

    def _bind(self, name, conv, specs):
        original = conv.forward
        had_local, previous = 'forward' in vars(conv), vars(conv).get('forward')
        weight, bias = conv.weight, conv.bias

        def forward(current, *args, **kwargs):
            if current.weight is not weight or current.bias is not bias:
                raise RuntimeError('convolution parameter ownership changed: ' + name)
            if len(args) == 1 and not kwargs:
                inputs = args[0]
                if (current.training and torch.is_grad_enabled() and isinstance(inputs, torch.Tensor)
                        and inputs.ndim == 4 and inputs.device.type == 'cuda'
                        and torch.is_autocast_enabled() and torch.get_autocast_dtype('cuda') == torch.float16
                        and inputs.is_contiguous() and weight.requires_grad):
                    for spec in specs:
                        dtype = torch.float32 if spec['input_dtype'] == 'float32' else torch.float16
                        if (inputs.shape[0] in spec['batches'] and list(inputs.shape[1:]) == spec['input_tail']
                                and inputs.dtype == dtype and inputs.requires_grad == spec['input_gradient']):
                            return _ConvWeightGradient.apply(inputs, weight, bias, spec)
            return original(*args, **kwargs)

        installed = MethodType(forward, conv)
        conv.forward = installed
        self.bindings.append((conv, original, installed, had_local, previous, weight, bias))

    def close(self):
        for conv, _, installed, _, _, weight, bias in self.bindings:
            if vars(conv).get('forward') is not installed or conv.weight is not weight or conv.bias is not bias:
                raise RuntimeError('convolution binding or parameter ownership changed')
        for conv, original, _, had_local, previous, _, _ in reversed(self.bindings):
            if had_local:
                conv.forward = previous
            else:
                del conv.forward
            if conv.forward != original:
                raise RuntimeError('convolution forward did not restore')
        self.bindings.clear()
