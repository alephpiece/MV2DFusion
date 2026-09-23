"""Scoped checkpoint compatibility for ROCm Native sparse weights.

Old MMDetection3D replaces SparseModule's loader with a CUDA-layout loader.
During pretrained initialization only, keep Native modules on PyTorch's loader
after an explicit layout conversion. No operator or global class is changed.
"""
from contextlib import contextmanager
import math
from types import MethodType

import torch


def _native_shape(module):
    kernel = tuple(module.kernel_size)
    shape = (math.prod(kernel), module.in_channels, module.out_channels)
    if getattr(module, 'conv1x1', False):
        shape = (module.in_channels, module.out_channels)
    return shape


def native_sparse_weight(weight, module, metadata=None):
    target = _native_shape(module)
    if tuple(module.weight.shape) != target:
        raise ValueError('module does not use Native sparse weight storage')
    if tuple(weight.shape) == target:
        return weight
    kernel = tuple(module.kernel_size)
    layouts = []
    if tuple(weight.shape) == (*kernel, module.in_channels, module.out_channels):
        layouts.append('RSCK')
    if tuple(weight.shape) == (module.out_channels, *kernel, module.in_channels):
        layouts.append('KRSC')
    if len(layouts) == 2:
        # Framework _version=2 also labels Native storage; it is not a layout.
        explicit = (metadata or {}).get('layout')
        layouts = [explicit] if explicit in layouts else []
    if len(layouts) != 1:
        raise RuntimeError(f'unrecognized or ambiguous sparse checkpoint layout: {tuple(weight.shape)} -> {target}')
    if layouts[0] == 'RSCK':
        return weight.reshape(target)
    axes = (*range(1, weight.ndim), 0)
    return weight.permute(axes).contiguous().reshape(target)


@contextmanager
def native_sparse_checkpoint_loading(model):
    try:
        from spconv.pytorch.conv import SparseConvolution
    except ImportError:
        yield
        return

    bindings = []
    diagnostics = []

    def load(module, state_dict, prefix, metadata, strict, missing, unexpected, errors):
        key = prefix + 'weight'
        if key not in state_dict:
            raise RuntimeError(f'missing pretrained sparse weight: {key}')
        converted = state_dict.copy()
        converted[key] = native_sparse_weight(state_dict[key], module, metadata)
        diagnostics.append((missing, unexpected, errors))
        torch.nn.Module._load_from_state_dict(module, converted, prefix, metadata,
                                              strict, missing, unexpected, errors)

    try:
        for module in model.modules():
            if not isinstance(module, SparseConvolution) or tuple(module.weight.shape) != _native_shape(module):
                continue
            owned = '_load_from_state_dict' in module.__dict__
            previous = module.__dict__.get('_load_from_state_dict')
            installed = MethodType(load, module)
            module._load_from_state_dict = installed
            bindings.append((module, owned, previous, installed))
        yield
        # MMCV strict=False logs these errors; a frozen pretrained branch must
        # not silently retain random or partially loaded parameters.
        for missing, unexpected, errors in diagnostics:
            required_missing = [key for key in missing if not key.endswith('num_batches_tracked')]
            if required_missing or unexpected or errors:
                raise RuntimeError(f'incomplete pretrained state: missing={required_missing}, '
                                   f'unexpected={unexpected}, errors={errors}')
    finally:
        for module, owned, previous, installed in reversed(bindings):
            if module.__dict__.get('_load_from_state_dict') is not installed:
                raise RuntimeError('sparse checkpoint loader ownership changed')
            if owned:
                module._load_from_state_dict = previous
            else:
                del module._load_from_state_dict
