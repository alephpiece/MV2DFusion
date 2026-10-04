"""Opt-in FP32 attention for six decoder FlashAttention modules.

This replaces the inner forward precision as well as its backward, not the
surrounding projections, parameters, flags or dropout. Only the validated
Float Q/KV, dropout-zero, noncausal, unmasked contract is supported. The caller
must disable matmul TF32; this policy never changes process-wide backend flags.
Importing this module does not activate the policy.
MATH is the default; fused safe-softmax requires explicit selection and Triton.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from mmcv.runner import Fp16OptimizerHook, Hook


INNER_NAMES = tuple(
    f'fusion_bbox_head.transformer.decoder.layers.{index}.'
    'attentions.1.attn.attn.inner_attn' for index in range(6))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def math_attention(q, kv, causal=False, key_padding_mask=None):
    _require(causal is False and key_padding_mask is None,
             'unsupported attention mask/causal contract')
    _require(q.dtype == kv.dtype == torch.float32,
             'FP32 attention requires original Float Q/KV, not recast Half')
    _require(q.device == kv.device and q.device.type in ('cpu', 'cuda'),
             'unsupported attention device')
    _require(q.ndim == 4 and kv.ndim == 5 and kv.shape[2] == 2,
             'expected Q[B,T,H,D] and KV[B,S,2,H,D]')
    _require(q.shape[0] == kv.shape[0] and q.shape[2:] == kv.shape[3:],
             'incompatible attention dimensions')
    _require(q.shape[1] > 0 and kv.shape[1] > 0 and q.shape[-1] > 0,
             'empty attention dimensions are not supported')
    if q.is_cuda:
        _require(not torch.backends.cuda.matmul.allow_tf32,
                 'FP32 math attention requires matmul.allow_tf32=False')
    query = q.transpose(1, 2)
    key, value = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    with torch.autocast(device_type=q.device.type, enabled=False):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0., is_causal=False,
                scale=1. / math.sqrt(q.shape[-1]))
    return output.transpose(1, 2), None


class FlashFP32Policy:
    def __init__(self, model, enabled=False, backend='fp32-math'):
        _require(backend in ('fp32-math', 'fp32-gemm-fused-safe-softmax'),
                 'unsupported FP32 attention backend')
        self.model = model.module if hasattr(model, 'module') else model
        self.enabled = bool(enabled)
        self.backend = backend
        self.bindings = []
        self.names = []
        self.installed = False

    @staticmethod
    def validate(module):
        _require(isinstance(module, nn.Module) and
                 type(module).__name__ == 'FlashAttention',
                 'expected a FlashAttention inner module')
        _require(module.dropout_p == 0., 'unsupported nonzero inner dropout')
        _require(module.softmax_scale is None,
                 'unsupported configured softmax scaling')

    def install(self):
        _require(not self.installed, 'attention precision policy already installed')
        if not self.enabled:
            return self
        attention = None
        if self.backend == 'fp32-gemm-fused-safe-softmax':
            # Keep Triton optional for native and MATH users.
            from .attention_safe_softmax import fused_attention
            attention = fused_attention
        modules = [(name, module) for name, module in self.model.named_modules()
                   if type(module).__name__ == 'FlashAttention']
        _require(tuple(name for name, _ in modules) == INNER_NAMES,
                 'expected exactly the six decoder FlashAttention modules')
        for _, module in modules:
            self.validate(module)
        try:
            for name, module in modules:
                previous_local = 'forward' in vars(module)
                previous_value = vars(module).get('forward')
                original_bound = module.forward

                def forward(owner, q, kv, causal=False, key_padding_mask=None):
                    FlashFP32Policy.validate(owner)
                    implementation = math_attention if attention is None else attention
                    return implementation(q, kv, causal=causal,
                                          key_padding_mask=key_padding_mask)

                replacement = MethodType(forward, module)
                setattr(module, 'forward', replacement)
                self.bindings.append((module, previous_local, previous_value,
                                      original_bound, replacement))
                self.names.append(name)
        except BaseException:
            self.close()
            raise
        self.installed = True
        return self

    def close(self):
        conflicts = []
        for module, previous_local, previous_value, original_bound, replacement in reversed(self.bindings):
            if vars(module).get('forward') is not replacement:
                conflicts.append(type(module).__name__)
                continue  # Never overwrite another owner's later binding.
            if previous_local:
                setattr(module, 'forward', previous_value)
            else:
                delattr(module, 'forward')
            _require(module.forward == original_bound,
                     'attention forward restoration failed')
        self.bindings.clear()
        self.names.clear()
        self.installed = False
        if conflicts:
            raise RuntimeError('attention policy uninstall ownership conflict: ' +
                               ', '.join(conflicts))

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


class FlashFP32Hook(Hook):
    """Train hook installed after the optimizer's precision setup."""
    def __init__(self, backend='fp32-math'):
        _require(backend in ('fp32-math', 'fp32-gemm-fused-safe-softmax'),
                 'unsupported FP32 attention backend')
        self.backend = backend
        self.policy = None

    def before_run(self, runner):
        _require(self.policy is None, 'attention precision hook already active')
        _require(getattr(runner, 'eval_model', None) is None,
                 'FP32 attention with a separate eval_model is not supported')
        optimizers = [index for index, hook in enumerate(runner.hooks)
                      if isinstance(hook, Fp16OptimizerHook)]
        _require(len(optimizers) == 1 and
                 optimizers[0] < runner.hooks.index(self),
                 'attention precision hook must follow one Fp16OptimizerHook')
        _require(not torch.backends.cuda.matmul.allow_tf32,
                 'FP32 math attention requires matmul.allow_tf32=False')
        self.policy = FlashFP32Policy(runner.model, enabled=True, backend=self.backend)
        self.policy.install()
        if runner.meta is None:
            runner.meta = {}
        runner.meta['flash_backend'] = self.backend
        runner.logger.info(f'Attention precision backend: {self.backend} '
                           '(six decoder FlashAttention modules)')
        if self.backend == 'fp32-gemm-fused-safe-softmax':
            runner.logger.info('Experimental attention backend: bounded training comparisons '
                               'showed lower validation metrics than fp32-math. '
                               'Training-quality equivalence is not established; '
                               'evaluate separately with fp32-math.')

    def after_run(self, runner):
        self.close()

    def close(self):
        if self.policy is not None:
            self.policy.close()
            self.policy = None
