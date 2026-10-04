"""FP32 GEMM attention with fused safe-softmax and first-order gradients.

Loaded only when explicitly selected by the attention precision policy.
This backend keeps dense scores/probabilities; it is not streaming attention.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=[2], do_not_specialize_on_alignment=[2])
def _forward(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    maximum = tl.max(x, 0)
    e = tl.exp(x - maximum)
    p = e / tl.sum(e, 0)
    all_negative_inf = tl.sum((x == -float('inf')).to(tl.int32), 0) == BLOCK
    p = tl.where(all_negative_inf, 0., p)
    tl.store(Y + row * N + col, p, col < N)


@triton.jit(do_not_specialize=[3], do_not_specialize_on_alignment=[3])
def _backward(P, DY, DX, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    p = tl.load(P + row * N + col, col < N, other=0.)
    dy = tl.load(DY + row * N + col, col < N, other=0.)
    dx = p * (dy - tl.sum(p * dy, 0))
    tl.store(DX + row * N + col, dx, col < N)


class _FusedSafeSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        if (scores.dtype != torch.float32 or not scores.is_cuda or
                not scores.is_contiguous() or scores.numel() == 0):
            raise ValueError('safe-softmax requires nonempty contiguous device FP32 scores')
        n = scores.shape[-1]
        if scores.numel() > 2**31 - 1:
            raise ValueError('safe-softmax exceeds the supported 32-bit indexing range')
        block = triton.next_power_of_2(n)
        if block > 32768:
            raise ValueError('safe-softmax supports row lengths up to 32768')
        p = torch.empty_like(scores)
        _forward[(scores.numel() // n,)](
            scores, p, n, block, num_warps=8 if block >= 8192 else 4,
            enable_fp_fusion=False)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, dy):
        if torch.is_grad_enabled():
            raise RuntimeError('fused safe-softmax supports first-order gradients only')
        (p,) = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(p)
        n = p.shape[-1]
        _backward[(p.numel() // n,)](
            p, dy, dx, n, triton.next_power_of_2(n),
            num_warps=8 if n > 4096 else 4, enable_fp_fusion=False)
        return dx


def fused_attention(q, kv, causal=False, key_padding_mask=None):
    """Return attention output and no weights for unmasked FP32 Q/KV."""
    if causal is not False or key_padding_mask is not None:
        raise ValueError('fused attention does not support masks or causal attention')
    if q.dtype != kv.dtype or q.dtype != torch.float32:
        raise ValueError('fused attention requires original FP32 inputs')
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('fused attention requires matmul.allow_tf32=False')
    if q.ndim != 4 or kv.ndim != 5 or kv.shape[2] != 2:
        raise ValueError('expected Q[B,T,H,D] and KV[B,S,2,H,D]')
    if q.device != kv.device or q.device.type != 'cuda':
        raise ValueError('fused attention requires matching device inputs')
    if q.shape[0] != kv.shape[0] or q.shape[2:] != kv.shape[3:]:
        raise ValueError('incompatible attention dimensions')
    if any(size == 0 for size in q.shape) or kv.shape[1] == 0:
        raise ValueError('empty attention dimensions are not supported')
    if kv.shape[1] > 32768:
        raise ValueError('fused attention supports KV lengths up to 32768')
    if q.shape[0] * q.shape[1] * q.shape[2] * kv.shape[1] > 2**31 - 1:
        raise ValueError('attention scores exceed the supported 32-bit indexing range')
    with torch.autocast(device_type='cuda', enabled=False):
        # Preserve MATH Q/K pre-scaling, rather than scaling the score matrix.
        factor = math.sqrt(1. / math.sqrt(q.shape[-1]))
        query = q.transpose(1, 2) * factor
        key = kv[:, :, 0].transpose(1, 2) * factor
        value = kv[:, :, 1].transpose(1, 2)
        scores = query @ key.transpose(-2, -1)
        probability = _FusedSafeSoftmax.apply(scores)
        return (probability @ value).transpose(1, 2), None
