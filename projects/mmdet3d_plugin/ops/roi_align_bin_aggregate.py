"""FP32 adaptive average ROIAlign backward with bounded same-bin aggregation.

Coordinates, weights and atomic accumulation stay FP32. FMA matches the
installed kernel's arithmetic more closely. Nonfinite-gradient tiles and
unsupported sampling geometries retain native per-sample scatter. Aggregating
finite contributions changes reduction rounding; it is not bitwise equality.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _axis(position, extent: tl.constexpr):
    valid = (position >= -1.) & (position <= extent)
    position = tl.maximum(position, 0.)
    low = tl.minimum(position.to(tl.int32), extent - 1)
    high = tl.minimum(low + 1, extent - 1)
    position = tl.where(low >= extent - 1, low.to(tl.float32), position)
    fraction = position - low
    return low, high, 1. - fraction, fraction, valid


@triton.jit
def _position(start, bin_index, bin_size, sample, grid):
    # Keep original expression ordering. Fusion is controlled by launch option.
    return start + bin_index * bin_size + tl.div_rn((sample + .5) * bin_size, grid.to(tl.float32))


@triton.jit
def _scatter_native(DX, grad, base, ph, pw, sh, sw, bh, bw, gh, gw,
                    valid, HEIGHT: tl.constexpr, WIDTH: tl.constexpr):
    count = (gh * gw).to(tl.float32)
    for iy in range(gh):
        y = _position(sh, ph, bh, iy, gh)
        yl, yh, hy, ly, vy = _axis(y, HEIGHT)
        for ix in range(gw):
            x = _position(sw, pw, bw, ix, gw)
            xl, xh, hx, lx, vx = _axis(x, WIDTH)
            mask = valid & vy & vx
            tl.atomic_add(DX + base + yl * WIDTH + xl, tl.div_rn(grad * (hy * hx), count), mask, sem='relaxed')
            tl.atomic_add(DX + base + yl * WIDTH + xh, tl.div_rn(grad * (hy * lx), count), mask, sem='relaxed')
            tl.atomic_add(DX + base + yh * WIDTH + xl, tl.div_rn(grad * (ly * hx), count), mask, sem='relaxed')
            tl.atomic_add(DX + base + yh * WIDTH + xh, tl.div_rn(grad * (ly * lx), count), mask, sem='relaxed')


@triton.jit
def _axis_weight(pixel, start, bin_index, bin_size, grid, EXTENT: tl.constexpr, LIMIT: tl.constexpr):
    weight = tl.full(pixel.shape, 0., tl.float32)
    for sample in tl.static_range(LIMIT):
        position = _position(start, bin_index, bin_size, sample, grid)
        low, high, wl, wh, inside = _axis(position, EXTENT)
        value = tl.where(pixel == low, wl, 0.) + tl.where(pixel == high, wh, 0.)
        weight += tl.where((sample < grid) & inside, value, 0.)
    return weight


@triton.jit
def _scatter_aggregate(DX, grad, base, ph, pw, sh, sw, bh, bw, gh, gw,
                       valid, HEIGHT: tl.constexpr, WIDTH: tl.constexpr, LIMIT: tl.constexpr):
    first_y = _position(sh, ph, bh, 0, gh)
    first_x = _position(sw, pw, bw, 0, gw)
    y0 = tl.minimum(tl.maximum(first_y, 0.).to(tl.int32), HEIGHT - 1)
    x0 = tl.minimum(tl.maximum(first_x, 0.).to(tl.int32), WIDTH - 1)
    # Fixed scalar vectors let the compiler retain small grids in registers.
    # Unused terms for LIMIT=2 are dead after static scalar selection.
    wy0 = _axis_weight(y0, sh, ph, bh, gh, HEIGHT, LIMIT)
    wy1 = _axis_weight(y0 + 1, sh, ph, bh, gh, HEIGHT, LIMIT)
    wy2 = _axis_weight(y0 + 2, sh, ph, bh, gh, HEIGHT, LIMIT)
    wy3 = _axis_weight(y0 + 3, sh, ph, bh, gh, HEIGHT, LIMIT)
    wy4 = _axis_weight(y0 + 4, sh, ph, bh, gh, HEIGHT, LIMIT)
    wx0 = _axis_weight(x0, sw, pw, bw, gw, WIDTH, LIMIT)
    wx1 = _axis_weight(x0 + 1, sw, pw, bw, gw, WIDTH, LIMIT)
    wx2 = _axis_weight(x0 + 2, sw, pw, bw, gw, WIDTH, LIMIT)
    wx3 = _axis_weight(x0 + 3, sw, pw, bw, gw, WIDTH, LIMIT)
    wx4 = _axis_weight(x0 + 4, sw, pw, bw, gw, WIDTH, LIMIT)
    count = (gh * gw).to(tl.float32)
    for iy in tl.static_range(LIMIT + 1):
        # The installed DTK frontend rejects tuple[constexpr] here. Explicit
        # static selection keeps these vectors separate through specialization.
        if iy == 0:
            weight_y = wy0
        elif iy == 1:
            weight_y = wy1
        elif iy == 2:
            weight_y = wy2
        elif iy == 3:
            weight_y = wy3
        else:
            weight_y = wy4
        for ix in tl.static_range(LIMIT + 1):
            if ix == 0:
                weight_x = wx0
            elif ix == 1:
                weight_x = wx1
            elif ix == 2:
                weight_x = wx2
            elif ix == 3:
                weight_x = wx3
            else:
                weight_x = wx4
            # Zero-weight destinations have no contribution for finite grad.
            mask = valid & (y0 + iy < HEIGHT) & (x0 + ix < WIDTH) & (weight_y != 0.) & (weight_x != 0.)
            # Normalize the nonnegative aggregate coefficient before grad so
            # repeated samples do not create an avoidable grad*count overflow.
            # This additional reassociation is part of the numerical gate.
            value = grad * tl.div_rn(weight_y * weight_x, count)
            tl.atomic_add(DX + base + (y0 + iy) * WIDTH + x0 + ix, value, mask, sem='relaxed')


@triton.jit
def _backward(DY, ROIS, DX, CHANNELS: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
              PH: tl.constexpr, PW: tl.constexpr, SCALE: tl.constexpr, RATIO: tl.constexpr,
              ALIGNED: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    roi = tl.program_id(1)
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < CHANNELS * PH * PW
    channel = index // (PH * PW)
    ph, pw = index // PW % PH, index % PW
    grad = tl.load(DY + roi * CHANNELS * PH * PW + index, valid, 0.)
    batch = tl.load(ROIS + roi * 5).to(tl.int32)
    offset = .5 if ALIGNED else 0.
    sw = tl.load(ROIS + roi * 5 + 1) * SCALE - offset
    sh = tl.load(ROIS + roi * 5 + 2) * SCALE - offset
    ew = tl.load(ROIS + roi * 5 + 3) * SCALE - offset
    eh = tl.load(ROIS + roi * 5 + 4) * SCALE - offset
    rh, rw = eh - sh, ew - sw
    if not ALIGNED:
        rh, rw = tl.maximum(rh, 1.), tl.maximum(rw, 1.)
    bh, bw = tl.div_rn(rh, PH * 1.), tl.div_rn(rw, PW * 1.)
    if RATIO > 0:
        gh, gw = tl.full((), RATIO, tl.int32), tl.full((), RATIO, tl.int32)
    else:
        gh, gw = tl.ceil(bh).to(tl.int32), tl.ceil(bw).to(tl.int32)
    base = (batch * CHANNELS + channel) * HEIGHT * WIDTH
    # Restrict the rearranged accumulation to actual adaptive sampling.
    # Fixed grids permit arbitrarily dense/repeated samples, including
    # zero-area ROI where aggregating 16 same-point updates changed finite
    # results beyond the unchanged screen. Preserve their native scatter.
    eligible = (RATIO <= 0) & (gh > 0) & (gw > 0) & (gh <= LIMIT) & (gw <= LIMIT) & (gh * gw > 1)
    eligible &= (bh > 0.) & (bw > 0.) & (bh <= gh) & (bw <= gw)
    if eligible:
        # Rounded positions extremely close to integer boundaries can
        # exceed the real-arithmetic grid+1 bound by one address. Retain
        # the original scatter for that whole tile rather than truncating.
        yl0, _, _, _, _ = _axis(_position(sh, ph, bh, 0, gh), HEIGHT)
        _, yhl, _, _, _ = _axis(_position(sh, ph, bh, gh - 1, gh), HEIGHT)
        xl0, _, _, _, _ = _axis(_position(sw, pw, bw, 0, gw), WIDTH)
        _, xhl, _, _, _ = _axis(_position(sw, pw, bw, gw - 1, gw), WIDTH)
        overflow = tl.sum((valid & ((yhl - yl0 > LIMIT) | (xhl - xl0 > LIMIT))).to(tl.int32), 0)
        # Native emits all four sample weights, including 0*Inf -> NaN.
        # A tile with any nonfinite gradient must preserve that path.
        nonfinite = tl.sum((valid & ~((grad == grad) & (tl.abs(grad) < float('inf')))).to(tl.int32), 0)
        eligible = (overflow == 0) & (nonfinite == 0)
    if eligible:
        _scatter_aggregate(DX, grad, base, ph, pw, sh, sw, bh, bw, gh, gw,
                           valid, HEIGHT, WIDTH, LIMIT)
    else:
        _scatter_native(DX, grad, base, ph, pw, sh, sw, bh, bw, gh, gw, valid, HEIGHT, WIDTH)


def backward(grad_output, rois, input_shape, output_size, spatial_scale,
             sampling_ratio=0, aligned=True, limit=4):
    """Return FP32 dX on the caller's current stream, with no host data reads.

    Only average adaptive sampling is accepted here. The application policy
    delegates other contracts to the original MMCV function. ROI geometry and
    batch indices obey the native forward contract; NaN/Inf gradients are valid.
    """
    if (grad_output.dtype != torch.float32 or rois.dtype != torch.float32
            or not grad_output.is_cuda or rois.device != grad_output.device):
        raise ValueError('aggregation requires FP32 CUDA/HIP gradients and ROIs')
    if limit not in (2, 4) or sampling_ratio > 0:
        raise ValueError('aggregation requires adaptive sampling and limit 2 or 4')
    ph, pw = output_size
    if (grad_output.ndim != 4 or rois.shape != (grad_output.shape[0], 5)
            or tuple(grad_output.shape[2:]) != (ph, pw)):
        raise ValueError('inconsistent ROI gradient shapes')
    _, channels, height, width = input_shape
    if channels != grad_output.shape[1] or min(channels, height, width, ph, pw) <= 0:
        raise ValueError('invalid feature shape')
    with torch.cuda.device(grad_output.device):
        dx = grad_output.new_zeros(input_shape)
        if rois.shape[0] == 0:
            return dx
        grad_output, rois = grad_output.contiguous(), rois.contiguous()
        block = 256
        _backward[(triton.cdiv(channels * ph * pw, block), rois.shape[0])](
            grad_output, rois, dx, channels, height, width, ph, pw, float(spatial_scale),
            int(sampling_ratio), bool(aligned), limit, block,
            num_warps=4, enable_fp_fusion=True)
    return dx
