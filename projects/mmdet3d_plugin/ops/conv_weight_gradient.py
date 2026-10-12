"""First-order image Conv2d weight gradients for selected Half-input specifications."""
import torch
import triton
import triton.language as tl



@triton.jit
def _unfold_3x3(x, column, batch: tl.constexpr, channels: tl.constexpr,
                             height: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    spatial = height * width
    reduction = batch * spatial
    sample_channel = tl.program_id(1)
    sample = sample_channel // channels
    channel = sample_channel % channels
    position = tl.program_id(0) * block + tl.arange(0, block)
    row, col = position // width, position % width
    # One spatial tile processes all taps, keeping nearby input reads together
    # instead of revisiting the full activation separately for each tap.
    for tap in tl.static_range(9):
        input_row, input_col = row + tap // 3 - 1, col + tap % 3 - 1
        value = tl.load(x + sample_channel * spatial + input_row * width + input_col,
                        mask=(position < spatial) & (input_row >= 0) & (input_row < height) &
                             (input_col >= 0) & (input_col < width), other=0)
        tl.store(column + (channel * 9 + tap) * reduction + sample * spatial + position,
                 value, mask=position < spatial)


@triton.jit
def _pointwise_partials(x, dy, partial, cin: tl.constexpr, cout: tl.constexpr,
                   spatial: tl.constexpr, splits: tl.constexpr,
                   block_m: tl.constexpr, block_n: tl.constexpr,
                   block_k: tl.constexpr):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    cin_tiles = tl.cdiv(cin, block_n)
    o = tile // cin_tiles * block_m + tl.arange(0, block_m)
    c = tile % cin_tiles * block_n + tl.arange(0, block_n)
    batch = part // splits
    spatial_part = part % splits
    span = tl.cdiv(spatial, splits)
    begin = spatial_part * span
    accum = tl.full((block_m, block_n), 0, tl.float32)
    for offset in range(0, span, block_k):
        k = begin + offset + tl.arange(0, block_k)
        a = tl.load(dy + batch * cout * spatial + o[:, None] * spatial + k[None, :],
                    mask=(o[:, None] < cout) & (k[None, :] < spatial) &
                         (k[None, :] < begin + span), other=0)
        b = tl.load(x + batch * cin * spatial + c[None, :] * spatial + k[:, None],
                    mask=(c[None, :] < cin) & (k[:, None] < spatial) &
                         (k[:, None] < begin + span), other=0)
        accum += tl.dot(a, b)
    tl.store(partial + part * cout * cin + o[:, None] * cin + c[None, :],
             accum, mask=(o[:, None] < cout) & (c[None, :] < cin))


@triton.jit
def _sum_partials(partial, output, elements: tl.constexpr, parts: tl.constexpr,
                  block_parts: tl.constexpr, block_output: tl.constexpr):
    element = tl.program_id(0) * block_output + tl.arange(0, block_output)
    part = tl.arange(0, block_parts)
    values = tl.load(partial + part[:, None] * elements + element[None, :],
                     mask=(part[:, None] < parts) & (element[None, :] < elements),
                     other=0)
    total = tl.sum(values, axis=0)
    tl.store(output + element, total.to(output.dtype.element_ty), mask=element < elements)


@triton.jit
def _conv3x3_partials(x, dy, partial, cin: tl.constexpr, cout: tl.constexpr,
                             height: tl.constexpr, width: tl.constexpr, splits: tl.constexpr,
                             block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    tile, part, tap = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    spatial = height * width
    cin_tiles = tl.cdiv(cin, block_n)
    o = tile // cin_tiles * block_m + tl.arange(0, block_m)
    c = tile % cin_tiles * block_n + tl.arange(0, block_n)
    sample, spatial_part = part // splits, part % splits
    span = tl.cdiv(spatial, splits * block_k) * block_k
    begin = spatial_part * span
    accumulator = tl.full((block_m, block_n), 0, tl.float32)
    for offset in range(0, span, block_k):
        k = begin + offset + tl.arange(0, block_k)
        row, col = k // width + tap // 3 - 1, k % width + tap % 3 - 1
        a = tl.load(dy + (sample * cout + o[:, None]) * spatial + k[None, :],
                    mask=(o[:, None] < cout) & (k[None, :] < spatial), other=0)
        b = tl.load(x + (sample * cin + c[None, :]) * spatial + row[:, None] * width + col[:, None],
                    mask=(c[None, :] < cin) & (k[:, None] < spatial) &
                         (row[:, None] >= 0) & (row[:, None] < height) &
                         (col[:, None] >= 0) & (col[:, None] < width), other=0)
        accumulator += tl.dot(a, b)
    tl.store(partial + (tap * tl.num_programs(1) + part) * cout * cin +
             o[:, None] * cin + c[None, :], accumulator,
             mask=(o[:, None] < cout) & (c[None, :] < cin))


@triton.jit
def _conv3x3_sum(partial, output, elements: tl.constexpr, parts: tl.constexpr,
                            block_parts: tl.constexpr, block_output: tl.constexpr):
    e = tl.program_id(0) * block_output + tl.arange(0, block_output)
    tap, oc = e // elements, e % elements
    p = tl.arange(0, block_parts)
    value = tl.load(partial + (tap[None, :] * parts + p[:, None]) * elements + oc[None, :],
                    mask=(p[:, None] < parts) & (e[None, :] < 9 * elements), other=0)
    total = tl.sum(value, axis=0)
    tl.store(output + oc * 9 + tap, total.to(tl.float16), mask=e < 9 * elements)


@triton.jit
def _stem_partials(x, dy, partial, batch: tl.constexpr, cin: tl.constexpr,
                 cout: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
                 out_height: tl.constexpr, out_width: tl.constexpr,
                 splits: tl.constexpr, block_m: tl.constexpr,
                 block_n: tl.constexpr, block_k: tl.constexpr):
    spatial = out_height * out_width
    reduction = batch * spatial
    filter_width = cin * 49
    o = tl.program_id(0) * block_m + tl.arange(0, block_m)
    q = tl.program_id(1) * block_n + tl.arange(0, block_n)
    channel, tap = q // 49, q % 49
    part = tl.program_id(2)
    span = tl.cdiv(reduction, splits * block_k) * block_k
    accumulator = tl.full((block_m, block_n), 0, tl.float32)
    for offset in range(0, span, block_k):
        k = part * span + offset + tl.arange(0, block_k)
        sample, position = k // spatial, k % spatial
        row = (position[:, None] // out_width) * 2 + tap[None, :] // 7 - 3
        col = (position[:, None] % out_width) * 2 + tap[None, :] % 7 - 3
        a = tl.load(dy + (sample[None, :] * cout + o[:, None]) * spatial + position[None, :],
                    mask=(o[:, None] < cout) & (k[None, :] < reduction), other=0)
        b = tl.load(x + (sample[:, None] * cin + channel[None, :]) * height * width +
                    row * width + col,
                    mask=(channel[None, :] < cin) & (k[:, None] < reduction) &
                         (row >= 0) & (row < height) & (col >= 0) & (col < width), other=0)
        accumulator += tl.dot(a, b)
    tl.store(partial + (part * cout + o[:, None]) * filter_width + q[None, :],
             accumulator, mask=(o[:, None] < cout) & (q[None, :] < filter_width))


def _check_inputs(inputs, gradient, *, stem=False):
    if (inputs.ndim != 4 or gradient.ndim != 4 or inputs.device.type != 'cuda'
            or gradient.device != inputs.device
            or inputs.dtype != torch.float16 or gradient.dtype != torch.float16
            or not inputs.is_contiguous() or not gradient.is_contiguous()
            or inputs.shape[0] != gradient.shape[0]):
        raise ValueError('weight gradient requires contiguous CUDA/HIP Half operands')
    spatial = tuple((size + 1) // 2 for size in inputs.shape[2:]) if stem else inputs.shape[2:]
    if gradient.shape[2:] != spatial:
        raise ValueError('weight-gradient operand spatial shapes differ')


def unfold_weight_gradient(inputs, gradient):
    """Padded, stride-one 3x3: spatially grouped unfold and joint-batch GEMM."""
    _check_inputs(inputs, gradient)
    batch, channels, height, width = inputs.shape
    column = torch.empty((channels * 9, batch * height * width),
                         device=inputs.device, dtype=inputs.dtype)
    _unfold_3x3[(triton.cdiv(height * width, 1024), batch * channels)](
        inputs, column, batch, channels, height, width, 1024, num_warps=4)
    dy = gradient.permute(1, 0, 2, 3).reshape(gradient.shape[1], -1)
    return torch.mm(dy, column.t()).reshape(gradient.shape[1], channels, 3, 3)


def pointwise_weight_gradient(inputs, gradient, parts, *, output_dtype=torch.float16):
    """Direct 1x1 with FP32 partials and a single final output conversion."""
    _check_inputs(inputs, gradient)
    if output_dtype not in (torch.float16, torch.float32):
        raise ValueError('unsupported weight-gradient output dtype')
    batch, cin, height, width = inputs.shape
    cout = gradient.shape[1]
    partial = torch.empty((batch * parts, cout, cin), dtype=torch.float32, device=inputs.device)
    output = torch.empty((cout, cin, 1, 1), dtype=output_dtype, device=inputs.device)
    block_m = 16 if cout < 32 else 32
    _pointwise_partials[(triton.cdiv(cout, block_m) * triton.cdiv(cin, 64), batch * parts)](
        inputs, gradient, partial, cin, cout, height * width, parts,
        block_m, 64, 64, num_warps=4)
    _sum_partials[(triton.cdiv(cout * cin, 32),)](
        partial, output, cout * cin, batch * parts,
        triton.next_power_of_2(batch * parts), 32, num_warps=4)
    return output


def pointwise_gemm_weight_gradient(inputs, gradient):
    """Joint-batch 1x1 GEMM for the selected short-reduction calls."""
    _check_inputs(inputs, gradient)
    dy = gradient.permute(1, 0, 2, 3).reshape(gradient.shape[1], -1)
    x = inputs.permute(1, 0, 2, 3).reshape(inputs.shape[1], -1)
    return torch.mm(dy, x.t()).reshape(gradient.shape[1], inputs.shape[1], 1, 1)


def conv3x3_weight_gradient(inputs, gradient):
    """Per-sample, per-tap 3x3 partials; reduction also restores weight layout."""
    _check_inputs(inputs, gradient)
    batch, cin, height, width = inputs.shape
    cout, parts = gradient.shape[1], 2
    partial = torch.empty((9, batch * parts, cout, cin), dtype=torch.float32, device=inputs.device)
    output = torch.empty((cout, cin, 3, 3), dtype=torch.float16, device=inputs.device)
    _conv3x3_partials[(triton.cdiv(cout, 64) * triton.cdiv(cin, 64), batch * parts, 9)](
        inputs, gradient, partial, cin, cout, height, width, parts, 64, 64, 64, num_warps=4)
    _conv3x3_sum[(triton.cdiv(output.numel(), 64),)](
        partial, output, cout * cin, batch * parts,
        triton.next_power_of_2(batch * parts), 64, num_warps=4)
    return output


def stem_weight_gradient(inputs, gradient):
    """7x7/stride2/padding3 with joint-batch partials and shared reduction."""
    _check_inputs(inputs, gradient, stem=True)
    batch, cin, height, width = inputs.shape
    cout, oh, ow = gradient.shape[1:]
    parts = 64
    partial = torch.empty((parts, cout, cin * 49), dtype=torch.float32, device=inputs.device)
    output = torch.empty((cout, cin, 7, 7), dtype=torch.float16, device=inputs.device)
    _stem_partials[(triton.cdiv(cout, 64), triton.cdiv(cin * 49, 64), parts)](
        inputs, gradient, partial, batch, cin, cout, height, width, oh, ow,
        parts, 64, 64, 64, num_warps=4)
    _sum_partials[(triton.cdiv(output.numel(), 128),)](
        partial, output, output.numel(), parts, triton.next_power_of_2(parts), 128, num_warps=4)
    return output
