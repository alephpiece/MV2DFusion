"""GPU contracts against independent installed MMCV forward/backward.

Run from the repository with one visible GPU:
  python tests/test_roi_bin_aggregation.py --output-json result.json
Optional --fixtures PATH... adds actual saved ROI/dy backward cases. Feature
values are zero in those cases because average backward does not read them;
the ordinary tests separately check nonzero native forward outputs.

FP32 atomic backward is screened against an independent native repeat with the
unchanged v4 bounds. A fixed absolute 1e-6 gate rejected native repetitions of
the small legacy/clamped-ROI case itself on BW150. That original gate remains
in each report; exact forward, fallback and nonfinite-position checks remain.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

import torch
from mmcv.ops.roi_align import RoIAlign, RoIAlignFunction


def native_empty_probe():
    result = {}
    x = torch.zeros((1, 2, 5, 5), device='cuda', requires_grad=True)
    rois = torch.empty((0, 5), device='cuda')
    try:
        y = RoIAlignFunction.apply(x, rois, (7, 7), 1., 0, 'avg', True)
        torch.cuda.synchronize()
        result['forward_supported'] = True
        result['output_shape'] = list(y.shape)
        try:
            dx, = torch.autograd.grad(y.sum(), (x,))
            torch.cuda.synchronize()
            result['backward_supported'] = True
            result['zero_gradient'] = bool(torch.equal(dx, torch.zeros_like(dx)))
        except Exception as exc:
            result.update(backward_supported=False, backward_error=repr(exc))
    except Exception as exc:
        result.update(forward_supported=False, forward_error=repr(exc))
    return result


if '--probe-native-empty' in sys.argv:
    print(json.dumps(native_empty_probe()), flush=True)
    raise SystemExit(0)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
policy_module = importlib.import_module('projects.mmdet3d_plugin.core.hook.roi_bin_aggregation')
kernel = importlib.import_module('projects.mmdet3d_plugin.ops.roi_align_bin_aggregate')
Policy = policy_module.RoIAlignBinAggregationPolicy
Hook = policy_module.RoIAlignBinAggregationHook
REPORT = []
FIXTURES = []


def metric(actual, reference):
    """Chunked Float64 norm; also compare exact nonfinite positions/signs."""
    same_patterns = True
    maximum = difference_sq = reference_sq = 0.
    for begin in range(0, actual.numel(), 1048576):
        a = actual.flatten()[begin:begin + 1048576].double()
        b = reference.flatten()[begin:begin + 1048576].double()
        for pattern in (torch.isfinite, torch.isnan, torch.isposinf, torch.isneginf):
            same_patterns &= bool(torch.equal(pattern(a), pattern(b)))
        valid = torch.isfinite(a) & torch.isfinite(b)
        if bool(valid.any()):
            a, b = a[valid], b[valid]
            difference = a - b
            maximum = max(maximum, float(difference.abs().max()))
            difference_sq += float(difference.square().sum())
            reference_sq += float(b.square().sum())
    relative = (difference_sq / max(reference_sq, 1e-300)) ** .5
    return dict(patterns_equal=same_patterns, max_abs=maximum, relative_l2=relative,
                passed=same_patterns and maximum <= 1e-6 and relative <= 1e-6)


def with_native_repeat(comparison, noise):
    """Apply the existing v4 screen, without treating native noise as failure.

    Bounds use only the independent same-input native repeat; candidate errors
    never calibrate their own tolerance. Preserve the former fixed-gate result
    for audit, including when a native repeat itself exceeds that fixed gate.
    """
    relative = max(1e-6, 10 * noise['relative_l2'])
    absolute = max(1e-6, 20 * noise['max_abs'])
    result = dict(comparison)
    result.update(fixed_gate_passed=comparison['passed'], native_repeat=noise,
                  numerical_policy='v4-native-repeat', relative_l2_bound=relative, max_abs_bound=absolute,
                  passed=(comparison['patterns_equal'] and noise['patterns_equal']
                          and comparison['relative_l2'] <= relative and comparison['max_abs'] <= absolute))
    return result


def native(module, inputs, rois):
    return RoIAlignFunction.apply(inputs, rois, module.output_size,
                                 module.spatial_scale, module.sampling_ratio,
                                 module.pool_mode, module.aligned)


class RepeatNoiseGateTest(unittest.TestCase):
    def test_noise_calibrates_only_finite_error_and_keeps_fixed_result(self):
        comparison = dict(patterns_equal=True, max_abs=1.2e-6, relative_l2=1.2e-7, passed=False)
        noise = dict(patterns_equal=True, max_abs=1.2e-6, relative_l2=1e-7, passed=False)
        result = with_native_repeat(comparison, noise)
        self.assertTrue(result['passed'])
        self.assertFalse(result['fixed_gate_passed'])
        self.assertEqual(result['max_abs_bound'], 20 * noise['max_abs'])
        self.assertEqual(result['relative_l2_bound'], 1e-6)
        self.assertFalse(with_native_repeat(dict(comparison, patterns_equal=False), noise)['passed'])
        self.assertFalse(with_native_repeat(comparison, dict(noise, patterns_equal=False))['passed'])
        self.assertFalse(with_native_repeat(dict(comparison, relative_l2=1.1e-6), noise)['passed'])
        quiet = dict(noise, max_abs=0., relative_l2=0., passed=True)
        self.assertFalse(with_native_repeat(comparison, quiet)['passed'])


class BindingsTest(unittest.TestCase):
    def test_ownership_close_and_hook(self):
        model = torch.nn.Module()
        model.selected = torch.nn.Sequential(RoIAlign(7, .25), RoIAlign(7, .125))
        model.unselected = RoIAlign(7, .25)
        original_class = RoIAlign.forward
        policy = Policy(model, scale_limits={.25: 2}, module_prefixes=('selected',))
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with policy:
                self.assertEqual(policy.module_names, ['selected.0', 'selected.1'])
                self.assertIn('forward', vars(model.selected[0]))
                self.assertNotIn('forward', vars(model.unselected))
                self.assertIs(RoIAlign.forward, original_class)
                with self.assertRaises(RuntimeError):
                    Policy(model, module_prefixes=('selected',)).install()
                raise RuntimeError('sentinel')
        self.assertFalse(policy.bindings)
        self.assertNotIn('forward', vars(model.selected[0]))
        policy.close()
        hook = Hook(module_prefixes=('selected',), scale_limits={.25: 2})
        runner = SimpleNamespace(model=model, logger=mock.Mock())
        hook.before_run(runner)
        self.assertIsNotNone(hook.policy)
        hook.after_run(runner)
        self.assertIsNone(hook.policy)
        hook.close()

    def test_close_does_not_overwrite_another_owner(self):
        model = torch.nn.Sequential(RoIAlign(7))
        policy = Policy(model).install()
        owned = model[0].forward
        model[0].forward = MethodType(lambda current, *args: None, model[0])
        with self.assertRaises(RuntimeError):
            policy.close()
        model[0].forward = owned
        policy.close()
        self.assertNotIn('forward', vars(model[0]))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA/HIP required')
class GPUContractsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(440)

    def compare(self, module, x, rois, dy, label, *, limit=4, scale_limits=None):
        reference = native(module, x, rois)
        reference_dx, = torch.autograd.grad(reference, (x,), dy)
        repeated = native(module, x, rois)
        repeated_dx, = torch.autograd.grad(repeated, (x,), dy)
        self.assertTrue(torch.equal(repeated, reference), label + ': native forward repeat changed')
        noise = metric(repeated_dx, reference_dx)
        del repeated, repeated_dx
        with Policy(module, limit=limit, scale_limits=scale_limits):
            actual = module(input=x, rois=rois)
            actual_dx, = torch.autograd.grad(actual, (x,), dy)
        self.assertTrue(torch.equal(actual, reference), label + ': native forward changed')
        self.assertEqual(actual.dtype, reference.dtype)
        self.assertEqual(actual_dx.dtype, x.dtype)
        result = with_native_repeat(metric(actual_dx, reference_dx), noise)
        REPORT.append(dict(case=label, **result))
        self.assertTrue(result['passed'], (label, result))

    def test_adaptive_values_gradients_layout_scale_and_roi_derivative(self):
        x = torch.randn((2, 7, 8, 9), device='cuda', requires_grad=True)
        rois = torch.tensor([[0, -2., -2., 6., 6.], [1, 0., 0., 9., 8.],
                             [0, 7., 6., 11., 10.], [1, 3., 3., 1., 1.]], device='cuda')
        for ratio, aligned in ((0, True), (-1, True), (0, False)):
            for scale in (.25, .125):
                module = RoIAlign(7, spatial_scale=scale, sampling_ratio=ratio, aligned=aligned)
                scaled = rois.clone()
                scaled[:, 1:] /= scale
                dy = torch.randn((4, 7, 7, 7), device='cuda').contiguous(memory_format=torch.channels_last)
                with mock.patch.object(kernel, 'backward', wraps=kernel.backward) as called:
                    self.compare(module, x, scaled, dy, f'adaptive-{ratio}-{aligned}-{scale}',
                                 scale_limits={.25: 2})
                    self.assertEqual(called.call_count, 1)
                    self.assertEqual(called.call_args.kwargs['limit'], 2 if scale == .25 else 4)
        differentiable_rois = rois.clone().requires_grad_(True)
        module = RoIAlign(7)
        with Policy(module):
            dx, dr = torch.autograd.grad(module(x, differentiable_rois).sum(),
                                        (x, differentiable_rois), allow_unused=True)
        self.assertIsNone(dr)
        self.assertTrue(bool(torch.isfinite(dx).all()))

    def test_nonfinite_gradients_preserve_positions(self):
        x = torch.randn((1, 7, 8, 8), device='cuda', requires_grad=True)
        rois = torch.tensor([[0, .25, .25, 7.75, 7.75], [0, .25, .25, 7.75, 7.75],
                             [0, 2., 2., 6., 6.], [0, 30., 30., 31., 31.]], device='cuda')
        for limit in (2, 4):
            for label, value in (('positive-inf', float('inf')), ('negative-inf', -float('inf')),
                                 ('nan', float('nan')), ('opposite', float('inf'))):
                dy = torch.randn((4, 7, 7, 7), device='cuda') * .01
                dy[0, 0, 0, 0] = value
                dy[3, 0, 0, 0] = float('nan')
                if label == 'opposite':
                    dy[1, 0, 0, 0] = -float('inf')
                self.compare(RoIAlign(7), x, rois, dy, f'nonfinite-{limit}-{label}', limit=limit)

    def test_half_double_max_and_fixed_use_original(self):
        for dtype, mode, ratio in ((torch.float16, 'avg', 0), (torch.float64, 'avg', 0),
                                   (torch.float32, 'max', 0), (torch.float32, 'avg', 4)):
            x = torch.randn((1, 2, 5, 5), device='cuda', dtype=dtype, requires_grad=True)
            rois = torch.tensor([[0, .25, .25, 4.75, 4.75]], dtype=dtype, device='cuda')
            module = RoIAlign(1, sampling_ratio=ratio, pool_mode=mode)
            reference = native(module, x, rois)
            expected, = torch.autograd.grad(reference.sum(), (x,))
            with Policy(module), mock.patch.object(kernel, 'backward', side_effect=AssertionError('fallback used aggregation')):
                actual = module(x, rois)
                result, = torch.autograd.grad(actual.sum(), (x,))
            self.assertTrue(torch.equal(actual, reference))
            self.assertTrue(torch.equal(result, expected))
            REPORT.append(dict(case=f'fallback-{dtype}-{mode}-{ratio}', passed=True))

    def test_no_grad_no_input_grad_and_export_routes(self):
        module = RoIAlign(1)
        x = torch.randn((1, 2, 5, 5), device='cuda', requires_grad=True)
        rois = torch.tensor([[0, 0., 0., 4., 4.]], device='cuda')
        with Policy(module), mock.patch.object(policy_module._AggregatedRoIAlign, 'apply',
                                               side_effect=AssertionError('ineligible custom forward')):
            with torch.no_grad():
                self.assertFalse(module(x, rois).requires_grad)
            self.assertFalse(module(x.detach(), rois).requires_grad)
            for owner, name in ((torch.onnx, 'is_in_onnx_export'),
                                (torch.jit, 'is_tracing'), (torch.jit, 'is_scripting')):
                with mock.patch.object(owner, name, return_value=True):
                    actual = module(x, rois)
                self.assertTrue(torch.equal(actual, native(module, x, rois)))

    def test_empty_contract_and_independent_native_probe(self):
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-native-empty'],
                                   capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
        REPORT.append(dict(case='installed-empty-probe', observed=observed, passed=True))
        module = RoIAlign(7)
        x = torch.randn((1, 7, 8, 8), device='cuda', requires_grad=True)
        rois = torch.empty((0, 5), device='cuda')
        with Policy(module):
            y = module(x, rois)
            dx, = torch.autograd.grad(y.sum(), (x,))
        self.assertEqual(tuple(y.shape), (0, 7, 7, 7))
        self.assertTrue(torch.equal(dx, torch.zeros_like(x)))
        REPORT.append(dict(case='candidate-empty', passed=True))

    def test_nondefault_stream(self):
        module = RoIAlign(7)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn((2, 7, 16, 16), device='cuda', requires_grad=True)
            rois = torch.tensor([[0, .3, .4, 12.2, 13.1], [1, 1.1, 1.2, 15., 15.]], device='cuda')
            dy = torch.randn((2, 7, 7, 7), device='cuda')
            reference = native(module, x, rois)
            expected, = torch.autograd.grad(reference, (x,), dy)
            repeated = native(module, x, rois)
            repeated_dx, = torch.autograd.grad(repeated, (x,), dy)
            with Policy(module):
                actual = module(x, rois)
                result, = torch.autograd.grad(actual, (x,), dy)
            finished = torch.cuda.Event()
            finished.record()
        torch.cuda.current_stream().wait_event(finished)
        self.assertTrue(torch.equal(actual, reference))
        self.assertTrue(torch.equal(repeated, reference))
        comparison = with_native_repeat(metric(result, expected), metric(repeated_dx, expected))
        REPORT.append(dict(case='nondefault-stream', **comparison))
        self.assertTrue(comparison['passed'])

    def test_optional_real_gradient_fixtures(self):
        if not FIXTURES:
            self.skipTest('no actual ROI/dy fixtures supplied')
        for path in FIXTURES:
            fixture = torch.load(path, map_location='cpu', weights_only=True)
            params = fixture['parameters']
            x = torch.zeros(fixture['input_shape'], device='cuda', dtype=torch.float32, requires_grad=True)
            rois, dy = fixture['rois'].cuda(), fixture['grad_output'].cuda()
            for limit in (2, 4):
                module = RoIAlign(params['output_size'], params['spatial_scale'],
                                  params['sampling_ratio'], params['pool_mode'], params['aligned'])
                self.compare(module, x, rois, dy, f'{Path(path).name}-limit{limit}', limit=limit)
            del x, rois, dy, fixture
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--fixtures', type=Path, nargs='*', default=[])
    args, unittest_args = parser.parse_known_args()
    if args.output_json and args.output_json.exists():
        raise FileExistsError(args.output_json)
    FIXTURES[:] = args.fixtures
    result = unittest.main(argv=[sys.argv[0], *unittest_args], exit=False).result
    source = Path(inspect.getsourcefile(RoIAlignFunction))
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  skipped=[reason for _, reason in result.skipped], torch=torch.__version__,
                  numerical_policy='v4-native-repeat; original fixed gate retained per comparison',
                  native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  application_source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(policy_module.__file__), Path(kernel.__file__), Path(__file__).resolve())},
                  checks=REPORT)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
