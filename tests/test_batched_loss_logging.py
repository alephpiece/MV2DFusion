"""Loss/gradient and collective contracts against installed BaseDetector."""
import ast
from collections import OrderedDict
from contextlib import contextmanager
from datetime import timedelta
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'logging_under_test', ROOT / 'projects/mmdet3d_plugin/core/batched_loss_logging.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def native_parser():
    # Read the installed oracle without importing model or CUDA extensions.
    package = Path(importlib.util.find_spec('mmdet').origin).parent
    source = package / 'models/detectors/base.py'
    tree = ast.parse(source.read_text())
    detector = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == 'BaseDetector')
    function = next(n for n in detector.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_parse_losses')
    function.decorator_list = []
    namespace = dict(torch=torch, dist=dist, OrderedDict=OrderedDict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['_parse_losses']


@contextmanager
def collectives():
    original = dist.all_reduce
    calls = []
    def counted(value, *args, **kwargs):
        calls.append((tuple(value.shape), value.dtype))
        return original(value, *args, **kwargs)
    dist.all_reduce = counted
    try:
        yield calls
    finally:
        dist.all_reduce = original


def error(function, losses):
    try:
        function(None, losses)
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError('expected invalid input to fail')


def compare(native, losses, parameters):
    originals = [p.detach().clone() for p in parameters]
    with collectives() as a_calls:
        a, a_logs = native(None, losses)
    with collectives() as b_calls:
        b, b_logs = candidate.batched_parse_losses(None, losses)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert list(a_logs) == list(b_logs) and a_logs == b_logs
    da = torch.autograd.grad(a, parameters, retain_graph=True, allow_unused=True)
    db = torch.autograd.grad(b, parameters, retain_graph=True, allow_unused=True)
    for left, right in zip(da, db):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    for value, original in zip(parameters, originals):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
        assert value.grad is None
    assert len(a_calls) == 1 + len(a_logs)
    assert len(b_calls) == 1 + len({dtype for _, dtype in a_calls[1:]})


def distributed_worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        native = native_parser()
        p = torch.nn.Parameter(torch.tensor([.125 + rank, -2.25, .33]))
        q = torch.nn.Parameter(torch.tensor([.2 + rank, -1.125], dtype=torch.float64))
        a, b = p.square(), q.square()
        compare(native, OrderedDict(loss_main=a, accuracy=p.detach().mean(),
                                     aux_loss=[a, b], temperature=q.detach().sum()), (p, q))
        compare(native, OrderedDict(loss=a, accuracy=p.detach().mean(),
                                     aux_loss=[b]), (p, q))
        # Both ranks must reject count disagreement before per-value reductions.
        mismatch = OrderedDict(loss_main=a)
        if rank == 0:
            mismatch['accuracy'] = p.detach().mean()
        left = error(native, mismatch)
        dist.barrier()
        right = error(candidate.batched_parse_losses, mismatch)
        assert left == right and left[0] is AssertionError
        dist.barrier()
        for invalid in (OrderedDict(), OrderedDict(loss_main=[]),
                        OrderedDict(loss_main=a, accuracy=[]),
                        OrderedDict(loss_bad='invalid')):
            left = error(native, invalid)
            dist.barrier()
            right = error(candidate.batched_parse_losses, invalid)
            assert left == right, (left, right)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class BatchedLoggingTest(unittest.TestCase):
    def test_two_rank_values_gradients_and_error_boundaries(self):
        with tempfile.TemporaryDirectory(prefix='mv2d-logging-') as folder:
            mp.spawn(distributed_worker, args=(str(Path(folder) / 'store'),),
                     nprocs=2, join=True)

    def test_local_error_boundaries(self):
        native = native_parser()
        for invalid in ({'accuracy': torch.tensor([.5])}, {'loss_main': []}, {}):
            self.assertEqual(error(native, invalid),
                             error(candidate.batched_parse_losses, invalid))


if __name__ == '__main__':
    unittest.main()
