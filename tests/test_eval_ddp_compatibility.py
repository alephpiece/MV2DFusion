"""Exercise the evaluation DDP bridge with a real CPU process group."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from mmcv.parallel import MMDistributedDataParallel as NativeMMDDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.mmdet3d_plugin.core.parallel import MMDistributedDataParallel


class EvaluationDDPTest(unittest.TestCase):
    def test_cpu_forward_and_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('gloo', rank=0, world_size=1,
                                    init_method='file://' + str(Path(directory) / 'group'))
            try:
                model = torch.nn.Linear(3, 2)
                reference = copy.deepcopy(model)
                wrapped = MMDistributedDataParallel(model)
                inputs = torch.randn(4, 3)
                actual = wrapped(inputs)
                expected = reference(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                expected.sum().backward()
                for left, right in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            finally:
                dist.destroy_process_group()

    def test_existing_flag_is_preserved(self):
        def initialize_with_existing_flag(owner):
            owner._use_replicated_tensor_module = True

        with mock.patch.object(NativeMMDDP, '__init__', initialize_with_existing_flag):
            self.assertTrue(MMDistributedDataParallel()._use_replicated_tensor_module)


if __name__ == '__main__':
    unittest.main()
