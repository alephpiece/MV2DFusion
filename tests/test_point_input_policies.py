"""CPU checks of point policies at the target and voxelization boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from projects.fsdv2.single_stage_fsd import VoteSegmentor
from projects.fsdv2.single_stage_fsd_v2 import SingleStageFSDV2


class BoundaryReached(Exception):
    def __init__(self, points, batch_indices=None):
        self.points = points
        self.batch_indices = batch_indices


def stop_at_boundary(points, *unused):
    raise BoundaryReached(points)


class SegmentorStub(torch.nn.Module):
    filter_valid_points = VoteSegmentor.filter_valid_points
    forward_train = VoteSegmentor.forward_train
    simple_test = VoteSegmentor.simple_test

    def __init__(self, enabled=True, downsample=False):
        super().__init__()
        self.filter_invalid_points = enabled
        self.tanh_dims = []
        self.voxel_downsampling_size = [1, 1, 1] if downsample else None
        self.use_multiscale_features = False
        self.segmentation_head = SimpleNamespace(get_targets=stop_at_boundary)
        self.extract_feat = stop_at_boundary

    def voxel_layer(self, points):
        # A CPU voxelizer stand-in: a negative coordinate marks an invalid row.
        return points[:, :3].floor().to(torch.int32)

    def voxel_downsample(self, points):
        return [sample[::2] for sample in points]


class PointInputPoliciesTest(unittest.TestCase):
    def test_filter_preserves_rows_fields_batch_positions_and_input(self):
        points = [
            torch.tensor([[1., 2., 3., 91., 11.], [-1., 2., 3., 92., 12.],
                          [4., 5., 6., 93., 13.]]),
            torch.tensor([[2., -1., 3., 81., 21.], [6., 7., 8., 82., 22.]]),
        ]
        originals = [sample.clone() for sample in points]
        selected = SegmentorStub().filter_valid_points(points)
        self.assertEqual(len(selected), 2)
        for actual, expected in zip(selected, [points[0][[0, 2]], points[1][[1]]]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, original in zip(points, originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_filter_rejects_empty_or_fully_invalid_sample_without_renumbering(self):
        valid = torch.tensor([[1., 2., 3., 4.]])
        cases = [(torch.empty(0, 4), 'sample 1 has no input points'),
                 (torch.tensor([[1., -1., 2., 4.], [1., 2., -1., 5.]]),
                  'sample 1 has no valid points after voxelization')]
        for invalid, message in cases:
            with self.subTest(message=message):
                original = invalid.clone()
                with self.assertRaisesRegex(ValueError, message):
                    SegmentorStub().filter_valid_points([valid, invalid])
                torch.testing.assert_close(invalid, original, rtol=0, atol=0)

    def test_training_and_inference_filter_after_downsample_before_consumers(self):
        points = [torch.tensor([[1., 2., 3., 10.], [2., 3., 4., 20.],
                                [3., -1., 4., 30.], [4., 5., 6., 40.],
                                [5., 6., 7., 50.]])]
        original = points[0].clone()
        for training in (True, False):
            for enabled in (True, False):
                with self.subTest(training=training, enabled=enabled):
                    segmentor = SegmentorStub(enabled, downsample=True)
                    with self.assertRaises(BoundaryReached) as caught:
                        if training:
                            segmentor.forward_train(points, [{}], [], [])
                        else:
                            segmentor.simple_test(points, [{}])
                    expected = original[[0, 4] if enabled else [0, 2, 4]]
                    torch.testing.assert_close(caught.exception.points[0], expected,
                                               rtol=0, atol=0)
                    torch.testing.assert_close(points[0], original, rtol=0, atol=0)

    def test_raw_clipping_is_optional_but_predicted_centers_always_clip(self):
        class MixerStub:
            extract_feat = SingleStageFSDV2.extract_feat
            clip_points = SingleStageFSDV2.clip_points
            baseline_mode = False
            zero_virtual_feature = False
            point_cloud_range = [0., 0., 0., 10., 10., 10.]
            virtual_proj = staticmethod(lambda value: value[:, :1])
            ori_proj = staticmethod(lambda value: value)

            def voxelize_with_batch_idx(self, points, batch_indices):
                raise BoundaryReached(points, batch_indices)

        raw = torch.tensor([[-2., 5., 12., 99.], [4., 3., 2., 88.]])
        centers = torch.tensor([[12., -2., 5.]])
        for enabled in (True, False):
            with self.subTest(clip_raw_points=enabled):
                mixer = MixerStub()
                mixer.clip_raw_points = enabled
                origin = dict(seg_points=raw.clone(), seg_feats=torch.ones(2, 1),
                              batch_idx=torch.tensor([0, 1]))
                sampled = dict(seg_points=torch.tensor([[1., 2., 3., 77.]]),
                               center_preds=centers.clone(), seg_logits=torch.ones(1, 1),
                               seg_feats=torch.ones(1, 1), batch_idx=torch.tensor([1]))
                with self.assertRaises(BoundaryReached) as caught:
                    mixer.extract_feat(sampled, origin)
                expected_raw = raw.clone()
                if enabled:
                    expected_raw[0, 0] = 1e-5
                    expected_raw[0, 2] = 10. - 1e-5
                expected_center = torch.tensor([[10. - 1e-5, 1e-5, 5.]])
                torch.testing.assert_close(origin['seg_points'], expected_raw, rtol=0, atol=0)
                torch.testing.assert_close(sampled['center_preds'], expected_center, rtol=0, atol=0)
                torch.testing.assert_close(caught.exception.points,
                                           torch.cat([expected_raw[:, :3], expected_center]),
                                           rtol=0, atol=0)
                torch.testing.assert_close(caught.exception.batch_indices, torch.tensor([0, 1, 1]))


if __name__ == '__main__':
    unittest.main()
