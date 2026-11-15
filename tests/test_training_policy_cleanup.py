"""CPU exercise of the training API's owned-policy cleanup paths."""
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from mmcv import ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
api = importlib.import_module('projects.mmdet3d_plugin.core.apis.mmdet_train')
image_policy = importlib.import_module('projects.mmdet3d_plugin.core.hook.image_precision')
attention_policy = importlib.import_module('projects.mmdet3d_plugin.core.hook.attention_precision')
roi_policy = importlib.import_module('projects.mmdet3d_plugin.core.hook.roi_bin_aggregation')


class TrainingFailure(Exception):
    pass


class CleanupFailure(Exception):
    pass


class TrainingPolicyCleanupTest(unittest.TestCase):
    def exercise(self, failure=None, cleanup_failure=None, with_roi=True,
                 precision=True, point_failure=False):
        events = []

        def closer(name):
            def close():
                events.append(name)
                if cleanup_failure == name:
                    raise CleanupFailure(name)
            return close

        roi = roi_policy.RoIAlignBinAggregationHook()

        def install_roi(runner):
            roi.policy = SimpleNamespace(close=closer('roi'))

        roi.before_run = install_roi
        unknown = SimpleNamespace(close=closer('unknown'))

        def install_attention(runner):
            if failure == 'before_run':
                raise TrainingFailure('before_run')

        attention = SimpleNamespace(before_run=install_attention, close=closer('attention'))
        image = SimpleNamespace(close=closer('image'))

        class Runner:
            def __init__(self):
                self.hooks = []
                self.meta = None

            def register_training_hooks(self, *args):
                pass

            def register_hook(self, hook, **kwargs):
                self.hooks.append(hook)

            def run(self, loaders, workflow):
                for hook in self.hooks:
                    if hasattr(hook, 'before_run'):
                        hook.before_run(self)
                if failure == 'run':
                    raise TrainingFailure('run')
                for hook in self.hooks:
                    if hasattr(hook, 'after_run'):
                        hook.after_run(self)

        def close_point():
            events.append('point')
            if point_failure:
                raise CleanupFailure('point')

        model = SimpleNamespace(close_point_worker=close_point)
        model.cuda = lambda *args: model
        runner = Runner()
        cfg = ConfigDict(
            log_level='INFO', data=dict(samples_per_gpu=1, workers_per_gpu=0,
                shuffler_sampler={}, nonshuffler_sampler={}), gpu_ids=[0], seed=42,
            runner=dict(type='IterBasedRunner', max_iters=1), optimizer={},
            optimizer_config={}, lr_config={}, checkpoint_config={}, log_config={},
            work_dir='/tmp', workflow=[('train', 1)], resume_from=None, load_from=None,
            custom_hooks=([dict(type='roi'), dict(type='unknown')] if with_roi else []),
            image_precision='image-fp32' if precision else None,
            flash_attention_backend='fp32-math' if precision else None)
        error = None
        with ExitStack() as patches:
            replacements = dict(
                get_root_logger=mock.Mock(), build_dataloader=mock.Mock(return_value=[]),
                MMDataParallel=lambda value, **kwargs: SimpleNamespace(module=value),
                build_optimizer=mock.Mock(), build_runner=mock.Mock(return_value=runner),
                build_from_cfg=lambda value, registry: roi if value['type'] == 'roi' else unknown)
            for name, replacement in replacements.items():
                patches.enter_context(mock.patch.object(api, name, replacement))
            patches.enter_context(mock.patch.object(image_policy, 'ImageFP32Hook', return_value=image))
            patches.enter_context(mock.patch.object(attention_policy, 'FlashFP32Hook', return_value=attention))
            try:
                api.custom_train_detector(model, object(), cfg)
            except (TrainingFailure, CleanupFailure) as exc:
                error = exc
        self.assertNotIn('unknown', events)
        return events, error

    def test_normal_completion_and_failures_close_owned_policies(self):
        for failure in (None, 'before_run', 'run'):
            with self.subTest(failure=failure):
                events, error = self.exercise(failure=failure)
                # On success the real ROI after_run closes it once; the API's
                # final close is idempotent. Failures bypass after_run entirely.
                expected = (['roi', 'point', 'attention', 'image'] if failure is None
                            else ['point', 'attention', 'image', 'roi'])
                self.assertEqual(events, expected)
                if failure is None:
                    self.assertIsNone(error)
                else:
                    self.assertIsInstance(error, TrainingFailure)

    def test_a_failing_closer_does_not_skip_remaining_policies(self):
        for name in ('attention', 'image', 'roi'):
            with self.subTest(closer=name):
                events, error = self.exercise(failure='run', cleanup_failure=name)
                self.assertEqual(events, ['point', 'attention', 'image', 'roi'])
                self.assertIsInstance(error, CleanupFailure)
                self.assertIsInstance(error.__context__, TrainingFailure)

    def test_no_roi_and_point_failure_preserves_training_error(self):
        events, error = self.exercise(failure='run', with_roi=False, point_failure=True)
        self.assertEqual(events, ['point', 'attention', 'image'])
        self.assertIsInstance(error, TrainingFailure)
        events, error = self.exercise(with_roi=False, precision=False)
        self.assertEqual(events, ['point'])
        self.assertIsNone(error)


if __name__ == '__main__':
    unittest.main()
