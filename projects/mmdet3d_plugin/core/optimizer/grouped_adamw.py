"""Group identical AdamW options before MMCV initializes the LR schedule."""
import torch
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor


def _option_key(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', id(value))
    if isinstance(value, dict):
        return ('dict', tuple((key, _option_key(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_option_key(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__name__, value)
    return ('object', id(value))


@OPTIMIZER_BUILDERS.register_module()
class GroupedAdamWOptimizerConstructor(DefaultOptimizerConstructor):
    """Preserve paramwise options and batch parameters with identical options.

    Construct before LR-hook initialization. Parameter objects and order within
    each option class are preserved. Checkpoints use this grouped layout and
    can resume with the same constructor/configuration.
    """
    def __call__(self, model):
        optimizer = super().__call__(model)
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError('GroupedAdamWOptimizerConstructor requires AdamW')
        grouped, lookup = [], {}
        for group in optimizer.param_groups:
            options = {key: value for key, value in group.items() if key != 'params'}
            key = _option_key(options)
            if key in lookup:
                grouped[lookup[key]]['params'].extend(group['params'])
            else:
                lookup[key] = len(grouped)
                grouped.append({**options, 'params': list(group['params'])})
        optimizer.param_groups = grouped
        return optimizer
