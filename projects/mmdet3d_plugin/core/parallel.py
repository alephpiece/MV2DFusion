"""MMCV 1.x distributed forward compatibility for training and evaluation."""
from mmcv.parallel import MMDistributedDataParallel as NativeMMDDP


class MMDistributedDataParallel(NativeMMDDP):
    """Preserve MMCV's forward path when PyTorch omits the legacy flag."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_use_replicated_tensor_module'):
            self._use_replicated_tensor_module = False
