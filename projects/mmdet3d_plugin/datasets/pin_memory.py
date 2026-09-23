from collections.abc import Mapping

from mmcv.parallel import DataContainer
import torch


def _pin_data(value):
    if isinstance(value, torch.Tensor):
        return value.pin_memory() if value.device.type == 'cpu' else value
    if isinstance(value, Mapping):
        return type(value)({key: _pin_data(item) for key, item in value.items()})
    if isinstance(value, tuple):
        if hasattr(value, '_fields'):
            return type(value)(*(_pin_data(item) for item in value))
        return tuple(_pin_data(item) for item in value)
    if isinstance(value, list):
        return [_pin_data(item) for item in value]

    # MMCV scatter can rebuild tensor-owning objects (for example,
    # LiDARInstance3DBoxes) before its background copy stream is synchronized.
    # Keep these objects pageable so their constructors cannot read an
    # unfinished pinned transfer.
    return value


class PinMemoryDataContainer(DataContainer):
    """DataContainer variant traversable by the DataLoader pinning thread."""

    def pin_memory(self):
        return type(self)(
            _pin_data(self.data),
            stack=self.stack,
            padding_value=self.padding_value,
            cpu_only=self.cpu_only,
            pad_dims=self.pad_dims)


def wrap_data_containers(value):
    """Wrap transferable containers after MMCV collate; leave CPU-only data alone."""
    if isinstance(value, DataContainer):
        if value.cpu_only:
            return value
        return PinMemoryDataContainer(
            value.data,
            stack=value.stack,
            padding_value=value.padding_value,
            cpu_only=False,
            pad_dims=value.pad_dims)
    if isinstance(value, Mapping):
        return type(value)({key: wrap_data_containers(item)
                            for key, item in value.items()})
    if isinstance(value, tuple):
        if hasattr(value, '_fields'):
            return type(value)(*(wrap_data_containers(item) for item in value))
        return tuple(wrap_data_containers(item) for item in value)
    if isinstance(value, list):
        return [wrap_data_containers(item) for item in value]
    return value
