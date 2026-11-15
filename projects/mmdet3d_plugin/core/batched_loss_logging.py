"""Batch detached scalar logging reductions without changing training loss."""
from collections import OrderedDict
import torch
import torch.distributed as dist


def batched_parse_losses(self, losses):
    # The local means, list sums, total-loss order and length guard are copied
    # literally from the captured installed BaseDetector._parse_losses.
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        log_var_length = torch.tensor(len(log_vars), device=loss.device)
        dist.all_reduce(log_var_length)
        message = (f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}'
                   + ' keys: ' + ','.join(log_vars.keys()))
        assert log_var_length == len(log_vars) * dist.get_world_size(), \
            'loss log variables are different across GPUs!\n' + message
    log_vars['loss'] = loss
    groups = OrderedDict()
    for name, value in log_vars.items():
        if not isinstance(value, torch.Tensor):
            # Preserve the installed failures for empty lists / no-loss cases.
            if distributed:
                value.data.clone()
            value.item()
        groups.setdefault((value.device, value.dtype), []).append((name, value))
    host_values = {}
    for rows in groups.values():
        values = torch.stack([value.detach() for _, value in rows])
        if distributed:
            # Original divides each dtype-preserved scalar before summation.
            dist.all_reduce(values.div_(dist.get_world_size()))
        for (name, _), value in zip(rows, values.cpu().tolist()):
            host_values[name] = value
    return loss, OrderedDict((name, host_values[name]) for name in log_vars)
