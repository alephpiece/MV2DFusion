# Validated BW150 four-GPU profile; retain the original LR horizon.
_base_ = ['./mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc-grouped-optimizer.py']

model = dict(batch_loss_logging=True)
custom_imports = dict(
    imports=['projects.mmdet3d_plugin.core.hook.roi_bin_aggregation'],
    allow_failed_imports=False)
custom_hooks = [dict(
    type='RoIAlignBinAggregationHook', limit=4,
    module_prefixes=('img_roi_extractor', 'img_roi_head.roi_head.bbox_roi_extractor'),
    priority='NORMAL')]
