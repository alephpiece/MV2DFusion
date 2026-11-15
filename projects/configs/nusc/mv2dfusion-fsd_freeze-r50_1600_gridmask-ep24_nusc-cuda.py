# CUDA baseline: native precision, convolution and serial point/image execution.
_base_ = ['./mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py']

model = dict(
    point_image_overlap=False,
    pts_backbone=dict(
        clip_raw_points=False,
        segmentor=dict(filter_invalid_points=True)))
image_precision = None
image_conv_weight_gradient = False
flash_attention_backend = None
evaluation_flash_attention_backend = None
validate_during_training = True
data = dict(workers_per_gpu=2, pin_memory=False)
