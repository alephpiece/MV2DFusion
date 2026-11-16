# BW150 and CUDA execution profiles

The same source supports the validated BW150/DTK and L20/CUDA paths. Install
the matching dependencies and run from the project root. Dataset paths are
relative to `data/nuscenes/`; pretrained weights are under `weights/`.

Validation used PyTorch 2.5.1 and MMCV 1.6.1 with their matching CUDA or DTK
builds. The environment also needs the MMCV scatter patch for 3D box objects
and the current stream API, plus the compatible MMDetection3D NumPy-bool fix.
These are dependency patches; the application source does not install them.

All configurations below are in `projects/configs/nusc/`, with the prefix
`mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc`.
The base configuration retains the existing BW150 defaults.

| Suffix | Use | Execution |
| --- | --- | --- |
| `.py` | BW150 reference | Mixed image/2D detection, custom dW, FP32 fused training attention, point/image overlap |
| `-cuda.py` | CUDA reference | Native precision/convolution/Half FlashAttention, serial point/image execution |
| `-grouped-optimizer.py` | BW150 eight-GPU selection | BW reference plus grouped/foreach AdamW |
| `-bw150-4gpu.py` | BW150 four-GPU selection | Grouped AdamW, batched loss logging, FP32 ROI backward aggregation |
| `-cuda-optimized.py` | L20 four/eight-GPU selection | CUDA reference plus grouped/foreach AdamW |

Profiles are explicit: changing world size does not switch algorithms. They
retain the original 42192-iteration LR schedule and disable backbone/decoder
activation checkpointing. BW uses six DataLoader workers per GPU and pinned
data containers; CUDA uses two workers and ordinary containers. Keep worker
counts and OpenCV thread settings fixed within a performance comparison.

CUDA filters invalid dynamic-voxelization rows before segmentation targets and
VFE, and disables the BW extra clamp of original points. Predicted centers
remain clamped on both paths. These independent policies are controlled by
`model.pts_backbone.segmentor.filter_invalid_points` and
`model.pts_backbone.clip_raw_points`. Empty/fully invalid samples report an error.
BW defaults retain the existing preprocessing.

BW independent evaluation selects FP32 MATH; CUDA uses native Half FlashAttention.
The BW fused training path has previously observed validation-quality differences
from MATH; these profiles preserve that choice. HIP/Flex library discovery runs
before starting the point worker; CUDA and serial paths skip that initialization.

Launch training with the selected configuration and `--no-validate`; evaluate in
a separate process using `tools/test.py --launcher pytorch`. For an ablation of
the BW four-GPU selection, disable `model.batch_loss_logging` or remove the ROI hook.
The CUDA profile preserves the original online-validation default when
`--no-validate` is omitted; the BW fused-training profile uses separate evaluation.

Grouped optimizer checkpoints require the same parameter-group construction on
resume. Older singleton-group optimizer states are not converted automatically;
model-weight loading is independent of grouping. FP16 resume restores the saved
loss scaler through the registered optimizer hook and requires
`meta.fp16.loss_scaler` in the checkpoint. Use `load_from` for model-only loading.
Native checkpoint iteration counting and temporal/data-state restoration are
unchanged.
