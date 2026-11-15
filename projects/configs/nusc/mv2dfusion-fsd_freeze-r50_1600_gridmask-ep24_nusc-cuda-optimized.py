# Validated CUDA execution profile for four or eight GPUs.
_base_ = ['./mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc-cuda.py']

# Resume optimizer state with the same grouped layout.
optimizer = dict(constructor='GroupedAdamWOptimizerConstructor',
                 foreach=True, fused=False)
