# Optimizer checkpoints use grouped layout; resume with this configuration.
# Legacy singleton-group optimizer checkpoints require explicit conversion.
# Research configuration: identical-option groups with batched AdamW updates.
_base_ = ['./mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py']
optimizer = dict(constructor='GroupedAdamWOptimizerConstructor',
                 foreach=True, fused=False)
