

export MMDET_DATASETS=/data/liuxu/zhijia/MV2DFusion/data/nuscenes

# export HIP_VISIBLE_DEVICES=0

FASTPT_TORCH_LIB_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib
FASTPT_CUDA_LIB_PATH=/opt/dtk/cuda/cuda-12/lib64/
export LD_LIBRARY_PATH=$FASTPT_TORCH_LIB_PATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FASTPT_CUDA_LIB_PATH:$LD_LIBRARY_PATH

bash tools/dist_train.sh projects/configs/nusc/mv2dfusion-fsd_freeze-r50_1600_gridmask-ep24_nusc.py 2 --seed 42 2>&1 | tee 111.log
