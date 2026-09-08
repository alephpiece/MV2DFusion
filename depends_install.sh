pip3 install mmdet==2.25.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install mmsegmentation==0.28.0  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 注意：mmdet3d 1.0.0rc4 需要通过以下命令手动安装，因为它包含特殊参数
pip3 install mmdet3d==1.0.0rc4 --no-build-isolation --no-deps
pip3 install mmcv-lite==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 核心依赖
# spconv-cu114==2.2.3 安装DCU版本;
pip3 install opencv-python==4.11.0.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install opencv-python-headless==4.11.0.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numba==0.63.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install lyft-dataset-sdk==0.0.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install ipython==8.39.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install yapf==0.40.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install sympy==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install ipdb -i https://pypi.tuna.tsinghua.edu.cn/simple

pip3 install nuscenes-devkit==1.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install trimesh -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install av2==0.2.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
