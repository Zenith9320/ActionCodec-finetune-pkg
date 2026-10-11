# ============================================================
# G0.5 YAM ActionCodec 微调 —— 训练镜像（只装依赖，不含代码/数据）
#
# 代码（ActionCodec-finetune-pkg 目录）与数据（PFS）在训练任务里挂载，
# 镜像只负责提供运行环境：Python 3.10 + CUDA 12.8 + torch cu128 + 精简依赖。
# ============================================================

# CUDA 12.8 运行时（GPU 就绪）；Ubuntu 22.04 自带 Python 3.10
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120

# Python 3.10 + pip + git（本仓库 requires-python 3.10；git 供 GitPython 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip

# 国内 PyPI 镜像源（海外环境可去掉这两行）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# torch 2.7.1 (cu128) —— 单独装，走 PyTorch 轮子源（+cu128 自带 CUDA 运行时）
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/

# 其余第三方依赖（精简清单见 requirements.txt）
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
CMD ["bash"]
