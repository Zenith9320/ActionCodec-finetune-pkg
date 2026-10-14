#!/usr/bin/env bash
# ============================================================
# 一键安装运行环境（torch cu128 + 精简依赖）
# 用途：预设镜像 + git clone 的启动流程里，先跑本脚本，再跑 run_finetune.sh
#
# 镜像源可用环境变量覆盖（默认官方源）：
#   PYPI_INDEX_URL       PyPI 官方源
#   PYTORCH_INDEX_URL    PyTorch 官方 cu128 轮子源
# ============================================================
set -euo pipefail

# 屏蔽镜像自带的 pip 配置（NGC 容器写入了 pypi.ngc.nvidia.com，DNS 解析失败会导致重试/超时）
export PIP_CONFIG_FILE=/dev/null

PYTHON="${PYTHON:-python}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Python 版本检查（项目要求 3.10）
PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo '?')"
echo "[env] Python $("$PYTHON" --version 2>&1)"
if [ "$PYVER" != "3.10" ]; then
  echo "[env] ⚠ 期望 Python 3.10，当前 $PYVER，可能有兼容问题" >&2
fi

# 升级 pip（旧版 pip 装新 torch 可能报错）
echo "[env] 升级 pip ..."
"$PYTHON" -m pip install --upgrade pip -i "$PYPI_INDEX_URL"

# torch：NGC 预设镜像通常已自带 torch + CUDA，但可能是只支持到 sm_90 的旧版；
# 在 Blackwell（sm_100/sm_120）GPU 上 is_available() 仍返回 True，却无法执行 CUDA kernel。
# 因此除 is_available() 外，还需确认每张 GPU 的算力都在 torch 的 arch 列表里，否则强制重装 cu128。
if "$PYTHON" -c '
import torch
assert torch.cuda.is_available(), "no cuda"
archs = torch.cuda.get_arch_list()
for i in range(torch.cuda.device_count()):
    cap = torch.cuda.get_device_capability(i)
    sm = f"sm_{cap[0]}{cap[1]}"
    compute = f"compute_{cap[0]}{cap[1]}"
    assert sm in archs or compute in archs, f"unsupported arch {sm} on device {i}"
' >/dev/null 2>&1; then
  TVER="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
  CUDA="$("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
  echo "[env] 复用镜像自带 torch=$TVER cuda=$CUDA（算力匹配），跳过重装"
else
  echo "[env] 安装 torch==2.7.1 (cu128) ..."
  "$PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url "$PYTORCH_INDEX_URL"
fi

# 其余依赖
echo "[env] 安装 requirements.txt ..."
"$PYTHON" -m pip install -r requirements.txt -i "$PYPI_INDEX_URL"

echo "[env] 完成"
