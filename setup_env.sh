#!/usr/bin/env bash
# ============================================================
# 一键安装运行环境（torch cu128 + 精简依赖）
# 用途：预设镜像 + git clone 的启动流程里，先跑本脚本，再跑 run_finetune.sh
#
# 镜像源可用环境变量覆盖（默认阿里云）：
#   PYPI_INDEX_URL       PyPI 镜像
#   PYTORCH_INDEX_URL    PyTorch cu128 轮子源
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu128/}"

# Python 版本检查（项目要求 3.10）
PYVER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo '?')"
echo "[env] Python $("$PYTHON" --version 2>&1)"
if [ "$PYVER" != "3.10" ]; then
  echo "[env] ⚠ 期望 Python 3.10，当前 $PYVER，可能有兼容问题" >&2
fi

# 升级 pip（旧版 pip 装新 torch 可能报错）
echo "[env] 升级 pip ..."
"$PYTHON" -m pip install --upgrade pip -i "$PYPI_INDEX_URL"

# torch 2.7.1 (cu128)
echo "[env] 安装 torch==2.7.1 (cu128) ..."
"$PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url "$PYTORCH_INDEX_URL"

# 其余依赖
echo "[env] 安装 requirements.txt ..."
"$PYTHON" -m pip install -r requirements.txt -i "$PYPI_INDEX_URL"

echo "[env] 完成"
