#!/usr/bin/env bash
# ============================================================================
# 下载预训练 ActionCodec 码本 checkpoints/action_tokenizer.pt（约 484MB）
#
# 来源：Hugging Face OpenGalaxea/G05（Galaxea 官方 G0.5 模型仓库，根目录下）
# 依赖：huggingface-hub（setup_env.sh 已装，需先跑 setup_env.sh）
#
# 默认走国内 HF 镜像 hf-mirror.com；海外环境可 `export HF_ENDPOINT=` 关闭镜像。
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

REPO="${REPO:-OpenGalaxea/G05}"
CKPT_FILE="${CKPT_FILE:-action_tokenizer.pt}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p checkpoints

if [ -f "checkpoints/$CKPT_FILE" ]; then
  echo "[ckpt] 已存在 checkpoints/$CKPT_FILE，跳过下载"
  exit 0
fi

echo "[ckpt] 下载 $REPO/$CKPT_FILE ..."
huggingface-cli download "$REPO" "$CKPT_FILE" \
  --repo-type model \
  --local-dir checkpoints

echo "[ckpt] 完成 -> checkpoints/$CKPT_FILE"
