#!/usr/bin/env bash
# ============================================================================
# 下载预训练 ActionCodec 码本 checkpoints/action_tokenizer.pt（约 484MB）
#
# 来源：Zenith6666/G05-bucket（用户私有 bucket，公开只读）。直接从 bucket 的
#       resolve 端点 HTTP 下载，绕开上游 OpenGalaxea/G05 的 gated 认证限制。
# 实现：用 curl 直连，不依赖 huggingface_hub 的 bucket API（平台上是 0.35.0，
#       不支持 bucket），也不需要 HF_TOKEN。
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CKPT_FILE="${CKPT_FILE:-action_tokenizer.pt}"
# bucket 公开，无需 token；完整下载 URL 可用 CKPT_URL 覆盖
URL="${CKPT_URL:-https://huggingface.co/buckets/Zenith6666/G05-bucket/resolve/${CKPT_FILE}}"

mkdir -p checkpoints

if [ -f "checkpoints/$CKPT_FILE" ]; then
  echo "[ckpt] 已存在 checkpoints/$CKPT_FILE，跳过下载"
  exit 0
fi

echo "[ckpt] 下载 $URL ..."
curl -fL --retry 3 --retry-delay 2 -C - -o "checkpoints/$CKPT_FILE" "$URL"

echo "[ckpt] 完成 -> checkpoints/$CKPT_FILE（$(du -h "checkpoints/$CKPT_FILE" | cut -f1)）"
