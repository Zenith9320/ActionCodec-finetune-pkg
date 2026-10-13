#!/usr/bin/env bash
# ============================================================================
# YAM ActionCodec 评估脚本（可选，用于基线对照 / 微调后验收）
#
# 用法：
#   ./run_eval.sh                              # 评估预训练基线（微调前）
#   ./run_eval.sh checkpoints/action_tokenizer_yam.pt   # 评估微调后
#
# 输出：归一化空间逐键 RMSE（rmse_left_control / rmse_right_control 等）
#       + 各层码本利用率 utilization_l0..l3
# 结果同时落盘到 eval/eval_<ckpt名>.json（可用 EVAL_OUT 覆盖），便于上传 HF
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 数据集已由 convert_data.sh 转换到 data/yam/，此处无需软链
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${1:-checkpoints/action_tokenizer.pt}"
WORKERS="${WORKERS:-16}"
MICRO_BATCH="${MICRO_BATCH:-64}"
EVAL_OUT="${EVAL_OUT:-eval/eval_$(basename "$CKPT" .pt).json}"

echo "[eval] ckpt=$CKPT -> $EVAL_OUT"
python scripts/train_tokenizer.py \
  --task yam \
  --eval-only \
  --ckpt "$CKPT" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS" \
  --eval-out "$EVAL_OUT"
