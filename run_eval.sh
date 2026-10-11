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
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 数据集软链（与 run_finetune.sh 一致）
DATA_ROOT="${YAM_DATA_ROOT:-/mnt/pfs/yam_data}"
mkdir -p data/yam
if [ -d "$DATA_ROOT/TASK-YAM-0001_lerobot" ] && [ ! -e "data/yam/TASK-YAM-0001_lerobot" ]; then
  ln -s "$DATA_ROOT/TASK-YAM-0001_lerobot" "data/yam/TASK-YAM-0001_lerobot"
  echo "[data] 已软链: $DATA_ROOT/TASK-YAM-0001_lerobot -> data/yam/TASK-YAM-0001_lerobot"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CKPT="${1:-checkpoints/action_tokenizer.pt}"
WORKERS="${WORKERS:-16}"
MICRO_BATCH="${MICRO_BATCH:-64}"

echo "[eval] ckpt=$CKPT"
python scripts/train_tokenizer.py \
  --task yam \
  --eval-only \
  --ckpt "$CKPT" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS"
