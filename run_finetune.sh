#!/usr/bin/env bash
# ============================================================================
# YAM ActionCodec 微调脚本（G0.5 复现）
#
# 作用：在预训练码本 checkpoints/action_tokenizer.pt 上，用 YAM 双臂数据
#       微调 ActionCodec（自监督重建 + commitment loss）。
#
# 数据集：本地 data/yam/TASK-YAM-XXXX_lerobot/，由 convert_data.sh 从原始
#         MCAP 转换而来（本脚本只做存在性检查，不再做 PFS 软链）。
#
# 码本稳定性超参（ema_decay=0.99 / threshold_ema_dead=0 / quantizer_dropout=0）
# 已固化在 configs/task/yam.yaml，无需在此传参。
# ============================================================================
set -euo pipefail

# ── 路径与数据集检查 ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 数据集已由 convert_data.sh 转换到 data/yam/TASK-YAM-XXXX_lerobot/，此处只做检查。
DATASET_COUNT="$(find data/yam -maxdepth 1 -type d -name 'TASK-YAM-*_lerobot' 2>/dev/null | wc -l | tr -d ' ')"
# 默认用全部已下载的任务；可用 MAX_DATASETS=N 只取前 N 个做快速试跑。
MAX_DATASETS="${MAX_DATASETS:-$DATASET_COUNT}"
if [ "${DATASET_COUNT:-0}" -gt 0 ]; then
  echo "[data] 发现 $DATASET_COUNT 个 _lerobot 数据集"
else
  echo "[data] ⚠ 未找到 data/yam/TASK-YAM-*_lerobot" >&2
  echo "       请先转换原始 MCAP：RAW_DATA_ROOT=<原始数据路径> bash convert_data.sh" >&2
fi

# ── 环境 ─────────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# ── 超参（与本地验证一致；可用环境变量覆盖）─────────────────────────────
INIT_CKPT="${INIT_CKPT:-checkpoints/action_tokenizer.pt}"
OUT_CKPT="${OUT_CKPT:-checkpoints/action_tokenizer_yam.pt}"
EPOCHS="${EPOCHS:-2}"
MICRO_BATCH="${MICRO_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-5}"
WORKERS="${WORKERS:-16}"

echo "================================================================"
echo " YAM ActionCodec 微调"
echo "   init-ckpt : $INIT_CKPT"
echo "   out       : $OUT_CKPT"
echo "   freeze    : none   (全参微调，本地验证最优)"
echo "   epochs    : $EPOCHS   micro-batch: $MICRO_BATCH   grad-accum: $GRAD_ACCUM   lr: $LR"
echo "================================================================"

# ── 训练 ─────────────────────────────────────────────────────────────────
python scripts/train_tokenizer.py \
  --task yam \
  --init-ckpt "$INIT_CKPT" \
  --out "$OUT_CKPT" \
  --freeze none \
  --epochs "$EPOCHS" \
  --micro-batch "$MICRO_BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --workers "$WORKERS" \
  --max-datasets "$MAX_DATASETS" \
  --log-file finetune.jsonl

echo "[done] 微调完成 -> $OUT_CKPT"

# ── 训练后评估（逐键 RMSE + 码本利用率）─────────────────────────────────
echo "================================================================"
echo " 微调后评估"
echo "================================================================"
EVAL_OUT="${EVAL_OUT:-eval/eval_action_tokenizer_yam.json}"
python scripts/train_tokenizer.py \
  --task yam \
  --eval-only \
  --ckpt "$OUT_CKPT" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS" \
  --max-datasets "$MAX_DATASETS" \
  --eval-out "$EVAL_OUT"
echo "[done] 评估结果 -> $EVAL_OUT"
