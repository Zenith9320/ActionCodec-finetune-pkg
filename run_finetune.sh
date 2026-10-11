#!/usr/bin/env bash
# ============================================================================
# YAM ActionCodec 微调脚本（G0.5 复现 · 百度智能云训练任务）
#
# 作用：在预训练码本 checkpoints/action_tokenizer.pt 上，用 YAM 双臂数据
#       微调 ActionCodec（自监督重建 + commitment loss）。
#
# 数据集：通过 PFS（并行文件系统）挂载。本脚本把挂载点软链到 configs 期望
#         的路径 data/yam/TASK-YAM-0001_lerobot。
#         用环境变量 YAM_DATA_ROOT 指定挂载点（默认 /mnt/pfs/yam_data），
#         挂载点下需包含 LeRobot 数据集目录 TASK-YAM-0001_lerobot。
#
# 码本稳定性超参（ema_decay=0.99 / threshold_ema_dead=0 / quantizer_dropout=0）
# 已固化在 configs/task/yam.yaml，无需在此传参。
# ============================================================================
set -euo pipefail

# ── 路径与数据集挂载 ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${YAM_DATA_ROOT:-/mnt/pfs/yam_data}"

mkdir -p data/yam
if [ -d "$DATA_ROOT/TASK-YAM-0001_lerobot" ]; then
  if [ ! -e "data/yam/TASK-YAM-0001_lerobot" ]; then
    ln -s "$DATA_ROOT/TASK-YAM-0001_lerobot" "data/yam/TASK-YAM-0001_lerobot"
    echo "[data] 已软链: $DATA_ROOT/TASK-YAM-0001_lerobot -> data/yam/TASK-YAM-0001_lerobot"
  else
    echo "[data] 已存在: data/yam/TASK-YAM-0001_lerobot（跳过软链）"
  fi
else
  echo "[data] ⚠ 未找到 $DATA_ROOT/TASK-YAM-0001_lerobot，请确认 PFS 挂载路径" >&2
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
  --log-file finetune.jsonl

echo "[done] 微调完成 -> $OUT_CKPT"

# ── 训练后评估（逐键 RMSE + 码本利用率）─────────────────────────────────
echo "================================================================"
echo " 微调后评估"
echo "================================================================"
python scripts/train_tokenizer.py \
  --task yam \
  --eval-only \
  --ckpt "$OUT_CKPT" \
  --micro-batch "$MICRO_BATCH" \
  --workers "$WORKERS"
