#!/usr/bin/env bash
# ============================================================================
# 原始 MCAP → LeRobot v3.0 转换（在算力平台上运行一次即可）
#
# 输入：原始 YAM MCAP 数据（TASK-YAM-XXXX/mcap/success/*.mcap）
# 输出：data/yam/TASK-YAM-XXXX_lerobot/（configs/data/yam.yaml 直接读取）
#
# 用法（原始数据已通过 PFS 挂载，直接指过去转，无需拷贝）：
#   RAW_DATA_ROOT=/mnt/workspace/handzero-share/DATASET/TeleopData/YAM bash convert_data.sh
# 或先把原始数据拷到本地再转：
#   RAW_DATA_ROOT=data/raw bash convert_data.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

RAW_DATA_ROOT="${RAW_DATA_ROOT:-/mnt/workspace/handzero-share/DATASET/TeleopData/YAM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/yam}"
TASKS_JSON="${TASKS_JSON:-tools/yam/tasks_yam.json}"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================"
echo " MCAP → LeRobot 转换"
echo "   input  : $RAW_DATA_ROOT"
echo "   output : $OUTPUT_ROOT"
echo "   tasks  : $TASKS_JSON"
echo "================================================================"

python tools/yam/mcap_to_lerobot.py \
  --input-root "$RAW_DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --tasks-json "$TASKS_JSON"

# 数据规模变了（10 个任务），清掉旧 stats，让下次微调时自动遍历全量数据重算
if [ -f data/stats/yam_stats.json ]; then
  rm -f data/stats/yam_stats.json
  echo "[stats] 已删除旧 yam_stats.json（下次微调会自动重算全量统计量）"
fi

echo "[done] 转换完成。下一步：bash run_finetune.sh"
