#!/usr/bin/env bash
# ============================================================================
# 把 data/yam/TASK-YAM-XXXX_lerobot/ 上传到 HuggingFace 私有 dataset 仓库
#
# 与 download_dataset.sh 配套：上传/下载同一仓库、同一目录结构。
# 上传后目录保持 TASK-YAM-XXXX_lerobot/，训练端 data/yam/ 直接命中。
#
# 依赖：huggingface-hub（hf CLI，setup_env.sh 已装）
# 必需环境变量：
#   HF_TOKEN       HF 写权限 token（勿硬编码/勿提交）
# 可选：
#   HF_REPO_ID     数据集仓库，默认 Zenith6666/yam-lerobot
#   ONLY_TASK      只上传某个任务（如 TASK-YAM-0001），默认上传全部
#   HF_ENDPOINT    默认官方 https://huggingface.co；注意 hf-mirror 只读，不支持 push
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-lerobot}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token，勿硬编码）}"
ONLY_TASK="${ONLY_TASK:-}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "================================================================"
echo " 上传 LeRobot 数据集到 HF"
echo "   repo   : $HF_REPO_ID"
[ -n "$ONLY_TASK" ] && echo "   only   : ${ONLY_TASK}_lerobot"
echo "================================================================"

# 收集要上传的数据集目录
if [ -n "$ONLY_TASK" ]; then
  dirs=( "data/yam/${ONLY_TASK}_lerobot" )
else
  dirs=( data/yam/TASK-YAM-*_lerobot )
fi

found=0
for d in "${dirs[@]}"; do
  [ -d "$d" ] || { echo "[upload] ⚠ 跳过（本地不存在）: $d"; continue; }
  name="$(basename "$d")"
  echo "[upload] $d -> $HF_REPO_ID/$name/"
  # --private 仅在仓库不存在时生效（首次自动建私有 dataset 仓库）
  hf upload "$HF_REPO_ID" "$d" "$name" --repo-type dataset --private
  found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
  echo "[upload] ⚠ 未找到任何 data/yam/TASK-YAM-*_lerobot，请先转换数据" >&2
  exit 1
fi

echo "[done] 上传完成，共 $found 个数据集"
