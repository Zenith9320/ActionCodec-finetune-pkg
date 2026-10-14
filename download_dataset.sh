#!/usr/bin/env bash
# ============================================================================
# 从 HuggingFace 私有 dataset 仓库下载已转换好的 LeRobot 数据集到 data/yam/
#
# 与上传端配套：先把 data/yam/TASK-YAM-XXXX_lerobot/ 推到仓库，
# 本脚本在算力平台上把它们拉回本地，路径正好对上 configs/data/yam.yaml 的
# dataset_dirs（data/yam/TASK-YAM-XXXX_lerobot），训练端无需再改。
#
# 依赖：huggingface-hub（hf CLI，setup_env.sh 已装）
# 必需环境变量：
#   HF_TOKEN       HF 读权限 token（勿硬编码/勿提交）
# 可选：
#   HF_REPO_ID     数据集仓库，默认 Zenith6666/yam-lerobot
#   ONLY_TASK      只下载某个任务（如 TASK-YAM-0001），默认下载全部
#   HF_ENDPOINT    默认官方 https://huggingface.co；国内可 export HF_ENDPOINT=https://hf-mirror.com
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-lerobot}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 读权限 token，勿硬编码）}"
OUTPUT_DIR="${OUTPUT_DIR:-data/yam}"
ONLY_TASK="${ONLY_TASK:-}"
export HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "================================================================"
echo " 从 HF 下载 LeRobot 数据集"
echo "   repo   : $HF_REPO_ID"
echo "   to     : $OUTPUT_DIR"
if [ -n "$ONLY_TASK" ]; then
  echo "   only   : ${ONLY_TASK}_lerobot"
fi
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

if [ -n "$ONLY_TASK" ]; then
  # 只拉单个任务的子目录（--include 的 glob 相对仓库根匹配）
  hf download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR" \
    --include "${ONLY_TASK}_lerobot/**"
else
  hf download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR"
fi

echo "[done] 已下载到 $OUTPUT_DIR"
