#!/usr/bin/env bash
# ============================================================================
# 任务结束前，把微调产物上传到 HuggingFace（私有仓库），避免随临时容器销毁。
#
# 依赖：huggingface-hub（setup_env.sh 已装）
# 必需环境变量：
#   HF_REPO_ID     目标仓库，默认 "Zenith6666/yam-actioncodec"
#   HF_TOKEN       HF 写权限 token（平台上用密钥注入，勿硬编码/勿提交）
# 可选：
#   HF_ENDPOINT    默认官方 https://huggingface.co（注意 hf-mirror 只读，不支持 push）
#   UPLOAD_FILES    要传的文件列表（空格分隔），默认 action_tokenizer_yam.pt + yam_stats.json
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_REPO_ID="${HF_REPO_ID:-Zenith6666/yam-actioncodec}"
HF_TOKEN="${HF_TOKEN:?请设置 HF_TOKEN（HF 写权限 token，勿硬编码）}"
# 默认只传关键产物；数据文件(数据文件很大)如需传，用 UPLOAD_FILES 覆盖
UPLOAD_FILES="${UPLOAD_FILES:-checkpoints/action_tokenizer_yam.pt data/stats/yam_stats.json}"

echo "================================================================"
echo " 上传产物到 HuggingFace"
echo "   repo   : $HF_REPO_ID"
echo "   files  : $UPLOAD_FILES"
echo "================================================================"

HF_REPO_ID="$HF_REPO_ID" HF_TOKEN="$HF_TOKEN" UPLOAD_FILES="$UPLOAD_FILES" python - <<'PY'
import os
from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
token = os.environ["HF_TOKEN"]
files = os.environ["UPLOAD_FILES"].split()

api = HfApi(token=token)

# 私有仓库不存在则自动创建（已存在则 no-op）
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
print(f"[upload] 仓库就绪: {repo_id}")

for local in files:
    if not os.path.exists(local):
        print(f"[upload] ⚠ 跳过（本地不存在）: {local}")
        continue
    # 传到 repo 根目录，扁平命名（去掉目录前缀）
    remote = os.path.basename(local)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[upload] {local} -> {repo_id}/{remote}")

print("[upload] 完成")
PY
