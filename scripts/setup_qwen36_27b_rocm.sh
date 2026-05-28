#!/usr/bin/env bash
# 用同一套 llama.cpp ROCm 部署脚本，额外启动一个 Qwen3.6-27B GGUF OpenAI-compatible API。
#
# 重要：llama.cpp 只能直接跑 GGUF。Unsloth/HF safetensors 训练模型不能直接用本脚本部署；
# 你需要指定一个已经量化好的 Qwen3.6-27B GGUF 仓库/文件，或把 GGUF 上传到 /mnt。
#
# 用法示例：
#   MODEL_LOCAL_FILE=/mnt/qwen3.6-27b-q4_k_m.gguf bash scripts/setup_qwen36_27b_rocm.sh --restart --tunnel
#
# 或指定 HF / hf-mirror GGUF：
#   HF_REPO_ID=你的用户名/Qwen3.6-27B-GGUF #   MODEL_NAME=qwen3.6-27b-q4_k_m.gguf #   bash scripts/setup_qwen36_27b_rocm.sh --restart --tunnel
#
# 默认端口 8010，避免和 Sydney 的 8000 冲突。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/setup_sydney_rocm.sh"
if [[ ! -f "$BASE_SCRIPT" ]]; then
  echo "找不到基础脚本：$BASE_SCRIPT" >&2
  exit 1
fi

# 第二个模型必须使用独立 WORKDIR/RUN_DIR/PORT，否则会复用 Sydney 的 pid 和模型目录。
export WORKDIR="${WORKDIR:-/workspace/qwen36_27b_rocm}"
export PORT="${PORT:-8010}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b}"

# Qwen 27B 比 12B 大很多。192GB 显存可先用 Q4/Q5 + parallel 4。
# 如果你用 Q8，建议把 PARALLEL 降到 1-2，CTX_SIZE 降到 8192/16384。
export PARALLEL="${PARALLEL:-4}"
export CTX_SIZE="${CTX_SIZE:-16384}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"

# 用户应显式指定 GGUF。这里给占位默认，避免误下载不存在文件时不知道改哪里。
export HF_REPO_ID="${HF_REPO_ID:-unsloth/Qwen3.6-27B-GGUF}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-Q4_K_M.gguf}"
export MODEL_URL="${MODEL_URL:-${HF_ENDPOINT:-https://hf-mirror.com}/$HF_REPO_ID/resolve/main/$MODEL_NAME}"
export MODEL_URL_FALLBACKS="${MODEL_URL_FALLBACKS:-$MODEL_URL https://huggingface.co/$HF_REPO_ID/resolve/main/$MODEL_NAME}"

# 本地上传搜索目录。基础脚本会优先使用 MODEL_LOCAL_FILE 或同名 MODEL_NAME。
export MODEL_LOCAL_SEARCH_DIRS="${MODEL_LOCAL_SEARCH_DIRS:-/mnt /mnt/data /workspace /root}"

cat <<EOF
[Qwen3.6-27B ROCm]
WORKDIR=$WORKDIR
PORT=$PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
HF_REPO_ID=$HF_REPO_ID
MODEL_NAME=$MODEL_NAME
MODEL_LOCAL_FILE=${MODEL_LOCAL_FILE:-}
PARALLEL=$PARALLEL
CTX_SIZE=$CTX_SIZE

注意：如果默认 HF_REPO_ID/MODEL_NAME 不存在，请改成你实际的 Qwen3.6-27B GGUF 仓库和文件，
或上传 GGUF 到 /mnt 并设置 MODEL_LOCAL_FILE=/mnt/xxx.gguf。
EOF

exec bash "$BASE_SCRIPT" "$@"
