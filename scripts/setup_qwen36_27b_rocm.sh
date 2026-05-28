#!/usr/bin/env bash
# 用同一套 llama.cpp ROCm 部署脚本，额外启动一个 Qwen3.6-27B GGUF OpenAI-compatible API。
#
# 重要：llama.cpp 只能直接跑 GGUF。vLLM/FP8 在当前 ROCm 环境可能触发
# gdn_attention_core hipErrorIllegalAddress，所以这里默认走 GGUF。
#
# 用法示例：
#   MODEL_LOCAL_FILE=/mnt/Qwen3.6-27B-Q8_0.gguf bash scripts/setup_qwen36_27b_rocm.sh --restart
#
# 或指定 HF / hf-mirror GGUF：
#   HF_REPO_ID=ggml-org/Qwen3.6-27B-GGUF \
#   MODEL_NAME=Qwen3.6-27B-Q8_0.gguf \
#   bash scripts/setup_qwen36_27b_rocm.sh --restart
#
# 或指定 ModelScope GGUF：
#   MODEL_PROVIDER=modelscope \
#   MODELSCOPE_MODEL_ID=你的命名空间/Qwen3.6-27B-GGUF \
#   MODELSCOPE_FILE_PATH=Qwen3.6-27B-Q8_0.gguf \
#   bash scripts/setup_qwen36_27b_rocm.sh --restart
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
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-27b-q8-gguf}"

# 192GB 显存可尝试 Q8 + 较高 parallel/ctx；如 OOM，优先降 CTX_SIZE，然后降 PARALLEL 或改 Q6/Q5。
export PARALLEL="${PARALLEL:-100}"
export CTX_SIZE="${CTX_SIZE:-400000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"
export TEMP="${TEMP:-0.55}"
export TOP_P="${TOP_P:-0.90}"
export REPEAT_PENALTY="${REPEAT_PENALTY:-1.08}"

# 默认给 GGUF 仓库/文件名；如不存在或你有更好的量化，设置 HF/ModelScope 变量或 MODEL_LOCAL_FILE。
export MODEL_PROVIDER="${MODEL_PROVIDER:-auto}"
export HF_REPO_ID="${HF_REPO_ID:-ggml-org/Qwen3.6-27B-GGUF}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-Q8_0.gguf}"
export MODELSCOPE_MODEL_ID="${MODELSCOPE_MODEL_ID:-${QWEN_MS_GGUF_MODEL_ID:-}}"
export MODELSCOPE_FILE_PATH="${MODELSCOPE_FILE_PATH:-$MODEL_NAME}"
export MODEL_URL="${MODEL_URL:-${HF_ENDPOINT:-https://hf-mirror.com}/$HF_REPO_ID/resolve/main/$MODEL_NAME}"
export MODEL_URL_FALLBACKS="${MODEL_URL_FALLBACKS:-$MODEL_URL https://huggingface.co/$HF_REPO_ID/resolve/main/$MODEL_NAME}"

# 本地上传搜索目录。基础脚本会优先使用 MODEL_LOCAL_FILE 或同名 MODEL_NAME。
export MODEL_LOCAL_SEARCH_DIRS="${MODEL_LOCAL_SEARCH_DIRS:-/mnt /mnt/data /workspace /root}"

cat <<EOF
[Qwen3.6-27B GGUF ROCm]
WORKDIR=$WORKDIR
PORT=$PORT
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
HF_REPO_ID=$HF_REPO_ID
MODEL_PROVIDER=$MODEL_PROVIDER
MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-}
MODELSCOPE_FILE_PATH=${MODELSCOPE_FILE_PATH:-}
MODEL_NAME=$MODEL_NAME
MODEL_LOCAL_FILE=${MODEL_LOCAL_FILE:-}
PARALLEL=$PARALLEL
CTX_SIZE=$CTX_SIZE

如果默认 HF_REPO_ID/MODEL_NAME 不存在，请改成你实际的 Qwen3.6-27B GGUF 仓库和文件，
或上传 GGUF 到 /mnt 并设置 MODEL_LOCAL_FILE=/mnt/xxx.gguf。
EOF

exec bash "$BASE_SCRIPT" "$@"
