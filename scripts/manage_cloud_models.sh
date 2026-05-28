#!/usr/bin/env bash
# 管理云端两个 GGUF 模型：
#   1) Sydney source: llama.cpp GGUF OpenAI-compatible API
#   2) Qwen3.6-27B: llama.cpp GGUF OpenAI-compatible API
#
# 默认端口：
#   Sydney: http://127.0.0.1:8000/v1
#   Qwen:   http://127.0.0.1:8010/v1
#
# 用法：
#   bash scripts/manage_cloud_models.sh start-all
#   bash scripts/manage_cloud_models.sh restart-qwen
#   bash scripts/manage_cloud_models.sh logs-qwen

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STACK_DIR="${STACK_DIR:-/workspace/sydney_cloud_stack}"
MANAGER_ENV_FILE="${MANAGER_ENV_FILE:-$STACK_DIR/model-manager.env}"
if [[ -f "$MANAGER_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; . "$MANAGER_ENV_FILE"; set +a
fi

LOG_DIR="${LOG_DIR:-$STACK_DIR/logs}"
RUN_DIR="${RUN_DIR:-$STACK_DIR/run}"
APP_DIR="${APP_DIR:-$REPO_DIR}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-https://gh.llkk.cc/}"

SHARED_LLAMA_DIR="${SHARED_LLAMA_DIR:-/workspace/llama.cpp-rocm}"

SYDNEY_WORKDIR="${SYDNEY_WORKDIR:-/workspace/sydney_rocm}"
SYDNEY_LLAMA_DIR="${SYDNEY_LLAMA_DIR:-$SHARED_LLAMA_DIR}"
SYDNEY_HOST="${SYDNEY_HOST:-127.0.0.1}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
SYDNEY_MODEL_NAME="${SYDNEY_MODEL_NAME:-clever-sydney-4-12b-q8}"
SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-80}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-400000}"
SYDNEY_GPU_LAYERS="${SYDNEY_GPU_LAYERS:-999}"
SYDNEY_BATCH_SIZE="${SYDNEY_BATCH_SIZE:-512}"
SYDNEY_UBATCH_SIZE="${SYDNEY_UBATCH_SIZE:-512}"
SYDNEY_FORCE_SETUP="${SYDNEY_FORCE_SETUP:-0}"
SYDNEY_EXTRA_ENV="${SYDNEY_EXTRA_ENV:-}"

QWEN_WORKDIR="${QWEN_WORKDIR:-/workspace/qwen36_27b_rocm}"
QWEN_LLAMA_DIR="${QWEN_LLAMA_DIR:-$SHARED_LLAMA_DIR}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8010}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-q8-gguf}"
QWEN_GGUF_REPO_ID="${QWEN_GGUF_REPO_ID:-ggml-org/Qwen3.6-27B-GGUF}"
QWEN_GGUF_MODEL_NAME="${QWEN_GGUF_MODEL_NAME:-Qwen3.6-27B-Q8_0.gguf}"
QWEN_MODEL_PROVIDER="${QWEN_MODEL_PROVIDER:-${MODEL_PROVIDER:-auto}}" # auto / hf / modelscope
QWEN_MS_GGUF_MODEL_ID="${QWEN_MS_GGUF_MODEL_ID:-}"
QWEN_MS_GGUF_FILE_PATH="${QWEN_MS_GGUF_FILE_PATH:-$QWEN_GGUF_MODEL_NAME}"
QWEN_MODEL_LOCAL_FILE="${QWEN_MODEL_LOCAL_FILE:-}"
QWEN_MODEL_LOCAL_SEARCH_DIRS="${QWEN_MODEL_LOCAL_SEARCH_DIRS:-/mnt /mnt/data /workspace /root}"
QWEN_PARALLEL="${QWEN_PARALLEL:-80}"
QWEN_CTX_SIZE="${QWEN_CTX_SIZE:-262144}"
QWEN_GPU_LAYERS="${QWEN_GPU_LAYERS:-999}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-512}"
QWEN_UBATCH_SIZE="${QWEN_UBATCH_SIZE:-256}"
QWEN_TEMP="${QWEN_TEMP:-0.55}"
QWEN_TOP_P="${QWEN_TOP_P:-0.90}"
QWEN_REPEAT_PENALTY="${QWEN_REPEAT_PENALTY:-1.08}"
QWEN_LLAMA_ARG_FIT="${QWEN_LLAMA_ARG_FIT:-off}"
QWEN_LLAMA_CONT_BATCHING="${QWEN_LLAMA_CONT_BATCHING:-1}"
QWEN_FORCE_SETUP="${QWEN_FORCE_SETUP:-0}"
QWEN_EXTRA_ENV="${QWEN_EXTRA_ENV:-}"

START_ORDER="${START_ORDER:-qwen-first}" # qwen-first | sydney-first
MANAGER_KILL_PORT_FALLBACK="${MANAGER_KILL_PORT_FALLBACK:-1}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

log(){ printf '\033[1;36m[model-manager]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
have(){ command -v "$1" >/dev/null 2>&1; }

pid_from_file(){ [[ -s "$1" ]] && tr -dc '0-9' < "$1" || true; }
pid_alive(){ local p="$(pid_from_file "$1")"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

kill_port_fallback(){
  local port="$1" name="$2"
  [[ "$MANAGER_KILL_PORT_FALLBACK" == "1" ]] || return 0
  if have lsof; then
    local pids=""
    pids="$(lsof -ti TCP:"$port" 2>/dev/null | tr '\n' ' ' || true)"
    if [[ -n "$pids" ]]; then
      warn "按端口清理残留 $name 进程：port=$port pids=$pids"
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      sleep 2
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  elif have fuser; then
    warn "按端口清理残留 $name 进程：port=$port"
    fuser -k "${port}/tcp" 2>/dev/null || true
  fi
}

kill_pid_file(){
  local f="$1" name="${2:-process}" port="${3:-}"
  if pid_alive "$f"; then
    local pid pgid
    pid="$(pid_from_file "$f")"
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    log "停止 $name PID=$pid${pgid:+ PGID=$pgid}"
    if [[ -n "$pgid" ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then
      warn "$name 未正常退出，强制结束"
      if [[ -n "$pgid" ]]; then
        kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "$f"
  [[ -n "$port" ]] && kill_port_fallback "$port" "$name"
}

curl_models(){ local port="$1"; curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null || true; echo; }
print_gpu(){ if have rocm-smi; then rocm-smi || true; elif have nvidia-smi; then nvidia-smi || true; else echo "no rocm-smi/nvidia-smi found"; fi; }

qwen_model_path(){ echo "$QWEN_WORKDIR/models/$QWEN_GGUF_MODEL_NAME"; }
qwen_runtime_ready(){ [[ -x "$QWEN_WORKDIR/bin/start_sydney_server.sh" && -f "$(qwen_model_path)" ]]; }
sydney_runtime_ready(){ [[ -x "$SYDNEY_WORKDIR/bin/start_sydney_server.sh" ]]; }

status_sydney(){
  echo "--- Sydney llama.cpp GGUF ---"
  if [[ -x "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" ]]; then
    "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" || true
  else
    echo "status script not found: $SYDNEY_WORKDIR/bin/status_sydney_server.sh"
    curl_models "$SYDNEY_PORT"
  fi
}

status_qwen(){
  echo "--- Qwen3.6 llama.cpp GGUF ---"
  if [[ -x "$QWEN_WORKDIR/bin/status_sydney_server.sh" ]]; then
    "$QWEN_WORKDIR/bin/status_sydney_server.sh" || true
  else
    echo "status script not found: $QWEN_WORKDIR/bin/status_sydney_server.sh"
    curl_models "$QWEN_PORT"
  fi
  if [[ -f "$RUN_DIR/qwen-vllm.pid" ]]; then
    echo "legacy qwen-vllm pid file exists: $RUN_DIR/qwen-vllm.pid"
  fi
}

status_all(){ status_qwen; status_sydney; echo "--- GPU ---"; print_gpu; }

setup_sydney(){
  [[ -f "$SCRIPT_DIR/setup_sydney_rocm.sh" ]] || { err "找不到 $SCRIPT_DIR/setup_sydney_rocm.sh"; exit 1; }
  log "配置/启动 Sydney GGUF：port=$SYDNEY_PORT parallel=$SYDNEY_PARALLEL ctx=$SYDNEY_CTX_SIZE"
  # shellcheck disable=SC2086
  env HF_ENDPOINT="$HF_ENDPOINT" GITHUB_PROXY_PREFIX="$GITHUB_PROXY_PREFIX" \
    WORKDIR="$SYDNEY_WORKDIR" LLAMA_DIR="$SYDNEY_LLAMA_DIR" MODEL_LOCAL_FILE="$SYDNEY_MODEL_LOCAL_FILE" \
    HOST="$SYDNEY_HOST" PORT="$SYDNEY_PORT" PARALLEL="$SYDNEY_PARALLEL" CTX_SIZE="$SYDNEY_CTX_SIZE" \
    GPU_LAYERS="$SYDNEY_GPU_LAYERS" BATCH_SIZE="$SYDNEY_BATCH_SIZE" UBATCH_SIZE="$SYDNEY_UBATCH_SIZE" \
    USE_TUNNEL=0 $SYDNEY_EXTRA_ENV bash "$SCRIPT_DIR/setup_sydney_rocm.sh" --restart
}

start_sydney(){
  if [[ "$SYDNEY_FORCE_SETUP" == "1" || ! -x "$SYDNEY_WORKDIR/bin/start_sydney_server.sh" ]]; then
    setup_sydney
    return 0
  fi
  log "启动 Sydney: host=$SYDNEY_HOST port=$SYDNEY_PORT parallel=$SYDNEY_PARALLEL ctx=$SYDNEY_CTX_SIZE"
  # shellcheck disable=SC2086
  env HOST="$SYDNEY_HOST" PORT="$SYDNEY_PORT" PARALLEL="$SYDNEY_PARALLEL" CTX_SIZE="$SYDNEY_CTX_SIZE" \
    GPU_LAYERS="$SYDNEY_GPU_LAYERS" BATCH_SIZE="$SYDNEY_BATCH_SIZE" UBATCH_SIZE="$SYDNEY_UBATCH_SIZE" \
    $SYDNEY_EXTRA_ENV "$SYDNEY_WORKDIR/bin/start_sydney_server.sh"
}

stop_sydney(){
  if [[ -x "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh" ]]; then "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh" || true; fi
  kill_port_fallback "$SYDNEY_PORT" "sydney-llama-server"
}
restart_sydney(){ stop_sydney; start_sydney; }

stop_qwen_vllm_legacy(){ kill_pid_file "$RUN_DIR/qwen-vllm.pid" "legacy-qwen-vllm" "$QWEN_PORT"; }

setup_qwen(){
  [[ -f "$SCRIPT_DIR/setup_qwen36_27b_rocm.sh" ]] || { err "找不到 $SCRIPT_DIR/setup_qwen36_27b_rocm.sh"; exit 1; }
  stop_qwen_vllm_legacy
  log "配置/启动 Qwen3.6 GGUF：provider=$QWEN_MODEL_PROVIDER hf_repo=$QWEN_GGUF_REPO_ID ms_model=${QWEN_MS_GGUF_MODEL_ID:-<empty>} file=$QWEN_GGUF_MODEL_NAME"
  log "Qwen params: port=$QWEN_PORT parallel=$QWEN_PARALLEL ctx=$QWEN_CTX_SIZE batch=$QWEN_BATCH_SIZE ubatch=$QWEN_UBATCH_SIZE fit=$QWEN_LLAMA_ARG_FIT served=$QWEN_SERVED_MODEL_NAME"
  # shellcheck disable=SC2086
  env HF_ENDPOINT="$HF_ENDPOINT" GITHUB_PROXY_PREFIX="$GITHUB_PROXY_PREFIX" \
    WORKDIR="$QWEN_WORKDIR" LLAMA_DIR="$QWEN_LLAMA_DIR" \
    MODEL_PROVIDER="$QWEN_MODEL_PROVIDER" MODELSCOPE_MODEL_ID="$QWEN_MS_GGUF_MODEL_ID" MODELSCOPE_FILE_PATH="$QWEN_MS_GGUF_FILE_PATH" \
    HF_REPO_ID="$QWEN_GGUF_REPO_ID" MODEL_NAME="$QWEN_GGUF_MODEL_NAME" MODEL_LOCAL_FILE="$QWEN_MODEL_LOCAL_FILE" \
    MODEL_LOCAL_SEARCH_DIRS="$QWEN_MODEL_LOCAL_SEARCH_DIRS" SERVED_MODEL_NAME="$QWEN_SERVED_MODEL_NAME" \
    HOST="$QWEN_HOST" PORT="$QWEN_PORT" PARALLEL="$QWEN_PARALLEL" CTX_SIZE="$QWEN_CTX_SIZE" \
    GPU_LAYERS="$QWEN_GPU_LAYERS" BATCH_SIZE="$QWEN_BATCH_SIZE" UBATCH_SIZE="$QWEN_UBATCH_SIZE" \
    LLAMA_ARG_FIT="$QWEN_LLAMA_ARG_FIT" LLAMA_CONT_BATCHING="$QWEN_LLAMA_CONT_BATCHING" \
    TEMP="$QWEN_TEMP" TOP_P="$QWEN_TOP_P" REPEAT_PENALTY="$QWEN_REPEAT_PENALTY" \
    USE_TUNNEL=0 $QWEN_EXTRA_ENV bash "$SCRIPT_DIR/setup_qwen36_27b_rocm.sh" --restart
}

start_qwen(){
  stop_qwen_vllm_legacy
  if [[ "$QWEN_FORCE_SETUP" == "1" || ! -x "$QWEN_WORKDIR/bin/start_sydney_server.sh" || ! -f "$(qwen_model_path)" ]]; then
    setup_qwen
    return 0
  fi
  log "启动 Qwen GGUF: host=$QWEN_HOST port=$QWEN_PORT parallel=$QWEN_PARALLEL ctx=$QWEN_CTX_SIZE batch=$QWEN_BATCH_SIZE ubatch=$QWEN_UBATCH_SIZE fit=$QWEN_LLAMA_ARG_FIT"
  # shellcheck disable=SC2086
  env HOST="$QWEN_HOST" PORT="$QWEN_PORT" PARALLEL="$QWEN_PARALLEL" CTX_SIZE="$QWEN_CTX_SIZE" \
    GPU_LAYERS="$QWEN_GPU_LAYERS" BATCH_SIZE="$QWEN_BATCH_SIZE" UBATCH_SIZE="$QWEN_UBATCH_SIZE" \
    LLAMA_ARG_FIT="$QWEN_LLAMA_ARG_FIT" LLAMA_CONT_BATCHING="$QWEN_LLAMA_CONT_BATCHING" \
    TEMP="$QWEN_TEMP" TOP_P="$QWEN_TOP_P" REPEAT_PENALTY="$QWEN_REPEAT_PENALTY" \
    $QWEN_EXTRA_ENV "$QWEN_WORKDIR/bin/start_sydney_server.sh"
}

stop_qwen(){
  if [[ -x "$QWEN_WORKDIR/bin/stop_sydney_server.sh" ]]; then "$QWEN_WORKDIR/bin/stop_sydney_server.sh" || true; fi
  stop_qwen_vllm_legacy
  kill_port_fallback "$QWEN_PORT" "qwen-llama-server"
}
restart_qwen(){ stop_qwen; start_qwen; }

start_all(){
  if [[ "$START_ORDER" == "sydney-first" ]]; then start_sydney; start_qwen; else start_qwen; start_sydney; fi
}
stop_all(){ stop_qwen; stop_sydney; }
restart_all(){ stop_all; start_all; }

logs_qwen(){ touch "$QWEN_WORKDIR/logs/llama-server.log"; tail -f "$QWEN_WORKDIR/logs/llama-server.log"; }
logs_sydney(){ touch "$SYDNEY_WORKDIR/logs/llama-server.log"; tail -f "$SYDNEY_WORKDIR/logs/llama-server.log"; }

smoke_chat(){
  local port="$1" model="$2" text="$3"
  curl -fsS --max-time 120 -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @- <<JSON
{"model":"$model","messages":[{"role":"user","content":"$text"}],"max_tokens":48,"temperature":0.2}
JSON
  echo
}
test_qwen(){ smoke_chat "$QWEN_PORT" "$QWEN_SERVED_MODEL_NAME" "你好，简单回一句。"; }
test_sydney(){ smoke_chat "$SYDNEY_PORT" "$SYDNEY_MODEL_NAME" "Hi Sydney, say one short sentence."; }
test_all(){ test_qwen; test_sydney; }

write_env_template(){
  mkdir -p "$(dirname "$MANAGER_ENV_FILE")"
  [[ -f "$MANAGER_ENV_FILE" ]] && cp "$MANAGER_ENV_FILE" "$MANAGER_ENV_FILE.bak.$(date +%Y%m%d_%H%M%S)" || true
  cat > "$MANAGER_ENV_FILE" <<EOF
# model-manager persistent config: both models use llama.cpp GGUF
HF_ENDPOINT=$HF_ENDPOINT
GITHUB_PROXY_PREFIX=$GITHUB_PROXY_PREFIX
SHARED_LLAMA_DIR=$SHARED_LLAMA_DIR

SYDNEY_WORKDIR=$SYDNEY_WORKDIR
SYDNEY_PORT=$SYDNEY_PORT
SYDNEY_PARALLEL=$SYDNEY_PARALLEL
SYDNEY_CTX_SIZE=$SYDNEY_CTX_SIZE

QWEN_WORKDIR=$QWEN_WORKDIR
QWEN_PORT=$QWEN_PORT
QWEN_SERVED_MODEL_NAME=$QWEN_SERVED_MODEL_NAME
QWEN_GGUF_REPO_ID=$QWEN_GGUF_REPO_ID
QWEN_GGUF_MODEL_NAME=$QWEN_GGUF_MODEL_NAME
QWEN_MODEL_PROVIDER=$QWEN_MODEL_PROVIDER
QWEN_MS_GGUF_MODEL_ID=$QWEN_MS_GGUF_MODEL_ID
QWEN_MS_GGUF_FILE_PATH=$QWEN_MS_GGUF_FILE_PATH
QWEN_MODEL_LOCAL_FILE=$QWEN_MODEL_LOCAL_FILE
QWEN_PARALLEL=$QWEN_PARALLEL
QWEN_CTX_SIZE=$QWEN_CTX_SIZE
QWEN_BATCH_SIZE=$QWEN_BATCH_SIZE
QWEN_UBATCH_SIZE=$QWEN_UBATCH_SIZE
QWEN_LLAMA_ARG_FIT=$QWEN_LLAMA_ARG_FIT
QWEN_LLAMA_CONT_BATCHING=$QWEN_LLAMA_CONT_BATCHING
QWEN_TEMP=$QWEN_TEMP
QWEN_TOP_P=$QWEN_TOP_P
QWEN_REPEAT_PENALTY=$QWEN_REPEAT_PENALTY

START_ORDER=$START_ORDER
EOF
  log "已写入：$MANAGER_ENV_FILE"
}

print_env(){
  cat <<EOF
MANAGER_ENV_FILE=$MANAGER_ENV_FILE
STACK_DIR=$STACK_DIR
SCRIPT_DIR=$SCRIPT_DIR
APP_DIR=$APP_DIR

SYDNEY_WORKDIR=$SYDNEY_WORKDIR
SYDNEY_LLAMA_DIR=$SYDNEY_LLAMA_DIR
SYDNEY_PORT=$SYDNEY_PORT
SYDNEY_PARALLEL=$SYDNEY_PARALLEL
SYDNEY_CTX_SIZE=$SYDNEY_CTX_SIZE

QWEN_WORKDIR=$QWEN_WORKDIR
QWEN_LLAMA_DIR=$QWEN_LLAMA_DIR
QWEN_PORT=$QWEN_PORT
QWEN_SERVED_MODEL_NAME=$QWEN_SERVED_MODEL_NAME
QWEN_GGUF_REPO_ID=$QWEN_GGUF_REPO_ID
QWEN_GGUF_MODEL_NAME=$QWEN_GGUF_MODEL_NAME
QWEN_MODEL_PROVIDER=$QWEN_MODEL_PROVIDER
QWEN_MS_GGUF_MODEL_ID=$QWEN_MS_GGUF_MODEL_ID
QWEN_MS_GGUF_FILE_PATH=$QWEN_MS_GGUF_FILE_PATH
QWEN_MODEL_LOCAL_FILE=$QWEN_MODEL_LOCAL_FILE
QWEN_PARALLEL=$QWEN_PARALLEL
QWEN_CTX_SIZE=$QWEN_CTX_SIZE
QWEN_BATCH_SIZE=$QWEN_BATCH_SIZE
QWEN_UBATCH_SIZE=$QWEN_UBATCH_SIZE
QWEN_LLAMA_ARG_FIT=$QWEN_LLAMA_ARG_FIT
QWEN_LLAMA_CONT_BATCHING=$QWEN_LLAMA_CONT_BATCHING
QWEN_MODEL_PATH=$(qwen_model_path)

START_ORDER=$START_ORDER
EOF
}

usage(){
  cat <<'USAGE'
Usage: bash scripts/manage_cloud_models.sh <command>

Commands:
  status             查看两个模型和 GPU 状态
  start-all          启动两个 GGUF 模型，默认 Qwen -> Sydney
  stop-all           停止两个模型
  restart-all        重启两个模型

  setup-qwen         强制配置/下载/启动 Qwen GGUF
  start-qwen         启动 Qwen GGUF
  stop-qwen          停止 Qwen GGUF，并清理旧 vLLM 残留
  restart-qwen       重启 Qwen GGUF
  logs-qwen          跟踪 Qwen llama.cpp 日志
  test-qwen          发一条 Chat Completions 测试请求

  setup-sydney       强制配置/下载/启动 Sydney GGUF
  start-sydney       启动 Sydney GGUF
  stop-sydney        停止 Sydney GGUF
  restart-sydney     重启 Sydney GGUF
  logs-sydney        跟踪 Sydney 日志
  test-sydney        发一条 Chat Completions 测试请求

  test-all           测试两个模型接口
  env                打印当前管理参数
  write-env          写入持久配置：/workspace/sydney_cloud_stack/model-manager.env

Examples:
  bash scripts/manage_cloud_models.sh stop-qwen
  bash scripts/manage_cloud_models.sh setup-qwen
  bash scripts/manage_cloud_models.sh start-all
  QWEN_GGUF_MODEL_NAME=Qwen3.6-27B-Q6_K.gguf QWEN_FORCE_SETUP=1 bash scripts/manage_cloud_models.sh restart-qwen
  QWEN_MODEL_LOCAL_FILE=/mnt/Qwen3.6-27B-Q8_0.gguf bash scripts/manage_cloud_models.sh setup-qwen
  QWEN_MODEL_PROVIDER=modelscope QWEN_MS_GGUF_MODEL_ID=你的命名空间/Qwen3.6-27B-GGUF QWEN_MS_GGUF_FILE_PATH=Qwen3.6-27B-Q8_0.gguf bash scripts/manage_cloud_models.sh setup-qwen
USAGE
}

cmd="${1:-status}"
case "$cmd" in
  status) status_all ;;
  start-all) start_all ;;
  stop-all) stop_all ;;
  restart-all) restart_all ;;
  setup-qwen) setup_qwen ;;
  start-qwen) start_qwen ;;
  stop-qwen) stop_qwen ;;
  restart-qwen) restart_qwen ;;
  logs-qwen) logs_qwen ;;
  test-qwen) test_qwen ;;
  setup-sydney) setup_sydney ;;
  start-sydney) start_sydney ;;
  stop-sydney) stop_sydney ;;
  restart-sydney) restart_sydney ;;
  logs-sydney) logs_sydney ;;
  test-sydney) test_sydney ;;
  test-all) test_all ;;
  env) print_env ;;
  write-env) write_env_template ;;
  -h|--help|help) usage ;;
  *) err "未知命令：$cmd"; usage; exit 2 ;;
esac
