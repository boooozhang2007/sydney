#!/usr/bin/env bash
# 管理云端两个模型：
#   Sydney llama.cpp: /workspace/sydney_rocm, port 8000
#   Qwen vLLM:      /workspace/sydney_cloud_stack/run/qwen-vllm.pid, port 8010
#
# 用法：
#   bash scripts/manage_cloud_models.sh status
#   bash scripts/manage_cloud_models.sh stop-qwen
#   bash scripts/manage_cloud_models.sh start-qwen
#   bash scripts/manage_cloud_models.sh restart-qwen
#   bash scripts/manage_cloud_models.sh stop-sydney
#   bash scripts/manage_cloud_models.sh start-sydney
#   bash scripts/manage_cloud_models.sh restart-sydney
#   bash scripts/manage_cloud_models.sh stop-all
#   bash scripts/manage_cloud_models.sh logs-qwen
#   bash scripts/manage_cloud_models.sh logs-sydney

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/workspace/sydney_NEWBING}"
STACK_DIR="${STACK_DIR:-/workspace/sydney_cloud_stack}"
LOG_DIR="$STACK_DIR/logs"
RUN_DIR="$STACK_DIR/run"

SYDNEY_WORKDIR="${SYDNEY_WORKDIR:-/workspace/sydney_rocm}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-100}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-400000}"

QWEN_PORT="${QWEN_PORT:-8010}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/workspace/modelscope/qwen36_27b_fp8}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-fp8}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.45}"
QWEN_AUTO_MEMORY_UTIL="${QWEN_AUTO_MEMORY_UTIL:-0}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-64}"
QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-65536}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-1}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
QWEN_USE_V1="${QWEN_USE_V1:-0}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-1}"
QWEN_DISABLE_CUDA_GRAPH="${QWEN_DISABLE_CUDA_GRAPH:-1}"
QWEN_CLEAR_COMPILE_CACHE="${QWEN_CLEAR_COMPILE_CACHE:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

log(){ printf '\033[1;36m[model-manager]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

pid_alive(){ [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
kill_pid_file(){
  local f="$1" name="${2:-process}"
  if pid_alive "$f"; then
    local pid; pid="$(cat "$f")"
    log "停止 $name PID=$pid"
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$f"
}

status_sydney(){
  echo "--- Sydney llama.cpp ---"
  if [[ -x "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" ]]; then
    "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" || true
  else
    echo "status script not found: $SYDNEY_WORKDIR/bin/status_sydney_server.sh"
    curl -fsS "http://127.0.0.1:$SYDNEY_PORT/v1/models" || true
    echo
  fi
}

status_qwen(){
  echo "--- Qwen vLLM ---"
  local pf="$RUN_DIR/qwen-vllm.pid"
  if pid_alive "$pf"; then
    local pid; pid="$(cat "$pf")"
    echo "qwen-vllm: running PID=$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
    echo
  else
    echo "qwen-vllm: stopped"
  fi
  curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" || true
  echo
}

status_all(){
  status_sydney
  status_qwen
  echo "--- GPU ---"
  rocm-smi 2>/dev/null || nvidia-smi 2>/dev/null || true
}

start_sydney(){
  if [[ ! -x "$SYDNEY_WORKDIR/bin/start_sydney_server.sh" ]]; then
    err "找不到 Sydney 启动脚本：$SYDNEY_WORKDIR/bin/start_sydney_server.sh"
    err "请先运行 deploy_cloud_full_rocm.sh 或 setup_sydney_rocm.sh"
    exit 1
  fi
  log "启动 Sydney: port=$SYDNEY_PORT parallel=$SYDNEY_PARALLEL ctx=$SYDNEY_CTX_SIZE"
  PORT="$SYDNEY_PORT" PARALLEL="$SYDNEY_PARALLEL" CTX_SIZE="$SYDNEY_CTX_SIZE" \
    "$SYDNEY_WORKDIR/bin/start_sydney_server.sh"
}

stop_sydney(){
  if [[ -x "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh" ]]; then
    "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh"
  else
    warn "Sydney stop script not found"
  fi
}

restart_sydney(){ stop_sydney; start_sydney; }

calc_qwen_gpu_util(){
  if [[ "$QWEN_AUTO_MEMORY_UTIL" != "1" ]]; then
    echo "$QWEN_GPU_MEMORY_UTILIZATION"
    return 0
  fi
  python3 - <<PY
import os
fallback=float(os.environ.get('QWEN_GPU_MEMORY_UTILIZATION','0.45'))
try:
 import torch
 free,total=torch.cuda.mem_get_info()
 free_gib=free/1024**3; total_gib=total/1024**3
 util=max(0.10,min(fallback,(free_gib-8.0)/total_gib))
 print(f'{util:.3f}')
except Exception:
 print(fallback)
PY
}

start_qwen(){
  if pid_alive "$RUN_DIR/qwen-vllm.pid"; then
    log "Qwen vLLM 已在运行：PID=$(cat "$RUN_DIR/qwen-vllm.pid")"
    return 0
  fi
  if [[ ! -f "$QWEN_MODEL_DIR/config.json" ]]; then
    err "Qwen 模型目录无 config.json：$QWEN_MODEL_DIR"
    err "请先运行 deploy_cloud_full_rocm.sh 下载 ModelScope 模型。"
    exit 1
  fi
  if [[ "$QWEN_CLEAR_COMPILE_CACHE" == "1" ]]; then
    rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
  fi
  local util; util="$(calc_qwen_gpu_util)"
  export HF_ENDPOINT="$HF_ENDPOINT"
  export VLLM_USE_MODELSCOPE="True"
  export VLLM_USE_V1="$QWEN_USE_V1"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
  local args=(
    --host 127.0.0.1
    --port "$QWEN_PORT"
    --model "$QWEN_MODEL_DIR"
    --served-model-name "$QWEN_SERVED_MODEL_NAME"
    --dtype "$QWEN_DTYPE"
    --tensor-parallel-size "$QWEN_TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$util"
    --max-model-len "$QWEN_MAX_MODEL_LEN"
    --max-num-seqs "$QWEN_MAX_NUM_SEQS"
    --max-num-batched-tokens "$QWEN_MAX_NUM_BATCHED_TOKENS"
  )
  [[ "$QWEN_ENFORCE_EAGER" == "1" ]] && args+=(--enforce-eager)
  [[ "$QWEN_DISABLE_CUDA_GRAPH" == "1" ]] && args+=(--disable-cudagraph)
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra=( $VLLM_EXTRA_ARGS )
    args+=("${extra[@]}")
  fi
  log "启动 Qwen vLLM: port=$QWEN_PORT util=$util seqs=$QWEN_MAX_NUM_SEQS max_len=$QWEN_MAX_MODEL_LEN"
  log "args: ${args[*]}"
  nohup python3 -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$LOG_DIR/qwen-vllm.log" 2>&1 &
  echo $! > "$RUN_DIR/qwen-vllm.pid"
  for i in {1..300}; do
    if curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then
      log "Qwen vLLM ready"
      return 0
    fi
    if ! pid_alive "$RUN_DIR/qwen-vllm.pid"; then
      rm -f "$RUN_DIR/qwen-vllm.pid"
      err "Qwen vLLM failed. tail log:"
      tail -n 120 "$LOG_DIR/qwen-vllm.log" || true
      exit 1
    fi
    sleep 2
  done
  err "Qwen vLLM startup timeout. tail log:"
  tail -n 120 "$LOG_DIR/qwen-vllm.log" || true
  exit 1
}

stop_qwen(){ kill_pid_file "$RUN_DIR/qwen-vllm.pid" "qwen-vllm"; }
restart_qwen(){ stop_qwen; start_qwen; }

logs_qwen(){ tail -f "$LOG_DIR/qwen-vllm.log"; }
logs_sydney(){ tail -f "$SYDNEY_WORKDIR/logs/llama-server.log"; }

usage(){
  cat <<USAGE
Usage: bash scripts/manage_cloud_models.sh <command>

Commands:
  status             查看两个模型和 GPU 状态
  start-sydney       启动 Sydney llama.cpp
  stop-sydney        停止 Sydney llama.cpp
  restart-sydney     重启 Sydney llama.cpp
  logs-sydney        跟踪 Sydney 日志
  start-qwen         启动 Qwen vLLM
  stop-qwen          停止 Qwen vLLM
  restart-qwen       重启 Qwen vLLM
  logs-qwen          跟踪 Qwen 日志
  start-all          启动两个模型
  stop-all           停止两个模型
  restart-all        重启两个模型

Common env:
  SYDNEY_PARALLEL=100 SYDNEY_CTX_SIZE=400000
  QWEN_GPU_MEMORY_UTILIZATION=0.45 QWEN_MAX_NUM_SEQS=64 QWEN_MAX_NUM_BATCHED_TOKENS=65536
  QWEN_USE_V1=0 QWEN_ENFORCE_EAGER=1 QWEN_DISABLE_CUDA_GRAPH=1
USAGE
}

cmd="${1:-status}"
case "$cmd" in
  status) status_all ;;
  start-sydney) start_sydney ;;
  stop-sydney) stop_sydney ;;
  restart-sydney) restart_sydney ;;
  logs-sydney) logs_sydney ;;
  start-qwen) start_qwen ;;
  stop-qwen) stop_qwen ;;
  restart-qwen) restart_qwen ;;
  logs-qwen) logs_qwen ;;
  start-all) start_sydney; start_qwen ;;
  stop-all) stop_qwen; stop_sydney ;;
  restart-all) stop_qwen; stop_sydney; start_sydney; start_qwen ;;
  -h|--help|help) usage ;;
  *) err "未知命令：$cmd"; usage; exit 2 ;;
esac
