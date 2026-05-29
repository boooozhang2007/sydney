#!/usr/bin/env bash
# 管理云端两个模型：
#   1) Sydney source: llama.cpp GGUF OpenAI-compatible API
#   2) Qwen3.6-27B-FP8: vLLM OpenAI-compatible API
#
# 默认端口：
#   Sydney: http://127.0.0.1:8000/v1
#   Qwen:   http://127.0.0.1:8010/v1

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
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-}"

SHARED_LLAMA_DIR="${SHARED_LLAMA_DIR:-/workspace/llama.cpp-rocm}"

SYDNEY_WORKDIR="${SYDNEY_WORKDIR:-/workspace/sydney_rocm}"
SYDNEY_LLAMA_DIR="${SYDNEY_LLAMA_DIR:-$SHARED_LLAMA_DIR}"
SYDNEY_HOST="${SYDNEY_HOST:-127.0.0.1}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
SYDNEY_MODEL_NAME="${SYDNEY_MODEL_NAME:-clever-sydney-4-12b-q8}"
SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-64}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-262144}"
SYDNEY_GPU_LAYERS="${SYDNEY_GPU_LAYERS:-999}"
SYDNEY_BATCH_SIZE="${SYDNEY_BATCH_SIZE:-512}"
SYDNEY_UBATCH_SIZE="${SYDNEY_UBATCH_SIZE:-512}"
SYDNEY_FORCE_SETUP="${SYDNEY_FORCE_SETUP:-0}"
SYDNEY_EXTRA_ENV="${SYDNEY_EXTRA_ENV:-}"

QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8010}"
QWEN_MODEL_PROVIDER="${QWEN_MODEL_PROVIDER:-modelscope}"
QWEN_MS_MODEL_ID="${QWEN_MS_MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/workspace/modelscope/qwen36_27b_fp8}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-fp8}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.40}"
QWEN_AUTO_MEMORY_UTIL="${QWEN_AUTO_MEMORY_UTIL:-1}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-64}"
QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-1}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
QWEN_TRUST_REMOTE_CODE="${QWEN_TRUST_REMOTE_CODE:-1}"
QWEN_USE_V1="${QWEN_USE_V1:-}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-1}"
QWEN_DISABLE_CUDA_GRAPH="${QWEN_DISABLE_CUDA_GRAPH:-1}"
QWEN_CLEAR_COMPILE_CACHE="${QWEN_CLEAR_COMPILE_CACHE:-1}"
QWEN_STARTUP_TIMEOUT_SEC="${QWEN_STARTUP_TIMEOUT_SEC:-900}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# ROCm/vLLM 0.20.x 对 Qwen3.6 的 GDN/aiter/chunked prefill 组合比较敏感。
# safe-mode 保持 max-num-seqs=64，但降低单次 batched tokens，并关闭容易触发 HIP illegal access 的启动路径。
QWEN_ROCM_SAFE_MODE="${QWEN_ROCM_SAFE_MODE:-1}"
QWEN_SAFE_MAX_NUM_BATCHED_TOKENS="${QWEN_SAFE_MAX_NUM_BATCHED_TOKENS:-8192}"
QWEN_LANGUAGE_MODEL_ONLY="${QWEN_LANGUAGE_MODEL_ONLY:-1}"
QWEN_ENABLE_CHUNKED_PREFILL="${QWEN_ENABLE_CHUNKED_PREFILL:-0}"
QWEN_ASYNC_SCHEDULING="${QWEN_ASYNC_SCHEDULING:-0}"
QWEN_DISABLE_ASYNC_OUTPUT_PROC="${QWEN_DISABLE_ASYNC_OUTPUT_PROC:-1}"
QWEN_PATCH_GDN_WARMUP="${QWEN_PATCH_GDN_WARMUP:-1}"
QWEN_USE_AITER="${QWEN_USE_AITER:-0}"
QWEN_REASONING_PARSER="${QWEN_REASONING_PARSER:-}"

START_ORDER="${START_ORDER:-sydney-first}" # sydney-first | qwen-first
MANAGER_KILL_PORT_FALLBACK="${MANAGER_KILL_PORT_FALLBACK:-1}"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$QWEN_MODEL_DIR"

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
    if [[ -n "$pgid" ]]; then kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; else kill -TERM "$pid" 2>/dev/null || true; fi
    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then
      warn "$name 未正常退出，强制结束"
      if [[ -n "$pgid" ]]; then kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; else kill -KILL "$pid" 2>/dev/null || true; fi
    fi
  fi
  rm -f "$f"
  [[ -n "$port" ]] && kill_port_fallback "$port" "$name"
}

curl_models(){ local port="$1"; curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null || true; echo; }
print_gpu(){ if have rocm-smi; then rocm-smi || true; elif have nvidia-smi; then nvidia-smi || true; else echo "no rocm-smi/nvidia-smi found"; fi; }

status_sydney(){
  echo "--- Sydney llama.cpp GGUF ---"
  if [[ -x "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" ]]; then "$SYDNEY_WORKDIR/bin/status_sydney_server.sh" || true; else curl_models "$SYDNEY_PORT"; fi
}

status_qwen(){
  echo "--- Qwen vLLM ---"
  local pf="$RUN_DIR/qwen-vllm.pid"
  if pid_alive "$pf"; then
    local pid; pid="$(pid_from_file "$pf")"
    echo "qwen-vllm: running PID=$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
    echo
  else
    echo "qwen-vllm: stopped"
  fi
  curl_models "$QWEN_PORT"
}

status_all(){ status_sydney; status_qwen; echo "--- GPU ---"; print_gpu; }

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
  if [[ "$SYDNEY_FORCE_SETUP" == "1" || ! -x "$SYDNEY_WORKDIR/bin/start_sydney_server.sh" ]]; then setup_sydney; return 0; fi
  log "启动 Sydney: host=$SYDNEY_HOST port=$SYDNEY_PORT parallel=$SYDNEY_PARALLEL ctx=$SYDNEY_CTX_SIZE"
  # shellcheck disable=SC2086
  env HOST="$SYDNEY_HOST" PORT="$SYDNEY_PORT" PARALLEL="$SYDNEY_PARALLEL" CTX_SIZE="$SYDNEY_CTX_SIZE" \
    GPU_LAYERS="$SYDNEY_GPU_LAYERS" BATCH_SIZE="$SYDNEY_BATCH_SIZE" UBATCH_SIZE="$SYDNEY_UBATCH_SIZE" \
    $SYDNEY_EXTRA_ENV "$SYDNEY_WORKDIR/bin/start_sydney_server.sh"
}

stop_sydney(){ if [[ -x "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh" ]]; then "$SYDNEY_WORKDIR/bin/stop_sydney_server.sh" || true; fi; kill_port_fallback "$SYDNEY_PORT" "sydney-llama-server"; }
restart_sydney(){ stop_sydney; start_sydney; }

modelscope_download_qwen_fp8(){
  if [[ -f "$QWEN_MODEL_DIR/config.json" ]]; then log "Qwen FP8 已存在：$QWEN_MODEL_DIR"; return 0; fi
  log "Qwen 模型下载 provider=$QWEN_MODEL_PROVIDER（默认/优先 ModelScope）"
  log "用 ModelScope 下载 Qwen FP8：$QWEN_MS_MODEL_ID -> $QWEN_MODEL_DIR"
  QWEN_MS_MODEL_ID="$QWEN_MS_MODEL_ID" QWEN_MODEL_DIR="$QWEN_MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
model_id=os.environ['QWEN_MS_MODEL_ID']; local_dir=os.environ['QWEN_MODEL_DIR']
Path(local_dir).mkdir(parents=True, exist_ok=True)
try:
    from modelscope import snapshot_download
except Exception:
    from modelscope.hub.snapshot_download import snapshot_download
for kwargs in (dict(model_id=model_id, local_dir=local_dir), dict(model_id=model_id, cache_dir=local_dir)):
    try:
        print('[modelscope] snapshot_download', kwargs, flush=True); snapshot_download(**kwargs); break
    except TypeError:
        continue
PY
  if [[ ! -f "$QWEN_MODEL_DIR/config.json" ]]; then
    local found; found="$(find "$QWEN_MODEL_DIR" -maxdepth 5 -name config.json -type f | head -n 1 || true)"
    [[ -n "$found" ]] || { err "Qwen FP8 下载后未找到 config.json：$QWEN_MODEL_DIR"; exit 1; }
    QWEN_MODEL_DIR="$(dirname "$found")"; log "自动定位 Qwen 模型目录：$QWEN_MODEL_DIR"
  fi
}

calc_qwen_gpu_util(){
  if [[ "$QWEN_AUTO_MEMORY_UTIL" != "1" ]]; then echo "$QWEN_GPU_MEMORY_UTILIZATION"; return 0; fi
  "$PYTHON_BIN" - <<'PY'
import os
fallback=float(os.environ.get('QWEN_GPU_MEMORY_UTILIZATION','0.40'))
try:
 import torch
 free,total=torch.cuda.mem_get_info(); free_gib=free/1024**3; total_gib=total/1024**3
 print(f'{max(0.10,min(fallback,(free_gib-8.0)/total_gib)):.3f}')
except Exception:
 print(fallback)
PY
}

check_vllm_available(){
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('vllm') else 1)
PY
  then err "当前 Python 环境没有 vLLM：$PYTHON_BIN"; exit 1; fi
}

vllm_help_file(){ echo "$RUN_DIR/vllm-api-server.help"; }
refresh_vllm_help(){ local hf; hf="$(vllm_help_file)"; [[ -s "$hf" && "${VLLM_REFRESH_HELP:-0}" != "1" ]] || "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help > "$hf" 2>&1 || true; }
vllm_supports(){ local flag="$1"; refresh_vllm_help; grep -q -- "$flag" "$(vllm_help_file)" 2>/dev/null; }

vllm_envs_file(){ echo "$RUN_DIR/vllm-envs.list"; }
refresh_vllm_envs(){
  local ef; ef="$(vllm_envs_file)"
  [[ -s "$ef" && "${VLLM_REFRESH_ENVS:-0}" != "1" ]] && return 0
  "$PYTHON_BIN" - <<'PY' > "$ef" 2>/dev/null || true
try:
    import vllm.envs as envs
    names = set()
    for obj_name in ('environment_variables', 'ENV_VARS'):
        obj = getattr(envs, obj_name, None)
        if isinstance(obj, dict):
            names.update(str(k) for k in obj.keys())
    names.update(n for n in dir(envs) if n.isupper())
    for n in sorted(names):
        print(n)
except Exception:
    pass
PY
}
vllm_env_known(){ local name="$1"; refresh_vllm_envs; grep -qx -- "$name" "$(vllm_envs_file)" 2>/dev/null; }
export_vllm_env_if_known(){
  local name="$1" default_value="$2"
  if vllm_env_known "$name"; then
    if [[ -z "${!name+x}" ]]; then export "$name=$default_value"; else export "$name=${!name}"; fi
  else
    [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 不识别环境变量 $name，已跳过，避免 unknown env 警告。"
  fi
}

append_vllm_switch(){
  local arr_name="$1" flag="$2"; local -n _arr="$arr_name"
  if vllm_supports "$flag"; then _arr+=("$flag"); else [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 不支持参数 $flag，已跳过。"; fi
}
append_vllm_option(){
  local arr_name="$1" flag="$2" value="$3"; local -n _arr="$arr_name"
  if vllm_supports "$flag"; then _arr+=("$flag" "$value"); else [[ "${4:-0}" == "1" ]] && warn "当前 vLLM 不支持参数 $flag，已跳过。"; fi
}
append_vllm_disable_bool(){
  local arr_name="$1" flag="$2"; local -n _arr="$arr_name"
  local no_flag="--no-${flag#--}"
  local disable_flag="$flag"
  if [[ "$flag" == --enable-* ]]; then disable_flag="--disable-${flag#--enable-}"; fi
  if vllm_supports "$no_flag"; then
    _arr+=("$no_flag")
  elif [[ "$disable_flag" != "$flag" ]] && vllm_supports "$disable_flag"; then
    _arr+=("$disable_flag")
  else
    [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 未暴露关闭 $flag 的 CLI 参数，已跳过。"
  fi
}

apply_qwen_rocm_safe_defaults(){
  [[ "$QWEN_ROCM_SAFE_MODE" == "1" ]] || return 0
  if [[ "$QWEN_MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ && "$QWEN_SAFE_MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ && "$QWEN_MAX_NUM_BATCHED_TOKENS" -gt "$QWEN_SAFE_MAX_NUM_BATCHED_TOKENS" ]]; then
    warn "ROCm safe-mode: QWEN_MAX_NUM_BATCHED_TOKENS $QWEN_MAX_NUM_BATCHED_TOKENS -> $QWEN_SAFE_MAX_NUM_BATCHED_TOKENS（并发 seqs 仍为 $QWEN_MAX_NUM_SEQS）"
    QWEN_MAX_NUM_BATCHED_TOKENS="$QWEN_SAFE_MAX_NUM_BATCHED_TOKENS"
  fi
}

prepare_qwen_gdn_patch(){
  [[ "$QWEN_PATCH_GDN_WARMUP" == "1" ]] || return 0
  local patch_dir="$STACK_DIR/vllm_patches"
  mkdir -p "$patch_dir"
  cat > "$patch_dir/sitecustomize.py" <<'PY'
# Runtime workaround for ROCm/vLLM/Qwen3.6 GDN warmup HIP illegal memory access.
import builtins
import os
import sys

_TARGETS = (
    "vllm.model_executor.layers.mamba.gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
)

def _noop_warmup(self, *args, **kwargs):
    return None

def _patch_module(mod):
    if mod is None or getattr(mod, "_sydney_rocm_gdn_warmup_patch", False):
        return
    patched = 0
    for _, obj in list(vars(mod).items()):
        if isinstance(obj, type):
            for attr in list(vars(obj).keys()):
                if attr.startswith("_warmup") and "kernel" in attr:
                    try:
                        setattr(obj, attr, _noop_warmup)
                        patched += 1
                    except Exception:
                        pass
    if patched:
        setattr(mod, "_sydney_rocm_gdn_warmup_patch", True)
        try:
            sys.stderr.write(f"[sydney-vllm-patch] disabled {patched} GDN warmup kernel method(s) in {mod.__name__}\n")
        except Exception:
            pass

def _patch_loaded():
    for name in _TARGETS:
        _patch_module(sys.modules.get(name))

if os.environ.get("VLLM_QWEN_GDN_DISABLE_WARMUP") == "1":
    _orig_import = builtins.__import__
    def _import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        if "gdn" in name or "mamba" in name or name.startswith("vllm"):
            _patch_loaded()
        return mod
    builtins.__import__ = _import_hook
    _patch_loaded()
PY
  export VLLM_QWEN_GDN_DISABLE_WARMUP=1
  case ":${PYTHONPATH:-}:" in
    *:"$patch_dir":*) ;;
    *) export PYTHONPATH="$patch_dir${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  log "ROCm safe-mode: 已启用 Qwen GDN warmup runtime patch，可用 QWEN_PATCH_GDN_WARMUP=0 关闭。"
}

apply_qwen_vllm_envs(){
  export HF_ENDPOINT="$HF_ENDPOINT"
  export VLLM_USE_MODELSCOPE="True"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
  if [[ -n "$QWEN_USE_V1" ]]; then
    export_vllm_env_if_known VLLM_USE_V1 "$QWEN_USE_V1" 1
  fi
  if [[ "$QWEN_ROCM_SAFE_MODE" == "1" ]]; then
    export_vllm_env_if_known VLLM_ROCM_USE_AITER "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_LINEAR "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_MOE "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_RMSNORM "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_PAGED_ATTN "$QWEN_USE_AITER" 0
    prepare_qwen_gdn_patch
  fi
}

start_qwen(){
  if pid_alive "$RUN_DIR/qwen-vllm.pid"; then log "Qwen vLLM 已在运行：PID=$(pid_from_file "$RUN_DIR/qwen-vllm.pid")"; return 0; fi
  check_vllm_available
  modelscope_download_qwen_fp8
  [[ "$QWEN_CLEAR_COMPILE_CACHE" == "1" ]] && rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
  local util; util="$(calc_qwen_gpu_util)"
  apply_qwen_rocm_safe_defaults
  apply_qwen_vllm_envs
  local args=(
    --host "$QWEN_HOST"
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
  [[ "$QWEN_TRUST_REMOTE_CODE" == "1" ]] && append_vllm_switch args "--trust-remote-code"
  [[ "$QWEN_ENFORCE_EAGER" == "1" ]] && append_vllm_switch args "--enforce-eager"
  [[ "$QWEN_LANGUAGE_MODEL_ONLY" == "1" ]] && append_vllm_switch args "--language-model-only"
  [[ -n "$QWEN_REASONING_PARSER" ]] && append_vllm_option args "--reasoning-parser" "$QWEN_REASONING_PARSER"
  [[ "$QWEN_ENABLE_CHUNKED_PREFILL" == "0" ]] && append_vllm_disable_bool args "--enable-chunked-prefill" 1
  [[ "$QWEN_ASYNC_SCHEDULING" == "0" ]] && append_vllm_disable_bool args "--async-scheduling" 0
  [[ "$QWEN_DISABLE_ASYNC_OUTPUT_PROC" == "1" ]] && append_vllm_switch args "--disable-async-output-proc"
  if [[ "$QWEN_DISABLE_CUDA_GRAPH" == "1" ]]; then
    if vllm_supports "--disable-cudagraph"; then args+=(--disable-cudagraph); else warn "当前 vLLM 无 --disable-cudagraph；已使用 --enforce-eager 并跳过 VLLM_USE_V1 unknown env。"; fi
  fi
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then # shellcheck disable=SC2206
    extra=( $VLLM_EXTRA_ARGS ); args+=("${extra[@]}")
  fi
  log "启动 Qwen vLLM: port=$QWEN_PORT util=$util seqs=$QWEN_MAX_NUM_SEQS max_len=$QWEN_MAX_MODEL_LEN"
  log "args: ${args[*]}"
  nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$LOG_DIR/qwen-vllm.log" 2>&1 &
  echo $! > "$RUN_DIR/qwen-vllm.pid"
  local loops=$(( QWEN_STARTUP_TIMEOUT_SEC / 2 )); [[ "$loops" -lt 1 ]] && loops=1
  for ((i=1; i<=loops; i++)); do
    if curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then log "Qwen vLLM ready"; return 0; fi
    if ! pid_alive "$RUN_DIR/qwen-vllm.pid"; then rm -f "$RUN_DIR/qwen-vllm.pid"; err "Qwen vLLM failed. tail log:"; tail -n 160 "$LOG_DIR/qwen-vllm.log" || true; exit 1; fi
    sleep 2
  done
  err "Qwen vLLM startup timeout. tail log:"; tail -n 160 "$LOG_DIR/qwen-vllm.log" || true; exit 1
}

stop_qwen(){ kill_pid_file "$RUN_DIR/qwen-vllm.pid" "qwen-vllm" "$QWEN_PORT"; }
restart_qwen(){ stop_qwen; start_qwen; }

start_all(){ if [[ "$START_ORDER" == "qwen-first" ]]; then start_qwen; start_sydney; else start_sydney; start_qwen; fi; }
stop_all(){ stop_qwen; stop_sydney; }
restart_all(){ stop_all; start_all; }

logs_qwen(){ touch "$LOG_DIR/qwen-vllm.log"; tail -f "$LOG_DIR/qwen-vllm.log"; }
logs_sydney(){ touch "$SYDNEY_WORKDIR/logs/llama-server.log"; tail -f "$SYDNEY_WORKDIR/logs/llama-server.log"; }

smoke_chat(){
  local port="$1" model="$2" text="$3"
  curl -fsS --max-time 120 -X POST "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON
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
# model-manager persistent config: Sydney llama.cpp + Qwen vLLM
HF_ENDPOINT=$HF_ENDPOINT
GITHUB_PROXY_PREFIX=$GITHUB_PROXY_PREFIX
SHARED_LLAMA_DIR=$SHARED_LLAMA_DIR

SYDNEY_WORKDIR=$SYDNEY_WORKDIR
SYDNEY_PORT=$SYDNEY_PORT
SYDNEY_PARALLEL=$SYDNEY_PARALLEL
SYDNEY_CTX_SIZE=$SYDNEY_CTX_SIZE

QWEN_PORT=$QWEN_PORT
QWEN_MODEL_PROVIDER=$QWEN_MODEL_PROVIDER
QWEN_MS_MODEL_ID=$QWEN_MS_MODEL_ID
QWEN_MODEL_DIR=$QWEN_MODEL_DIR
QWEN_SERVED_MODEL_NAME=$QWEN_SERVED_MODEL_NAME
QWEN_GPU_MEMORY_UTILIZATION=$QWEN_GPU_MEMORY_UTILIZATION
QWEN_AUTO_MEMORY_UTIL=$QWEN_AUTO_MEMORY_UTIL
QWEN_MAX_MODEL_LEN=$QWEN_MAX_MODEL_LEN
QWEN_MAX_NUM_SEQS=$QWEN_MAX_NUM_SEQS
QWEN_MAX_NUM_BATCHED_TOKENS=$QWEN_MAX_NUM_BATCHED_TOKENS
QWEN_USE_V1=$QWEN_USE_V1
QWEN_ENFORCE_EAGER=$QWEN_ENFORCE_EAGER
QWEN_DISABLE_CUDA_GRAPH=$QWEN_DISABLE_CUDA_GRAPH

START_ORDER=$START_ORDER
EOF
  log "已写入：$MANAGER_ENV_FILE"
}

print_env(){
  cat <<EOF
SYDNEY_PARALLEL=$SYDNEY_PARALLEL
SYDNEY_CTX_SIZE=$SYDNEY_CTX_SIZE
QWEN_MODEL_PROVIDER=$QWEN_MODEL_PROVIDER
QWEN_MS_MODEL_ID=$QWEN_MS_MODEL_ID
QWEN_MODEL_DIR=$QWEN_MODEL_DIR
QWEN_MAX_NUM_SEQS=$QWEN_MAX_NUM_SEQS
QWEN_MAX_NUM_BATCHED_TOKENS=$QWEN_MAX_NUM_BATCHED_TOKENS
QWEN_GPU_MEMORY_UTILIZATION=$QWEN_GPU_MEMORY_UTILIZATION
START_ORDER=$START_ORDER
EOF
}

usage(){
  cat <<'USAGE'
Usage: bash scripts/manage_cloud_models.sh <command>

Commands:
  status/start-all/stop-all/restart-all
  start-qwen stop-qwen restart-qwen logs-qwen test-qwen
  setup-sydney start-sydney stop-sydney restart-sydney logs-sydney test-sydney
  test-all env write-env

Defaults: APP/model concurrency 64, Sydney parallel 64, Qwen vLLM max-num-seqs 64.
USAGE
}

cmd="${1:-status}"
case "$cmd" in
  status) status_all ;;
  start-all) start_all ;;
  stop-all) stop_all ;;
  restart-all) restart_all ;;
  start-qwen|setup-qwen) start_qwen ;;
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
