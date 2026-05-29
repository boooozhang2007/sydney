#!/usr/bin/env bash
# 云端全流程部署：
#   Sydney source：llama.cpp + GGUF，127.0.0.1:8000
#   Qwen3.6-27B-FP8：vLLM OpenAI-compatible API，127.0.0.1:8010
#   控制台：FastAPI，127.0.0.1:7860
#   只暴露一个 Cloudflare Tunnel：控制台，不暴露模型端口
#
# 用法：
#   bash scripts/deploy_cloud_full_rocm.sh
#   bash scripts/deploy_cloud_full_rocm.sh --console-only   # 只启动/暴露控制台，不重启模型

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/workspace/sydney_NEWBING}"
REPO_URL="${REPO_URL:-https://gitee.com/qzonez/sydney.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_PROXY_PREFIX="${GIT_PROXY_PREFIX:-}"
# 只加速 GitHub/Hugging Face；apt/pip 保持当前镜像环境默认源。
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-7860}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
QWEN_PORT="${QWEN_PORT:-8010}"

APP_CONCURRENCY="${APP_CONCURRENCY:-64}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-64}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-262144}"
SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"

# Qwen FP8 + vLLM 参数。ROCm 上默认保守，优先稳定；要冲吞吐再逐步调高。
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
VLLM_HELP_TIMEOUT_SEC="${VLLM_HELP_TIMEOUT_SEC:-45}"
# ROCm 镜像里 `python -m vllm.entrypoints.openai.api_server --help` 可能导入很慢甚至卡住；
# 默认不做动态探测，只传 vLLM 0.20.x ROCm 常用稳定参数。需要动态兼容时可设为 0。
VLLM_SKIP_OPTION_PROBE="${VLLM_SKIP_OPTION_PROBE:-1}"
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
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"

LLAMA_DIR="${LLAMA_DIR:-/workspace/sydney_rocm/llama.cpp}"

CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH:-/mnt/cloudflared}"
CLOUDFLARED_URL="${CLOUDFLARED_URL:-https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS:-$CLOUDFLARED_URL https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"

STACK_DIR="${STACK_DIR:-/workspace/sydney_cloud_stack}"
LOG_DIR="$STACK_DIR/logs"
RUN_DIR="$STACK_DIR/run"
BIN_DIR="$STACK_DIR/bin"
CONSOLE_ONLY="0"
SKIP_MODEL_START="0"

usage(){
  cat <<'USAGE'
云端全流程部署 Sydney + Qwen + Web 控制台，并且只把 Web 控制台暴露到 Cloudflare Tunnel。

Usage:
  bash scripts/deploy_cloud_full_rocm.sh [options]

Options:
  --console-only   只启动 Web 控制台和控制台 Tunnel，不启动/重启两个模型
  --skip-models    同 --console-only
  -h, --help       显示帮助

默认：
  Sydney: 127.0.0.1:8000  parallel=64 ctx_size=262144
  Qwen:   127.0.0.1:8010  max_num_seqs=64
  Console tunnel: Cloudflare -> http://127.0.0.1:7860
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --console-only|--skip-models) CONSOLE_ONLY="1"; SKIP_MODEL_START="1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

log(){ printf '\033[1;36m[cloud-stack]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
have(){ command -v "$1" >/dev/null 2>&1; }

ensure_dirs(){ mkdir -p "$STACK_DIR" "$LOG_DIR" "$RUN_DIR" "$BIN_DIR" "$QWEN_MODEL_DIR"; }

apt_install(){
  if have apt-get; then
    log "安装系统依赖"
    apt-get update -y || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git curl wget aria2 ca-certificates procps lsof python3 python3-pip jq || true
  fi
}

pip_install(){
  log "安装/检查 Python 依赖，使用当前环境默认 pip 源"
  python3 -m pip install -U modelscope uvicorn fastapi python-dotenv httpx pydantic
}

check_vllm(){
  if [[ "$SKIP_MODEL_START" == "1" ]]; then
    warn "console-only: 跳过 vLLM 检查。"
    return 0
  fi
  if [[ "$SKIP_VLLM_CHECK" == "1" ]]; then
    warn "已跳过 vLLM 检查：SKIP_VLLM_CHECK=1"
    return 0
  fi
  log "检查当前环境 vLLM"
  if ! python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec('vllm')
if spec is None:
    print('vLLM not found', file=sys.stderr)
    sys.exit(1)
import vllm
print('vLLM OK:', getattr(vllm, '__version__', 'unknown'))
PY
  then
    err "当前 Python 环境没有可用 vLLM。请切换到自带 vLLM/ROCm 的镜像，或手动安装后重跑。"
    exit 1
  fi
}

clone_or_update_repo(){
  if [[ -f "$APP_DIR/app.py" ]]; then
    log "源码已存在：$APP_DIR"
    cd "$APP_DIR"
    if [[ -n "$REPO_URL" && -d .git ]]; then
      log "设置源码 origin：$REPO_URL"
      git remote set-url origin "$REPO_URL" || true
    fi
    git pull --ff-only || warn "git pull 失败，继续使用本地源码"
    return 0
  fi
  if [[ -z "$REPO_URL" ]]; then
    err "APP_DIR 不存在且 REPO_URL 未设置。请设置 REPO_URL=https://gitee.com/qzonez/sydney.git 或你的仓库地址"
    exit 1
  fi
  mkdir -p "$(dirname "$APP_DIR")"
  log "克隆源码：$REPO_URL -> $APP_DIR"
  git -c http.version=HTTP/1.1 clone --depth 1 --branch "$GIT_BRANCH" "${GIT_PROXY_PREFIX}${REPO_URL}" "$APP_DIR" \
    || git -c http.version=HTTP/1.1 clone --depth 1 --branch "$GIT_BRANCH" "$REPO_URL" "$APP_DIR"
}

install_repo_requirements(){
  cd "$APP_DIR"
  if [[ -f requirements-dataset.txt ]]; then
    log "安装项目 requirements-dataset.txt"
    python3 -m pip install -r requirements-dataset.txt
  fi
}

modelscope_download_qwen_fp8(){
  if [[ -f "$QWEN_MODEL_DIR/config.json" ]]; then
    log "Qwen FP8 已存在：$QWEN_MODEL_DIR"
    return 0
  fi
  log "Qwen 模型下载 provider=$QWEN_MODEL_PROVIDER（默认/优先 ModelScope）"
  log "用 ModelScope 下载 Qwen FP8：$QWEN_MS_MODEL_ID -> $QWEN_MODEL_DIR"
  QWEN_MS_MODEL_ID="$QWEN_MS_MODEL_ID" QWEN_MODEL_DIR="$QWEN_MODEL_DIR" python3 - <<'PY'
import os
from pathlib import Path
model_id = os.environ['QWEN_MS_MODEL_ID']
local_dir = os.environ['QWEN_MODEL_DIR']
Path(local_dir).mkdir(parents=True, exist_ok=True)
try:
    from modelscope import snapshot_download
except Exception:
    from modelscope.hub.snapshot_download import snapshot_download

for kwargs in (
    dict(model_id=model_id, local_dir=local_dir),
    dict(model_id=model_id, cache_dir=local_dir),
):
    try:
        print('[modelscope] snapshot_download', kwargs, flush=True)
        snapshot_download(**kwargs)
        break
    except TypeError:
        continue
PY
  if [[ ! -f "$QWEN_MODEL_DIR/config.json" ]]; then
    local found
    found="$(find "$QWEN_MODEL_DIR" -maxdepth 5 -name config.json -type f | head -n 1 || true)"
    if [[ -n "$found" ]]; then
      QWEN_MODEL_DIR="$(dirname "$found")"
      log "自动定位 Qwen 模型目录：$QWEN_MODEL_DIR"
    else
      err "Qwen FP8 下载后未找到 config.json，请检查 QWEN_MS_MODEL_ID=$QWEN_MS_MODEL_ID"
      exit 1
    fi
  fi
}

start_sydney(){
  cd "$APP_DIR"
  log "启动 Sydney source 模型：127.0.0.1:$SYDNEY_PORT"
  MODEL_LOCAL_FILE="$SYDNEY_MODEL_LOCAL_FILE" \
  GITHUB_PROXY_PREFIX="$GIT_PROXY_PREFIX" \
  HF_ENDPOINT="$HF_ENDPOINT" \
  WORKDIR=/workspace/sydney_rocm \
  LLAMA_DIR="$LLAMA_DIR" \
  PORT="$SYDNEY_PORT" \
  PARALLEL="$SYDNEY_PARALLEL" \
  CTX_SIZE="$SYDNEY_CTX_SIZE" \
  USE_TUNNEL=0 \
  bash scripts/setup_sydney_rocm.sh --restart
}

detect_qwen_gpu_memory_utilization(){
  if [[ "$QWEN_AUTO_MEMORY_UTIL" != "1" ]]; then
    echo "$QWEN_GPU_MEMORY_UTILIZATION"
    return 0
  fi
  python3 - <<PY
import os
fallback = float(os.environ.get('QWEN_GPU_MEMORY_UTILIZATION', '0.40'))
try:
    import torch
    if not torch.cuda.is_available():
        print(fallback); raise SystemExit
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    total_gib = total / 1024**3
    util = max(0.10, min(fallback, (free_gib - 8.0) / total_gib))
    print(f"{util:.3f}")
except Exception:
    print(fallback)
PY
}

vllm_help_file(){ echo "$RUN_DIR/vllm-api-server.help"; }
refresh_vllm_help(){
  local hf tmp rc=0
  hf="$(vllm_help_file)"
  if [[ "$VLLM_SKIP_OPTION_PROBE" == "1" ]]; then
    [[ "${_VLLM_HELP_SKIP_LOGGED:-0}" == "1" ]] || warn "已跳过 vLLM CLI 参数探测：VLLM_SKIP_OPTION_PROBE=1"
    _VLLM_HELP_SKIP_LOGGED=1
    : > "$hf"
    return 0
  fi
  [[ "${_VLLM_HELP_PROBED:-0}" == "1" ]] && return 0
  [[ -e "$hf" && "${VLLM_REFRESH_HELP:-0}" != "1" ]] && return 0
  _VLLM_HELP_PROBED=1
  tmp="$hf.tmp"
  rm -f "$tmp"
  log "探测 vLLM CLI 参数（最多 ${VLLM_HELP_TIMEOUT_SEC}s；首次导入 vLLM 可能较慢）"
  if have timeout; then
    timeout "$VLLM_HELP_TIMEOUT_SEC" python3 -m vllm.entrypoints.openai.api_server --help > "$tmp" 2>&1 || rc=$?
  else
    python3 -m vllm.entrypoints.openai.api_server --help > "$tmp" 2>&1 || rc=$?
  fi
  if [[ "$rc" == "0" && -s "$tmp" ]]; then
    mv "$tmp" "$hf"
    return 0
  fi
  warn "vLLM CLI 参数探测失败/超时（rc=$rc），将跳过无法确认的可选参数，避免卡在 --help。"
  [[ -s "$tmp" ]] && tail -n 40 "$tmp" >&2 || true
  : > "$hf"
  rm -f "$tmp"
}
vllm_supports(){ local flag="$1"; refresh_vllm_help; grep -q -- "$flag" "$(vllm_help_file)" 2>/dev/null; }

vllm_envs_file(){ echo "$RUN_DIR/vllm-envs.list"; }
refresh_vllm_envs(){
  local ef; ef="$(vllm_envs_file)"
  [[ "${_VLLM_ENVS_PROBED:-0}" == "1" ]] && return 0
  [[ -e "$ef" && "${VLLM_REFRESH_ENVS:-0}" != "1" ]] && return 0
  _VLLM_ENVS_PROBED=1
  if have timeout; then
    timeout "$VLLM_HELP_TIMEOUT_SEC" python3 - <<'PY' > "$ef" 2>/dev/null || : > "$ef"
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
  else
    python3 - <<'PY' > "$ef" 2>/dev/null || : > "$ef"
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
  fi
}
vllm_env_known(){ local name="$1"; refresh_vllm_envs; grep -qx -- "$name" "$(vllm_envs_file)" 2>/dev/null; }
export_vllm_env_if_known(){
  local name="$1" default_value="$2"
  if vllm_env_known "$name"; then
    if [[ -z "${!name+x}" ]]; then export "$name=$default_value"; else export "$name=${!name}"; fi
  else
    [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 不识别环境变量 $name，已跳过，避免 unknown env 警告。"
  fi
  return 0
}
append_vllm_switch(){
  local arr_name="$1" flag="$2"; local -n _arr="$arr_name"
  if [[ "$VLLM_SKIP_OPTION_PROBE" == "1" ]]; then
    case "$flag" in
      --trust-remote-code|--enforce-eager) _arr+=("$flag") ;;
      --language-model-only|--disable-async-output-proc|--disable-cudagraph) [[ "${3:-0}" == "1" ]] && warn "已跳过可选参数 $flag：VLLM_SKIP_OPTION_PROBE=1" ;;
      *) [[ "${3:-0}" == "1" ]] && warn "已跳过未知可选参数 $flag：VLLM_SKIP_OPTION_PROBE=1" ;;
    esac
    return 0
  fi
  if vllm_supports "$flag"; then _arr+=("$flag"); else [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 不支持参数 $flag，已跳过。"; fi
  return 0
}
append_vllm_option(){
  local arr_name="$1" flag="$2" value="$3"; local -n _arr="$arr_name"
  if [[ "$VLLM_SKIP_OPTION_PROBE" == "1" ]]; then
    [[ "${4:-0}" == "1" ]] && warn "已跳过可选参数 $flag：VLLM_SKIP_OPTION_PROBE=1"
    return 0
  fi
  if vllm_supports "$flag"; then _arr+=("$flag" "$value"); else [[ "${4:-0}" == "1" ]] && warn "当前 vLLM 不支持参数 $flag，已跳过。"; fi
  return 0
}
append_vllm_disable_bool(){
  local arr_name="$1" flag="$2"; local -n _arr="$arr_name"
  if [[ "$VLLM_SKIP_OPTION_PROBE" == "1" ]]; then
    [[ "${3:-0}" == "1" ]] && warn "已跳过布尔可选参数 $flag：VLLM_SKIP_OPTION_PROBE=1"
    return 0
  fi
  local no_flag="--no-${flag#--}"
  local disable_flag="$flag"
  if [[ "$flag" == --enable-* ]]; then disable_flag="--disable-${flag#--enable-}"; fi
  if vllm_supports "$no_flag"; then _arr+=("$no_flag"); elif [[ "$disable_flag" != "$flag" ]] && vllm_supports "$disable_flag"; then _arr+=("$disable_flag"); else [[ "${3:-0}" == "1" ]] && warn "当前 vLLM 未暴露关闭 $flag 的 CLI 参数，已跳过。"; fi
  return 0
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
import builtins
import os
import sys
_TARGETS = ("vllm.model_executor.layers.mamba.gdn_linear_attn", "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
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
                        setattr(obj, attr, _noop_warmup); patched += 1
                    except Exception:
                        pass
    if patched:
        setattr(mod, "_sydney_rocm_gdn_warmup_patch", True)
        try: sys.stderr.write(f"[sydney-vllm-patch] disabled {patched} GDN warmup kernel method(s) in {mod.__name__}\n")
        except Exception: pass
def _patch_loaded():
    for name in _TARGETS: _patch_module(sys.modules.get(name))
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
  case ":${PYTHONPATH:-}:" in *:"$patch_dir":*) ;; *) export PYTHONPATH="$patch_dir${PYTHONPATH:+:$PYTHONPATH}" ;; esac
  log "ROCm safe-mode: 已启用 Qwen GDN warmup runtime patch，可用 QWEN_PATCH_GDN_WARMUP=0 关闭。"
}
apply_qwen_vllm_envs(){
  export HF_ENDPOINT="$HF_ENDPOINT"
  export VLLM_USE_MODELSCOPE="True"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
  if [[ -n "$QWEN_USE_V1" ]]; then export_vllm_env_if_known VLLM_USE_V1 "$QWEN_USE_V1" 1; fi
  if [[ "$QWEN_ROCM_SAFE_MODE" == "1" ]]; then
    export_vllm_env_if_known VLLM_ROCM_USE_AITER "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_LINEAR "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_MOE "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_RMSNORM "$QWEN_USE_AITER" 0
    export_vllm_env_if_known VLLM_ROCM_USE_AITER_PAGED_ATTN "$QWEN_USE_AITER" 0
    prepare_qwen_gdn_patch
  fi
}

start_qwen_vllm(){
  modelscope_download_qwen_fp8
  if [[ -f "$RUN_DIR/qwen-vllm.pid" ]] && ! kill -0 "$(cat "$RUN_DIR/qwen-vllm.pid")" 2>/dev/null; then
    rm -f "$RUN_DIR/qwen-vllm.pid"
  fi
  if [[ "$QWEN_CLEAR_COMPILE_CACHE" == "1" ]]; then
    rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
  fi
  if [[ -f "$RUN_DIR/qwen-vllm.pid" ]] && kill -0 "$(cat "$RUN_DIR/qwen-vllm.pid")" 2>/dev/null; then
    log "Qwen vLLM 已在运行：PID=$(cat "$RUN_DIR/qwen-vllm.pid")"
    return 0
  fi
  local effective_gpu_util
  effective_gpu_util="$(detect_qwen_gpu_memory_utilization)"
  log "启动 Qwen3.6-27B-FP8 vLLM：127.0.0.1:$QWEN_PORT"
  log "model=$QWEN_MODEL_DIR max_model_len=$QWEN_MAX_MODEL_LEN max_num_seqs=$QWEN_MAX_NUM_SEQS gpu_memory_utilization=$effective_gpu_util"
  apply_qwen_rocm_safe_defaults
  apply_qwen_vllm_envs
  log "准备构造 vLLM 参数；如果长时间停住，通常是在探测 vLLM --help。"
  local vllm_args=(
    --host 127.0.0.1
    --port "$QWEN_PORT"
    --model "$QWEN_MODEL_DIR"
    --served-model-name "$QWEN_SERVED_MODEL_NAME"
    --dtype "$QWEN_DTYPE"
    --tensor-parallel-size "$QWEN_TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$effective_gpu_util"
    --max-model-len "$QWEN_MAX_MODEL_LEN"
    --max-num-seqs "$QWEN_MAX_NUM_SEQS"
    --max-num-batched-tokens "$QWEN_MAX_NUM_BATCHED_TOKENS"
  )
  [[ "$QWEN_TRUST_REMOTE_CODE" == "1" ]] && append_vllm_switch vllm_args "--trust-remote-code"
  [[ "$QWEN_ENFORCE_EAGER" == "1" ]] && append_vllm_switch vllm_args "--enforce-eager"
  [[ "$QWEN_LANGUAGE_MODEL_ONLY" == "1" ]] && append_vllm_switch vllm_args "--language-model-only"
  [[ -n "$QWEN_REASONING_PARSER" ]] && append_vllm_option vllm_args "--reasoning-parser" "$QWEN_REASONING_PARSER"
  [[ "$QWEN_ENABLE_CHUNKED_PREFILL" == "0" ]] && append_vllm_disable_bool vllm_args "--enable-chunked-prefill" 1
  [[ "$QWEN_ASYNC_SCHEDULING" == "0" ]] && append_vllm_disable_bool vllm_args "--async-scheduling" 0
  [[ "$QWEN_DISABLE_ASYNC_OUTPUT_PROC" == "1" ]] && append_vllm_switch vllm_args "--disable-async-output-proc"
  if [[ "$QWEN_DISABLE_CUDA_GRAPH" == "1" ]]; then
    if vllm_supports "--disable-cudagraph"; then
      vllm_args+=(--disable-cudagraph)
    else
      warn "当前 vLLM 无 --disable-cudagraph；已使用 --enforce-eager 并跳过 VLLM_USE_V1 unknown env。"
    fi
  fi
  if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_args=( $VLLM_EXTRA_ARGS )
    vllm_args+=("${extra_args[@]}")
  fi
  log "vLLM args: ${vllm_args[*]}"
  nohup python3 -m vllm.entrypoints.openai.api_server "${vllm_args[@]}" \
    > "$LOG_DIR/qwen-vllm.log" 2>&1 &
  echo $! > "$RUN_DIR/qwen-vllm.pid"
  local loops=$(( QWEN_STARTUP_TIMEOUT_SEC / 2 ))
  [[ "$loops" -lt 1 ]] && loops=1
  for ((i=1; i<=loops; i++)); do
    if curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then
      log "Qwen vLLM 已就绪"
      return 0
    fi
    if ! kill -0 "$(cat "$RUN_DIR/qwen-vllm.pid")" 2>/dev/null; then
      rm -f "$RUN_DIR/qwen-vllm.pid"
      err "Qwen vLLM 启动失败，最近日志："
      tail -n 160 "$LOG_DIR/qwen-vllm.log" || true
      exit 1
    fi
    sleep 2
  done
  err "Qwen vLLM 启动超时，日志：$LOG_DIR/qwen-vllm.log"
  tail -n 160 "$LOG_DIR/qwen-vllm.log" || true
  exit 1
}

write_cloud_env(){
  cd "$APP_DIR"
  if [[ -f .env ]]; then cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)" || true; fi
  log "写入云端 .env：Sydney source + Qwen FP8(vLLM) aux 全复用"
  cat > .env <<EOF
TEACHER_BASE_URL=http://127.0.0.1:$SYDNEY_PORT/v1
TEACHER_API_KEY=sk-local
TEACHER_MODEL=clever-sydney-4-12b-q8
TEACHER_API_PROTOCOL=legacy_chat_completions
SOURCE_PROMPT_MODE=legacy_chat
SOURCE_USE_DEFAULT_STOPS=false

SIMULATOR_BASE_URL=http://127.0.0.1:$QWEN_PORT/v1
SIMULATOR_API_KEY=sk-local
SIMULATOR_MODEL=$QWEN_SERVED_MODEL_NAME
SIMULATOR_API_PROTOCOL=chat_completions

TRANSLATOR_BASE_URL=http://127.0.0.1:$QWEN_PORT/v1
TRANSLATOR_API_KEY=sk-local
TRANSLATOR_MODEL=$QWEN_SERVED_MODEL_NAME
TRANSLATOR_API_PROTOCOL=chat_completions

JUDGE_BASE_URL=http://127.0.0.1:$QWEN_PORT/v1
JUDGE_API_KEY=sk-local
JUDGE_MODEL=$QWEN_SERVED_MODEL_NAME
JUDGE_API_PROTOCOL=chat_completions

APP_DEFAULT_COUNT=3000
APP_DEFAULT_CONCURRENCY=$APP_CONCURRENCY
APP_DEFAULT_MAX_TURNS=20
APP_DEFAULT_TRANSLATE_TO_ZH=true
APP_DEFAULT_SAME_AUX_MODEL=true
APP_DEFAULT_USE_JUDGE=true
APP_DEFAULT_TARGET_MODEL=qwen36_27b
APP_DEFAULT_TRAIN_MODE=qlora
APP_DEFAULT_INCLUDE_NEEDS_REVIEW=false
APP_DEFAULT_ONLY_DIALOGUE_DISTILLATION=true

MODEL_TIMEOUT=600
MODEL_RETRIES=3
MODEL_RETRY_BACKOFF=2
TEACHER_TIMEOUT=600
TEACHER_RETRIES=3
SIMULATOR_TIMEOUT=600
SIMULATOR_RETRIES=3
TRANSLATOR_TIMEOUT=600
TRANSLATOR_RETRIES=3
JUDGE_TIMEOUT=600
JUDGE_RETRIES=3

GENERATION_MIN_TURNS=6
GENERATION_PRESERVE_LENGTH=true
SOURCE_TEMPERATURE=0.78
SOURCE_TOP_P=0.92
SOURCE_FREQUENCY_PENALTY=0.35
SOURCE_PRESENCE_PENALTY=0.25
SOURCE_REPEAT_PENALTY=1.12
SOURCE_MAX_TOKENS=512

DISABLE_JOB_RESTORE=false
JOB_RESTORE_LIMIT=3
JOB_API_LOG_LIMIT=120
JOB_LOG_PERSIST_KEEP=120
JOB_PERSIST_MIN_INTERVAL_MS=1000
JOB_PERSIST_EVERY_UPDATES=25
JOB_PERSIST_RETRIES=8
EOF
}

start_app(){
  cd "$APP_DIR"
  if [[ -f "$RUN_DIR/app.pid" ]] && kill -0 "$(cat "$RUN_DIR/app.pid")" 2>/dev/null; then
    log "Web 控制台已在运行：PID=$(cat "$RUN_DIR/app.pid")"
    return 0
  fi
  log "启动 Web 控制台：http://$APP_HOST:$APP_PORT"
  nohup python3 -m uvicorn app:app --host "$APP_HOST" --port "$APP_PORT" > "$LOG_DIR/app.log" 2>&1 &
  echo $! > "$RUN_DIR/app.pid"
  for i in {1..120}; do
    if curl -fsS "http://127.0.0.1:$APP_PORT/api/config" >/dev/null 2>&1; then
      log "Web 控制台已就绪"
      return 0
    fi
    sleep 1
  done
  err "Web 控制台启动超时，日志：$LOG_DIR/app.log"
  tail -n 80 "$LOG_DIR/app.log" || true
  exit 1
}

ensure_cloudflared(){
  local bin="$STACK_DIR/cloudflared"
  if [[ -x "$bin" ]]; then echo "$bin"; return 0; fi
  if [[ -f "$CLOUDFLARED_LOCAL_PATH" ]]; then cp "$CLOUDFLARED_LOCAL_PATH" "$bin" && chmod +x "$bin" && echo "$bin" && return 0; fi
  for u in $CLOUDFLARED_URL_FALLBACKS; do
    log "下载 cloudflared: $u" >&2
    if curl -L --retry 3 --retry-delay 3 --connect-timeout 20 --max-time 180 --speed-time 30 --speed-limit 1024 -o "$bin.tmp" "$u"; then
      mv "$bin.tmp" "$bin" && chmod +x "$bin" && echo "$bin" && return 0
    fi
  done
  err "cloudflared 下载失败。你可以上传到 /mnt/cloudflared 后重跑。"
  exit 1
}

start_console_tunnel(){
  local cfbin
  cfbin="$(ensure_cloudflared)"
  if [[ -f "$RUN_DIR/cloudflared.pid" ]] && kill -0 "$(cat "$RUN_DIR/cloudflared.pid")" 2>/dev/null; then
    log "控制台 CF tunnel 已在运行：PID=$(cat "$RUN_DIR/cloudflared.pid")"
    if ! grep -q "127.0.0.1:$APP_PORT" "$LOG_DIR/cloudflared-console.log" 2>/dev/null; then
      warn "已有 cloudflared 可能不是指向控制台端口 $APP_PORT，重启控制台 tunnel。"
      kill "$(cat "$RUN_DIR/cloudflared.pid")" 2>/dev/null || true
      sleep 2
      kill -9 "$(cat "$RUN_DIR/cloudflared.pid")" 2>/dev/null || true
      rm -f "$RUN_DIR/cloudflared.pid"
      nohup "$cfbin" tunnel --url "http://127.0.0.1:$APP_PORT" > "$LOG_DIR/cloudflared-console.log" 2>&1 &
      echo $! > "$RUN_DIR/cloudflared.pid"
    fi
  else
    log "启动唯一 Cloudflare Tunnel -> http://127.0.0.1:$APP_PORT"
    nohup "$cfbin" tunnel --url "http://127.0.0.1:$APP_PORT" > "$LOG_DIR/cloudflared-console.log" 2>&1 &
    echo $! > "$RUN_DIR/cloudflared.pid"
  fi
  sleep 8
  local url
  url="$(grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "$LOG_DIR/cloudflared-console.log" | tail -n 1 || true)"
  if [[ -n "$url" ]]; then
    echo
    echo "================ 控制台公网地址 ================"
    echo "$url"
    echo "================================================="
  else
    warn "暂未解析到 trycloudflare URL。查看：tail -f $LOG_DIR/cloudflared-console.log"
  fi
}

write_management_scripts(){
  cat > "$BIN_DIR/status_all.sh" <<EOF
#!/usr/bin/env bash
set -e
bash "$APP_DIR/scripts/manage_cloud_models.sh" status
EOF
  cat > "$BIN_DIR/stop_all.sh" <<EOF
#!/usr/bin/env bash
set -e
bash "$APP_DIR/scripts/manage_cloud_models.sh" stop-all
for name in cloudflared app; do
  pid_file="$RUN_DIR/\$name.pid"
  if [[ -f "\$pid_file" ]]; then
    pid="\$(cat "\$pid_file")"
    kill "\$pid" 2>/dev/null || true
    sleep 1
    kill -9 "\$pid" 2>/dev/null || true
    rm -f "\$pid_file"
  fi
done
EOF
  chmod +x "$BIN_DIR/status_all.sh" "$BIN_DIR/stop_all.sh"
}

main(){
  ensure_dirs
  apt_install
  pip_install
  check_vllm
  clone_or_update_repo
  install_repo_requirements
  write_management_scripts
  if [[ "$SKIP_MODEL_START" == "1" ]]; then
    warn "console-only: 不启动/重启 Sydney 和 Qwen；只写 .env、启动 Web 控制台和控制台 tunnel。"
  else
    start_sydney
    start_qwen_vllm
  fi
  write_cloud_env
  start_app
  start_console_tunnel
  echo
  echo "控制台说明："
  echo "  Cloudflare Tunnel 只指向 Web 控制台 http://127.0.0.1:$APP_PORT"
  echo "  模型端口仅本机可访问：Sydney=$SYDNEY_PORT, Qwen=$QWEN_PORT"
  echo "  页面生成数据时会让 Sydney(source) 与 Qwen(simulator/translator/judge) 在内网互相调用。"
  echo
  echo "管理命令："
  echo "  bash $APP_DIR/scripts/manage_cloud_models.sh status"
  echo "  bash $APP_DIR/scripts/manage_cloud_models.sh restart-qwen"
  echo "  $BIN_DIR/status_all.sh"
  echo "  $BIN_DIR/stop_all.sh"
  echo "日志："
  echo "  tail -f $LOG_DIR/app.log"
  echo "  tail -f $LOG_DIR/qwen-vllm.log"
  echo "  tail -f $LOG_DIR/cloudflared-console.log"
  echo "  tail -f /workspace/sydney_rocm/logs/llama-server.log"
}

main "$@"
