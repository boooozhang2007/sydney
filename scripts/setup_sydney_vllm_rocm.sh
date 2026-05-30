#!/usr/bin/env bash
# 把 Free_Sydney_V2_13b_HF (或同系 HF 模型) 用 vLLM 0.21 部署成 OpenAI 兼容 API,
# 跑在 AMD/ROCm 上, 复用 manage_cloud_models.sh 里 Qwen 那套 ROCm-safe 默认值.
#
# 设计目标:
#   - 走 HF 镜像 (hf-mirror.com) 或 ModelScope, 适配国内网络
#   - vLLM 已预装 (0.21), 不再 pip install
#   - cont-batching + paged attention + prefix caching 三件套全开
#   - 可选 cloudflared tunnel, 暴露公网 HTTPS
#   - 启动后输出可直接贴到 .env 的 TEACHER_* 配置片段
#
# 快速用法:
#   bash scripts/setup_sydney_vllm_rocm.sh --start
#   bash scripts/setup_sydney_vllm_rocm.sh --start --tunnel
#   bash scripts/setup_sydney_vllm_rocm.sh --restart
#   bash scripts/setup_sydney_vllm_rocm.sh --stop
#
# 常用环境变量见下方 SYDNEY_VLLM_* 部分.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKDIR="${WORKDIR:-/workspace/sydney_vllm}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
RUN_DIR="${RUN_DIR:-$WORKDIR/run}"
BIN_DIR="${BIN_DIR:-$WORKDIR/bin}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/models}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# 模型选择. 默认 Free_Sydney_V2_13b_HF, 同系也可换:
#   FPHam/Free_Sydney_13b_HF
#   FPHam/Sydney_Overthinker_13b_HF
SYDNEY_VLLM_HF_REPO_ID="${SYDNEY_VLLM_HF_REPO_ID:-FPHam/Free_Sydney_V2_13b_HF}"
SYDNEY_VLLM_SERVED_MODEL_NAME="${SYDNEY_VLLM_SERVED_MODEL_NAME:-free-sydney-v2-13b}"
SYDNEY_VLLM_LOCAL_DIR="${SYDNEY_VLLM_LOCAL_DIR:-$MODEL_DIR/free_sydney_v2_13b}"

# 国内镜像. 默认 hf-mirror, 失败可切 ModelScope.
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SYDNEY_VLLM_PROVIDER="${SYDNEY_VLLM_PROVIDER:-hf}"        # hf | modelscope
SYDNEY_VLLM_MS_MODEL_ID="${SYDNEY_VLLM_MS_MODEL_ID:-}"     # 可选 ModelScope 镜像 id

# 服务端口 / 监听
SYDNEY_VLLM_HOST="${SYDNEY_VLLM_HOST:-0.0.0.0}"
SYDNEY_VLLM_PORT="${SYDNEY_VLLM_PORT:-8000}"

# vLLM 关键参数. 192GB AMD 上目标是高 max-num-seqs + prefix caching + bf16.
# LLaMA-2-13B native ctx=4096; 多轮日常聊天 4096 已够 20 轮.
SYDNEY_VLLM_MAX_MODEL_LEN="${SYDNEY_VLLM_MAX_MODEL_LEN:-4096}"
SYDNEY_VLLM_MAX_NUM_SEQS="${SYDNEY_VLLM_MAX_NUM_SEQS:-256}"
SYDNEY_VLLM_MAX_NUM_BATCHED_TOKENS="${SYDNEY_VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
SYDNEY_VLLM_GPU_MEMORY_UTIL="${SYDNEY_VLLM_GPU_MEMORY_UTIL:-0.80}"
SYDNEY_VLLM_TENSOR_PARALLEL_SIZE="${SYDNEY_VLLM_TENSOR_PARALLEL_SIZE:-1}"
SYDNEY_VLLM_DTYPE="${SYDNEY_VLLM_DTYPE:-bfloat16}"
SYDNEY_VLLM_TRUST_REMOTE_CODE="${SYDNEY_VLLM_TRUST_REMOTE_CODE:-1}"
SYDNEY_VLLM_ENFORCE_EAGER="${SYDNEY_VLLM_ENFORCE_EAGER:-1}"
SYDNEY_VLLM_DISABLE_CUDA_GRAPH="${SYDNEY_VLLM_DISABLE_CUDA_GRAPH:-1}"
SYDNEY_VLLM_ENABLE_PREFIX_CACHING="${SYDNEY_VLLM_ENABLE_PREFIX_CACHING:-1}"
SYDNEY_VLLM_ENABLE_CHUNKED_PREFILL="${SYDNEY_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
SYDNEY_VLLM_ROCM_SAFE_MODE="${SYDNEY_VLLM_ROCM_SAFE_MODE:-1}"
SYDNEY_VLLM_USE_AITER="${SYDNEY_VLLM_USE_AITER:-0}"
SYDNEY_VLLM_DISABLE_ASYNC_OUTPUT_PROC="${SYDNEY_VLLM_DISABLE_ASYNC_OUTPUT_PROC:-0}"
SYDNEY_VLLM_CLEAR_COMPILE_CACHE="${SYDNEY_VLLM_CLEAR_COMPILE_CACHE:-1}"
SYDNEY_VLLM_STARTUP_TIMEOUT_SEC="${SYDNEY_VLLM_STARTUP_TIMEOUT_SEC:-900}"
SYDNEY_VLLM_EXTRA_ARGS="${SYDNEY_VLLM_EXTRA_ARGS:-}"

# FPHam Sydney 系列 tokenizer 没自带 chat_template, transformers v4.44+ 不再兜底,
# 所以必须显式传 --chat-template, 否则 /v1/chat/completions 会直接 HTTP 400.
# 默认写一份 Vicuna v1.1 模板 (USER:/ASSISTANT:), Free_Sydney_V2_13b 用这个.
# 想换成 alpaca/llama2 风格可把 SYDNEY_VLLM_CHAT_TEMPLATE 指到自己的 .jinja 文件.
SYDNEY_VLLM_CHAT_TEMPLATE="${SYDNEY_VLLM_CHAT_TEMPLATE:-$WORKDIR/chat_template_vicuna_v1.1.jinja}"
SYDNEY_VLLM_AUTO_WRITE_CHAT_TEMPLATE="${SYDNEY_VLLM_AUTO_WRITE_CHAT_TEMPLATE:-1}"

# Cloudflare Tunnel 同 setup_sydney_rocm.sh
USE_TUNNEL="${USE_TUNNEL:-0}"
CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH:-/mnt/cloudflared}"
CLOUDFLARED_URL="${CLOUDFLARED_URL:-https://ghfast.top/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS:-$CLOUDFLARED_URL https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"

START_AFTER_SETUP="0"; STOP_AFTER_SETUP="0"; RESTART_AFTER_SETUP="0"; STATUS_ONLY="0"

usage(){ cat <<'USAGE'
vLLM Sydney (ROCm) setup helper.

Options:
  --start         配置后启动 vLLM
  --stop          停止 vLLM 和 tunnel
  --restart       改了 PARALLEL/CTX 等必须用它
  --status        查看运行状态
  --tunnel        启动 cloudflared tunnel (会先 --start)
  -h, --help      显示帮助

Examples:
  bash scripts/setup_sydney_vllm_rocm.sh --start
  SYDNEY_VLLM_MAX_NUM_SEQS=384 bash scripts/setup_sydney_vllm_rocm.sh --restart
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START_AFTER_SETUP=1 ;;
    --stop) STOP_AFTER_SETUP=1 ;;
    --restart) RESTART_AFTER_SETUP=1; START_AFTER_SETUP=1 ;;
    --status) STATUS_ONLY=1 ;;
    --tunnel) USE_TUNNEL=1; START_AFTER_SETUP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac; shift
done

log(){ printf '\033[1;36m[setup-sydney-vllm]\033[0m %s\n' "$*" >&2; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
have(){ command -v "$1" >/dev/null 2>&1; }

ensure_dirs(){ mkdir -p "$WORKDIR" "$LOG_DIR" "$RUN_DIR" "$BIN_DIR" "$MODEL_DIR" "$SYDNEY_VLLM_LOCAL_DIR"; }

pid_file(){ echo "$RUN_DIR/sydney-vllm.pid"; }
tunnel_pid_file(){ echo "$RUN_DIR/sydney-vllm-tunnel.pid"; }
tunnel_log_file(){ echo "$LOG_DIR/cloudflared.log"; }
tunnel_url_file(){ echo "$RUN_DIR/sydney-vllm-tunnel.url"; }

pid_alive(){ local p="${1:-}"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }
read_pid(){ [[ -s "$1" ]] && tr -dc '0-9' < "$1" || true; }

check_vllm_available(){
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('vllm') else 1)
PY
  then err "Python 环境没有 vllm: $PYTHON_BIN"; exit 1; fi
}

download_via_hf(){
  log "HF 镜像下载: $SYDNEY_VLLM_HF_REPO_ID -> $SYDNEY_VLLM_LOCAL_DIR (HF_ENDPOINT=$HF_ENDPOINT)"
  HF_ENDPOINT="$HF_ENDPOINT" \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  SYDNEY_VLLM_HF_REPO_ID="$SYDNEY_VLLM_HF_REPO_ID" \
  SYDNEY_VLLM_LOCAL_DIR="$SYDNEY_VLLM_LOCAL_DIR" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
repo = os.environ['SYDNEY_VLLM_HF_REPO_ID']
local = os.environ['SYDNEY_VLLM_LOCAL_DIR']
Path(local).mkdir(parents=True, exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=repo,
    local_dir=local,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json","*.txt","*.model","*.safetensors","*.bin","tokenizer*","*.py"],
)
print("[hf] done")
PY
}

download_via_ms(){
  local mid="${SYDNEY_VLLM_MS_MODEL_ID:-}"
  [[ -z "$mid" ]] && { err "SYDNEY_VLLM_MS_MODEL_ID 未设置, 无法走 ModelScope"; exit 1; }
  log "ModelScope 下载: $mid -> $SYDNEY_VLLM_LOCAL_DIR"
  SYDNEY_VLLM_MS_MODEL_ID="$mid" SYDNEY_VLLM_LOCAL_DIR="$SYDNEY_VLLM_LOCAL_DIR" \
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
mid = os.environ['SYDNEY_VLLM_MS_MODEL_ID']
local = os.environ['SYDNEY_VLLM_LOCAL_DIR']
Path(local).mkdir(parents=True, exist_ok=True)
try:
    from modelscope import snapshot_download
except Exception:
    from modelscope.hub.snapshot_download import snapshot_download
for kw in (dict(model_id=mid, local_dir=local), dict(model_id=mid, cache_dir=local)):
    try:
        snapshot_download(**kw); break
    except TypeError:
        continue
print("[ms] done")
PY
}

ensure_model_downloaded(){
  if [[ -f "$SYDNEY_VLLM_LOCAL_DIR/config.json" ]]; then
    log "模型已就绪: $SYDNEY_VLLM_LOCAL_DIR"
    return 0
  fi
  case "$SYDNEY_VLLM_PROVIDER" in
    hf) download_via_hf ;;
    modelscope) download_via_ms ;;
    *) err "未知 SYDNEY_VLLM_PROVIDER=$SYDNEY_VLLM_PROVIDER"; exit 1 ;;
  esac
  if [[ ! -f "$SYDNEY_VLLM_LOCAL_DIR/config.json" ]]; then
    local found
    found="$(find "$SYDNEY_VLLM_LOCAL_DIR" -maxdepth 5 -name config.json -type f | head -n 1 || true)"
    [[ -n "$found" ]] || { err "下载后未找到 config.json: $SYDNEY_VLLM_LOCAL_DIR"; exit 1; }
    SYDNEY_VLLM_LOCAL_DIR="$(dirname "$found")"
    log "自动定位模型目录: $SYDNEY_VLLM_LOCAL_DIR"
  fi
}

ensure_chat_template(){
  # 用户给了路径 -> 必须真存在
  if [[ "$SYDNEY_VLLM_AUTO_WRITE_CHAT_TEMPLATE" != "1" ]]; then
    [[ -f "$SYDNEY_VLLM_CHAT_TEMPLATE" ]] || { err "chat template 不存在: $SYDNEY_VLLM_CHAT_TEMPLATE"; exit 1; }
    return 0
  fi
  [[ -f "$SYDNEY_VLLM_CHAT_TEMPLATE" ]] && { log "chat template 已就绪: $SYDNEY_VLLM_CHAT_TEMPLATE"; return 0; }
  log "写入默认 Vicuna v1.1 chat template -> $SYDNEY_VLLM_CHAT_TEMPLATE"
  mkdir -p "$(dirname "$SYDNEY_VLLM_CHAT_TEMPLATE")"
  cat > "$SYDNEY_VLLM_CHAT_TEMPLATE" <<'JINJA'
{%- if messages[0]['role'] == 'system' -%}
{{- messages[0]['content'].strip() + ' ' -}}
{%- set loop_messages = messages[1:] -%}
{%- else -%}
{{- "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " -}}
{%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
{%- if message['role'] == 'user' -%}
{{- 'USER: ' + message['content'].strip() + ' ' -}}
{%- elif message['role'] == 'assistant' -%}
{{- 'ASSISTANT: ' + message['content'].strip() + eos_token + ' ' -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- 'ASSISTANT:' -}}
{%- endif -%}
JINJA
}

apply_rocm_envs(){
  export HF_ENDPOINT="$HF_ENDPOINT"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
  if [[ "$SYDNEY_VLLM_ROCM_SAFE_MODE" == "1" ]]; then
    export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-$SYDNEY_VLLM_USE_AITER}"
    export VLLM_ROCM_USE_AITER_LINEAR="${VLLM_ROCM_USE_AITER_LINEAR:-$SYDNEY_VLLM_USE_AITER}"
    export VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-$SYDNEY_VLLM_USE_AITER}"
    export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-$SYDNEY_VLLM_USE_AITER}"
    export VLLM_ROCM_USE_AITER_PAGED_ATTN="${VLLM_ROCM_USE_AITER_PAGED_ATTN:-$SYDNEY_VLLM_USE_AITER}"
  fi
}

build_args(){
  local -n _a="$1"
  _a=(
    --host "$SYDNEY_VLLM_HOST"
    --port "$SYDNEY_VLLM_PORT"
    --model "$SYDNEY_VLLM_LOCAL_DIR"
    --served-model-name "$SYDNEY_VLLM_SERVED_MODEL_NAME"
    --dtype "$SYDNEY_VLLM_DTYPE"
    --tensor-parallel-size "$SYDNEY_VLLM_TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$SYDNEY_VLLM_GPU_MEMORY_UTIL"
    --max-model-len "$SYDNEY_VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$SYDNEY_VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$SYDNEY_VLLM_MAX_NUM_BATCHED_TOKENS"
    --chat-template "$SYDNEY_VLLM_CHAT_TEMPLATE"
  )
  [[ "$SYDNEY_VLLM_TRUST_REMOTE_CODE" == "1" ]] && _a+=(--trust-remote-code)
  [[ "$SYDNEY_VLLM_ENFORCE_EAGER" == "1" ]] && _a+=(--enforce-eager)
  [[ "$SYDNEY_VLLM_ENABLE_PREFIX_CACHING" == "1" ]] && _a+=(--enable-prefix-caching)
  [[ "$SYDNEY_VLLM_ENABLE_CHUNKED_PREFILL" == "1" ]] && _a+=(--enable-chunked-prefill)
  [[ "$SYDNEY_VLLM_DISABLE_ASYNC_OUTPUT_PROC" == "1" ]] && _a+=(--disable-async-output-proc)
  if [[ -n "$SYDNEY_VLLM_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $SYDNEY_VLLM_EXTRA_ARGS ); _a+=("${extra[@]}")
  fi
}

start_vllm(){
  if pid_alive "$(read_pid "$(pid_file)")"; then
    log "Sydney vLLM 已在运行: PID=$(read_pid "$(pid_file)")"
    return 0
  fi
  check_vllm_available
  ensure_model_downloaded
  ensure_chat_template
  [[ "$SYDNEY_VLLM_CLEAR_COMPILE_CACHE" == "1" ]] && rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true
  apply_rocm_envs
  local args; build_args args
  log "vLLM args: ${args[*]}"
  log "日志: $LOG_DIR/sydney-vllm.log"
  nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server "${args[@]}" \
    > "$LOG_DIR/sydney-vllm.log" 2>&1 &
  echo $! > "$(pid_file)"
  local loops=$(( SYDNEY_VLLM_STARTUP_TIMEOUT_SEC / 2 )); [[ $loops -lt 1 ]] && loops=1
  for ((i=1; i<=loops; i++)); do
    if curl -fsS "http://127.0.0.1:$SYDNEY_VLLM_PORT/v1/models" >/dev/null 2>&1; then
      log "Sydney vLLM ready: http://127.0.0.1:$SYDNEY_VLLM_PORT/v1"
      return 0
    fi
    if ! pid_alive "$(read_pid "$(pid_file)")"; then
      rm -f "$(pid_file)"; err "vLLM 启动失败, tail log:"; tail -n 200 "$LOG_DIR/sydney-vllm.log" || true; exit 1
    fi
    sleep 2
  done
  err "vLLM 启动超时, tail log:"; tail -n 200 "$LOG_DIR/sydney-vllm.log" || true; exit 1
}

stop_vllm(){
  local p; p="$(read_pid "$(pid_file)")"
  if pid_alive "$p"; then
    log "停止 sydney-vllm PID=$p"
    kill -TERM "$p" 2>/dev/null || true
    for _ in {1..20}; do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true
  fi
  rm -f "$(pid_file)"
  if have lsof; then
    local pids; pids="$(lsof -ti TCP:"$SYDNEY_VLLM_PORT" 2>/dev/null | tr '\n' ' ' || true)"
    [[ -n "$pids" ]] && { warn "按端口清理残留: $pids"; kill $pids 2>/dev/null || true; sleep 1; kill -9 $pids 2>/dev/null || true; }
  fi
}

ensure_cloudflared(){
  local bin="$BIN_DIR/cloudflared"
  if [[ -x "$bin" ]]; then printf '%s' "$bin"; return 0; fi
  if [[ -x "$CLOUDFLARED_LOCAL_PATH" ]]; then
    cp "$CLOUDFLARED_LOCAL_PATH" "$bin"; chmod +x "$bin"; printf '%s' "$bin"; return 0
  fi
  for url in $CLOUDFLARED_URL_FALLBACKS; do
    log "下载 cloudflared: $url"
    if curl -fsSL --connect-timeout 15 --max-time 600 -o "$bin.tmp" "$url"; then
      # 真二进制 ~37MB; 镜像源损坏时常返回几十字节的错误页. 至少 1MB + ELF magic.
      sz="$(stat -c '%s' "$bin.tmp" 2>/dev/null || echo 0)"
      if [[ "$sz" -gt 1000000 ]] && head -c 4 "$bin.tmp" | grep -q $'\x7fELF'; then
        mv "$bin.tmp" "$bin"; chmod +x "$bin"; printf '%s' "$bin"; return 0
      fi
      warn "cloudflared 源损坏 (size=$sz, 非 ELF): $url"
    fi
    rm -f "$bin.tmp"
  done
  err "cloudflared 下载失败, 可手动放到 $CLOUDFLARED_LOCAL_PATH"; exit 1
}

start_tunnel(){
  local p; p="$(read_pid "$(tunnel_pid_file)")"
  pid_alive "$p" && { log "tunnel 已在运行 PID=$p"; return 0; }
  local bin; bin="$(ensure_cloudflared)"
  : > "$(tunnel_log_file)"
  nohup "$bin" tunnel --no-autoupdate --url "http://127.0.0.1:$SYDNEY_VLLM_PORT" \
    > "$(tunnel_log_file)" 2>&1 &
  echo $! > "$(tunnel_pid_file)"
  local url=""
  for _ in {1..60}; do
    url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$(tunnel_log_file)" | head -n 1 || true)"
    [[ -n "$url" ]] && break
    sleep 1
  done
  if [[ -n "$url" ]]; then
    echo "$url" > "$(tunnel_url_file)"
    log "Tunnel URL: $url"
  else
    warn "未抓到 trycloudflare URL, 请查看: $(tunnel_log_file)"
  fi
}

stop_tunnel(){
  local p; p="$(read_pid "$(tunnel_pid_file)")"
  pid_alive "$p" && { log "停止 tunnel PID=$p"; kill -TERM "$p" 2>/dev/null || true; sleep 1; kill -KILL "$p" 2>/dev/null || true; }
  rm -f "$(tunnel_pid_file)"
}

print_env_snippet(){
  local base="http://127.0.0.1:$SYDNEY_VLLM_PORT/v1"
  if [[ -s "$(tunnel_url_file)" ]]; then
    base="$(cat "$(tunnel_url_file)")/v1"
  fi
  cat <<EOF

=== 复制到 .env ===
TEACHER_BASE_URL=$base
TEACHER_API_KEY=sk-local
TEACHER_MODEL=$SYDNEY_VLLM_SERVED_MODEL_NAME
TEACHER_API_PROTOCOL=chat_completions
TEACHER_TIMEOUT=180

# 可选: 配合 vLLM 高并发, 把 HTTP 池放大到至少 max-num-seqs 同级
HTTP_POOL_CONNECTIONS=$SYDNEY_VLLM_MAX_NUM_SEQS
HTTP_MAX_CONNECTIONS=$(( SYDNEY_VLLM_MAX_NUM_SEQS * 2 ))
TRANSLATION_CONCURRENCY=$SYDNEY_VLLM_MAX_NUM_SEQS
==================

EOF
}

status(){
  echo "--- Sydney vLLM ---"
  local p; p="$(read_pid "$(pid_file)")"
  if pid_alive "$p"; then
    echo "sydney-vllm: running PID=$p"
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true; echo
  else
    echo "sydney-vllm: stopped"
  fi
  curl -fsS --max-time 5 "http://127.0.0.1:$SYDNEY_VLLM_PORT/v1/models" 2>/dev/null || true; echo
  if [[ -s "$(tunnel_url_file)" ]]; then echo "tunnel: $(cat "$(tunnel_url_file)")"; fi
  if have rocm-smi; then rocm-smi || true; elif have nvidia-smi; then nvidia-smi || true; fi
}

ensure_dirs

if [[ "$STATUS_ONLY" == "1" ]]; then status; exit 0; fi
if [[ "$STOP_AFTER_SETUP" == "1" && "$START_AFTER_SETUP" != "1" ]]; then stop_tunnel; stop_vllm; exit 0; fi
if [[ "$RESTART_AFTER_SETUP" == "1" ]]; then stop_tunnel; stop_vllm; fi
if [[ "$START_AFTER_SETUP" == "1" ]]; then
  start_vllm
  [[ "$USE_TUNNEL" == "1" ]] && start_tunnel
  print_env_snippet
fi
