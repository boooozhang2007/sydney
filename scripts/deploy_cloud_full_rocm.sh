#!/usr/bin/env bash
# 云端全流程部署：
#   Sydney source：llama.cpp + GGUF，127.0.0.1:8000
#   Qwen3.6-27B-FP8：vLLM OpenAI-compatible API，127.0.0.1:8010
#   控制台：FastAPI，127.0.0.1:7860
#   只暴露一个 Cloudflare Tunnel：控制台，不暴露模型端口
#
# 用法：
#   REPO_URL=https://github.com/boooozhang2007/sydney.git bash scripts/deploy_cloud_full_rocm.sh
#
# 质量优先默认：QWEN_MS_MODEL_ID=Qwen/Qwen3.6-27B-FP8

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/workspace/sydney_NEWBING}"
REPO_URL="${REPO_URL:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_PROXY_PREFIX="${GIT_PROXY_PREFIX:-https://gh.llkk.cc/}"
# 只加速 GitHub/Hugging Face；apt/pip 保持当前镜像环境默认源。
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-7860}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
QWEN_PORT="${QWEN_PORT:-8010}"

APP_CONCURRENCY="${APP_CONCURRENCY:-100}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-100}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-400000}"

# Qwen FP8 + vLLM 参数。
QWEN_MS_MODEL_ID="${QWEN_MS_MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/workspace/modelscope/qwen36_27b_fp8}"
# 注意：vLLM 的 gpu-memory-utilization 是按整卡总显存计算，不是按剩余显存计算。
# 如果 Sydney llama.cpp 已占显存，0.92 会要求约 176GiB，可能启动失败；默认自动按剩余显存限幅。
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.50}"
QWEN_AUTO_MEMORY_UTIL="${QWEN_AUTO_MEMORY_UTIL:-1}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-100}"
QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-131072}"
QWEN_TENSOR_PARALLEL_SIZE="${QWEN_TENSOR_PARALLEL_SIZE:-1}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-fp8}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# ROCm 上 torch.compile / cudagraph 偶发 hipErrorIllegalAddress，默认关闭以换稳定性。
QWEN_DISABLE_CUDA_GRAPH="${QWEN_DISABLE_CUDA_GRAPH:-1}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-1}"
QWEN_USE_V1="${QWEN_USE_V1:-0}"
QWEN_CLEAR_COMPILE_CACHE="${QWEN_CLEAR_COMPILE_CACHE:-1}"
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"

SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"
LLAMA_DIR="${LLAMA_DIR:-/workspace/llama.cpp-rocm}"

CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH:-/mnt/cloudflared}"
CLOUDFLARED_URL="${CLOUDFLARED_URL:-https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS:-$CLOUDFLARED_URL https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"

STACK_DIR="${STACK_DIR:-/workspace/sydney_cloud_stack}"
LOG_DIR="$STACK_DIR/logs"
RUN_DIR="$STACK_DIR/run"
BIN_DIR="$STACK_DIR/bin"

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
  # 只安装工作台轻量依赖；vLLM 环境按你的要求假设镜像已自带，不在这里 pip 安装/升级。
  python3 -m pip install -U modelscope uvicorn fastapi python-dotenv httpx pydantic
}

check_vllm(){
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
    git pull --ff-only || warn "git pull 失败，继续使用本地源码"
    return 0
  fi
  if [[ -z "$REPO_URL" ]]; then
    err "APP_DIR 不存在且 REPO_URL 未设置。请设置 REPO_URL=https://github.com/xxx/sydney_NEWBING.git"
    exit 1
  fi
  mkdir -p "$(dirname "$APP_DIR")"
  log "克隆源码：$REPO_URL -> $APP_DIR"
  git clone --depth 1 --branch "$GIT_BRANCH" "${GIT_PROXY_PREFIX}${REPO_URL}" "$APP_DIR" \
    || git clone --depth 1 --branch "$GIT_BRANCH" "$REPO_URL" "$APP_DIR"
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
    # 有些 modelscope 版本会下载到 local_dir/model_id 子目录，尝试自动定位。
    local found
    found="$(find "$QWEN_MODEL_DIR" -maxdepth 4 -name config.json -type f | head -n 1 || true)"
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
fallback = float(os.environ.get('QWEN_GPU_MEMORY_UTILIZATION', '0.50'))
try:
    import torch
    if not torch.cuda.is_available():
        print(fallback); raise SystemExit
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    total_gib = total / 1024**3
    # vLLM 检查 requested = total * util，必须小于启动时 free。
    # 留 8GiB 给框架/碎片/后续波动。
    util = max(0.10, min(fallback, (free_gib - 8.0) / total_gib))
    print(f"{util:.3f}")
except Exception:
    print(fallback)
PY
}

start_qwen_vllm(){
  modelscope_download_qwen_fp8
  # 如果上次 hip illegal memory access 失败，旧进程/缓存可能污染后续启动。
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
  export HF_ENDPOINT="$HF_ENDPOINT"
  export VLLM_USE_MODELSCOPE="True"
  export VLLM_USE_V1="$QWEN_USE_V1"
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
  export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
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
  if [[ "$QWEN_ENFORCE_EAGER" == "1" ]]; then
    vllm_args+=(--enforce-eager)
  fi
  if [[ "$QWEN_DISABLE_CUDA_GRAPH" == "1" ]]; then
    vllm_args+=(--disable-cudagraph)
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
  for i in {1..900}; do
    if curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then
      log "Qwen vLLM 已就绪"
      return 0
    fi
    if ! kill -0 "$(cat "$RUN_DIR/qwen-vllm.pid")" 2>/dev/null; then
      rm -f "$RUN_DIR/qwen-vllm.pid"
      err "Qwen vLLM 启动失败，最近日志："
      tail -n 120 "$LOG_DIR/qwen-vllm.log" || true
      echo >&2
      err "如果是 Free memory 错误：降低 QWEN_GPU_MEMORY_UTILIZATION，或先降低/停止 Sydney：/workspace/sydney_rocm/bin/stop_sydney_server.sh"
      err "如果是 hipErrorIllegalAddress：保持 QWEN_USE_V1=0 QWEN_ENFORCE_EAGER=1 QWEN_DISABLE_CUDA_GRAPH=1；必要时降低 QWEN_MAX_NUM_SEQS/批量 tokens。"
      exit 1
    fi
    sleep 2
  done
  err "Qwen vLLM 启动超时，日志：$LOG_DIR/qwen-vllm.log"
  tail -n 120 "$LOG_DIR/qwen-vllm.log" || true
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

# vLLM 启动内存策略：QWEN_AUTO_MEMORY_UTIL=1 时脚本会按剩余显存自动降低 gpu_memory_utilization。
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
[[ -x /workspace/sydney_rocm/bin/status_sydney_server.sh ]] && /workspace/sydney_rocm/bin/status_sydney_server.sh || true
for name in qwen-vllm app cloudflared; do
  pid_file="$RUN_DIR/\$name.pid"
  if [[ -f "\$pid_file" ]] && kill -0 "\$(cat "\$pid_file")" 2>/dev/null; then
    echo "\$name: running PID=\$(cat "\$pid_file")"
  else
    echo "\$name: stopped"
  fi
done
curl -fsS http://127.0.0.1:$QWEN_PORT/v1/models || true
curl -fsS http://127.0.0.1:$APP_PORT/api/config >/dev/null && echo "app ok: http://127.0.0.1:$APP_PORT" || true
grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "$LOG_DIR/cloudflared-console.log" 2>/dev/null | tail -n 1 || true
EOF
  cat > "$BIN_DIR/stop_all.sh" <<EOF
#!/usr/bin/env bash
set -e
[[ -x /workspace/sydney_rocm/bin/stop_sydney_server.sh ]] && /workspace/sydney_rocm/bin/stop_sydney_server.sh || true
for name in cloudflared app qwen-vllm; do
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
  start_sydney
  start_qwen_vllm
  write_cloud_env
  start_app
  start_console_tunnel
  echo
  echo "管理命令："
  echo "  $BIN_DIR/status_all.sh"
  echo "  $BIN_DIR/stop_all.sh"
  echo "日志："
  echo "  tail -f $LOG_DIR/app.log"
  echo "  tail -f $LOG_DIR/qwen-vllm.log"
  echo "  tail -f $LOG_DIR/cloudflared-console.log"
  echo "  tail -f /workspace/sydney_rocm/logs/llama-server.log"
}

main "$@"
