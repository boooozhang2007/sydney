#!/usr/bin/env bash
# 云端全流程部署：
#   Sydney source：llama.cpp + GGUF，127.0.0.1:8000
#   Qwen3.6-27B：llama.cpp + GGUF，127.0.0.1:8010
#   控制台：FastAPI，127.0.0.1:7860
#   只暴露一个 Cloudflare Tunnel：控制台，不暴露模型端口
#
# 用法：
#   REPO_URL=https://github.com/boooozhang2007/sydney.git bash scripts/deploy_cloud_full_rocm.sh
#
# 注意：Qwen 已从 vLLM/FP8 改为 GGUF/llama.cpp，避开 ROCm vLLM 的 gdn_attention_core hipErrorIllegalAddress。

set -Eeuo pipefail

APP_DIR="${APP_DIR:-/workspace/sydney_NEWBING}"
REPO_URL="${REPO_URL:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_PROXY_PREFIX="${GIT_PROXY_PREFIX:-https://github.akams.cn/}"
# 只加速 GitHub/Hugging Face；apt/pip 保持当前镜像环境默认源。
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-7860}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
QWEN_PORT="${QWEN_PORT:-8010}"

APP_CONCURRENCY="${APP_CONCURRENCY:-80}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-80}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-400000}"
SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"

# Qwen GGUF + llama.cpp 参数。
# 如果默认仓库/文件名与你实际使用的不一致，设置 QWEN_GGUF_REPO_ID/QWEN_GGUF_MODEL_NAME，或直接上传并设置 QWEN_MODEL_LOCAL_FILE=/mnt/xxx.gguf。
QWEN_WORKDIR="${QWEN_WORKDIR:-/workspace/qwen36_27b_rocm}"
QWEN_GGUF_REPO_ID="${QWEN_GGUF_REPO_ID:-ggml-org/Qwen3.6-27B-GGUF}"
QWEN_GGUF_MODEL_NAME="${QWEN_GGUF_MODEL_NAME:-Qwen3.6-27B-Q8_0.gguf}"
QWEN_MODEL_PROVIDER="${QWEN_MODEL_PROVIDER:-${MODEL_PROVIDER:-auto}}" # auto / hf / modelscope
QWEN_MS_GGUF_MODEL_ID="${QWEN_MS_GGUF_MODEL_ID:-}"
QWEN_MS_GGUF_FILE_PATH="${QWEN_MS_GGUF_FILE_PATH:-$QWEN_GGUF_MODEL_NAME}"
QWEN_MODEL_LOCAL_FILE="${QWEN_MODEL_LOCAL_FILE:-}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-q8-gguf}"
QWEN_PARALLEL="${QWEN_PARALLEL:-80}"
QWEN_CTX_SIZE="${QWEN_CTX_SIZE:-262144}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-512}"
QWEN_UBATCH_SIZE="${QWEN_UBATCH_SIZE:-256}"
QWEN_LLAMA_ARG_FIT="${QWEN_LLAMA_ARG_FIT:-off}"
QWEN_LLAMA_CONT_BATCHING="${QWEN_LLAMA_CONT_BATCHING:-1}"
QWEN_TEMP="${QWEN_TEMP:-0.55}"
QWEN_TOP_P="${QWEN_TOP_P:-0.90}"
QWEN_REPEAT_PENALTY="${QWEN_REPEAT_PENALTY:-1.08}"

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

ensure_dirs(){ mkdir -p "$STACK_DIR" "$LOG_DIR" "$RUN_DIR" "$BIN_DIR" "$QWEN_WORKDIR"; }

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
  python3 -m pip install -U uvicorn fastapi python-dotenv httpx pydantic
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

stop_legacy_qwen_vllm(){
  if [[ -f "$RUN_DIR/qwen-vllm.pid" ]]; then
    local pid
    pid="$(cat "$RUN_DIR/qwen-vllm.pid" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      warn "停止旧 vLLM Qwen 进程：PID=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/qwen-vllm.pid"
  fi
}

start_sydney(){
  cd "$APP_DIR"
  log "启动 Sydney source GGUF：127.0.0.1:$SYDNEY_PORT"
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

start_qwen_gguf(){
  cd "$APP_DIR"
  stop_legacy_qwen_vllm
  log "启动 Qwen3.6 GGUF：127.0.0.1:$QWEN_PORT"
  log "provider=$QWEN_MODEL_PROVIDER hf_repo=$QWEN_GGUF_REPO_ID ms_model=${QWEN_MS_GGUF_MODEL_ID:-<empty>} file=$QWEN_GGUF_MODEL_NAME parallel=$QWEN_PARALLEL ctx=$QWEN_CTX_SIZE batch=$QWEN_BATCH_SIZE ubatch=$QWEN_UBATCH_SIZE fit=$QWEN_LLAMA_ARG_FIT"
  HF_ENDPOINT="$HF_ENDPOINT" \
  GITHUB_PROXY_PREFIX="$GIT_PROXY_PREFIX" \
  WORKDIR="$QWEN_WORKDIR" \
  LLAMA_DIR="$LLAMA_DIR" \
  MODEL_PROVIDER="$QWEN_MODEL_PROVIDER" \
  MODELSCOPE_MODEL_ID="$QWEN_MS_GGUF_MODEL_ID" \
  MODELSCOPE_FILE_PATH="$QWEN_MS_GGUF_FILE_PATH" \
  HF_REPO_ID="$QWEN_GGUF_REPO_ID" \
  MODEL_NAME="$QWEN_GGUF_MODEL_NAME" \
  MODEL_LOCAL_FILE="$QWEN_MODEL_LOCAL_FILE" \
  SERVED_MODEL_NAME="$QWEN_SERVED_MODEL_NAME" \
  PORT="$QWEN_PORT" \
  PARALLEL="$QWEN_PARALLEL" \
  CTX_SIZE="$QWEN_CTX_SIZE" \
  BATCH_SIZE="$QWEN_BATCH_SIZE" \
  UBATCH_SIZE="$QWEN_UBATCH_SIZE" \
  LLAMA_ARG_FIT="$QWEN_LLAMA_ARG_FIT" \
  LLAMA_CONT_BATCHING="$QWEN_LLAMA_CONT_BATCHING" \
  TEMP="$QWEN_TEMP" \
  TOP_P="$QWEN_TOP_P" \
  REPEAT_PENALTY="$QWEN_REPEAT_PENALTY" \
  USE_TUNNEL=0 \
  bash scripts/setup_qwen36_27b_rocm.sh --restart
}

write_cloud_env(){
  cd "$APP_DIR"
  if [[ -f .env ]]; then cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)" || true; fi
  log "写入云端 .env：Sydney source + Qwen GGUF aux 全复用"
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
  clone_or_update_repo
  install_repo_requirements
  write_management_scripts
  start_qwen_gguf
  start_sydney
  write_cloud_env
  start_app
  start_console_tunnel
  echo
  echo "管理命令："
  echo "  bash $APP_DIR/scripts/manage_cloud_models.sh status"
  echo "  bash $APP_DIR/scripts/manage_cloud_models.sh restart-qwen"
  echo "  $BIN_DIR/status_all.sh"
  echo "  $BIN_DIR/stop_all.sh"
  echo "日志："
  echo "  tail -f $LOG_DIR/app.log"
  echo "  tail -f $LOG_DIR/cloudflared-console.log"
  echo "  tail -f /workspace/sydney_rocm/logs/llama-server.log"
  echo "  tail -f $QWEN_WORKDIR/logs/llama-server.log"
}

main "$@"
