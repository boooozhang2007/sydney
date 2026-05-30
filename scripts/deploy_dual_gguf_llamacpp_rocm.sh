#!/usr/bin/env bash
# 自包含一键部署：Sydney GGUF + Qwen3.6-27B Unsloth GGUF 全部走 llama.cpp/ROCm。
# 不调用本仓库其它 shell 脚本；只复用 app.py/web 控制台代码（若 APP_DIR 不存在则自动 clone）。
#
# 默认端口：
#   Sydney source: 127.0.0.1:8000/v1
#   Qwen aux:      127.0.0.1:8010/v1
#   Web 控制台:    127.0.0.1:7860
#   Cloudflare Tunnel 只暴露 Web 控制台，不暴露模型 API。
#
# 快速使用：
#   bash scripts/deploy_dual_gguf_llamacpp_rocm.sh
#   bash scripts/deploy_dual_gguf_llamacpp_rocm.sh --status
#   bash scripts/deploy_dual_gguf_llamacpp_rocm.sh --stop

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKDIR="${WORKDIR:-/workspace/sydney_dual_gguf}"
LLAMA_DIR="${LLAMA_DIR:-$WORKDIR/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/models}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
RUN_DIR="${RUN_DIR:-$WORKDIR/run}"
BIN_DIR="${BIN_DIR:-$WORKDIR/bin}"

# 若当前目录就是项目，默认使用当前项目；否则 clone 到 /workspace/sydney_NEWBING。
APP_DIR="${APP_DIR:-}"
REPO_URL="${REPO_URL:-https://gitee.com/qzonez/sydney.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_PROXY_PREFIX="${GIT_PROXY_PREFIX:-}"

# llama.cpp 镜像源。也可覆盖为 GitHub 原始源。
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://gitcode.com/GitHub_Trending/ll/llama.cpp.git}"
GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
SKIP_APT="${SKIP_APT:-0}"

# 两个服务均默认并发 64。llama.cpp 的 --ctx-size 是总上下文；
# 262144 / 64 = 每 slot 约 4096 tokens。需要更长单轮上下文可增大 *_CTX_SIZE。
PARALLEL="${PARALLEL:-64}"
SYDNEY_PARALLEL="${SYDNEY_PARALLEL:-$PARALLEL}"
QWEN_PARALLEL="${QWEN_PARALLEL:-$PARALLEL}"
SYDNEY_CTX_SIZE="${SYDNEY_CTX_SIZE:-262144}"
QWEN_CTX_SIZE="${QWEN_CTX_SIZE:-262144}"
GPU_LAYERS="${GPU_LAYERS:-999}"
SYDNEY_BATCH_SIZE="${SYDNEY_BATCH_SIZE:-1024}"
SYDNEY_UBATCH_SIZE="${SYDNEY_UBATCH_SIZE:-512}"
QWEN_BATCH_SIZE="${QWEN_BATCH_SIZE:-2048}"
QWEN_UBATCH_SIZE="${QWEN_UBATCH_SIZE:-512}"

SYDNEY_HOST="${SYDNEY_HOST:-127.0.0.1}"
SYDNEY_PORT="${SYDNEY_PORT:-8000}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8010}"
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-7860}"

# Sydney GGUF：沿用你的 Clever Sydney Q8。
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SYDNEY_HF_REPO_ID="${SYDNEY_HF_REPO_ID:-FPHam/Clever_Sydney-4_12b_GGUF}"
SYDNEY_MODEL_FILE="${SYDNEY_MODEL_FILE:-Clever_Sydney-4_12b_Q8_0_o.gguf}"
SYDNEY_MODEL_PATH="${SYDNEY_MODEL_PATH:-$MODEL_DIR/$SYDNEY_MODEL_FILE}"
SYDNEY_MODEL_LOCAL_FILE="${SYDNEY_MODEL_LOCAL_FILE:-}"
SYDNEY_SERVED_MODEL_NAME="${SYDNEY_SERVED_MODEL_NAME:-clever-sydney-4-12b-q8}"

# Qwen3.6 Unsloth GGUF：默认 ModelScope 仓库和 UD-Q4_K_XL，速度/质量折中较好。
QWEN_MS_MODEL_ID="${QWEN_MS_MODEL_ID:-unsloth/Qwen3.6-27B-GGUF}"
QWEN_GGUF_FILE="${QWEN_GGUF_FILE:-Qwen3.6-27B-UD-Q4_K_XL.gguf}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$MODEL_DIR/$QWEN_GGUF_FILE}"
QWEN_MODEL_LOCAL_FILE="${QWEN_MODEL_LOCAL_FILE:-}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen3.6-27b-unsloth-gguf}"

# 采样默认：Sydney 保留原设置；Qwen 用 Unsloth/Qwen 常用 coding/thinking 友好参数。
SYDNEY_TEMP="${SYDNEY_TEMP:-0.82}"
SYDNEY_TOP_P="${SYDNEY_TOP_P:-0.94}"
SYDNEY_REPEAT_PENALTY="${SYDNEY_REPEAT_PENALTY:-1.16}"
QWEN_TEMP="${QWEN_TEMP:-0.6}"
QWEN_TOP_P="${QWEN_TOP_P:-0.95}"
QWEN_TOP_K="${QWEN_TOP_K:-20}"
QWEN_MIN_P="${QWEN_MIN_P:-0.0}"
QWEN_REPEAT_PENALTY="${QWEN_REPEAT_PENALTY:-1.05}"

LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
SYDNEY_EXTRA_ARGS="${SYDNEY_EXTRA_ARGS:-}"
QWEN_EXTRA_ARGS="${QWEN_EXTRA_ARGS:-}"
USE_JINJA_QWEN="${USE_JINJA_QWEN:-1}"
USE_METRICS="${USE_METRICS:-1}"

APP_CONCURRENCY="${APP_CONCURRENCY:-64}"
USE_TUNNEL="${USE_TUNNEL:-1}"
KILL_PORT_FALLBACK="${KILL_PORT_FALLBACK:-1}"
CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH:-/mnt/cloudflared}"
CLOUDFLARED_URL="${CLOUDFLARED_URL:-https://ghfast.top/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS:-$CLOUDFLARED_URL https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"

CMD="start"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start|start) CMD="start" ;;
    --restart|restart) CMD="restart" ;;
    --stop|stop) CMD="stop" ;;
    --status|status) CMD="status" ;;
    --console-only) CMD="console-only" ;;
    --no-tunnel) USE_TUNNEL="0" ;;
    --force-rebuild) FORCE_REBUILD="1" ;;
    --skip-apt) SKIP_APT="1" ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

log(){ printf '\033[1;36m[dual-gguf]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err(){ printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
have(){ command -v "$1" >/dev/null 2>&1; }

ensure_dirs(){ mkdir -p "$WORKDIR" "$MODEL_DIR" "$LOG_DIR" "$RUN_DIR" "$BIN_DIR"; }

as_root_prefix(){
  if [[ "$(id -u)" == "0" ]]; then echo ""; elif have sudo; then echo "sudo"; else echo ""; fi
}

apt_install(){
  [[ "$SKIP_APT" == "1" ]] && { warn "已跳过 apt 安装"; return 0; }
  have apt-get || { warn "未检测到 apt-get，跳过系统依赖安装"; return 0; }
  local prefix; prefix="$(as_root_prefix)"
  if [[ -z "$prefix" && "$(id -u)" != "0" ]]; then
    warn "当前非 root 且无 sudo，跳过 apt。"
    return 0
  fi
  log "安装系统依赖"
  $prefix apt-get update -y || true
  DEBIAN_FRONTEND=noninteractive $prefix apt-get install -y \
    git cmake ninja-build build-essential pkg-config \
    curl wget aria2 ca-certificates python3 python3-pip jq procps lsof || true
}

pip_install(){
  log "安装/检查 Python 依赖"
  python3 -m pip install -U modelscope huggingface_hub uvicorn fastapi python-dotenv httpx pydantic >/dev/null
}

resolve_app_dir(){
  if [[ -n "$APP_DIR" && -f "$APP_DIR/app.py" ]]; then
    return 0
  fi
  if [[ -f "$REPO_DIR/app.py" ]]; then
    APP_DIR="$REPO_DIR"
    return 0
  fi
  if [[ -f "$(pwd)/app.py" ]]; then
    APP_DIR="$(pwd)"
    return 0
  fi
  APP_DIR="${APP_DIR:-/workspace/sydney_NEWBING}"
  if [[ -f "$APP_DIR/app.py" ]]; then
    return 0
  fi
  log "克隆控制台源码：$REPO_URL -> $APP_DIR"
  mkdir -p "$(dirname "$APP_DIR")"
  git -c http.version=HTTP/1.1 clone --depth 1 --branch "$GIT_BRANCH" "${GIT_PROXY_PREFIX}${REPO_URL}" "$APP_DIR" \
    || git -c http.version=HTTP/1.1 clone --depth 1 --branch "$GIT_BRANCH" "$REPO_URL" "$APP_DIR"
}

install_app_requirements(){
  resolve_app_dir
  cd "$APP_DIR"
  [[ -f requirements-dataset.txt ]] && python3 -m pip install -r requirements-dataset.txt >/dev/null
}

detect_rocm_root(){
  if [[ -d "${ROCM_PATH:-}" ]]; then echo "$ROCM_PATH"; elif [[ -d /opt/rocm ]]; then echo /opt/rocm; else echo ""; fi
}

detect_amd_arch(){
  local arch="${AMDGPU_TARGETS:-}"
  [[ -n "$arch" ]] && { echo "$arch"; return 0; }
  if have rocminfo; then arch="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1 || true)"; fi
  if [[ -z "$arch" ]] && have rocm_agent_enumerator; then arch="$(rocm_agent_enumerator 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1 || true)"; fi
  [[ -z "$arch" ]] && { arch="gfx942"; warn "未能自动检测 AMDGPU_TARGETS，临时使用 $arch；可手动覆盖。"; }
  echo "$arch"
}

clone_llama_cpp(){
  if [[ -d "$LLAMA_DIR/.git" ]]; then
    log "llama.cpp 已存在：$LLAMA_DIR"
    return 0
  fi
  [[ -d "$LLAMA_DIR" && ! -d "$LLAMA_DIR/.git" ]] && rm -rf "$LLAMA_DIR"
  local urls=()
  if [[ -n "$GITHUB_PROXY_PREFIX" && "$LLAMA_CPP_REPO" == *github.com/* ]]; then
    urls+=("${GITHUB_PROXY_PREFIX}${LLAMA_CPP_REPO}")
  fi
  urls+=("$LLAMA_CPP_REPO" "https://github.com/ggml-org/llama.cpp.git")
  local u
  for u in "${urls[@]}"; do
    log "克隆 llama.cpp：$u"
    if git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999 \
      clone --depth 1 --single-branch "$u" "$LLAMA_DIR"; then
      return 0
    fi
    warn "clone 失败：$u"
    rm -rf "$LLAMA_DIR"
    sleep 2
  done
  err "llama.cpp 克隆失败"
  exit 1
}

build_llama_cpp(){
  local server_bin="$LLAMA_DIR/build/bin/llama-server"
  if [[ "$FORCE_REBUILD" != "1" && -x "$server_bin" ]]; then
    log "检测到已编译 llama-server：$server_bin"
    return 0
  fi
  local rocm arch
  rocm="$(detect_rocm_root)"
  arch="$(detect_amd_arch)"
  if [[ -n "$rocm" ]]; then
    export ROCM_PATH="$rocm"
    export PATH="$ROCM_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
  fi
  log "编译 llama.cpp ROCm/HIP：AMDGPU_TARGETS=$arch"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$arch" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" -j"$(nproc)"
  [[ -x "$server_bin" ]] || { err "编译后找不到 $server_bin"; exit 1; }
}

file_ok(){ local f="$1"; local min="${2:-1000000000}"; [[ -f "$f" ]] && [[ "$(stat -c '%s' "$f" 2>/dev/null || echo 0)" -gt "$min" ]]; }
gb(){ awk -v b="${1:-0}" 'BEGIN {printf "%.2f", b/1000000000}'; }

download_with_aria2_or_curl(){
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  log "直链下载：$url"
  if have aria2c; then
    aria2c -x 16 -s 16 -k 1M -c \
      --retry-wait=5 --max-tries=20 --timeout=60 --connect-timeout=30 \
      --allow-overwrite=true --auto-file-renaming=false \
      -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  elif have curl; then
    curl -fL --retry 20 --retry-delay 5 --connect-timeout 30 -C - -o "$out" "$url"
  elif have wget; then
    wget -c --tries=20 --timeout=60 -O "$out" "$url"
  else
    err "没有 aria2c/curl/wget"
    exit 1
  fi
}

download_sydney(){
  if file_ok "$SYDNEY_MODEL_PATH"; then
    local s; s="$(stat -c '%s' "$SYDNEY_MODEL_PATH")"
    log "Sydney 模型已存在：$SYDNEY_MODEL_PATH ($(gb "$s") GB)"
    return 0
  fi
  if [[ -n "$SYDNEY_MODEL_LOCAL_FILE" && -f "$SYDNEY_MODEL_LOCAL_FILE" ]]; then
    log "使用本地 Sydney 模型：$SYDNEY_MODEL_LOCAL_FILE"
    cp -f "$SYDNEY_MODEL_LOCAL_FILE" "$SYDNEY_MODEL_PATH"
    return 0
  fi
  log "下载 Sydney GGUF：$SYDNEY_HF_REPO_ID/$SYDNEY_MODEL_FILE"
  if have hf; then
    HF_ENDPOINT="$HF_ENDPOINT" HF_HUB_DISABLE_XET=1 hf download "$SYDNEY_HF_REPO_ID" "$SYDNEY_MODEL_FILE" --local-dir "$MODEL_DIR" || true
  fi
  if ! file_ok "$SYDNEY_MODEL_PATH"; then
    download_with_aria2_or_curl "$HF_ENDPOINT/$SYDNEY_HF_REPO_ID/resolve/main/$SYDNEY_MODEL_FILE" "$SYDNEY_MODEL_PATH" || true
  fi
  file_ok "$SYDNEY_MODEL_PATH" || { err "Sydney GGUF 下载失败：$SYDNEY_MODEL_PATH"; exit 1; }
}

download_qwen_modelscope(){
  if file_ok "$QWEN_MODEL_PATH"; then
    local s; s="$(stat -c '%s' "$QWEN_MODEL_PATH")"
    log "Qwen GGUF 已存在：$QWEN_MODEL_PATH ($(gb "$s") GB)"
    return 0
  fi
  if [[ -n "$QWEN_MODEL_LOCAL_FILE" && -f "$QWEN_MODEL_LOCAL_FILE" ]]; then
    log "使用本地 Qwen GGUF：$QWEN_MODEL_LOCAL_FILE"
    cp -f "$QWEN_MODEL_LOCAL_FILE" "$QWEN_MODEL_PATH"
    return 0
  fi
  log "用 ModelScope 下载 Qwen Unsloth GGUF：$QWEN_MS_MODEL_ID / $QWEN_GGUF_FILE"
  QWEN_MS_MODEL_ID="$QWEN_MS_MODEL_ID" QWEN_GGUF_FILE="$QWEN_GGUF_FILE" QWEN_MODEL_PATH="$QWEN_MODEL_PATH" MODEL_DIR="$MODEL_DIR" \
  python3 - <<'PY'
import os, shutil
from pathlib import Path

model_id = os.environ["QWEN_MS_MODEL_ID"]
file_name = os.environ["QWEN_GGUF_FILE"]
model_dir = Path(os.environ["MODEL_DIR"])
target = Path(os.environ["QWEN_MODEL_PATH"])
model_dir.mkdir(parents=True, exist_ok=True)

try:
    from modelscope import snapshot_download
except Exception:
    from modelscope.hub.snapshot_download import snapshot_download

kwargs_list = [
    dict(model_id=model_id, allow_file_pattern=file_name, local_dir=str(model_dir)),
    dict(model_id=model_id, allow_patterns=[file_name], local_dir=str(model_dir)),
    dict(model_id=model_id, local_dir=str(model_dir)),
    dict(model_id=model_id, cache_dir=str(model_dir)),
]
last = None
root = None
for kwargs in kwargs_list:
    try:
        print("[modelscope] snapshot_download", kwargs, flush=True)
        root = snapshot_download(**kwargs)
        break
    except TypeError as e:
        last = e
    except Exception as e:
        last = e
if root is None:
    raise RuntimeError(f"ModelScope download failed: {last}")

candidates = [Path(root) / file_name, model_dir / file_name]
candidates += list(Path(root).rglob(file_name)) if Path(root).exists() else []
candidates += list(model_dir.rglob(file_name))
src = next((p for p in candidates if p.is_file() and p.stat().st_size > 1_000_000_000), None)
if src is None:
    raise FileNotFoundError(f"cannot find large GGUF after download: {file_name}")
target.parent.mkdir(parents=True, exist_ok=True)
if src.resolve() != target.resolve():
    shutil.copyfile(src, target)
print(f"[modelscope] ready: {target}", flush=True)
PY
  if ! file_ok "$QWEN_MODEL_PATH"; then
    warn "ModelScope Python 未得到目标文件，尝试 resolve 直链。"
    download_with_aria2_or_curl "https://modelscope.cn/models/$QWEN_MS_MODEL_ID/resolve/master/$QWEN_GGUF_FILE" "$QWEN_MODEL_PATH" || true
  fi
  file_ok "$QWEN_MODEL_PATH" || { err "Qwen GGUF 下载失败：$QWEN_MODEL_PATH"; exit 1; }
}

server_bin(){ echo "$LLAMA_DIR/build/bin/llama-server"; }
server_supports(){
  local flag="$1"
  "$(server_bin)" --help 2>&1 | grep -q -- "$flag"
}

append_supported_switch(){
  local arr_name="$1" flag="$2"; local -n arr="$arr_name"
  if server_supports "$flag"; then arr+=("$flag"); fi
}

append_extra_args(){
  local arr_name="$1" extra="$2"; local -n arr="$arr_name"
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    local parts=( $extra )
    arr+=("${parts[@]}")
  fi
}

start_llama_server(){
  local name="$1" port="$2" host="$3" model="$4" alias="$5" ctx="$6" parallel="$7" batch="$8" ubatch="$9" temp="${10}" top_p="${11}" repeat="${12}" log_file="${13}" pid_file="${14}" extra_kind="${15}"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    log "$name 已在运行：PID=$(cat "$pid_file")"
    return 0
  fi
  kill_port_if_needed "$port" "$name"
  local args=(
    --host "$host"
    --port "$port"
    --model "$model"
    --alias "$alias"
    --ctx-size "$ctx"
    --n-gpu-layers "$GPU_LAYERS"
    --parallel "$parallel"
    --batch-size "$batch"
    --ubatch-size "$ubatch"
    --temp "$temp"
    --top-p "$top_p"
    --repeat-penalty "$repeat"
    --cont-batching
  )
  if [[ "$USE_METRICS" == "1" ]]; then
    append_supported_switch args "--metrics"
    append_supported_switch args "--slots"
  fi
  if [[ "$extra_kind" == "qwen" ]]; then
    if [[ "$USE_JINJA_QWEN" == "1" ]]; then append_supported_switch args "--jinja"; fi
    args+=(--top-k "$QWEN_TOP_K" --min-p "$QWEN_MIN_P")
    append_extra_args args "$QWEN_EXTRA_ARGS"
  else
    append_extra_args args "$SYDNEY_EXTRA_ARGS"
  fi
  append_extra_args args "$LLAMA_EXTRA_ARGS"

  log "启动 $name: http://$host:$port/v1 parallel=$parallel ctx=$ctx"
  log "$name args: ${args[*]}"
  nohup "$(server_bin)" "${args[@]}" > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  for _ in {1..240}; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      log "$name 已就绪"
      return 0
    fi
    if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      rm -f "$pid_file"
      err "$name 启动失败，最近日志："
      tail -n 160 "$log_file" || true
      exit 1
    fi
    sleep 2
  done
  err "$name 启动超时，最近日志："
  tail -n 160 "$log_file" || true
  exit 1
}

kill_pid(){
  local pid_file="$1" name="$2" port="${3:-}"
  if [[ -f "$pid_file" ]]; then
    local pid; pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "停止 $name PID=$pid"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
  if [[ -n "$port" ]] && have lsof; then
    local pids; pids="$(lsof -ti TCP:"$port" 2>/dev/null | tr '\n' ' ' || true)"
    if [[ -n "$pids" ]]; then
      warn "按端口清理 $name 残留：$port $pids"
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      sleep 1
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

kill_port_if_needed(){
  local port="$1" name="$2"
  [[ "$KILL_PORT_FALLBACK" == "1" ]] || return 0
  have lsof || return 0
  local pids
  pids="$(lsof -ti TCP:"$port" 2>/dev/null | tr '\n' ' ' || true)"
  [[ -z "$pids" ]] && return 0
  warn "端口 $port 已被占用，准备清理旧的 $name 进程：$pids"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 2
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
}

write_env(){
  resolve_app_dir
  cd "$APP_DIR"
  [[ -f .env ]] && cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)" || true
  log "写入控制台 .env：Sydney(source) + Qwen(GGUF aux)"
  cat > .env <<EOF
TEACHER_BASE_URL=http://127.0.0.1:$SYDNEY_PORT/v1
TEACHER_API_KEY=sk-local
TEACHER_MODEL=$SYDNEY_SERVED_MODEL_NAME
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
APP_DEFAULT_TARGET_MODEL=qwen36_27b_gguf
APP_DEFAULT_TRAIN_MODE=qlora
APP_DEFAULT_INCLUDE_NEEDS_REVIEW=false
APP_DEFAULT_ONLY_DIALOGUE_DISTILLATION=true

MODEL_TIMEOUT=600
MODEL_RETRIES=3
MODEL_RETRY_BACKOFF=2
TEACHER_TIMEOUT=600
SIMULATOR_TIMEOUT=600
TRANSLATOR_TIMEOUT=600
JUDGE_TIMEOUT=600

SOURCE_TEMPERATURE=0.78
SOURCE_TOP_P=0.92
SOURCE_MAX_TOKENS=512
EOF
}

start_app(){
  resolve_app_dir
  if [[ -f "$RUN_DIR/app.pid" ]] && kill -0 "$(cat "$RUN_DIR/app.pid")" 2>/dev/null; then
    log "Web 控制台已在运行：PID=$(cat "$RUN_DIR/app.pid")"
    return 0
  fi
  kill_port_if_needed "$APP_PORT" "Web 控制台"
  cd "$APP_DIR"
  log "启动 Web 控制台：http://$APP_HOST:$APP_PORT"
  nohup python3 -m uvicorn app:app --host "$APP_HOST" --port "$APP_PORT" > "$LOG_DIR/app.log" 2>&1 &
  echo $! > "$RUN_DIR/app.pid"
  for _ in {1..120}; do
    if curl -fsS "http://127.0.0.1:$APP_PORT/api/config" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:$APP_PORT/" >/dev/null 2>&1; then
      log "Web 控制台已就绪"
      return 0
    fi
    if ! kill -0 "$(cat "$RUN_DIR/app.pid")" 2>/dev/null; then
      err "Web 控制台启动失败，最近日志："
      tail -n 120 "$LOG_DIR/app.log" || true
      exit 1
    fi
    sleep 1
  done
  err "Web 控制台启动超时，最近日志："
  tail -n 120 "$LOG_DIR/app.log" || true
  exit 1
}

ensure_cloudflared(){
  local bin="$WORKDIR/cloudflared"
  if [[ -x "$bin" ]]; then echo "$bin"; return 0; fi
  if [[ -f "$CLOUDFLARED_LOCAL_PATH" ]]; then
    cp "$CLOUDFLARED_LOCAL_PATH" "$bin" && chmod +x "$bin" && echo "$bin" && return 0
  fi
  local u
  for u in $CLOUDFLARED_URL_FALLBACKS; do
    log "下载 cloudflared: $u" >&2
    if curl -L --retry 3 --retry-delay 3 --connect-timeout 20 --max-time 180 --speed-time 30 --speed-limit 1024 -o "$bin.tmp" "$u"; then
      # 真二进制 ~37MB; 镜像源损坏时常返回几十字节错误页. 至少 1MB + ELF magic.
      sz="$(stat -c '%s' "$bin.tmp" 2>/dev/null || echo 0)"
      if [[ "$sz" -gt 1000000 ]] && head -c 4 "$bin.tmp" | grep -q $'\x7fELF'; then
        mv "$bin.tmp" "$bin" && chmod +x "$bin" && echo "$bin" && return 0
      fi
      warn "cloudflared 源损坏 (size=$sz, 非 ELF): $u"
      rm -f "$bin.tmp"
    fi
  done
  err "cloudflared 下载失败。可上传到 $CLOUDFLARED_LOCAL_PATH 或设置 CLOUDFLARED_URL。"
  exit 1
}

start_tunnel(){
  [[ "$USE_TUNNEL" == "1" ]] || { warn "USE_TUNNEL=0，跳过公网 tunnel"; return 0; }
  local cfbin; cfbin="$(ensure_cloudflared)"
  if [[ -f "$RUN_DIR/cloudflared.pid" ]] && kill -0 "$(cat "$RUN_DIR/cloudflared.pid")" 2>/dev/null; then
    log "控制台 tunnel 已在运行：PID=$(cat "$RUN_DIR/cloudflared.pid")"
  else
    log "启动唯一 Cloudflare Tunnel -> http://127.0.0.1:$APP_PORT"
    nohup "$cfbin" tunnel --url "http://127.0.0.1:$APP_PORT" > "$LOG_DIR/cloudflared-console.log" 2>&1 &
    echo $! > "$RUN_DIR/cloudflared.pid"
  fi
  sleep 8
  local url; url="$(grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "$LOG_DIR/cloudflared-console.log" | tail -n 1 || true)"
  if [[ -n "$url" ]]; then
    echo
    echo "================ 控制台公网地址 ================"
    echo "$url"
    echo "================================================="
  else
    warn "暂未解析到 trycloudflare URL。查看：tail -f $LOG_DIR/cloudflared-console.log"
  fi
}

stop_all(){
  kill_pid "$RUN_DIR/cloudflared.pid" cloudflared
  kill_pid "$RUN_DIR/app.pid" app "$APP_PORT"
  kill_pid "$RUN_DIR/qwen-llama.pid" qwen-llama "$QWEN_PORT"
  kill_pid "$RUN_DIR/sydney-llama.pid" sydney-llama "$SYDNEY_PORT"
}

status_one(){
  local name="$1" pid_file="$2" port="$3"
  echo "--- $name ---"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    local pid; pid="$(cat "$pid_file")"
    echo "running PID=$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
    echo
  else
    echo "stopped"
  fi
  [[ -n "$port" ]] && curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null || true
  echo
}

status_all(){
  status_one "Sydney llama.cpp GGUF" "$RUN_DIR/sydney-llama.pid" "$SYDNEY_PORT"
  status_one "Qwen3.6 Unsloth llama.cpp GGUF" "$RUN_DIR/qwen-llama.pid" "$QWEN_PORT"
  echo "--- Web 控制台 ---"
  if [[ -f "$RUN_DIR/app.pid" ]] && kill -0 "$(cat "$RUN_DIR/app.pid")" 2>/dev/null; then echo "running PID=$(cat "$RUN_DIR/app.pid") http://127.0.0.1:$APP_PORT"; else echo "stopped"; fi
  echo "--- Tunnel ---"
  if [[ -f "$RUN_DIR/cloudflared.pid" ]] && kill -0 "$(cat "$RUN_DIR/cloudflared.pid")" 2>/dev/null; then
    echo "running PID=$(cat "$RUN_DIR/cloudflared.pid")"
    grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "$LOG_DIR/cloudflared-console.log" 2>/dev/null | tail -n 1 || true
  else
    echo "stopped"
  fi
  echo "--- GPU ---"
  if have rocm-smi; then rocm-smi || true; elif have nvidia-smi; then nvidia-smi || true; fi
}

start_models(){
  start_llama_server \
    "Sydney" "$SYDNEY_PORT" "$SYDNEY_HOST" "$SYDNEY_MODEL_PATH" "$SYDNEY_SERVED_MODEL_NAME" \
    "$SYDNEY_CTX_SIZE" "$SYDNEY_PARALLEL" "$SYDNEY_BATCH_SIZE" "$SYDNEY_UBATCH_SIZE" \
    "$SYDNEY_TEMP" "$SYDNEY_TOP_P" "$SYDNEY_REPEAT_PENALTY" \
    "$LOG_DIR/sydney-llama.log" "$RUN_DIR/sydney-llama.pid" "sydney"
  start_llama_server \
    "Qwen3.6-Unsloth-GGUF" "$QWEN_PORT" "$QWEN_HOST" "$QWEN_MODEL_PATH" "$QWEN_SERVED_MODEL_NAME" \
    "$QWEN_CTX_SIZE" "$QWEN_PARALLEL" "$QWEN_BATCH_SIZE" "$QWEN_UBATCH_SIZE" \
    "$QWEN_TEMP" "$QWEN_TOP_P" "$QWEN_REPEAT_PENALTY" \
    "$LOG_DIR/qwen-llama.log" "$RUN_DIR/qwen-llama.pid" "qwen"
}

full_start(){
  ensure_dirs
  apt_install
  pip_install
  install_app_requirements
  clone_llama_cpp
  build_llama_cpp
  download_sydney
  download_qwen_modelscope
  write_env
  start_models
  start_app
  start_tunnel
  echo
  echo "完成：两个模型均为 llama.cpp，默认并发均为 64。"
  echo "  Sydney: http://127.0.0.1:$SYDNEY_PORT/v1  model=$SYDNEY_SERVED_MODEL_NAME parallel=$SYDNEY_PARALLEL ctx=$SYDNEY_CTX_SIZE"
  echo "  Qwen:   http://127.0.0.1:$QWEN_PORT/v1  model=$QWEN_SERVED_MODEL_NAME parallel=$QWEN_PARALLEL ctx=$QWEN_CTX_SIZE"
  echo "  控制台: http://127.0.0.1:$APP_PORT  tunnel 只暴露控制台"
  echo "日志："
  echo "  tail -f $LOG_DIR/sydney-llama.log"
  echo "  tail -f $LOG_DIR/qwen-llama.log"
  echo "  tail -f $LOG_DIR/app.log"
  echo "  tail -f $LOG_DIR/cloudflared-console.log"
}

console_only(){
  ensure_dirs
  pip_install
  install_app_requirements
  write_env
  start_app
  start_tunnel
}

case "$CMD" in
  start) full_start ;;
  restart) stop_all; full_start ;;
  stop) stop_all ;;
  status) ensure_dirs; status_all ;;
  console-only) console_only ;;
  *) err "未知命令：$CMD"; exit 2 ;;
esac
