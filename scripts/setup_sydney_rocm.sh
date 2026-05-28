#!/usr/bin/env bash
# 自动配置 AMD/ROCm 环境，把 Clever Sydney GGUF 部署成 OpenAI-compatible API。
#
# 适用场景：
#   - Ubuntu 22.04 + ROCm 镜像/云 Notebook
#   - AMD GPU，已预装 ROCm / torch-rocm / ModelScope 镜像也可
#   - 模型：FPHam/Clever_Sydney-4_12b_GGUF 的 Q8_0 GGUF
#
# 功能：
#   1. 安装基础依赖
#   2. 下载/编译 llama.cpp ROCm/HIP 版本
#   3. 下载 Sydney GGUF 到持久目录
#   4. 生成 start/stop/status 脚本
#   5. 可选启动 llama-server 和 Cloudflare Tunnel
#   6. 输出本地工作台 .env 配置片段
#
# 快速用法：
#   bash scripts/setup_sydney_rocm.sh --start
#   bash scripts/setup_sydney_rocm.sh --start --tunnel
#
# 常用环境变量：
#   WORKDIR=/workspace/sydney_rocm
#   MODEL_URL=https://hf-mirror.com/FPHam/Clever_Sydney-4_12b_GGUF/resolve/main/Clever_Sydney-4_12b_Q8_0_o.gguf
#   HF_ENDPOINT=https://hf-mirror.com
#   MODEL_NAME=Clever_Sydney-4_12b_Q8_0_o.gguf
#   MODEL_LOCAL_FILE=/mnt/Clever_Sydney-4_12b_Q8_0_o.gguf
#   MODEL_LOCAL_SEARCH_DIRS=/mnt /mnt/data /workspace /root
#   SERVED_MODEL_NAME=clever-sydney-4-12b-q8
#   PORT=8000
#   CTX_SIZE=32768   # llama.cpp server 常把 ctx 作为总上下文；PARALLEL=8 时约等于每 slot 4096
#   GPU_LAYERS=999
#   PARALLEL=8
#   TEMP=0.82
#   TOP_P=0.94
#   REPEAT_PENALTY=1.16
#   FORCE_REBUILD=0
#   USE_TUNNEL=0
#   CLOUDFLARED_URL=https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
#   CLOUDFLARED_LOCAL_PATH=/mnt/cloudflared

set -Eeuo pipefail

WORKDIR="${WORKDIR:-/workspace/sydney_rocm}"
LLAMA_DIR="${LLAMA_DIR:-$WORKDIR/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$WORKDIR/models}"
LOG_DIR="${LOG_DIR:-$WORKDIR/logs}"
RUN_DIR="${RUN_DIR:-$WORKDIR/run}"
BIN_DIR="${BIN_DIR:-$WORKDIR/bin}"

HF_REPO_ID="${HF_REPO_ID:-FPHam/Clever_Sydney-4_12b_GGUF}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# GitHub 加速只用于克隆 llama.cpp；不改 apt/pip 等其他源。
GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-https://gh.llkk.cc/}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
MODEL_NAME="${MODEL_NAME:-Clever_Sydney-4_12b_Q8_0_o.gguf}"
MODEL_URL="${MODEL_URL:-$HF_ENDPOINT/$HF_REPO_ID/resolve/main/$MODEL_NAME}"
# 备用源会按顺序尝试。国内环境默认优先 hf-mirror，失败后再试 Hugging Face 官方。
MODEL_URL_FALLBACKS="${MODEL_URL_FALLBACKS:-$MODEL_URL https://huggingface.co/$HF_REPO_ID/resolve/main/$MODEL_NAME}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"
# 如果你已手动上传模型，脚本会先在这些位置查找并复制/链接到 MODEL_PATH。
# 默认覆盖 /mnt 和 /mnt/data，适合云 Notebook 上传文件。
MODEL_LOCAL_SEARCH_DIRS="${MODEL_LOCAL_SEARCH_DIRS:-/mnt /mnt/data /workspace /root}"
MODEL_LOCAL_FILE="${MODEL_LOCAL_FILE:-}"
# hardlink 可能跨盘失败；失败自动 copy。也可设置 MODEL_LINK_MODE=symlink/copy。
MODEL_LINK_MODE="${MODEL_LINK_MODE:-copy}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-clever-sydney-4-12b-q8}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# 对 llama.cpp server，较稳的经验是 CTX_SIZE ≈ PARALLEL × 单 slot 目标上下文。
# 默认 PARALLEL=8 时给总 ctx 32768，约每 slot 4096；若 OOM 可降到 16384/8192。
CTX_SIZE="${CTX_SIZE:-32768}"
GPU_LAYERS="${GPU_LAYERS:-999}"
# llama.cpp server 的 --parallel 表示并行 slot 数。显存 192GB 这类 AMD 环境可先用 8。
# 注意 KV cache 主要随 CTX_SIZE 增长；如果 OOM 就降低 CTX_SIZE 或 PARALLEL。
PARALLEL="${PARALLEL:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
TEMP="${TEMP:-0.82}"
TOP_P="${TOP_P:-0.94}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.16}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"
USE_TUNNEL="${USE_TUNNEL:-0}"
# cloudflared 下载在国内经常卡住；默认给多个镜像源和短超时。
CLOUDFLARED_URL="${CLOUDFLARED_URL:-https://gh.llkk.cc/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS:-$CLOUDFLARED_URL https://ghproxy.net/https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64}"
# 如果你已手动把 cloudflared 上传到 /mnt，脚本会优先复制这个文件。
CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH:-/mnt/cloudflared}"
START_AFTER_SETUP="0"
STOP_AFTER_SETUP="0"
RESTART_AFTER_SETUP="0"
STATUS_ONLY="0"
SKIP_APT="${SKIP_APT:-0}"

usage() {
  cat <<'USAGE'
自动配置 AMD/ROCm Sydney GGUF OpenAI-compatible API

Usage:
  bash scripts/setup_sydney_rocm.sh [options]

Options:
  --start          配置完成后启动 llama-server
  --stop           停止已启动的 llama-server/cloudflared
  --restart        停止旧服务后重新启动，修改 PARALLEL/CTX_SIZE 后必须用它
  --status         查看服务状态
  --tunnel         启动 Cloudflare Tunnel，生成公网 HTTPS URL
  --force-rebuild  强制重新编译 llama.cpp
  --skip-apt       跳过 apt 安装，适合无 sudo/root 环境
  -h, --help       显示帮助

Examples:
  bash scripts/setup_sydney_rocm.sh --start
  bash scripts/setup_sydney_rocm.sh --start --tunnel

环境变量示例：
  CTX_SIZE=32768 PARALLEL=8 PORT=8000 bash scripts/setup_sydney_rocm.sh --start
  WORKDIR=/mnt/data/sydney_rocm bash scripts/setup_sydney_rocm.sh --start --tunnel
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START_AFTER_SETUP="1" ;;
    --stop) STOP_AFTER_SETUP="1" ;;
    --restart) RESTART_AFTER_SETUP="1"; START_AFTER_SETUP="1" ;;
    --status) STATUS_ONLY="1" ;;
    --tunnel) USE_TUNNEL="1" ;;
    --force-rebuild) FORCE_REBUILD="1" ;;
    --skip-apt) SKIP_APT="1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

log() { printf '\033[1;36m[setup-sydney-rocm]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

as_root_prefix() {
  if [[ "$(id -u)" == "0" ]]; then
    echo ""
  elif have_cmd sudo; then
    echo "sudo"
  else
    echo ""
  fi
}

apt_install_if_possible() {
  if [[ "$SKIP_APT" == "1" ]]; then
    warn "已跳过 apt 安装。"
    return 0
  fi
  if ! have_cmd apt-get; then
    warn "未检测到 apt-get，跳过系统依赖安装。"
    return 0
  fi
  local prefix
  prefix="$(as_root_prefix)"
  if [[ -z "$prefix" && "$(id -u)" != "0" ]]; then
    warn "当前不是 root 且无 sudo，跳过 apt 安装。若编译失败，请手动安装 git/cmake/ninja/curl 等。"
    return 0
  fi

  log "安装基础依赖：git cmake ninja-build curl wget aria2 python3-pip rocm 运行所需工具"
  $prefix apt-get update -y
  DEBIAN_FRONTEND=noninteractive $prefix apt-get install -y \
    git cmake ninja-build build-essential pkg-config \
    curl wget aria2 ca-certificates python3 python3-pip jq procps lsof
}

detect_rocm_root() {
  if [[ -d "${ROCM_PATH:-}" ]]; then
    echo "$ROCM_PATH"
  elif [[ -d /opt/rocm ]]; then
    echo /opt/rocm
  else
    echo ""
  fi
}

detect_amd_arch() {
  local arch="${AMDGPU_TARGETS:-}"
  if [[ -n "$arch" ]]; then
    echo "$arch"
    return 0
  fi
  if have_cmd rocminfo; then
    arch="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1 || true)"
  fi
  if [[ -z "$arch" ]] && have_cmd rocm_agent_enumerator; then
    arch="$(rocm_agent_enumerator 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | head -n 1 || true)"
  fi
  if [[ -z "$arch" ]]; then
    # 常见云上 MI300/MI250/MI210 默认值。若不对，用户可手动 AMDGPU_TARGETS=gfxxxx 覆盖。
    arch="gfx942"
    warn "未能自动检测 AMDGPU_TARGETS，临时使用 $arch。若编译失败，请运行 rocminfo 查看并设置 AMDGPU_TARGETS=gfxxxx。"
  fi
  echo "$arch"
}

ensure_dirs() {
  mkdir -p "$WORKDIR" "$MODEL_DIR" "$LOG_DIR" "$RUN_DIR" "$BIN_DIR"
}

clone_or_update_llama_cpp() {
  if [[ -d "$LLAMA_DIR/.git" ]]; then
    log "llama.cpp 已存在：$LLAMA_DIR"
    return 0
  fi

  # 上次失败可能留下空目录，先清理。
  if [[ -d "$LLAMA_DIR" && ! -d "$LLAMA_DIR/.git" ]]; then
    warn "检测到不完整 llama.cpp 目录，清理后重试：$LLAMA_DIR"
    rm -rf "$LLAMA_DIR"
  fi

  local urls=()
  if [[ -n "$GITHUB_PROXY_PREFIX" ]]; then
    urls+=("${GITHUB_PROXY_PREFIX}${LLAMA_CPP_REPO}")
  fi
  urls+=("$LLAMA_CPP_REPO")

  local url
  for url in "${urls[@]}"; do
    log "克隆 llama.cpp 到 $LLAMA_DIR：$url"
    # 避免部分网络 HTTP/2 framing layer 问题，强制 HTTP/1.1，并降低并发。
    if git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999       clone --depth 1 --single-branch "$url" "$LLAMA_DIR"; then
      return 0
    fi
    warn "clone 失败：$url"
    rm -rf "$LLAMA_DIR"
    sleep 2
  done

  err "llama.cpp 克隆失败。可手动上传/预置到 LLAMA_DIR=$LLAMA_DIR，或设置 GITHUB_PROXY_PREFIX 为可用代理。"
  exit 1
}

build_llama_cpp() {
  local server_bin="$LLAMA_DIR/build/bin/llama-server"
  if [[ "$FORCE_REBUILD" != "1" && -x "$server_bin" ]]; then
    log "检测到已编译 llama-server：$server_bin"
    return 0
  fi

  local rocm_root arch
  rocm_root="$(detect_rocm_root)"
  arch="$(detect_amd_arch)"

  if [[ -z "$rocm_root" ]]; then
    warn "未检测到 /opt/rocm。继续尝试编译；若失败，请确认当前镜像已安装 ROCm。"
  else
    export ROCM_PATH="$rocm_root"
    export PATH="$ROCM_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
    log "ROCm 路径：$ROCM_PATH"
  fi
  log "AMDGPU_TARGETS=$arch"

  log "开始编译 llama.cpp ROCm/HIP 版本"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$arch" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" -j"$(nproc)"

  if [[ ! -x "$server_bin" ]]; then
    err "编译完成但找不到 $server_bin"
    exit 1
  fi
  log "llama-server 编译完成：$server_bin"
}

model_file_ok() {
  [[ -f "$MODEL_PATH" ]] || return 1
  local size
  size="$(stat -c '%s' "$MODEL_PATH" 2>/dev/null || echo 0)"
  [[ "$size" -gt 1000000000 ]]
}

print_download_progress_hint() {
  local size="0"
  if [[ -f "$MODEL_PATH" ]]; then
    size="$(stat -c '%s' "$MODEL_PATH" 2>/dev/null || echo 0)"
  fi
  log "当前已下载：$(awk "BEGIN {printf "%.2f", $size/1000000000}") GB -> $MODEL_PATH"
}

download_with_hf_cli() {
  if ! have_cmd huggingface-cli; then
    return 1
  fi
  log "尝试 huggingface-cli 下载，HF_ENDPOINT=$HF_ENDPOINT"
  HF_ENDPOINT="$HF_ENDPOINT" huggingface-cli download "$HF_REPO_ID" "$MODEL_NAME" \
    --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
}

download_with_aria2_or_curl() {
  local url="$1"
  log "尝试下载源：$url"
  print_download_progress_hint
  if have_cmd aria2c; then
    # aria2c 对国内镜像/大文件断点续传更稳。
    aria2c -x 16 -s 16 -k 1M -c \
      --retry-wait=5 --max-tries=20 --timeout=60 --connect-timeout=30 \
      -d "$MODEL_DIR" -o "$MODEL_NAME" "$url"
  elif have_cmd curl; then
    curl -L --retry 20 --retry-delay 5 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
      -C - -o "$MODEL_PATH" "$url"
  elif have_cmd wget; then
    wget -c --tries=20 --timeout=60 -O "$MODEL_PATH" "$url"
  else
    err "没有 aria2c/curl/wget，无法下载模型。"
    exit 1
  fi
}

find_local_uploaded_model() {
  # 1) 用户显式指定完整文件路径。
  if [[ -n "$MODEL_LOCAL_FILE" && -f "$MODEL_LOCAL_FILE" ]]; then
    echo "$MODEL_LOCAL_FILE"
    return 0
  fi

  # 2) 在常见上传目录查找同名文件。
  local d
  for d in $MODEL_LOCAL_SEARCH_DIRS; do
    [[ -d "$d" ]] || continue
    if [[ -f "$d/$MODEL_NAME" ]]; then
      echo "$d/$MODEL_NAME"
      return 0
    fi
  done

  # 3) 兜底：在常见目录下浅层查找 Sydney GGUF。
  for d in $MODEL_LOCAL_SEARCH_DIRS; do
    [[ -d "$d" ]] || continue
    local found
    found="$(find "$d" -maxdepth 3 -type f \( -name "$MODEL_NAME" -o -iname '*Sydney*.gguf' -o -iname '*sydney*.gguf' \) 2>/dev/null | head -n 1 || true)"
    if [[ -n "$found" ]]; then
      echo "$found"
      return 0
    fi
  done
  return 1
}

use_local_uploaded_model_if_any() {
  if model_file_ok; then
    return 0
  fi

  local src=""
  src="$(find_local_uploaded_model || true)"
  if [[ -z "$src" ]]; then
    return 1
  fi

  local size
  size="$(stat -c '%s' "$src" 2>/dev/null || echo 0)"
  if [[ "$size" -lt 1000000000 ]]; then
    warn "找到本地文件但体积过小，忽略：$src"
    return 1
  fi

  mkdir -p "$MODEL_DIR"
  if [[ "$(readlink -f "$src")" == "$(readlink -f "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")" ]]; then
    log "已使用本地上传模型：$MODEL_PATH"
    return 0
  fi

  log "发现本地上传模型：$src ($(awk "BEGIN {printf "%.2f", $size/1000000000}") GB)"
  log "准备放到运行路径：$MODEL_PATH"
  if [[ "$MODEL_LINK_MODE" == "symlink" ]]; then
    ln -sf "$src" "$MODEL_PATH"
  elif [[ "$MODEL_LINK_MODE" == "hardlink" ]]; then
    ln -f "$src" "$MODEL_PATH" 2>/dev/null || cp -f "$src" "$MODEL_PATH"
  else
    cp -f "$src" "$MODEL_PATH"
  fi
  return 0
}

download_model() {
  if use_local_uploaded_model_if_any && model_file_ok; then
    local size
    size="$(stat -c '%s' "$MODEL_PATH" 2>/dev/null || echo 0)"
    log "本地上传模型就绪：$MODEL_PATH ($(awk "BEGIN {printf "%.2f", $size/1000000000}") GB)"
    return 0
  fi

  if model_file_ok; then
    local size
    size="$(stat -c '%s' "$MODEL_PATH" 2>/dev/null || echo 0)"
    log "模型已存在：$MODEL_PATH ($(awk "BEGIN {printf "%.2f", $size/1000000000}") GB)"
    return 0
  fi

  log "下载 Sydney GGUF 到：$MODEL_PATH"
  log "默认国内镜像：$HF_ENDPOINT"

  # 先尝试 HF CLI + hf-mirror。失败不退出，继续 aria2/curl 直链。
  download_with_hf_cli || warn "huggingface-cli 下载失败或不可用，切换到直链断点下载。"

  if ! model_file_ok; then
    # 去重后按顺序尝试所有 URL。
    local tried=""
    local url
    for url in $MODEL_URL_FALLBACKS; do
      [[ -z "$url" ]] && continue
      if [[ " $tried " == *" $url "* ]]; then
        continue
      fi
      tried="$tried $url"
      download_with_aria2_or_curl "$url" || warn "下载源失败：$url"
      if model_file_ok; then
        break
      fi
    done
  fi

  local size
  size="$(stat -c '%s' "$MODEL_PATH" 2>/dev/null || echo 0)"
  if [[ "$size" -lt 1000000000 ]]; then
    err "模型文件过小，疑似下载失败：$MODEL_PATH"
    err "建议重试：HF_ENDPOINT=https://hf-mirror.com bash scripts/setup_sydney_rocm.sh --start"
    err "或手动下载到该路径后再运行脚本。"
    exit 1
  fi
  log "模型下载完成：$MODEL_PATH ($(awk "BEGIN {printf "%.2f", $size/1000000000}") GB)"
}

write_runtime_scripts() {
  local start_script="$BIN_DIR/start_sydney_server.sh"
  local stop_script="$BIN_DIR/stop_sydney_server.sh"
  local status_script="$BIN_DIR/status_sydney_server.sh"

  cat > "$start_script" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
WORKDIR="$WORKDIR"
LLAMA_DIR="$LLAMA_DIR"
MODEL_PATH="$MODEL_PATH"
LOG_DIR="$LOG_DIR"
RUN_DIR="$RUN_DIR"
SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
HOST="\${HOST:-$HOST}"
PORT="\${PORT:-$PORT}"
CTX_SIZE="\${CTX_SIZE:-$CTX_SIZE}"
GPU_LAYERS="\${GPU_LAYERS:-$GPU_LAYERS}"
PARALLEL="\${PARALLEL:-$PARALLEL}"
BATCH_SIZE="\${BATCH_SIZE:-$BATCH_SIZE}"
UBATCH_SIZE="\${UBATCH_SIZE:-$UBATCH_SIZE}"
TEMP="\${TEMP:-$TEMP}"
TOP_P="\${TOP_P:-$TOP_P}"
REPEAT_PENALTY="\${REPEAT_PENALTY:-$REPEAT_PENALTY}"
USE_TUNNEL="\${USE_TUNNEL:-$USE_TUNNEL}"
CLOUDFLARED_URL="${CLOUDFLARED_URL}"
CLOUDFLARED_URL_FALLBACKS="${CLOUDFLARED_URL_FALLBACKS}"
CLOUDFLARED_LOCAL_PATH="${CLOUDFLARED_LOCAL_PATH}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="\$ROCM_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="\$ROCM_PATH/lib:\${LD_LIBRARY_PATH:-}"
mkdir -p "\$LOG_DIR" "\$RUN_DIR"
SERVER_BIN="\$LLAMA_DIR/build/bin/llama-server"
if [[ ! -x "\$SERVER_BIN" ]]; then
  echo "找不到 llama-server: \$SERVER_BIN" >&2
  exit 1
fi
if [[ -f "\$RUN_DIR/llama-server.pid" ]] && kill -0 "\$(cat "\$RUN_DIR/llama-server.pid")" 2>/dev/null; then
  echo "llama-server 已在运行，PID=\$(cat "\$RUN_DIR/llama-server.pid")"
  echo "注意：修改 PARALLEL/CTX_SIZE 不会作用到已运行进程。请执行：$BIN_DIR/stop_sydney_server.sh && PARALLEL=8 CTX_SIZE=32768 $BIN_DIR/start_sydney_server.sh"
else
  echo "启动 llama-server: http://127.0.0.1:\$PORT/v1"
  echo "参数：CTX_SIZE=\$CTX_SIZE PARALLEL=\$PARALLEL BATCH_SIZE=\$BATCH_SIZE UBATCH_SIZE=\$UBATCH_SIZE"
  nohup "\$SERVER_BIN" \
    --host "\$HOST" \
    --port "\$PORT" \
    --model "\$MODEL_PATH" \
    --alias "\$SERVED_MODEL_NAME" \
    --ctx-size "\$CTX_SIZE" \
    --n-gpu-layers "\$GPU_LAYERS" \
    --parallel "\$PARALLEL" \
    --batch-size "\$BATCH_SIZE" \
    --ubatch-size "\$UBATCH_SIZE" \
    --temp "\$TEMP" \
    --top-p "\$TOP_P" \
    --repeat-penalty "\$REPEAT_PENALTY" \
    --cont-batching \
    > "\$LOG_DIR/llama-server.log" 2>&1 &
  echo \$! > "\$RUN_DIR/llama-server.pid"
fi

for i in {1..180}; do
  if curl -fsS "http://127.0.0.1:\$PORT/health" >/dev/null 2>&1 || curl -fsS "http://127.0.0.1:\$PORT/v1/models" >/dev/null 2>&1; then
    echo "llama-server 已就绪"
    break
  fi
  if ! kill -0 "\$(cat "\$RUN_DIR/llama-server.pid")" 2>/dev/null; then
    echo "llama-server 启动失败，最近日志：" >&2
    tail -n 80 "\$LOG_DIR/llama-server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

if [[ "\$USE_TUNNEL" == "1" ]]; then
  CLOUDFLARED_BIN="\$WORKDIR/cloudflared"
  if [[ ! -x "\$CLOUDFLARED_BIN" ]]; then
    if [[ -f "\$CLOUDFLARED_LOCAL_PATH" ]]; then
      echo "使用手动上传的 cloudflared: \$CLOUDFLARED_LOCAL_PATH"
      cp "\$CLOUDFLARED_LOCAL_PATH" "\$CLOUDFLARED_BIN"
      chmod +x "\$CLOUDFLARED_BIN"
    fi
  fi
  if [[ ! -x "\$CLOUDFLARED_BIN" ]]; then
    echo "下载 cloudflared"
    downloaded="0"
    for u in \$CLOUDFLARED_URL_FALLBACKS; do
      echo "尝试 cloudflared 源: \$u"
      rm -f "\$CLOUDFLARED_BIN.tmp"
      if curl -L --retry 3 --retry-delay 3 --connect-timeout 20 --max-time 180 --speed-time 30 --speed-limit 1024 \
        -o "\$CLOUDFLARED_BIN.tmp" "\$u"; then
        if [[ -s "\$CLOUDFLARED_BIN.tmp" ]]; then
          mv "\$CLOUDFLARED_BIN.tmp" "\$CLOUDFLARED_BIN"
          chmod +x "\$CLOUDFLARED_BIN"
          downloaded="1"
          break
        fi
      fi
      echo "cloudflared 源失败: \$u"
    done
    if [[ "\$downloaded" != "1" ]]; then
      echo "cloudflared 下载失败。服务本身已在本机启动： http://127.0.0.1:\$PORT/v1" >&2
      echo "解决方式 1：不用 --tunnel，改用平台端口转发/公开端口 8000" >&2
      echo "解决方式 2：手动上传 cloudflared 到 \$CLOUDFLARED_BIN 并 chmod +x" >&2
      echo "解决方式 3：设置 CLOUDFLARED_URL 为你可访问的下载地址后重跑" >&2
      exit 0
    fi
  fi
  if [[ -f "\$RUN_DIR/cloudflared.pid" ]] && kill -0 "\$(cat "\$RUN_DIR/cloudflared.pid")" 2>/dev/null; then
    echo "cloudflared 已在运行，PID=\$(cat "\$RUN_DIR/cloudflared.pid")"
  else
    echo "启动 Cloudflare Tunnel"
    nohup "\$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:\$PORT" > "\$LOG_DIR/cloudflared.log" 2>&1 &
    echo \$! > "\$RUN_DIR/cloudflared.pid"
  fi
  sleep 6
  TUNNEL_URL="\$(grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "\$LOG_DIR/cloudflared.log" | tail -n 1 || true)"
  if [[ -n "\$TUNNEL_URL" ]]; then
    echo
    echo "公网 URL: \$TUNNEL_URL"
    echo "TEACHER_BASE_URL=\$TUNNEL_URL/v1"
  else
    echo "暂未解析到 tunnel URL，请查看：tail -f \$LOG_DIR/cloudflared.log"
  fi
fi

echo
echo "本机测试："
echo "curl http://127.0.0.1:\$PORT/v1/models"
echo
echo "本地数据工厂 .env："
echo "TEACHER_BASE_URL=http://127.0.0.1:\$PORT/v1"
echo "TEACHER_API_KEY=sk-local"
echo "TEACHER_MODEL=\$SERVED_MODEL_NAME"
echo "TEACHER_API_PROTOCOL=legacy_chat_completions"
echo "SOURCE_PROMPT_MODE=legacy_chat"
echo "SOURCE_USE_DEFAULT_STOPS=false"
EOF

  cat > "$stop_script" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
RUN_DIR="$RUN_DIR"
for name in cloudflared llama-server; do
  pid_file="\$RUN_DIR/\$name.pid"
  if [[ -f "\$pid_file" ]]; then
    pid="\$(cat "\$pid_file")"
    if kill -0 "\$pid" 2>/dev/null; then
      echo "停止 \$name PID=\$pid"
      kill "\$pid" || true
      sleep 2
      kill -9 "\$pid" 2>/dev/null || true
    fi
    rm -f "\$pid_file"
  fi
done
EOF

  cat > "$status_script" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
RUN_DIR="$RUN_DIR"
LOG_DIR="$LOG_DIR"
PORT="$PORT"
for name in llama-server cloudflared; do
  pid_file="\$RUN_DIR/\$name.pid"
  if [[ -f "\$pid_file" ]] && kill -0 "\$(cat "\$pid_file")" 2>/dev/null; then
    pid="\$(cat "\$pid_file")"
    echo "\$name: running PID=\$pid"
    if [[ "\$name" == "llama-server" ]]; then
      tr '\0' ' ' < "/proc/\$pid/cmdline" 2>/dev/null || true
      echo
    fi
  else
    echo "\$name: stopped"
  fi
done
echo
curl -fsS "http://127.0.0.1:\$PORT/v1/models" || true
echo
if [[ -f "\$LOG_DIR/cloudflared.log" ]]; then
  grep -oE 'https://[-a-zA-Z0-9.]+trycloudflare.com' "\$LOG_DIR/cloudflared.log" | tail -n 1 || true
fi
EOF

  chmod +x "$start_script" "$stop_script" "$status_script"
  log "已生成运行脚本："
  log "  $start_script"
  log "  $stop_script"
  log "  $status_script"
}

print_env_hint() {
  cat <<EOF

==================== 配置完成 ====================
工作目录：$WORKDIR
模型文件：$MODEL_PATH
服务端口：$PORT

启动：
  $BIN_DIR/start_sydney_server.sh
停止：
  $BIN_DIR/stop_sydney_server.sh
状态：
  $BIN_DIR/status_sydney_server.sh
日志：
  tail -f $LOG_DIR/llama-server.log

本地数据工厂 .env 可填：
  TEACHER_BASE_URL=http://<这台机器IP或Tunnel域名>:$PORT/v1
  TEACHER_API_KEY=sk-local
  TEACHER_MODEL=$SERVED_MODEL_NAME
  TEACHER_API_PROTOCOL=legacy_chat_completions
  SOURCE_PROMPT_MODE=legacy_chat
  SOURCE_USE_DEFAULT_STOPS=false

如果使用 --tunnel，启动脚本会打印：
  TEACHER_BASE_URL=https://xxxx.trycloudflare.com/v1
==================================================
EOF
}

main() {
  ensure_dirs
  write_runtime_scripts

  if [[ "$STOP_AFTER_SETUP" == "1" ]]; then
    "$BIN_DIR/stop_sydney_server.sh"
    exit 0
  fi
  if [[ "$RESTART_AFTER_SETUP" == "1" ]]; then
    "$BIN_DIR/stop_sydney_server.sh" || true
  fi
  if [[ "$STATUS_ONLY" == "1" ]]; then
    "$BIN_DIR/status_sydney_server.sh"
    exit 0
  fi

  apt_install_if_possible

  # pip 安装 huggingface_hub 不是强依赖；失败不影响 curl 下载。
  if have_cmd python3 && have_cmd pip3; then
    python3 -m pip install -q --upgrade huggingface_hub >/dev/null 2>&1 || true
  fi

  clone_or_update_llama_cpp
  build_llama_cpp
  download_model
  write_runtime_scripts
  print_env_hint

  if [[ "$START_AFTER_SETUP" == "1" ]]; then
    "$BIN_DIR/start_sydney_server.sh"
  fi
}

main "$@"
