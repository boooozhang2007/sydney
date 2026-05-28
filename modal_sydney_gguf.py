"""Deploy Clever Sydney GGUF on Modal as an OpenAI-compatible endpoint.

用途：
  把 Hugging Face 上的 GGUF 开源 Sydney 模型部署成 llama.cpp server，
  暴露 OpenAI-compatible `/v1/chat/completions`，然后在本地工作台页面
  的 Sydney/source 配置里使用。

部署：
  modal deploy modal_sydney_gguf.py

获取 URL：
  部署完成后 Modal 会输出一个类似：
  https://<workspace>--sydney-gguf-openai-server.modal.run

本地工作台配置：
  TEACHER_BASE_URL=<上面的 URL>/v1
  TEACHER_MODEL=clever-sydney-4-12b-q8
  TEACHER_API_PROTOCOL=chat_completions

说明：
  - 这个脚本不会加载 Qwen/Gemma 训练模型，只部署 GGUF Sydney/source。
  - 模型约 13GB 级别，首次启动会下载到 Modal Volume；后续冷启动复用缓存。
  - 默认 0 个常驻容器，scaledown_window 较短，省 credits。
"""

from __future__ import annotations

import os
import shutil
import shlex
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "sydney-gguf-openai-server"

MODEL_URL = (
    "https://huggingface.co/FPHam/Clever_Sydney-4_12b_GGUF/resolve/main/"
    "Clever_Sydney-4_12b_Q8_0_o.gguf"
)
MODEL_DIR_STR = "/models"
MODEL_DIR = Path(MODEL_DIR_STR)
MODEL_PATH = MODEL_DIR / "Clever_Sydney-4_12b_Q8_0_o.gguf"
SERVED_MODEL_NAME = "clever-sydney-4-12b-q8"

# llama.cpp server 端口。Modal web_server 会把这个端口暴露成 HTTPS。
PORT = 8000

# 省钱默认：不常驻，空闲 2 分钟自动缩容。
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(os.getenv("SCALEDOWN_WINDOW", "120"))

# 省 credits 默认用 A10G。12B Q8 + 4k ctx 通常可试；
# 如果 OOM 或吞吐太慢，部署前设置 MODAL_GPU=L40S。
GPU = os.getenv("MODAL_GPU", "A10G")

# 推理参数可用环境变量覆盖。
N_CTX = int(os.getenv("N_CTX", "4096"))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))
PARALLEL = int(os.getenv("PARALLEL", "1"))
CONT_BATCHING = os.getenv("CONT_BATCHING", "1") not in {"0", "false", "False"}

volume = modal.Volume.from_name("sydney-clever-gguf-cache", create_if_missing=True)

# 优先使用 llama.cpp 官方/社区 CUDA server 镜像。
# 如果该 tag 后续变动导致不可用，可把 LLAMA_CPP_IMAGE 改为你确认存在的镜像，
# 或切换到自己构建 llama.cpp 的 Modal Image。
LLAMA_CPP_IMAGE = os.getenv(
    "LLAMA_CPP_IMAGE",
    "ghcr.io/ggml-org/llama.cpp:server-cuda",
)

image = (
    modal.Image.from_registry(LLAMA_CPP_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("curl", "ca-certificates")
    .pip_install("huggingface_hub>=0.23.0", "httpx>=0.27.0")
)

app = modal.App(APP_NAME)

runtime_env = {
    "N_CTX": str(N_CTX),
    "N_GPU_LAYERS": str(N_GPU_LAYERS),
    "PARALLEL": str(PARALLEL),
    "CONT_BATCHING": "1" if CONT_BATCHING else "0",
}


def _find_llama_server() -> str:
    """查找镜像里 llama.cpp server 可执行文件。"""

    candidates = [
        os.getenv("LLAMA_SERVER_BIN", ""),
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
        "/app/llama-server",
        "/server",
        "llama-server",
    ]
    for item in candidates:
        if not item:
            continue
        if "/" in item and Path(item).exists():
            return item
        if "/" not in item:
            try:
                subprocess.run(["bash", "-lc", f"command -v {shlex.quote(item)}"], check=True, capture_output=True)
                return item
            except Exception:
                pass
    raise RuntimeError(
        "Cannot find llama-server binary in image. "
        "Set LLAMA_CPP_IMAGE or LLAMA_SERVER_BIN to a valid llama.cpp server image/binary."
    )


def _download_model_if_needed() -> None:
    """下载 GGUF 到 Modal Volume。

    使用 huggingface_hub 的 hf_hub_download 支持断点/缓存；下载完成后 commit volume。
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000_000:
        print(f"[modal-sydney] model exists: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e9:.2f} GB)")
        return

    print(f"[modal-sydney] downloading model from Hugging Face: {MODEL_URL}")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="FPHam/Clever_Sydney-4_12b_GGUF",
        filename="Clever_Sydney-4_12b_Q8_0_o.gguf",
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != MODEL_PATH.resolve():
        shutil.copy2(downloaded_path, MODEL_PATH)
    print(f"[modal-sydney] model ready: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e9:.2f} GB)")
    volume.commit()


def _wait_until_ready(timeout_s: int = 180) -> None:
    import httpx

    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                print(f"[modal-sydney] llama.cpp server ready: {resp.status_code} {resp.text[:200]}")
                return
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"llama.cpp server did not become ready in {timeout_s}s. last_error={last_error}")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    env=runtime_env,
    volumes={MODEL_DIR_STR: volume},
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=60 * 60,
    startup_timeout=20 * 60,
)
@modal.web_server(PORT, startup_timeout=20 * 60)
def serve():
    """Start llama.cpp server and expose it via Modal HTTPS."""

    _download_model_if_needed()
    llama_server = _find_llama_server()

    cmd = [
        llama_server,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--model",
        str(MODEL_PATH),
        "--alias",
        SERVED_MODEL_NAME,
        "--ctx-size",
        str(N_CTX),
        "--n-gpu-layers",
        str(N_GPU_LAYERS),
        "--parallel",
        str(PARALLEL),
    ]
    if CONT_BATCHING:
        cmd.append("--cont-batching")

    # 有些 llama.cpp 版本支持 --jinja，有些不支持。需要时可通过环境变量打开。
    if os.getenv("LLAMA_ENABLE_JINJA", "0") in {"1", "true", "True"}:
        cmd.append("--jinja")

    print("[modal-sydney] starting:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.Popen(cmd)
    _wait_until_ready()


@app.local_entrypoint()
def main():
    print("Deploy with:")
    print("  modal deploy modal_sydney_gguf.py")
    print()
    print("After deploy, use the Modal web URL as:")
    print("  TEACHER_BASE_URL=https://<your-modal-url>/v1")
    print(f"  TEACHER_MODEL={SERVED_MODEL_NAME}")
    print("  TEACHER_API_PROTOCOL=chat_completions")
