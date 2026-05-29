from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from auto_reviewer import apply_duplicate_penalty, review_sample
from data_generator import (
    ModelConfig,
    OpenAICompatibleClient,
    generate_dialogue_sample,
    make_generation_specs,
    translate_dialogue_sample,
    to_sharegpt,
    write_jsonl,
)
from deduper import DedupIndex, conversation_text, fingerprint
from notebook_exporter import TARGET_MODELS, export_unsloth_notebook

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REVIEWED_DIR = DATA_DIR / "reviewed"
REJECTED_DIR = DATA_DIR / "rejected"
EXPORT_DIR = DATA_DIR / "exports"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "workspace.sqlite"
DB_BUSY_TIMEOUT_MS = 30000
DB_CORRUPTION_HINTS = (
    "database disk image is malformed",
    "file is not a database",
    "database schema is corrupt",
)

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
_THREAD_CLIENTS = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_bool(name: str, default: bool = False) -> bool:
    value = (os.getenv(name, "") or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except Exception:
        value = default
    return max(lo, min(hi, value))


def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


def get_thread_model_client(
    config: ModelConfig | None,
    *,
    role: str,
    pool_connections: int,
) -> Optional[OpenAICompatibleClient]:
    """Reuse one HTTP client per worker thread/model role.

    ThreadPoolExecutor 会复用 worker 线程。把 OpenAICompatibleClient 放到
    thread-local 后，同一 worker 连续处理多个样本时可以复用 keep-alive/TLS
    连接，避免每条样本重新建 httpx.Client 和连接池。
    """

    if config is None or not config.ready:
        return None
    pool_connections = max(16, int(pool_connections or 16))
    setattr(config, "pool_connections", pool_connections)
    cache = getattr(_THREAD_CLIENTS, "clients", None)
    if cache is None:
        cache = {}
        _THREAD_CLIENTS.clients = cache
    key = (
        role,
        config.base_url,
        config.model,
        config.api_key,
        config.api_protocol,
        float(config.timeout),
        int(config.retries),
        float(config.retry_backoff),
        pool_connections,
    )
    client = cache.get(key)
    if client is None:
        client = OpenAICompatibleClient(config)
        cache[key] = client
    return client



def ensure_dirs() -> None:
    for path in [DATA_DIR, RAW_DIR, REVIEWED_DIR, REJECTED_DIR, EXPORT_DIR, JOBS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    return conn


def is_sqlite_corruption_error(exc: BaseException) -> bool:
    """Return True for errors where the SQLite file should be rebuilt/recovered."""

    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    text = str(exc).lower()
    return any(hint in text for hint in DB_CORRUPTION_HINTS)


def init_db() -> None:
    """初始化本地 SQLite 工作区。"""

    ensure_dirs()
    with db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                spec_json TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                sharegpt_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                review_json TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                duplicate_of TEXT,
                source TEXT NOT NULL DEFAULT 'teacher'
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_score ON samples(score)")
        conn.commit()


def job_file(job_id: str) -> Path:
    """单个后台任务的持久化文件路径。

    只保存任务进度、日志、模型名等非敏感信息，不保存 API Key。
    """

    return JOBS_DIR / f"{Path(job_id).name}.json"


JOB_LOG_KEEP = env_int("JOB_LOG_KEEP", 240, 20, 800)
JOB_LOG_PERSIST_KEEP = env_int("JOB_LOG_PERSIST_KEEP", 120, 20, 400)
JOB_RESTORE_LIMIT = env_int("JOB_RESTORE_LIMIT", 10, 0, 80)
JOB_API_LOG_LIMIT = env_int("JOB_API_LOG_LIMIT", 120, 0, 300)
JOB_API_ERROR_LIMIT = env_int("JOB_API_ERROR_LIMIT", 80, 0, 300)
JOB_RESULT_ITEM_LIMIT = env_int("JOB_RESULT_ITEM_LIMIT", 50, 0, 500)
# 高并发时不要每条日志/计数都立刻落盘；否则 Windows 上容易和杀软/索引器抢文件句柄。
JOB_PERSIST_MIN_INTERVAL_MS = env_int("JOB_PERSIST_MIN_INTERVAL_MS", 1000, 0, 60000)
JOB_PERSIST_EVERY_UPDATES = env_int("JOB_PERSIST_EVERY_UPDATES", 25, 1, 1000)
JOB_PERSIST_RETRIES = env_int("JOB_PERSIST_RETRIES", 8, 1, 50)


def compact_job_for_persist(job: Dict[str, Any]) -> Dict[str, Any]:
    """压缩 job 快照，避免 3000+ 大任务把 data/jobs 写成超大 JSON。"""

    slim = {k: v for k, v in job.items() if not str(k).startswith("_")}
    slim["logs"] = list(slim.get("logs") or [])[-JOB_LOG_PERSIST_KEEP:]
    slim["errors"] = list(slim.get("errors") or [])[-JOB_API_ERROR_LIMIT:]
    result = slim.get("result")
    if isinstance(result, dict):
        result = dict(result)
        if isinstance(result.get("generated"), list):
            result["generated_count"] = len(result["generated"])
            result["generated"] = result["generated"][-JOB_RESULT_ITEM_LIMIT:]
        slim["result"] = result
    return slim


def compact_job_for_api(job: Dict[str, Any]) -> Dict[str, Any]:
    """返回给前端的轻量 job，避免恢复任务时页面被大日志卡死。"""

    slim = compact_job_for_persist(job)
    slim["logs"] = list(job.get("logs") or [])[-JOB_API_LOG_LIMIT:]
    slim["errors"] = list(job.get("errors") or [])[-JOB_API_ERROR_LIMIT:]
    return slim


def enrich_job_progress(job: Dict[str, Any]) -> Dict[str, Any]:
    """给任务快照补充前端需要的 progress 字段。"""

    total = max(1, int(job.get("total", 0) or 0))
    completed = int(job.get("completed", 0) or 0)
    enriched = json.loads(json.dumps(job, ensure_ascii=False))
    enriched["progress"] = {
        "completed": completed,
        "total": int(job.get("total", 0) or 0),
        "percent": round((completed / total) * 100, 1),
    }
    return enriched


def persist_job_locked(job_id: str, *, force: bool = False) -> None:
    """把内存中的 job 原子写入 data/jobs。

    调用方必须已经持有 JOBS_LOCK。

    高并发生成时 job_log/job_inc 会非常频繁。Windows 下对同一个 JSON 文件
    高频 replace 偶尔会被杀软/索引器/另一个读取者短暂占用，抛出 WinError 5。
    所以这里做三件事：
    1. 非关键更新按时间/次数节流落盘；
    2. 每次写唯一 tmp 文件，避免 tmp 名称冲突；
    3. replace 失败时短暂重试，最终只告警，不让样本生成失败。
    """

    job = JOBS.get(job_id)
    if not job:
        return

    now = time.monotonic()
    if not force:
        dirty = int(job.get("_persist_dirty_count", 0) or 0) + 1
        last = float(job.get("_last_persist_monotonic", 0.0) or 0.0)
        job["_persist_dirty_count"] = dirty
        if dirty < JOB_PERSIST_EVERY_UPDATES and (now - last) * 1000 < JOB_PERSIST_MIN_INTERVAL_MS:
            return

    job["_persist_dirty_count"] = 0
    job["_last_persist_monotonic"] = now
    ensure_dirs()
    path = job_file(job_id)
    payload = json.dumps(compact_job_for_persist(job), ensure_ascii=False, indent=2)
    last_exc: Exception | None = None

    for attempt in range(1, JOB_PERSIST_RETRIES + 1):
        tmp = path.with_name(
            f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            # PermissionError/WinError 5 通常是瞬时文件句柄占用，退避后重试。
            time.sleep(min(0.05 * attempt, 0.5))

    print(f"[Sydney Data Factory] warning: failed to persist job {job_id}: {last_exc}")


def load_jobs_from_disk() -> None:
    """服务启动时恢复最近任务面板。

    注意：如果 uvicorn --reload 或进程重启发生在生成中，后台线程无法真正恢复。
    这里会把旧的 queued/running 任务标记为 interrupted，避免前端一直显示“生成中”。
    浏览器普通刷新不会重启服务，因此仍可继续看到实时进度。
    """

    ensure_dirs()
    if env_bool("DISABLE_JOB_RESTORE", False) or JOB_RESTORE_LIMIT <= 0:
        print("[Sydney Data Factory] job snapshot restore disabled")
        return
    loaded = 0
    now = utc_now()
    for path in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:JOB_RESTORE_LIMIT]:
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            job_id = str(job.get("id") or path.stem)
            job = compact_job_for_persist(job)
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["finished_at"] = now
                job["updated_at"] = now
                logs = job.setdefault("logs", [])
                logs.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "level": "warn",
                        "item": None,
                        "message": "服务进程重启，后台生成线程已丢失；该任务无法继续，请重新启动生成。",
                    }
                )
            with JOBS_LOCK:
                JOBS[job_id] = job
                persist_job_locked(job_id, force=True)
            loaded += 1
        except Exception:
            # 损坏的 job 文件不影响主应用启动。
            continue
    if loaded:
        print(f"[Sydney Data Factory] restored {loaded} job snapshots from data/jobs")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc) or not env_bool("AUTO_REBUILD_CORRUPT_DB", True):
            raise
        result = rebuild_workspace_db_from_mirrors(reason=str(exc))
        print(
            "[Sydney Data Factory] workspace.sqlite was corrupt; "
            f"rebuilt from {result.get('mirror_samples', 0)} JSON mirrors"
        )
    load_jobs_from_disk()
    yield


app = FastAPI(title="Sydney Data Factory", version="1.0.0", lifespan=lifespan)

WEB_DIR = ROOT / "web"
WEB_DIR.mkdir(exist_ok=True)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


def app_defaults() -> Dict[str, Any]:
    """页面刷新后恢复的默认生成/导出参数。"""

    return {
        "count": env_int("APP_DEFAULT_COUNT", 5, 1, 10000),
        "concurrency": env_int("APP_DEFAULT_CONCURRENCY", 3, 1, 128),
        "max_turns": env_int("APP_DEFAULT_MAX_TURNS", 12, 2, 20),
        "translate_to_zh": env_bool("APP_DEFAULT_TRANSLATE_TO_ZH", True),
        "same_aux_model": env_bool("APP_DEFAULT_SAME_AUX_MODEL", False),
        "use_judge": env_bool("APP_DEFAULT_USE_JUDGE", False),
        "target_model_key": env_str("APP_DEFAULT_TARGET_MODEL", "qwen36_27b"),
        "train_mode": env_str("APP_DEFAULT_TRAIN_MODE", "qlora"),
        "include_needs_review": env_bool("APP_DEFAULT_INCLUDE_NEEDS_REVIEW", False),
        "only_dialogue_distillation": env_bool("APP_DEFAULT_ONLY_DIALOGUE_DISTILLATION", True),
        "source_prompt_mode": env_str("SOURCE_PROMPT_MODE", env_str("TEACHER_PROMPT_MODE", "legacy_chat")),
        "source_use_default_stops": env_bool("SOURCE_USE_DEFAULT_STOPS", False),
        "generation_preserve_length": env_bool("GENERATION_PRESERVE_LENGTH", True),
        "generation_min_turns": env_int("GENERATION_MIN_TURNS", 6, 2, 20),
        "source_temperature": env_str("SOURCE_TEMPERATURE", "0.78"),
        "source_top_p": env_str("SOURCE_TOP_P", "0.92"),
        "source_frequency_penalty": env_str("SOURCE_FREQUENCY_PENALTY", "0.35"),
        "source_presence_penalty": env_str("SOURCE_PRESENCE_PENALTY", "0.25"),
        "source_repeat_penalty": env_str("SOURCE_REPEAT_PENALTY", "1.12"),
    }


class EndpointPayload(BaseModel):
    """页面传入的 OpenAI-compatible endpoint。

    字段为空时会自动回退到 .env。
    """

    base_url: Optional[str] = ""
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    api_protocol: Optional[str] = None

    def to_config(self, prefix: str) -> ModelConfig:
        env = ModelConfig.from_env(prefix)
        return ModelConfig(
            base_url=(self.base_url or env.base_url or "").strip(),
            api_key=(self.api_key or env.api_key or "").strip(),
            model=(self.model or env.model or "").strip(),
            api_protocol=(self.api_protocol or env.api_protocol or "responses").strip(),
            timeout=env.timeout,
            retries=env.retries,
            retry_backoff=env.retry_backoff,
        )


class GenerateRequest(BaseModel):
    # 允许大批量任务。3000 条这类长任务会以后台 job 方式运行，前端轮询进度。
    count: int = Field(default=5, ge=1, le=10000)
    # 大批量数据允许更高并发；实际建议按 Teacher/Translator/Judge API 限额设置。
    concurrency: int = Field(default=3, ge=1, le=128)
    max_turns: int = Field(default=20, ge=2, le=20)
    # 默认启用：英文源对话 -> 中文翻译。Clever Sydney GGUF 的英文分布明显强于中文直出。
    translate_to_zh: bool = True
    # 勾选后 Human Simulator / Translator / Judge 共用同一个强模型配置。
    # 后端会从 Translator / Human Simulator / Judge 三组配置里选择第一份最可靠的可用配置。
    same_aux_model: bool = False
    # teacher 字段保留兼容旧前端/旧请求；现在语义是 Sydney/source 模型。
    teacher: EndpointPayload = Field(default_factory=EndpointPayload)
    # 与 Sydney/source 对话的人类模拟器模型。留空时使用本地真人短句模拟器，绝不复用 Sydney/source。
    simulator: EndpointPayload = Field(default_factory=EndpointPayload)
    # Translator：强模型负责把英文源对话本地化成中文训练数据。默认必须配置，除非关闭 translate_to_zh。
    translator: EndpointPayload = Field(default_factory=EndpointPayload)
    judge: EndpointPayload = Field(default_factory=EndpointPayload)
    use_judge: bool = False
    seed: Optional[int] = None


class ConfigTestRequest(BaseModel):
    endpoint: EndpointPayload
    prefix: str = "TEACHER"


class UpdateSampleRequest(BaseModel):
    messages: Optional[List[Dict[str, str]]] = None
    status: Optional[str] = None
    score: Optional[float] = None
    review: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    target_model_key: str = "qwen36_27b"
    train_mode: str = "qlora"
    include_needs_review: bool = False
    only_dialogue_distillation: bool = True


def _looks_no_auth_or_local(base_url: str) -> bool:
    """判断一个 endpoint 是否可能无需 API key。

    OpenAI/Anthropic 官方地址通常必须有 key；本地 llama.cpp、局域网网关、
    Modal 自建 endpoint 常常不需要。这里只用于“共用模型配置”的候选优先级，
    不作为硬性校验。
    """

    lower = (base_url or "").lower()
    return any(
        token in lower
        for token in (
            "127.0.0.1",
            "localhost",
            "0.0.0.0",
            "[::1]",
            "192.168.",
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
            "modal.run",
        )
    )


def _endpoint_payload_has_inline_value(payload: EndpointPayload) -> bool:
    """页面是否显式填写了某个字段。用于共用模型配置的优先级。"""

    return bool(
        (payload.base_url or "").strip()
        or (payload.api_key or "").strip()
        or (payload.model or "").strip()
        or (payload.api_protocol or "").strip()
    )


def choose_shared_aux_config(req: GenerateRequest) -> tuple[ModelConfig | None, str]:
    """选择 Human/Translator/Judge 共用的强模型配置。

    选择策略：
    1. 只在 SIMULATOR / TRANSLATOR / JUDGE 之间选择，绝不复用 Sydney/source。
    2. 页面显式填写的配置优先于 .env 默认值。
    3. 带 API key 或看起来是本地/自建无鉴权 endpoint 的配置优先。
    4. 分数相同时偏向 Translator，因为默认流水线最依赖翻译质量。
    """

    candidates = [
        ("translator", "TRANSLATOR", req.translator, 3),
        ("simulator", "SIMULATOR", req.simulator, 2),
        ("judge", "JUDGE", req.judge, 1),
    ]
    best: tuple[int, ModelConfig, str] | None = None
    for name, prefix, payload, tie_breaker in candidates:
        cfg = payload.to_config(prefix)
        if not cfg.ready:
            continue
        score = tie_breaker
        if _endpoint_payload_has_inline_value(payload):
            score += 100
        if cfg.api_key:
            score += 20
        if _looks_no_auth_or_local(cfg.base_url):
            score += 10
        # 没有 key 的官方地址大概率不可用；仍保留为低优先级候选，便于兼容无鉴权代理。
        if not cfg.api_key and not _looks_no_auth_or_local(cfg.base_url):
            score -= 8
        if best is None or score > best[0]:
            best = (score, cfg, name)
    if best is None:
        return None, ""
    return best[1], best[2]


def row_to_sample(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "score": row["score"],
        "spec": json.loads(row["spec_json"]),
        "messages": json.loads(row["messages_json"]),
        "conversations": json.loads(row["sharegpt_json"]),
        "metadata": json.loads(row["metadata_json"]),
        "review": json.loads(row["review_json"]),
        "text_hash": row["text_hash"],
        "duplicate_of": row["duplicate_of"],
        "source": row["source"],
    }


def save_sample(sample: Dict[str, Any]) -> None:
    """保存或更新样本。"""

    messages = sample.get("messages", [])
    sample["conversations"] = to_sharegpt(messages)
    text_hash = fingerprint(conversation_text(messages))
    with db() as conn:
        conn.execute(
            """
            INSERT INTO samples (
                id, created_at, updated_at, status, score, spec_json, messages_json,
                sharegpt_json, metadata_json, review_json, text_hash, duplicate_of, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                score=excluded.score,
                spec_json=excluded.spec_json,
                messages_json=excluded.messages_json,
                sharegpt_json=excluded.sharegpt_json,
                metadata_json=excluded.metadata_json,
                review_json=excluded.review_json,
                text_hash=excluded.text_hash,
                duplicate_of=excluded.duplicate_of,
                source=excluded.source
            """,
            (
                sample["id"],
                sample.get("created_at") or utc_now(),
                sample.get("updated_at") or utc_now(),
                sample.get("status", "needs_review"),
                float(sample.get("score", 0) or 0),
                json.dumps(sample.get("spec", {}), ensure_ascii=False),
                json.dumps(messages, ensure_ascii=False),
                json.dumps(sample.get("conversations", []), ensure_ascii=False),
                json.dumps(sample.get("metadata", {}), ensure_ascii=False),
                json.dumps(sample.get("review", {}), ensure_ascii=False),
                text_hash,
                sample.get("duplicate_of"),
                sample.get("source", "teacher"),
            ),
        )
        conn.commit()


def get_existing_texts() -> List[tuple[str, str]]:
    try:
        with db() as conn:
            rows = conn.execute("SELECT id, messages_json FROM samples").fetchall()
        return [(row["id"], conversation_text(json.loads(row["messages_json"]))) for row in rows]
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        return [(sample["id"], conversation_text(sample.get("messages", []))) for sample in load_all_mirror_samples()]


def get_existing_topic_counts() -> Dict[str, int]:
    """统计历史样本的 topic_id 覆盖次数，用于新批次少见话题优先。

    这样做可以避免“每次生成 100 条都碰巧重复同几个话题”的长期偏差。
    老样本如果还没有 topic_id，会自动跳过，不影响兼容。
    """

    counts: Counter[str] = Counter()
    try:
        with db() as conn:
            rows = conn.execute("SELECT spec_json FROM samples").fetchall()
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        rows = [{"spec_json": json.dumps(sample.get("spec", {}), ensure_ascii=False)} for sample in load_all_mirror_samples()]
    for row in rows:
        try:
            spec = json.loads(row["spec_json"])
        except Exception:
            continue
        topic_id = str(spec.get("topic_id") or "").strip()
        if topic_id:
            counts[topic_id] += 1
    return dict(counts)


def write_status_mirror(sample: Dict[str, Any]) -> None:
    """把每条样本镜像成独立 JSON，便于手工检查和版本管理。"""

    status = sample.get("status", "needs_review")
    folder = REVIEWED_DIR if status == "accepted" else REJECTED_DIR if status == "rejected" else RAW_DIR
    # 状态变更时清理其它目录里的旧镜像，避免 reviewed/rejected 同时出现同一 id。
    for old_folder in (REVIEWED_DIR, REJECTED_DIR, RAW_DIR):
        old_path = old_folder / f"{sample['id']}.json"
        if old_folder != folder and old_path.exists():
            old_path.unlink()
    path = folder / f"{sample['id']}.json"
    path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


def mirror_default_status(path: Path) -> str:
    if path.parent == REVIEWED_DIR:
        return "accepted"
    if path.parent == REJECTED_DIR:
        return "rejected"
    return "needs_review"


def iter_sample_mirror_paths() -> List[Path]:
    """Return per-sample JSON mirrors; batch JSONL exports are intentionally ignored."""

    paths: List[Path] = []
    for folder in (RAW_DIR, REVIEWED_DIR, REJECTED_DIR):
        if folder.exists():
            paths.extend(sorted(p for p in folder.glob("*.json") if p.is_file()))
    return paths


def load_mirror_sample(path: Path) -> Dict[str, Any]:
    sample = json.loads(path.read_text(encoding="utf-8"))
    sample["id"] = str(sample.get("id") or path.stem)
    if sample.get("status") not in {"accepted", "needs_review", "rejected"}:
        sample["status"] = mirror_default_status(path)
    now = utc_now()
    sample.setdefault("created_at", now)
    sample.setdefault("updated_at", now)
    sample.setdefault("spec", {})
    sample.setdefault("messages", [])
    sample.setdefault("conversations", to_sharegpt(sample.get("messages", [])))
    sample.setdefault("metadata", {})
    sample.setdefault("review", {})
    sample.setdefault("score", float(sample.get("review", {}).get("overall", 0) or 0))
    sample.setdefault("text_hash", fingerprint(conversation_text(sample.get("messages", []))))
    sample.setdefault("duplicate_of", None)
    sample.setdefault("source", "teacher")
    return sample


def load_all_mirror_samples() -> List[Dict[str, Any]]:
    """Load JSON mirrors as a fallback when SQLite is unreadable.

    If stale duplicates exist across raw/reviewed/rejected, keep the newest file.
    """

    seen: Dict[str, tuple[float, Dict[str, Any]]] = {}
    for path in iter_sample_mirror_paths():
        try:
            sample = load_mirror_sample(path)
        except Exception:
            continue
        old = seen.get(sample["id"])
        mtime = path.stat().st_mtime
        if old is None or mtime >= old[0]:
            seen[sample["id"]] = (mtime, sample)
    return [item[1] for item in seen.values()]


def stats_from_mirrors(error: str = "") -> Dict[str, Any]:
    samples = load_all_mirror_samples()
    buckets: Dict[str, List[float]] = {"accepted": [], "needs_review": [], "rejected": []}
    for sample in samples:
        status = sample.get("status", "needs_review")
        buckets.setdefault(status, []).append(float(sample.get("score", 0) or 0))
    by_status = {
        status: {"count": len(scores), "avg_score": round(sum(scores) / len(scores), 2) if scores else 0}
        for status, scores in buckets.items()
        if scores
    }
    return {
        "total": len(samples),
        "by_status": by_status,
        "source": "json_mirrors",
        "warning": "SQLite 数据库不可读，当前统计来自 data/raw|reviewed|rejected 镜像 JSON。请重建 workspace.sqlite。",
        "db_error": error,
    }


def rebuild_workspace_db_from_mirrors(reason: str = "") -> Dict[str, Any]:
    """Move current SQLite files aside and rebuild them from JSON mirrors."""

    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved: List[str] = []
    for suffix in ("", "-wal", "-shm"):
        path = DATA_DIR / f"workspace.sqlite{suffix}"
        if path.exists():
            dst = DATA_DIR / f"workspace.sqlite{suffix}.corrupt.{ts}"
            path.replace(dst)
            moved.append(str(dst))

    init_db()
    samples = load_all_mirror_samples()
    rows = []
    for sample in samples:
        messages = sample.get("messages", [])
        conversations = sample.get("conversations") or to_sharegpt(messages)
        text_hash = sample.get("text_hash") or fingerprint(conversation_text(messages))
        rows.append(
            (
                sample["id"],
                sample.get("created_at") or utc_now(),
                sample.get("updated_at") or utc_now(),
                sample.get("status", "needs_review"),
                float(sample.get("score", 0) or 0),
                json.dumps(sample.get("spec", {}), ensure_ascii=False),
                json.dumps(messages, ensure_ascii=False),
                json.dumps(conversations, ensure_ascii=False),
                json.dumps(sample.get("metadata", {}), ensure_ascii=False),
                json.dumps(sample.get("review", {}), ensure_ascii=False),
                text_hash,
                sample.get("duplicate_of"),
                sample.get("source", "teacher"),
            )
        )

    with db() as conn:
        conn.executemany(
            """
            INSERT INTO samples (
                id, created_at, updated_at, status, score, spec_json, messages_json,
                sharegpt_json, metadata_json, review_json, text_hash, duplicate_of, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                score=excluded.score,
                spec_json=excluded.spec_json,
                messages_json=excluded.messages_json,
                sharegpt_json=excluded.sharegpt_json,
                metadata_json=excluded.metadata_json,
                review_json=excluded.review_json,
                text_hash=excluded.text_hash,
                duplicate_of=excluded.duplicate_of,
                source=excluded.source
            """,
            rows,
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) as n FROM samples").fetchone()["n"]
        by_status = {
            row["status"]: row["n"]
            for row in conn.execute("SELECT status, COUNT(*) as n FROM samples GROUP BY status").fetchall()
        }
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    return {
        "ok": integrity == "ok",
        "reason": reason,
        "moved": moved,
        "mirror_samples": len(samples),
        "total": total,
        "by_status": by_status,
        "integrity_check": integrity,
    }


def list_samples_from_mirrors(status: str, q: str, limit: int, offset: int, error: str = "") -> Dict[str, Any]:
    needle = q.strip().lower()
    filtered = []
    for sample in load_all_mirror_samples():
        if status != "all" and sample.get("status") != status:
            continue
        haystack = ""
        if needle:
            haystack = json.dumps(
                {
                    "messages": sample.get("messages", []),
                    "spec": sample.get("spec", {}),
                    "review": sample.get("review", {}),
                },
                ensure_ascii=False,
            ).lower()
            if needle not in haystack:
                continue
        filtered.append(sample)
    filtered.sort(key=lambda s: (float(s.get("score", 0) or 0), str(s.get("updated_at") or "")), reverse=True)
    items = []
    for sample in filtered[max(0, offset) : max(0, offset) + limit]:
        preview = " / ".join(m.get("content", "")[:80] for m in sample.get("messages", []) if m.get("role") != "system")[:220]
        items.append(
            {
                "id": sample["id"],
                "status": sample.get("status", "needs_review"),
                "score": float(sample.get("score", 0) or 0),
                "theme": sample.get("spec", {}).get("theme", ""),
                "scene": sample.get("spec", {}).get("scene", ""),
                "tags": sample.get("review", {}).get("tags", []),
                "preview": preview,
                "updated_at": sample.get("updated_at", ""),
            }
        )
    return {
        "items": items,
        "total": len(filtered),
        "source": "json_mirrors",
        "warning": "SQLite 数据库不可读，当前列表来自镜像 JSON。",
        "db_error": error,
    }


def get_sample_from_mirror(sample_id: str) -> Optional[Dict[str, Any]]:
    safe_id = Path(sample_id).stem
    candidates = []
    for folder in (RAW_DIR, REVIEWED_DIR, REJECTED_DIR):
        path = folder / f"{safe_id}.json"
        if path.exists():
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        return load_mirror_sample(candidates[0])
    except Exception:
        return None


def job_update(job_id: str, **fields: Any) -> None:
    """线程安全更新后台生成任务状态。"""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = utc_now()
        force = any(k in fields for k in {"status", "started_at", "finished_at", "result"})
        persist_job_locked(job_id, force=force)


def job_log(
    job_id: str,
    message: str,
    *,
    level: str = "info",
    item: int | None = None,
    kind: str = "log",
    role: str | None = None,
    turn: int | None = None,
) -> None:
    """追加任务事件，前端会轮询显示。

    kind=chat 的事件会在页面中以实时对话气泡展示；普通 kind 仍作为系统事件。
    """

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        event = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "item": item, "message": message, "kind": kind}
        if role:
            event["role"] = role
        if turn is not None:
            event["turn"] = turn
        logs.append(event)
        # 避免长时间批量生成时内存无限增长。
        if len(logs) > JOB_LOG_KEEP:
            del logs[: len(logs) - JOB_LOG_KEEP]
        job["updated_at"] = utc_now()
        # 高并发下错误日志可能瞬间很多，不强制每条落盘，避免 Windows replace 抢占。
        persist_job_locked(job_id)


def job_inc(job_id: str, **increments: int) -> None:
    """线程安全递增任务计数。"""

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        for key, value in increments.items():
            job[key] = int(job.get(key, 0)) + int(value)
        job["updated_at"] = utc_now()
        persist_job_locked(job_id)


def job_snapshot(job_id: str) -> Dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            return json.loads(json.dumps(compact_job_for_api(job), ensure_ascii=False))

    # 兜底：如果内存中没有，但 data/jobs 里有快照，也允许前端恢复查看。
    path = job_file(job_id)
    if path.exists():
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            with JOBS_LOCK:
                JOBS[job_id] = job
            return json.loads(json.dumps(compact_job_for_api(job), ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"job snapshot corrupted: {exc}") from exc
    raise HTTPException(status_code=404, detail="job not found")


def generate_and_review_one(
    *,
    job_id: str,
    idx: int,
    spec: Dict[str, Any],
    source_cfg: ModelConfig,
    simulator_cfg: ModelConfig | None,
    translator_cfg: ModelConfig | None,
    judge_cfg: ModelConfig | None,
    max_turns: int,
    translate_to_zh: bool,
    defer_translation: bool = False,
    generation_pool_connections: int = 16,
) -> Dict[str, Any]:
    """在线程池中生成并审核单条样本。

    新模式：Sydney/source 模型与 Human simulator 模型先逐轮英文对话，再可选翻译成中文。
    去重和落库在主 job 线程里串行处理，避免并发写入造成重复判断不稳定。
    """

    job_log(
        job_id,
        f"开始英文源逐轮对话：{spec.get('theme_en') or spec.get('theme')} | turns={min(int(spec.get('turns') or max_turns), max_turns)} | translate_to_zh={translate_to_zh}",
        item=idx,
    )
    source_client = get_thread_model_client(
        source_cfg,
        role="source",
        pool_connections=generation_pool_connections,
    )
    if source_client is None:
        raise RuntimeError("source model config missing")
    simulator_client = get_thread_model_client(
        simulator_cfg,
        role="simulator",
        pool_connections=generation_pool_connections,
    )
    translator_client = (
        get_thread_model_client(
            translator_cfg,
            role="translator_inline",
            pool_connections=generation_pool_connections,
        )
        if translator_cfg and translator_cfg.ready and translate_to_zh and not defer_translation
        else None
    )
    sample = generate_dialogue_sample(
        source_client,
        simulator_client,
        spec,
        max_turns=max_turns,
        translator_client=translator_client,
        translate_to_zh=bool(translate_to_zh and not defer_translation),
        on_event=lambda msg, **ev: job_log(job_id, msg, item=idx, **ev),
    )
    if translate_to_zh and defer_translation:
        job_log(job_id, "英文源逐轮对话完成，翻译进入独立并发队列", item=idx)
        return {"idx": idx, "spec": spec, "sample": sample, "review": None, "text": conversation_text(sample["messages"]), "needs_translation": True}

    text = conversation_text(sample["messages"])
    turns = sum(1 for m in sample["messages"] if m.get("role") in {"user", "assistant"})
    job_log(job_id, f"逐轮对话完成：{turns} 条 user/assistant 消息，开始自动审核", item=idx)

    judge_client = get_thread_model_client(
        judge_cfg,
        role="judge_inline",
        pool_connections=generation_pool_connections,
    )
    review = review_sample(sample, spec, judge_client=judge_client)
    job_log(
        job_id,
        f"审核完成：status={review.get('status')} overall={float(review.get('overall', 0) or 0):.2f}",
        item=idx,
    )
    return {"idx": idx, "spec": spec, "sample": sample, "review": review, "text": text}


def run_generation_job(
    job_id: str,
    req: GenerateRequest,
    source_cfg: ModelConfig,
    simulator_cfg: ModelConfig | None,
    translator_cfg: ModelConfig | None,
    judge_cfg: ModelConfig | None,
) -> None:
    """后台生成任务：并发生成英文源双模型对话，可选翻译中文，串行去重入库。"""

    job_update(job_id, status="running", started_at=utc_now())
    job_log(
        job_id,
        (
            f"任务启动：count={req.count}, concurrency={req.concurrency}, max_turns={req.max_turns}, "
            f"source={source_cfg.model}({source_cfg.api_protocol}), "
            f"simulator={simulator_cfg.model + '(' + simulator_cfg.api_protocol + ')' if simulator_cfg else 'local_human_simulator'}, "
            f"translator={translator_cfg.model + '(' + translator_cfg.api_protocol + ')' if translator_cfg else 'disabled'}"
        ),
    )

    specs = make_generation_specs(
        req.count,
        seed=req.seed,
        # 开源 Sydney GGUF 的英文分布明显强于中文；即使关闭翻译，也保留英文源对话。
        source_language="en",
        target_language="zh-CN",
        previous_topic_counts=get_existing_topic_counts(),
    )
    existing = get_existing_texts()
    dedup_index = DedupIndex(existing)
    generated: List[Dict[str, Any]] = []

    def finalize_result(result: Dict[str, Any]) -> None:
        """Serial DB/dedup section kept small; model work has already finished."""

        nonlocal generated
        idx = int(result["idx"])
        spec = result["spec"]
        sample = result["sample"]
        review = result.get("review")
        text = result["text"]
        if review is None:
            raise RuntimeError("internal error: review missing before finalize")

        is_dup, dup_id, dup_score = dedup_index.find(text)
        if is_dup:
            review = apply_duplicate_penalty(review, dup_id, dup_score)
            sample["duplicate_of"] = dup_id
            job_log(
                job_id,
                f"近重复，自动丢弃：duplicate_of={dup_id}, similarity={dup_score:.3f}",
                level="warn",
                item=idx,
            )

        sample["review"] = review
        sample["status"] = review.get("status", "needs_review")
        sample["score"] = float(review.get("overall", 0) or 0)
        sample["updated_at"] = utc_now()
        save_sample(sample)
        write_status_mirror(sample)

        generated.append(
            {
                "id": sample["id"],
                "status": sample["status"],
                "score": sample["score"],
                "theme": spec.get("theme"),
            }
        )
        existing.append((sample["id"], text))
        dedup_index.add(sample["id"], text)

        status = sample["status"]
        job_inc(
            job_id,
            completed=1,
            generated=1,
            accepted=1 if status == "accepted" else 0,
            needs_review=1 if status == "needs_review" else 0,
            rejected=1 if status == "rejected" else 0,
        )
        job_log(
            job_id,
            f"已保存：{sample['id']} | {status} | score={sample['score']:.2f}",
            level="ok" if status == "accepted" else "warn" if status == "needs_review" else "error",
            item=idx,
        )

    def translate_and_review_result(result: Dict[str, Any]) -> Dict[str, Any]:
        idx = int(result["idx"])
        spec = result["spec"]
        sample = result["sample"]
        if translator_cfg is None or not translator_cfg.ready:
            raise RuntimeError("translator config missing")
        translator_client = get_thread_model_client(
            translator_cfg,
            role="translator_async",
            pool_connections=max(translation_workers, req.concurrency, 16),
        )
        if translator_client is None:
            raise RuntimeError("translator config missing")
        translated = translate_dialogue_sample(
            sample,
            translator_client,
            spec,
            on_event=lambda msg, **ev: job_log(job_id, msg, item=idx, **ev),
        )
        text = conversation_text(translated["messages"])
        turns = sum(1 for m in translated["messages"] if m.get("role") in {"user", "assistant"})
        job_log(job_id, f"翻译后样本完成：{turns} 条 user/assistant 消息，开始自动审核", item=idx)
        judge_client = get_thread_model_client(
            judge_cfg,
            role="judge_async",
            pool_connections=max(translation_workers, req.concurrency, 16),
        )
        review = review_sample(translated, spec, judge_client=judge_client)
        job_log(
            job_id,
            f"审核完成：status={review.get('status')} overall={float(review.get('overall', 0) or 0):.2f}",
            item=idx,
        )
        return {"idx": idx, "spec": spec, "sample": translated, "review": review, "text": text}

    def record_failure(idx: int, spec: Dict[str, Any], exc: Exception) -> None:
        err = {"index": str(idx), "error": str(exc), "theme": spec.get("theme", "")}
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.setdefault("errors", []).append(err)
        # completed means "finalized sample count".  If generation succeeded
        # but the async translation/review stage failed, it still consumed one
        # requested item, so this is correct for both stages.
        job_inc(job_id, completed=1, failed=1)
        job_log(job_id, f"失败：{exc}", level="error", item=idx)

    try:
        # Translation is moved to a separate pool so a long 5k-token translation
        # request no longer occupies Sydney/source generation slots. This keeps
        # the design intact but hides translation latency behind later samples.
        defer_translation = bool(
            req.translate_to_zh
            and translator_cfg is not None
            and translator_cfg.ready
            and env_bool("ASYNC_TRANSLATION_STAGE", True)
        )
        # 旧版 async translation 默认只给 4 个 worker；在 generation concurrency=64
        # 时，英文对话会快速排队到翻译阶段，最终 completed 吞吐反而低于修复前。
        # 默认让翻译并发跟随请求并发，必要时可用 TRANSLATION_CONCURRENCY 手动限流。
        translation_workers = env_int("TRANSLATION_CONCURRENCY", req.concurrency, 1, 128)
        if not defer_translation:
            translation_workers = 1
        http_pool_connections = env_int(
            "HTTP_POOL_CONNECTIONS",
            max(req.concurrency, translation_workers, 16),
            1,
            1024,
        )
        job_update(
            job_id,
            async_translation=defer_translation,
            translation_concurrency=translation_workers if defer_translation else 0,
            http_pool_connections=http_pool_connections,
        )
        job_log(
            job_id,
            (
                "并发配置："
                f"generation_workers={req.concurrency}, "
                f"translation_workers={translation_workers if defer_translation else 0}, "
                f"http_pool_connections={http_pool_connections}"
            ),
        )
        with ThreadPoolExecutor(max_workers=req.concurrency) as gen_pool, ThreadPoolExecutor(max_workers=translation_workers) as trans_pool:
            gen_futures = {
                gen_pool.submit(
                    generate_and_review_one,
                    job_id=job_id,
                    idx=idx,
                    spec=spec,
                    source_cfg=source_cfg,
                    simulator_cfg=simulator_cfg,
                    translator_cfg=translator_cfg,
                    judge_cfg=judge_cfg,
                    max_turns=req.max_turns,
                    translate_to_zh=req.translate_to_zh,
                    defer_translation=defer_translation,
                    generation_pool_connections=http_pool_connections,
                ): (idx, spec)
                for idx, spec in enumerate(specs, start=1)
            }
            trans_futures: Dict[Any, tuple[int, Dict[str, Any]]] = {}
            job_update(job_id, queued=len(gen_futures))

            while gen_futures or trans_futures:
                done_gen, _ = wait(gen_futures.keys(), timeout=0.2, return_when=FIRST_COMPLETED) if gen_futures else (set(), set())
                for future in done_gen:
                    idx, spec = gen_futures.pop(future)
                    try:
                        result = future.result()
                        if result.get("needs_translation"):
                            tf = trans_pool.submit(translate_and_review_result, result)
                            trans_futures[tf] = (idx, spec)
                        else:
                            finalize_result(result)
                    except Exception as exc:  # noqa: BLE001
                        record_failure(idx, spec, exc)

                trans_timeout = 0.0 if gen_futures else 0.2
                done_trans, _ = wait(trans_futures.keys(), timeout=trans_timeout, return_when=FIRST_COMPLETED) if trans_futures else (set(), set())
                for future in done_trans:
                    idx, spec = trans_futures.pop(future)
                    try:
                        finalize_result(future.result())
                    except Exception as exc:  # noqa: BLE001
                        record_failure(idx, spec, exc)

        batch_path = None
        if generated:
            batch_path = RAW_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}.jsonl"
            with db() as conn:
                placeholders = ",".join("?" for _ in generated)
                rows = conn.execute(
                    f"SELECT * FROM samples WHERE id IN ({placeholders})",
                    [x["id"] for x in generated],
                ).fetchall()
            write_jsonl(str(batch_path), [row_to_sample(row) for row in rows])

        final_status = "completed"
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["result"] = {
                    "generated": generated,
                    "errors": job.get("errors", []),
                    "batch_path": str(batch_path.relative_to(ROOT)) if batch_path else None,
                }
                persist_job_locked(job_id, force=True)
        job_update(job_id, status=final_status, finished_at=utc_now())
        job_log(job_id, f"任务完成：成功 {len(generated)} 条，失败 {job_snapshot(job_id).get('failed', 0)} 条", level="ok")
    except Exception as exc:  # noqa: BLE001
        job_update(job_id, status="failed", finished_at=utc_now())
        job_log(job_id, f"任务级失败：{exc}", level="error")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """返回本地单页工作台。

    开发阶段优先读取 web/index.html，这样前端样式可以独立迭代，
    不需要把几万行 HTML/CSS/JS 都塞回 Python 字符串里。
    """

    ui_path = ROOT / "web" / "index.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return INDEX_HTML


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    teacher = ModelConfig.from_env("TEACHER")
    simulator = ModelConfig.from_env("SIMULATOR")
    translator = ModelConfig.from_env("TRANSLATOR")
    judge = ModelConfig.from_env("JUDGE")
    return {
        "teacher": {
            "base_url": teacher.base_url,
            "model": teacher.model,
            "api_protocol": teacher.api_protocol,
            "has_key": bool(teacher.api_key),
        },
        "simulator": {
            "base_url": simulator.base_url,
            "model": simulator.model,
            "api_protocol": simulator.api_protocol,
            "has_key": bool(simulator.api_key),
        },
        "translator": {
            "base_url": translator.base_url,
            "model": translator.model,
            "api_protocol": translator.api_protocol,
            "has_key": bool(translator.api_key),
        },
        "judge": {
            "base_url": judge.base_url,
            "model": judge.model,
            "api_protocol": judge.api_protocol,
            "has_key": bool(judge.api_key),
        },
        "defaults": app_defaults(),
        "target_models": TARGET_MODELS,
    }


@app.post("/api/config/test")
def test_config(req: ConfigTestRequest) -> Dict[str, Any]:
    cfg = req.endpoint.to_config(req.prefix)
    try:
        data = OpenAICompatibleClient(cfg).ping()
        return {"ok": True, "response": data}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/api/stats")
def stats() -> Dict[str, Any]:
    try:
        with db() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n, AVG(score) as avg_score FROM samples GROUP BY status"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) as n FROM samples").fetchone()["n"]
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        return stats_from_mirrors(str(exc))
    by_status = {
        row["status"]: {"count": row["n"], "avg_score": round(row["avg_score"] or 0, 2)}
        for row in rows
    }
    return {"total": total, "by_status": by_status, "source": "sqlite"}



@app.post("/api/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    """启动后台生成任务，立即返回 job_id，前端通过 /api/jobs/{job_id} 轮询进度。"""

    source_cfg = req.teacher.to_config("TEACHER")
    if not source_cfg.ready:
        raise HTTPException(
            status_code=400,
            detail="Sydney/source 模型未配置：请填写 TEACHER_BASE_URL / TEACHER_MODEL 或在页面输入。",
        )

    shared_aux_cfg = None
    shared_aux_source = ""
    if req.same_aux_model:
        shared_aux_cfg, shared_aux_source = choose_shared_aux_config(req)
        if not shared_aux_cfg:
            raise HTTPException(
                status_code=400,
                detail=(
                    "已勾选 Human/Translator/Judge 使用相同模型，但未找到可用强模型配置。"
                    "请至少填写 Human、Translator 或 Judge 中任意一组 Base URL + Model。"
                ),
            )

    simulator_cfg = shared_aux_cfg if shared_aux_cfg else req.simulator.to_config("SIMULATOR")
    if not simulator_cfg.ready:
        # 关键修复：不要复用 Sydney/source 当 user 模拟器。
        # 复用 source 会导致模型自己和自己聊天，产生复读、任务腔、客服腔垃圾数据。
        simulator_cfg = None

    translator_cfg = None
    if req.translate_to_zh:
        translator_cfg = shared_aux_cfg if shared_aux_cfg else req.translator.to_config("TRANSLATOR")
        if not translator_cfg.ready:
            raise HTTPException(
                status_code=400,
                detail=(
                    "已启用“英文源输出后翻译为中文”，但 Translator 未配置。"
                    "请填写 TRANSLATOR_BASE_URL / TRANSLATOR_MODEL，或勾选共用强模型并填写 Human/Translator/Judge 任意一组，或在页面关闭翻译。"
                ),
            )

    judge_cfg = None
    if req.use_judge:
        judge_cfg = shared_aux_cfg if shared_aux_cfg else req.judge.to_config("JUDGE")
        if not judge_cfg.ready:
            # 不再复用 Sydney/source 做 Judge；它上下文短且不是评审模型，容易误判/超上下文。
            judge_cfg = None

    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "total": req.count,
        "queued": req.count,
        "completed": 0,
        "generated": 0,
        "failed": 0,
        "accepted": 0,
        "needs_review": 0,
        "rejected": 0,
        "concurrency": req.concurrency,
        "async_translation": None,
        "translation_concurrency": None,
        "http_pool_connections": None,
        "max_turns": req.max_turns,
        "translate_to_zh": req.translate_to_zh,
        "same_aux_model": req.same_aux_model,
        "shared_aux_source": shared_aux_source,
        "logs": [],
        "errors": [],
        "result": None,
        "mode": "two_model_dialogue_distillation",
        "teacher": {
            # 兼容旧前端/旧 job 文件命名；现在 teacher 表示 Sydney/source 模型。
            "base_url": source_cfg.base_url,
            "model": source_cfg.model,
            "api_protocol": source_cfg.api_protocol,
        },
        "source": {
            "base_url": source_cfg.base_url,
            "model": source_cfg.model,
            "api_protocol": source_cfg.api_protocol,
        },
        "simulator": {
            "base_url": simulator_cfg.base_url if simulator_cfg else "",
            "model": simulator_cfg.model if simulator_cfg else "local_human_simulator",
            "api_protocol": simulator_cfg.api_protocol if simulator_cfg else "local",
            "reused_source": False,
            "local_fallback": simulator_cfg is None,
            "shared_aux_model": bool(req.same_aux_model and simulator_cfg),
        },
        "translator": {
            "base_url": translator_cfg.base_url if translator_cfg else "",
            "model": translator_cfg.model if translator_cfg else "",
            "api_protocol": translator_cfg.api_protocol if translator_cfg else "",
            "enabled": bool(translator_cfg),
            "shared_aux_model": bool(req.same_aux_model and translator_cfg),
        },
        "judge": {
            "base_url": judge_cfg.base_url if judge_cfg else "",
            "model": judge_cfg.model if judge_cfg else "",
            "api_protocol": judge_cfg.api_protocol if judge_cfg else "",
            "enabled": bool(judge_cfg),
            "shared_aux_model": bool(req.same_aux_model and judge_cfg),
        },
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        persist_job_locked(job_id, force=True)
    job_log(job_id, "任务已入队，后台线程即将启动")
    if req.same_aux_model:
        job_log(job_id, f"已启用共用强模型：来源={shared_aux_source or 'unknown'}，model={shared_aux_cfg.model if shared_aux_cfg else ''}")

    thread = threading.Thread(
        target=run_generation_job,
        args=(job_id, req, source_cfg, simulator_cfg, translator_cfg, judge_cfg),
        daemon=True,
        name=f"sydney-generate-{job_id[:8]}",
    )
    thread.start()
    return {"ok": True, "job_id": job_id, "job": job_snapshot(job_id)}


@app.get("/api/jobs")
def list_jobs(limit: int = 20, active: bool = False) -> Dict[str, Any]:
    """列出最近后台任务，用于页面刷新后恢复进度面板。

    - active=true：只返回 queued/running 的活动任务
    - active=false：返回最近任务，包括 completed/failed/interrupted
    """

    limit = max(1, min(100, int(limit or 20)))
    active_status = {"queued", "running"}
    with JOBS_LOCK:
        jobs = [json.loads(json.dumps(compact_job_for_api(job), ensure_ascii=False)) for job in JOBS.values()]
    if active:
        jobs = [job for job in jobs if job.get("status") in active_status]
    jobs.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
    return {"items": [enrich_job_progress(job) for job in jobs[:limit]], "total": len(jobs)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    return enrich_job_progress(job_snapshot(job_id))




@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, Any]:
    """删除单个任务快照。只影响进度面板，不删除已入库样本。"""

    safe_id = Path(job_id).name
    with JOBS_LOCK:
        JOBS.pop(safe_id, None)
    path = job_file(safe_id)
    if path.exists():
        path.unlink()
    return {"ok": True, "deleted": safe_id}


@app.post("/api/jobs/clear")
def clear_jobs() -> Dict[str, Any]:
    """清空所有任务快照。只影响进度面板，不删除数据集样本。"""

    with JOBS_LOCK:
        JOBS.clear()
    removed = 0
    for path in JOBS_DIR.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    for path in JOBS_DIR.glob("*.json.tmp"):
        try:
            path.unlink()
        except Exception:
            pass
    return {"ok": True, "removed": removed}


@app.get("/api/samples")
def list_samples(
    status: str = "accepted",
    q: str = "",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    allowed = {"accepted", "needs_review", "rejected", "all"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail="status must be accepted/needs_review/rejected/all")
    limit = max(1, min(200, limit))
    params: List[Any] = []
    where = []
    if status != "all":
        where.append("status = ?")
        params.append(status)
    if q:
        where.append("(messages_json LIKE ? OR spec_json LIKE ? OR review_json LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    try:
        with db() as conn:
            rows = conn.execute(
                f"SELECT * FROM samples {where_sql} ORDER BY score DESC, updated_at DESC LIMIT ? OFFSET ?",
                params + [limit, max(0, offset)],
            ).fetchall()
            total = conn.execute(f"SELECT COUNT(*) as n FROM samples {where_sql}", params).fetchone()["n"]
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        return list_samples_from_mirrors(status, q, limit, max(0, offset), str(exc))

    items = []
    for row in rows:
        sample = row_to_sample(row)
        preview = " / ".join(m["content"][:80] for m in sample["messages"] if m["role"] != "system")[:220]
        items.append(
            {
                "id": sample["id"],
                "status": sample["status"],
                "score": sample["score"],
                "theme": sample["spec"].get("theme", ""),
                "scene": sample["spec"].get("scene", ""),
                "tags": sample.get("review", {}).get("tags", []),
                "preview": preview,
                "updated_at": sample["updated_at"],
            }
        )
    return {"items": items, "total": total, "source": "sqlite"}


@app.get("/api/samples/{sample_id}")
def get_sample(sample_id: str) -> Dict[str, Any]:
    try:
        with db() as conn:
            row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        sample = get_sample_from_mirror(sample_id)
        if sample:
            return sample
        raise HTTPException(status_code=503, detail=f"SQLite 数据库不可读，且镜像中找不到样本：{exc}")
    if not row:
        raise HTTPException(status_code=404, detail="sample not found")
    return row_to_sample(row)


@app.put("/api/samples/{sample_id}")
def update_sample(sample_id: str, req: UpdateSampleRequest) -> Dict[str, Any]:
    sample = get_sample(sample_id)
    if req.messages is not None:
        cleaned = []
        for msg in req.messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role not in {"system", "user", "assistant"}:
                raise HTTPException(status_code=400, detail="role must be system/user/assistant")
            if content:
                cleaned.append({"role": role, "content": content})
        sample["messages"] = cleaned
    if req.status is not None:
        if req.status not in {"accepted", "needs_review", "rejected"}:
            raise HTTPException(status_code=400, detail="bad status")
        sample["status"] = req.status
    if req.score is not None:
        sample["score"] = max(0.0, min(10.0, float(req.score)))
        sample.setdefault("review", {})["overall"] = sample["score"]
    if req.review is not None:
        sample["review"] = req.review
    if req.metadata is not None:
        sample["metadata"] = req.metadata
    sample["updated_at"] = utc_now()
    save_sample(sample)
    write_status_mirror(sample)
    return {"ok": True, "sample": sample}


@app.post("/api/export")
def export(req: ExportRequest) -> Dict[str, Any]:
    if req.target_model_key not in TARGET_MODELS:
        raise HTTPException(status_code=400, detail="未知目标模型")
    if req.train_mode not in {"qlora", "lora", "fft"}:
        raise HTTPException(status_code=400, detail="train_mode must be qlora/lora/fft")

    statuses = ["accepted"] + (["needs_review"] if req.include_needs_review else [])
    placeholders = ",".join("?" for _ in statuses)
    where_extra = " AND source IN (?, ?)" if req.only_dialogue_distillation else ""
    params: List[Any] = list(statuses)
    if req.only_dialogue_distillation:
        # 未翻译样本和英文->中文翻译样本都属于双模型逐轮蒸馏。
        params.extend(["two_model_dialogue", "english_dialogue_translated_zh"])
    try:
        with db() as conn:
            rows = conn.execute(
                f"SELECT * FROM samples WHERE status IN ({placeholders}){where_extra} ORDER BY score DESC, updated_at DESC",
                params,
            ).fetchall()
        samples = [row_to_sample(row) for row in rows]
    except sqlite3.DatabaseError as exc:
        if not is_sqlite_corruption_error(exc):
            raise
        samples = []
        allowed_sources = {"two_model_dialogue", "english_dialogue_translated_zh"}
        for sample in load_all_mirror_samples():
            if sample.get("status") not in statuses:
                continue
            if req.only_dialogue_distillation and sample.get("source") not in allowed_sources:
                continue
            samples.append(sample)
        samples.sort(key=lambda s: (float(s.get("score", 0) or 0), str(s.get("updated_at") or "")), reverse=True)
    if not samples:
        raise HTTPException(
            status_code=400,
            detail="没有可导出的样本。请先生成并通过审核；如需导出旧版一次性生成样本，请取消“只导出双模型蒸馏样本”。",
        )

    chatml_rows = []
    sharegpt_rows = []
    for s in samples:
        metadata = {"score": s["score"], "status": s["status"], **s.get("metadata", {})}
        chatml_rows.append({"id": s["id"], "messages": s["messages"], "metadata": metadata})
        sharegpt_rows.append({"id": s["id"], "conversations": s["conversations"], "metadata": metadata})

    chatml_path = EXPORT_DIR / "train_chatml.jsonl"
    sharegpt_path = EXPORT_DIR / "train_sharegpt.jsonl"
    notebook_path = EXPORT_DIR / "train_sydney.ipynb"
    write_jsonl(str(chatml_path), chatml_rows)
    write_jsonl(str(sharegpt_path), sharegpt_rows)
    export_unsloth_notebook(
        notebook_path,
        target_model_key=req.target_model_key,
        train_mode=req.train_mode,
        dataset_filename=chatml_path.name,
    )
    return {
        "ok": True,
        "count": len(samples),
        "files": {
            "chatml": f"/download/{chatml_path.name}",
            "sharegpt": f"/download/{sharegpt_path.name}",
            "notebook": f"/download/{notebook_path.name}",
        },
        "paths": {
            "chatml": str(chatml_path),
            "sharegpt": str(sharegpt_path),
            "notebook": str(notebook_path),
        },
    }


@app.get("/download/{filename}")
def download(filename: str):
    safe = Path(filename).name
    path = EXPORT_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, filename=safe)


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sydney Data Factory</title>
  <style>
    :root{
      --bg:#f5f5f7;--bg2:#ececf0;--glass:rgba(255,255,255,.72);--glass2:rgba(255,255,255,.55);
      --ink:#1d1d1f;--muted:#6e6e73;--line:rgba(60,60,67,.16);--blue:#007aff;--green:#34c759;--orange:#ff9f0a;--red:#ff3b30;
      --shadow:0 18px 60px rgba(0,0,0,.10);--soft:0 8px 24px rgba(0,0,0,.08);--r:18px;
    }
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;min-height:100vh;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:radial-gradient(circle at 12% 0%,#dbeafe 0,transparent 32%),radial-gradient(circle at 92% 8%,#ffe4e6 0,transparent 26%),linear-gradient(180deg,var(--bg),var(--bg2));}
    body:before{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(180deg,rgba(255,255,255,.8),rgba(255,255,255,.18));}
    .shell{width:min(1680px,96vw);margin:18px auto 32px;position:relative}.top{display:grid;grid-template-columns:1.5fr .95fr;gap:14px;align-items:stretch}.hero,.stat,.panel{border:1px solid var(--line);background:var(--glass);backdrop-filter:blur(26px) saturate(1.35);box-shadow:var(--shadow)}.hero{border-radius:24px;padding:22px 24px;overflow:hidden;position:relative}.traffic{display:flex;gap:8px;margin-bottom:18px}.dot{width:12px;height:12px;border-radius:50%}.dot.red{background:#ff5f57}.dot.yellow{background:#febc2e}.dot.green{background:#28c840}.eyebrow{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.18em;text-transform:uppercase}.title{font-size:36px;line-height:1;margin:8px 0 8px;letter-spacing:-1.3px;font-weight:780}.subtitle{color:var(--muted);font-size:14px;line-height:1.65;max-width:780px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.stat{border-radius:20px;padding:16px}.stat b{display:block;font-size:26px;letter-spacing:-.8px}.stat span,.hint,.label{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:340px minmax(520px,1fr) 390px;gap:14px;margin-top:14px;align-items:start}.panel{border-radius:22px;overflow:hidden}.panelHead{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.42)}.panel h2{font-size:14px;margin:0;font-weight:760}.foldBtn{border:1px solid var(--line);background:rgba(255,255,255,.66);border-radius:999px;width:26px;height:26px;cursor:pointer;color:var(--muted)}.panelBody{transition:max-height .25s ease,opacity .2s ease}.panel.collapsed .panelBody{max-height:0!important;opacity:0;overflow:hidden}.panel.collapsed .foldBtn{transform:rotate(-90deg)}
    details.section{padding:0;border-bottom:1px solid var(--line)}details.section:last-child{border-bottom:0}details.section>summary{list-style:none;cursor:pointer;padding:13px 16px;font-size:13px;font-weight:720;display:flex;align-items:center;justify-content:space-between}details.section>summary::-webkit-details-marker{display:none}details.section>summary:after{content:"⌄";color:var(--muted);transition:.15s}details.section:not([open])>summary:after{transform:rotate(-90deg)}.sectionInner{padding:0 16px 16px}.label{margin:9px 0 6px;font-weight:650}.input,.select,.textarea{width:100%;border:1px solid var(--line);background:rgba(255,255,255,.78);color:var(--ink);border-radius:12px;padding:9px 10px;outline:none;font:12px ui-monospace,"SF Mono","Cascadia Code",monospace;box-shadow:inset 0 1px 0 rgba(255,255,255,.6)}.input:focus,.select:focus,.textarea:focus{border-color:rgba(0,122,255,.55);box-shadow:0 0 0 4px rgba(0,122,255,.12)}.row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.btn{border:1px solid rgba(0,122,255,.22);background:linear-gradient(180deg,#2997ff,#007aff);color:white;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:760;cursor:pointer;box-shadow:0 8px 16px rgba(0,122,255,.18);transition:.16s;min-height:30px}.btn:hover{transform:translateY(-1px);filter:brightness(1.03)}.btn.secondary{background:rgba(255,255,255,.72);color:var(--ink);border-color:var(--line);box-shadow:none}.btn.danger{background:linear-gradient(180deg,#ff6961,#ff3b30);border-color:rgba(255,59,48,.25)}.btn.good{background:linear-gradient(180deg,#4ade80,#34c759);border-color:rgba(52,199,89,.25)}.btn:disabled{opacity:.55;cursor:not-allowed;transform:none}.hint{line-height:1.5}.tabs{display:flex;gap:7px;padding:10px;border-bottom:1px solid var(--line);background:rgba(255,255,255,.38);align-items:center}.tab{border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.68);color:var(--muted);padding:7px 10px;font-size:12px;cursor:pointer}.tab.active{color:white;background:var(--blue);border-color:var(--blue)}.list{max-height:760px;overflow:auto;padding:10px}.card{border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.68);padding:12px;margin-bottom:9px;cursor:pointer;box-shadow:var(--soft)}.card:hover{border-color:rgba(0,122,255,.35)}.card.active{outline:3px solid rgba(0,122,255,.18);border-color:rgba(0,122,255,.55)}.cardTop{display:flex;justify-content:space-between;gap:12px}.score{font-family:ui-monospace,monospace;color:var(--blue);font-weight:900}.badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:2px 7px;font-size:11px;color:var(--muted);margin:6px 5px 0 0;background:rgba(255,255,255,.55)}.preview{font-size:13px;color:#3a3a3c;line-height:1.5;margin-top:8px}.workbench{display:grid;grid-template-rows:auto auto minmax(360px,1fr);gap:12px;padding:12px}.jobPanel{border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.58);padding:12px;box-shadow:var(--soft)}.progressLine{height:8px;border-radius:999px;overflow:hidden;background:rgba(60,60,67,.12)}#jobBar{height:100%;width:0%;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .25s}.liveChat{border:1px solid var(--line);border-radius:22px;background:linear-gradient(180deg,rgba(255,255,255,.7),rgba(255,255,255,.48));min-height:430px;max-height:620px;overflow:auto;padding:16px;box-shadow:var(--soft)}.liveEmpty{text-align:center;color:var(--muted);padding:80px 20px}.chat{padding:16px;max-height:520px;overflow:auto}.bubble{max-width:82%;padding:10px 12px;border-radius:18px;margin:9px 0;line-height:1.55;white-space:pre-wrap;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.04)}.bubble.system{max-width:100%;font-size:12px;color:var(--muted);border:1px dashed var(--line);background:rgba(255,255,255,.55);box-shadow:none}.bubble.user{margin-left:auto;background:#007aff;color:white;border-bottom-right-radius:7px}.bubble.assistant{background:rgba(229,229,234,.92);color:#1d1d1f;border-bottom-left-radius:7px}.bubble.event{max-width:100%;margin:7px auto;text-align:center;background:transparent;box-shadow:none;color:var(--muted);font-size:11px;padding:3px}.bubble.warn{max-width:100%;background:rgba(255,159,10,.12);color:#8a4b00}.jsonbox{height:220px;resize:vertical}.meta{font-family:ui-monospace,"SF Mono",monospace;font-size:11px;color:var(--muted);word-break:break-all}.reason{border-left:3px solid var(--blue);padding:8px 10px;background:rgba(0,122,255,.08);border-radius:9px;margin:8px 0;color:#303035}.toast{position:fixed;right:18px;bottom:18px;width:min(520px,90vw);z-index:20}.toast div{background:rgba(255,255,255,.88);border:1px solid var(--line);backdrop-filter:blur(20px);box-shadow:var(--shadow);border-radius:14px;padding:10px 12px;margin-top:8px}.links a{color:var(--blue);display:block;margin:7px 0}.spinner{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.5);border-top-color:white;border-radius:999px;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:1280px){.top,.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.liveChat{max-height:520px}}
  </style>
</head>
<body>
  <div class="shell">
    <div class="top">
      <div class="hero">
        <div class="traffic"><span class="dot red"></span><span class="dot yellow"></span><span class="dot green"></span></div>
        <div class="eyebrow">LOCAL DIALOGUE DATA FACTORY</div>
        <div class="title">Sydney 数据工厂</div>
        <div class="subtitle">逐轮对话蒸馏：Human Simulator 和 Sydney/source 都会收到明确的私聊环境与 transcript 上下文。中间区域实时展示生成中的对话，而不是只看日志。</div>
      </div>
      <div class="stats">
        <div class="stat"><b id="stTotal">0</b><span>总样本</span></div>
        <div class="stat"><b id="stAccepted">0</b><span>自动通过</span></div>
        <div class="stat"><b id="stReview">0</b><span>待复核</span></div>
        <div class="stat"><b id="stRejected">0</b><span>已丢弃</span></div>
      </div>
    </div>
    <div class="grid">
      <aside class="panel" id="controlPanel">
        <div class="panelHead"><h2>生成控制台</h2><button class="foldBtn" onclick="togglePanel('controlPanel')">⌄</button></div>
        <div class="panelBody">
          <details class="section" open><summary>Sydney/source</summary><div class="sectionInner">
            <div class="label">Base URL</div><input id="teacherBase" class="input" placeholder="OpenAI-compatible Base URL" />
            <div class="label">API Key</div><input id="teacherKey" class="input" type="password" placeholder="可留空" />
            <div class="label">Model</div><input id="teacherModel" class="input" placeholder="你的开源 Sydney 模型名" />
            <div class="label">Protocol</div><select id="teacherProtocol" class="select"><option value="responses">Responses</option><option value="chat_completions">Chat Completions</option><option value="claude_messages">Claude Messages</option></select>
            <button class="btn secondary" style="margin-top:10px" onclick="testTeacher()">测试</button>
            <div class="hint" style="margin-top:8px">训练样本 system 固定为：<code>You are a helpful assistant.</code></div>
          </div></details>
          <details class="section"><summary>共用强模型</summary><div class="sectionInner">
            <label class="hint"><input id="sameAuxModel" type="checkbox" onchange="syncAuxUi()" /> Human / Translator / Judge 使用相同配置</label>
            <div class="hint" style="margin-top:8px">只会在 Human、Translator、Judge 之间共用，不会复用 Sydney/source。</div>
          </div></details>
          <details class="section"><summary>Human Simulator</summary><div class="sectionInner">
            <div class="label">Base URL</div><input id="simulatorBase" class="input" placeholder="留空则使用本地短句模拟器" />
            <div class="label">API Key</div><input id="simulatorKey" class="input" type="password" />
            <div class="label">Model</div><input id="simulatorModel" class="input" placeholder="gpt / claude / glm / qwen ..." />
            <div class="label">Protocol</div><select id="simulatorProtocol" class="select"><option value="responses">Responses</option><option value="chat_completions">Chat Completions</option><option value="claude_messages">Claude Messages</option></select>
            <button class="btn secondary" style="margin-top:10px" onclick="testSimulator()">测试</button>
            <div class="hint" style="margin-top:8px">日常短聊；英文 ≤20 词，中文 ≤20 字；每轮明确带 transcript。</div>
          </div></details>
          <details class="section"><summary>Translator</summary><div class="sectionInner">
            <label class="hint"><input id="translateToZh" type="checkbox" checked /> 英文源对话后翻译为中文</label>
            <div class="label">Base URL</div><input id="translatorBase" class="input" placeholder="强模型 Base URL" />
            <div class="label">API Key</div><input id="translatorKey" class="input" type="password" />
            <div class="label">Model</div><input id="translatorModel" class="input" placeholder="推荐强中文模型" />
            <div class="label">Protocol</div><select id="translatorProtocol" class="select"><option value="responses">Responses</option><option value="chat_completions">Chat Completions</option><option value="claude_messages">Claude Messages</option></select>
            <button class="btn secondary" style="margin-top:10px" onclick="testTranslator()">测试</button>
          </div></details>
          <details class="section"><summary>Judge</summary><div class="sectionInner">
            <div class="label">Base URL</div><input id="judgeBase" class="input" />
            <div class="label">API Key</div><input id="judgeKey" class="input" type="password" />
            <div class="label">Model</div><input id="judgeModel" class="input" />
            <div class="label">Protocol</div><select id="judgeProtocol" class="select"><option value="responses">Responses</option><option value="chat_completions">Chat Completions</option><option value="claude_messages">Claude Messages</option></select>
            <label class="hint" style="display:block;margin-top:10px"><input id="useJudge" type="checkbox" /> 使用 LLM Judge</label>
          </div></details>
          <details class="section" open><summary>生成参数</summary><div class="sectionInner">
            <div class="row"><div><div class="label">数量</div><input id="count" class="input" type="number" min="1" max="10000" value="5" /></div><div><div class="label">并发</div><input id="concurrency" class="input" type="number" min="1" max="128" value="3" /></div></div>
            <div class="label">最大轮数</div><input id="maxTurns" class="input" type="number" min="2" max="20" value="12" />
            <div class="label">随机种子</div><input id="seed" class="input" type="number" placeholder="可空" />
            <button id="genBtn" class="btn" style="margin-top:12px" onclick="generate()">生成 + 审核</button>
          </div></details>
          <details class="section"><summary>导出训练文件</summary><div class="sectionInner">
            <div class="label">目标模型</div><select id="targetModel" class="select"><option value="qwen36_27b">Qwen3.6-27B</option><option value="gemma4_31b">Gemma 4 31B</option></select>
            <div class="label">训练模式</div><select id="trainMode" class="select"><option value="qlora">QLoRA</option><option value="lora">LoRA</option><option value="fft">FFT</option></select>
            <label class="hint"><input id="includeReview" type="checkbox" /> 包含 needs_review</label>
            <label class="hint"><input id="onlyDistill" type="checkbox" checked /> 只导出逐轮蒸馏样本</label>
            <button class="btn good" style="margin-top:10px" onclick="exportData()">导出 JSONL + Notebook</button>
            <div id="exportLinks" class="links hint"></div>
          </div></details>
        </div>
      </aside>
      <main class="panel" id="livePanel">
        <div class="panelHead"><h2>实时生成</h2><button class="foldBtn" onclick="togglePanel('livePanel')">⌄</button></div>
        <div class="panelBody workbench">
          <div id="jobPanel" class="jobPanel" style="display:none">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:center"><div><b id="jobStatus">idle</b><div id="jobCounts" class="hint">-</div></div><button class="btn secondary" onclick="clearLiveChat()">清空显示</button></div>
            <div class="progressLine" style="margin-top:10px"><div id="jobBar"></div></div>
          </div>
          <div class="liveChat" id="liveChat"><div class="liveEmpty">生成时这里会像聊天窗口一样实时显示 user / Sydney 对话</div></div>
          <details class="section"><summary>样本列表</summary><div class="sectionInner" style="padding:0"><div class="tabs"><button class="tab active" data-status="accepted" onclick="setStatus('accepted')">accepted</button><button class="tab" data-status="needs_review" onclick="setStatus('needs_review')">review</button><button class="tab" data-status="rejected" onclick="setStatus('rejected')">rejected</button><button class="tab" data-status="all" onclick="setStatus('all')">all</button><input id="search" class="input" style="margin-left:auto;max-width:240px" placeholder="搜索" onkeydown="if(event.key==='Enter') loadSamples()" /></div><div class="list" id="sampleList"></div></div></details>
        </div>
      </main>
      <aside class="panel" id="reviewPanel">
        <div class="panelHead"><h2>样本审核器</h2><button class="foldBtn" onclick="togglePanel('reviewPanel')">⌄</button></div>
        <div class="panelBody">
          <details class="section" open><summary>对话预览</summary><div class="chat" id="chat"><div class="hint">选择样本查看对话。</div></div></details>
          <details class="section"><summary>编辑样本</summary><div class="sectionInner">
            <div class="row"><div><div class="label">状态</div><select id="editStatus" class="select"><option>accepted</option><option>needs_review</option><option>rejected</option></select></div><div><div class="label">评分</div><input id="editScore" class="input" type="number" min="0" max="10" step="0.1" /></div></div>
            <div class="label">messages JSON</div><textarea id="jsonEdit" class="textarea jsonbox" spellcheck="false"></textarea>
            <div class="row" style="margin-top:10px"><button class="btn good" onclick="saveSample()">保存</button><button class="btn danger" onclick="quickReject()">丢弃</button></div>
          </div></details>
          <details class="section" open><summary>审核理由</summary><div class="sectionInner"><div id="reviewReasons" class="hint"></div><div class="label">样本 ID</div><div id="sampleId" class="meta">-</div></div></details>
        </div>
      </aside>
    </div>
  </div>
  <div class="toast" id="toast"></div>
<script>
const state = {status:'accepted', selected:null, selectedSample:null, currentJob:null, pollTimer:null};
const JOB_STORAGE_KEY = 'sydney_current_job_id';
const LAST_JOB_STORAGE_KEY = 'sydney_last_job_id';
function isActiveJob(job){return job && ['queued','running'].includes(job.status)}
function rememberJob(jobId){state.currentJob=jobId||null; if(jobId){localStorage.setItem(JOB_STORAGE_KEY,jobId)}else{localStorage.removeItem(JOB_STORAGE_KEY)}}
function togglePanel(id){document.getElementById(id)?.classList.toggle('collapsed')}
function toast(msg){const box=document.getElementById('toast'); const d=document.createElement('div'); d.textContent=msg; box.appendChild(d); setTimeout(()=>d.remove(),5200)}
function errorText(x){if(!x) return ''; if(typeof x==='string') return x; if(Array.isArray(x)) return x.map(errorText).filter(Boolean).join('；'); if(typeof x==='object'){if(x.msg){let loc=Array.isArray(x.loc)?x.loc.join('.'):x.loc; return (loc?loc+': ':'')+x.msg} if(x.message) return x.message; if(x.detail) return errorText(x.detail); try{return JSON.stringify(x)}catch(e){return String(x)}} return String(x)}
async function api(path, opts={}){const res=await fetch(path,{headers:{'Content-Type':'application/json'},...opts}); const txt=await res.text(); let data={}; try{data=txt?JSON.parse(txt):{}}catch(e){data={raw:txt}} if(!res.ok) throw new Error(errorText(data.detail||data.error||data.raw||txt||res.statusText)); return data}
function endpoint(prefix){return {base_url:document.getElementById(prefix+'Base').value.trim(), api_key:document.getElementById(prefix+'Key').value.trim(), model:document.getElementById(prefix+'Model').value.trim(), api_protocol:document.getElementById(prefix+'Protocol').value}}
function firstReadyAuxEndpoint(){for(const p of ['translator','simulator','judge']){const ep=endpoint(p); if(ep.base_url&&ep.model) return {prefix:p, endpoint:ep}} return {prefix:'translator', endpoint:endpoint('translator')}}
function effectiveEndpoint(prefix){if(sameAuxModel.checked && ['simulator','translator','judge'].includes(prefix)){return firstReadyAuxEndpoint().endpoint} return endpoint(prefix)}
function syncAuxUi(){if(sameAuxModel.checked){const shared=firstReadyAuxEndpoint(); toast('已启用共用强模型：'+(shared?.prefix||'translator'))}}
function applyDefaults(d){if(!d) return; count.value=d.count??5; concurrency.value=d.concurrency??3; maxTurns.value=d.max_turns??12; translateToZh.checked=!!d.translate_to_zh; sameAuxModel.checked=!!d.same_aux_model; useJudge.checked=!!d.use_judge; if(d.target_model_key) targetModel.value=d.target_model_key; if(d.train_mode) trainMode.value=d.train_mode; includeReview.checked=!!d.include_needs_review; onlyDistill.checked=d.only_dialogue_distillation!==false}
async function loadConfig(){try{const c=await api('/api/config'); teacherBase.value=c.teacher.base_url||''; teacherModel.value=c.teacher.model||''; teacherProtocol.value=c.teacher.api_protocol||'responses'; simulatorBase.value=c.simulator?.base_url||''; simulatorModel.value=c.simulator?.model||''; simulatorProtocol.value=c.simulator?.api_protocol||'responses'; translatorBase.value=c.translator?.base_url||''; translatorModel.value=c.translator?.model||''; translatorProtocol.value=c.translator?.api_protocol||'responses'; judgeBase.value=c.judge.base_url||''; judgeModel.value=c.judge.model||''; judgeProtocol.value=c.judge.api_protocol||'responses'; applyDefaults(c.defaults); if(c.teacher.has_key) teacherKey.placeholder='已从 .env 读取，可留空'; if(c.simulator?.has_key) simulatorKey.placeholder='已从 .env 读取，可留空'; if(c.translator?.has_key) translatorKey.placeholder='已从 .env 读取，可留空'; if(c.judge.has_key) judgeKey.placeholder='已从 .env 读取，可留空'}catch(e){toast('读取配置失败：'+e.message)}}
async function loadStats(){const s=await api('/api/stats'); stTotal.textContent=s.total||0; stAccepted.textContent=s.by_status?.accepted?.count||0; stReview.textContent=s.by_status?.needs_review?.count||0; stRejected.textContent=s.by_status?.rejected?.count||0}
function setStatus(st){state.status=st; document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x.dataset.status===st)); loadSamples()}
async function loadSamples(){await loadStats(); const q=encodeURIComponent(search.value.trim()); const data=await api(`/api/samples?status=${state.status}&q=${q}&limit=80`); const list=sampleList; list.innerHTML=''; if(!data.items.length){list.innerHTML='<div class="hint" style="padding:20px">没有样本。先点击生成。</div>'; return} data.items.forEach(item=>{const el=document.createElement('div'); el.className='card'+(state.selected===item.id?' active':''); el.onclick=()=>selectSample(item.id); const tags=(item.tags||[]).slice(0,5).map(t=>`<span class="badge">${escapeHtml(t)}</span>`).join(''); el.innerHTML=`<div class="cardTop"><b>${escapeHtml(item.theme||'Untitled')}</b><span class="score">${Number(item.score).toFixed(1)}</span></div><div>${tags}</div><div class="preview">${escapeHtml(item.preview||'')}</div><div class="meta" style="margin-top:8px">${item.status} · ${escapeHtml(item.id)}</div>`; list.appendChild(el)})}
async function selectSample(id){state.selected=id; const s=await api('/api/samples/'+id); state.selectedSample=s; sampleId.textContent=s.id; editStatus.value=s.status; editScore.value=s.score; jsonEdit.value=JSON.stringify(s.messages,null,2); renderChat(s); renderReasons(s); await loadSamples()}
function addBubble(container, role, text, extra=''){const b=document.createElement('div'); b.className='bubble '+role+(extra?' '+extra:''); b.textContent=text; container.appendChild(b); container.scrollTop=container.scrollHeight}
function clearLiveChat(){liveChat.innerHTML='<div class="liveEmpty">生成时这里会像聊天窗口一样实时显示 user / Sydney 对话</div>'}
function renderLiveEvents(job){const events=(job.logs||[]).slice(-240); liveChat.innerHTML=''; let visible=0; events.forEach(ev=>{if(ev.kind==='chat'){addBubble(liveChat, ev.role==='user'?'user':'assistant', ev.message); visible++}else if(ev.level==='error'||ev.level==='warn'||ev.kind==='warn'){addBubble(liveChat,'event',`[${ev.time}] ${ev.message}`, ev.level==='warn'?'warn':''); visible++}else if(ev.message&&(/任务启动|开始英文源|翻译完成|审核完成|已保存|任务完成/.test(ev.message))){addBubble(liveChat,'event',`[${ev.time}] ${ev.message}`); visible++}}); if(!visible) clearLiveChat()}
function renderChat(s){chat.innerHTML=''; s.messages.forEach(m=>{addBubble(chat,m.role,(m.role==='system'?'SYSTEM\n':'')+m.content)}); const src=s.metadata?.source_messages_en; if(Array.isArray(src)&&src.length){const details=document.createElement('details'); details.className='bubble system'; const summary=document.createElement('summary'); summary.textContent='查看英文源对话 metadata.source_messages_en'; details.appendChild(summary); const pre=document.createElement('pre'); pre.className='meta'; pre.textContent=JSON.stringify(src,null,2); details.appendChild(pre); chat.appendChild(details)}}
function renderReasons(s){const r=s.review||{}; const reasons=r.reasons||[]; reviewReasons.innerHTML=`<div class="reason">overall: ${Number(r.overall||s.score||0).toFixed(2)} · reviewer: ${escapeHtml(r.reviewer||'')}</div>`+reasons.map(x=>`<div class="reason">${escapeHtml(x)}</div>`).join('')+`<pre class="meta">${escapeHtml(JSON.stringify(r.scores||{},null,2))}</pre>`}
async function saveSample(){if(!state.selected) return toast('未选择样本'); let messages; try{messages=JSON.parse(jsonEdit.value)}catch(e){return toast('messages JSON 不合法：'+e.message)} await api('/api/samples/'+state.selected,{method:'PUT',body:JSON.stringify({messages,status:editStatus.value,score:Number(editScore.value)})}); toast('已保存'); await selectSample(state.selected); await loadStats()}
async function quickReject(){if(!state.selected) return; editStatus.value='rejected'; editScore.value=Math.min(Number(editScore.value||0),4); await saveSample()}
async function testTeacher(){try{const res=await api('/api/config/test',{method:'POST',body:JSON.stringify({endpoint:endpoint('teacher'),prefix:'TEACHER'})}); toast(res.ok?'Sydney/source OK':'Sydney/source 失败：'+res.error)}catch(e){toast('测试失败：'+e.message)}}
async function testSimulator(){try{const ep=effectiveEndpoint('simulator'); const res=await api('/api/config/test',{method:'POST',body:JSON.stringify({endpoint:ep,prefix:'SIMULATOR'})}); toast(res.ok?'Human Simulator OK':'Human Simulator 失败：'+res.error)}catch(e){toast('测试失败：'+e.message)}}
async function testTranslator(){try{const ep=effectiveEndpoint('translator'); const res=await api('/api/config/test',{method:'POST',body:JSON.stringify({endpoint:ep,prefix:'TRANSLATOR'})}); toast(res.ok?'Translator OK':'Translator 失败：'+res.error)}catch(e){toast('测试失败：'+e.message)}}
function renderJob(job){jobPanel.style.display='block'; const total=Number(job.total||0); const done=Number(job.completed||0); const pct=job.progress?job.progress.percent:(total?Math.round(done*1000/total)/10:0); jobStatus.textContent=`${job.status} · ${done}/${total} · ${pct}%`; jobBar.style.width=`${Math.max(0,Math.min(100,pct))}%`; const trans=job.translate_to_zh?'en→zh':'en only'; const translator=job.translator?.enabled?job.translator.model:'disabled'; const shared=job.same_aux_model?` · shared ${job.shared_aux_source||'on'}`:''; jobCounts.textContent=`accepted ${job.accepted||0} · review ${job.needs_review||0} · rejected ${job.rejected||0} · failed ${job.failed||0} · concurrency ${job.concurrency||0} · ${trans} · translator ${translator}${shared}`; renderLiveEvents(job)}
async function pollJob(jobId){try{const job=await api('/api/jobs/'+jobId); renderJob(job); await loadStats(); if(!isActiveJob(job)){if(state.pollTimer) clearInterval(state.pollTimer); state.pollTimer=null; if(state.currentJob===jobId){rememberJob(null); localStorage.setItem(LAST_JOB_STORAGE_KEY,jobId)} genBtn.disabled=false; genBtn.innerHTML='生成 + 审核'; toast(job.status==='completed'?'生成任务完成':job.status==='interrupted'?'任务被服务重启中断':'生成任务失败'); await loadSamples()}else{genBtn.disabled=true; genBtn.innerHTML='<span class="spinner"></span> 生成中'}}catch(e){if(state.pollTimer) clearInterval(state.pollTimer); state.pollTimer=null; rememberJob(null); genBtn.disabled=false; genBtn.innerHTML='生成 + 审核'; toast('轮询任务失败：'+e.message)}}
async function startPolling(jobId, showToast=false){rememberJob(jobId); if(state.pollTimer) clearInterval(state.pollTimer); genBtn.disabled=true; genBtn.innerHTML='<span class="spinner"></span> 生成中'; state.pollTimer=setInterval(()=>pollJob(jobId),1000); await pollJob(jobId); if(showToast) toast('已恢复任务进度：'+jobId.slice(0,8))}
async function resumeJob(){let jobId=localStorage.getItem(JOB_STORAGE_KEY); if(jobId){try{const job=await api('/api/jobs/'+jobId); renderJob(job); if(isActiveJob(job)){await startPolling(jobId,true); return}else{rememberJob(null); localStorage.setItem(LAST_JOB_STORAGE_KEY,jobId)}}catch(e){rememberJob(null)}} try{const active=await api('/api/jobs?active=true&limit=1'); if(active.items&&active.items.length){await startPolling(active.items[0].id,true); return}}catch(e){} /* 不再自动加载历史非活动 job，避免超大快照导致页面打开变慢。 */}
async function generate(){const btn=genBtn; btn.disabled=true; btn.innerHTML='<span class="spinner"></span> 生成中'; clearLiveChat(); try{const seedVal=seed.value.trim(); const data=await api('/api/generate',{method:'POST',body:JSON.stringify({count:Number(count.value||5),concurrency:Number(concurrency.value||3),max_turns:Number(maxTurns.value||12),translate_to_zh:translateToZh.checked,same_aux_model:sameAuxModel.checked,teacher:endpoint('teacher'),simulator:endpoint('simulator'),translator:endpoint('translator'),judge:endpoint('judge'),use_judge:useJudge.checked,seed:seedVal?Number(seedVal):null})}); renderJob(data.job); toast('任务已启动：'+data.job_id.slice(0,8)); await startPolling(data.job_id,false)}catch(e){toast('生成启动失败：'+e.message); rememberJob(null); btn.disabled=false; btn.innerHTML='生成 + 审核'}}
async function exportData(){try{const data=await api('/api/export',{method:'POST',body:JSON.stringify({target_model_key:targetModel.value,train_mode:trainMode.value,include_needs_review:includeReview.checked,only_dialogue_distillation:onlyDistill.checked})}); exportLinks.innerHTML=`导出 ${data.count} 条：<a href="${data.files.chatml}">train_chatml.jsonl</a><a href="${data.files.sharegpt}">train_sharegpt.jsonl</a><a href="${data.files.notebook}">train_sydney.ipynb</a>`; toast('导出完成')}catch(e){toast('导出失败：'+e.message)}}
function escapeHtml(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
loadConfig().then(()=>loadSamples()).then(loadStats).then(resumeJob);
</script>
</body>
</html>
"""


