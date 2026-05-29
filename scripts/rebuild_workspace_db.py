from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from deduper import conversation_text, fingerprint  # noqa: E402


def main() -> int:
    """Rebuild data/workspace.sqlite from per-sample JSON mirrors.

    Existing workspace.sqlite / -wal / -shm files are moved aside with a
    `.corrupt.<timestamp>` suffix before the clean database is created.
    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in ("", "-wal", "-shm"):
        path = app.DATA_DIR / f"workspace.sqlite{suffix}"
        if path.exists():
            dst = app.DATA_DIR / f"workspace.sqlite{suffix}.corrupt.{ts}"
            print(f"MOVE {path} -> {dst}")
            path.replace(dst)

    app.init_db()
    samples = app.load_all_mirror_samples()
    print(f"MIRROR_SAMPLES={len(samples)}")

    rows = []
    for sample in samples:
        messages = sample.get("messages", [])
        conversations = sample.get("conversations") or app.to_sharegpt(messages)
        text_hash = sample.get("text_hash") or fingerprint(conversation_text(messages))
        rows.append(
            (
                sample["id"],
                sample.get("created_at") or app.utc_now(),
                sample.get("updated_at") or app.utc_now(),
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

    with app.db() as conn:
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
        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        by_status = conn.execute("SELECT status, COUNT(*) FROM samples GROUP BY status ORDER BY status").fetchall()
        print("DB_TOTAL", total)
        print("DB_BY_STATUS", [(row[0], row[1]) for row in by_status])
        print("QUICK_CHECK", conn.execute("PRAGMA quick_check").fetchone()[0])
        print("INTEGRITY_CHECK", conn.execute("PRAGMA integrity_check").fetchone()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
