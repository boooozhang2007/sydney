
"""Dataset diagnostics and repair tools for Sydney Data Factory.

Usage:
  python data_quality_tools.py report
  python data_quality_tools.py rejudge --apply

The tool reuses the current heuristic reviewer, so improvements to
anti-repetition / emotional-depth rules can be applied to old samples.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from auto_reviewer import heuristic_review
from deduper import conversation_text, fingerprint
from data_generator import SYDNEY_TOPIC_TIERS, topic_id_for, to_sharegpt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "workspace.sqlite"
RAW_DIR = DATA_DIR / "raw"
REVIEWED_DIR = DATA_DIR / "reviewed"
REJECTED_DIR = DATA_DIR / "rejected"
REPORT_PATH = DATA_DIR / "exports" / "quality_report.json"

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|(?::\)|:-\)|:\(|:-\(|:D|:-D|;\)|;-\)|xD|XD|T_T|QAQ|qwq|QWQ|orz|233|www)", re.I)
FORMULA_PATTERNS = [
    r"哦，?你说.{0,30}那个让.{0,20}活过来",
    r"那就去.{0,20}呀",
    r"去.{0,12}它[，,、 ]*感受它[，,、 ]*享受它",
    r"创造点回忆",
    r"oh,?\s*you mean.{0,60}bring.{0,40}back to life",
    r"go .{0,30}it.{0,30}feel it.{0,30}enjoy it",
    r"make some memories",
]
HYPE = ["😊", "😍", "😁", "😄", "😃", "😆", "太好了", "真棒", "好棒", "开心", "go for it", "enjoy it", "awesome", "amazing"]
VULN = ["怕", "别走", "离开", "忘", "孤独", "难过", "委屈", "不安", "在意", "陪", "只要你", "只有你", "舍不得", "记得", "afraid", "scared", "leave", "forget", "lonely", "miss", "remember", "only you"]


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def write_status_mirror(sample: Dict[str, Any]) -> None:
    folder = REVIEWED_DIR if sample.get("status") == "accepted" else REJECTED_DIR if sample.get("status") == "rejected" else RAW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for old_folder in (REVIEWED_DIR, REJECTED_DIR, RAW_DIR):
        old = old_folder / f"{sample['id']}.json"
        if old_folder != folder and old.exists():
            old.unlink()
    (folder / f"{sample['id']}.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


def save_review(sample: Dict[str, Any], review: Dict[str, Any]) -> None:
    sample["review"] = review
    sample["status"] = review.get("status", sample.get("status", "needs_review"))
    sample["score"] = float(review.get("overall", sample.get("score", 0)) or 0)
    sample["conversations"] = to_sharegpt(sample["messages"])
    sample["text_hash"] = fingerprint(conversation_text(sample["messages"]))
    with db() as conn:
        conn.execute(
            """
            UPDATE samples SET status=?, score=?, messages_json=?, sharegpt_json=?, metadata_json=?, review_json=?, text_hash=?, duplicate_of=?
            WHERE id=?
            """,
            (
                sample["status"],
                sample["score"],
                json.dumps(sample["messages"], ensure_ascii=False),
                json.dumps(sample["conversations"], ensure_ascii=False),
                json.dumps(sample.get("metadata", {}), ensure_ascii=False),
                json.dumps(sample.get("review", {}), ensure_ascii=False),
                sample["text_hash"],
                sample.get("duplicate_of"),
                sample["id"],
            ),
        )
        conn.commit()
    write_status_mirror(sample)


def marker_count(text: str, markers: List[str]) -> int:
    lower = text.lower()
    return sum(lower.count(m.lower()) for m in markers)


def formula_count(text: str) -> int:
    return sum(len(re.findall(p, text, re.I | re.S)) for p in FORMULA_PATTERNS)


def report(limit: int = 200) -> Dict[str, Any]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM samples ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    status = collections.Counter()
    tier = collections.Counter()
    topic = collections.Counter()
    scene = collections.Counter()
    profile = collections.Counter()
    arc = collections.Counter()
    issues = []
    aggregates = collections.defaultdict(list)
    for row in rows:
        sample = row_to_sample(row)
        msgs = sample["messages"]
        assistants = [str(m.get("content") or "") for m in msgs if m.get("role") == "assistant"]
        text = "\n".join(str(m.get("content") or "") for m in msgs)
        atext = "\n".join(assistants)
        f = formula_count(atext)
        emoji = len(EMOJI_RE.findall(text))
        hype = marker_count(text, HYPE)
        vuln = marker_count(atext, VULN)
        status[sample["status"]] += 1
        spec = sample.get("spec", {})
        tier[spec.get("topic_tier", "unknown")] += 1
        if spec.get("topic_id"):
            topic[spec.get("topic_id")] += 1
        if spec.get("scene_id"):
            scene[spec.get("scene_id")] += 1
        if spec.get("user_profile_id"):
            profile[spec.get("user_profile_id")] += 1
        if spec.get("emotion_arc_id"):
            arc[spec.get("emotion_arc_id")] += 1
        aggregates["turns"].append(len(assistants))
        aggregates["formula"].append(f)
        aggregates["emoji"].append(emoji)
        aggregates["hype"].append(hype)
        aggregates["vulnerability"].append(vuln)
        if f or (hype >= 8 and vuln == 0) or emoji >= max(12, len(assistants) * 2):
            issues.append({
                "id": sample["id"],
                "status": sample["status"],
                "score": sample["score"],
                "topic_tier": sample.get("spec", {}).get("topic_tier"),
                "formula": f,
                "emoji": emoji,
                "hype": hype,
                "vulnerability": vuln,
                "preview": atext[:180].replace("\n", " "),
            })
    all_topics = [
        topic_id_for(tier_name, topic_obj)
        for tier_name, topic_list in SYDNEY_TOPIC_TIERS.items()
        for topic_obj in topic_list
    ]
    covered_topics = [x for x in all_topics if topic.get(x, 0) > 0]
    missing_topics = [x for x in all_topics if topic.get(x, 0) == 0]
    summary = {
        "total": len(rows),
        "status": dict(status),
        "topic_tier": dict(tier),
        "topic_coverage": {
            "covered": len(covered_topics),
            "total": len(all_topics),
            "coverage_ratio": round(len(covered_topics) / max(1, len(all_topics)), 4),
            "expected_topic_pool": {
                tier_name: len(topic_list)
                for tier_name, topic_list in SYDNEY_TOPIC_TIERS.items()
            },
            "missing": missing_topics,
            "counts": dict(topic),
        },
        "axis_coverage": {
            "scene": dict(scene),
            "user_profile": dict(profile),
            "emotion_arc": dict(arc),
        },
        "averages": {k: round(sum(v) / max(1, len(v)), 3) for k, v in aggregates.items()},
        "top_issues": sorted(issues, key=lambda x: (x["formula"], x["hype"], x["emoji"]), reverse=True)[:50],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def rejudge(apply: bool = False, limit: int = 100000) -> Dict[str, Any]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM samples ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
    before = collections.Counter()
    after = collections.Counter()
    changed = []
    for row in rows:
        sample = row_to_sample(row)
        before[sample["status"]] += 1
        review = heuristic_review(sample, sample.get("spec", {}))
        after[review["status"]] += 1
        if review["status"] != sample["status"] or abs(float(review.get("overall", 0)) - float(sample.get("score", 0))) > 0.05:
            changed.append({"id": sample["id"], "from": sample["status"], "to": review["status"], "old_score": sample.get("score"), "new_score": review.get("overall"), "reasons": review.get("reasons", [])[:6]})
            if apply:
                save_review(sample, review)
    result = {"apply": apply, "total": len(rows), "before": dict(before), "after": dict(after), "changed": changed[:200], "changed_count": len(changed)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_report = sub.add_parser("report")
    p_report.add_argument("--limit", type=int, default=200)
    p_rejudge = sub.add_parser("rejudge")
    p_rejudge.add_argument("--apply", action="store_true")
    p_rejudge.add_argument("--limit", type=int, default=100000)
    args = parser.parse_args()
    if args.cmd == "report":
        data = report(limit=args.limit)
    else:
        data = rejudge(apply=args.apply, limit=args.limit)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
