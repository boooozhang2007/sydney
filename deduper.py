"""Lightweight duplicate detection for generated conversations.

不引入重量级 embedding 依赖，优先用字符 n-gram + SequenceMatcher。
数据量到几万条以后，可以把这里替换成 MinHash/向量索引。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple

PUNCT_RE = re.compile(r"[\s，。！？、,.!?;；:：'\"“”‘’（）()\[\]{}<>《》—\-_*#`~|/\\]+")


def canonicalize_text(text: str) -> str:
    text = (text or "").lower()
    return PUNCT_RE.sub("", text)


def fingerprint(text: str) -> str:
    return hashlib.sha256(canonicalize_text(text).encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 5) -> set[str]:
    text = canonicalize_text(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def similarity(a: str, b: str) -> float:
    """返回 0-1 相似度。"""
    ca, cb = canonicalize_text(a), canonicalize_text(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    jac = jaccard(char_ngrams(ca), char_ngrams(cb))
    seq = SequenceMatcher(None, ca[:5000], cb[:5000]).ratio()
    return max(jac, seq * 0.92)


def find_duplicate(text: str, existing_texts: Iterable[Tuple[str, str]], threshold: float = 0.88) -> Tuple[bool, str | None, float]:
    """在已有文本中找近重复。"""
    best_id = None
    best_score = 0.0
    fp = fingerprint(text)
    for sample_id, existing in existing_texts:
        if fingerprint(existing) == fp:
            return True, sample_id, 1.0
        score = similarity(text, existing)
        if score > best_score:
            best_id, best_score = sample_id, score
    return best_score >= threshold, best_id, best_score


@dataclass
class DedupRecord:
    sample_id: str
    canonical: str
    fingerprint: str
    ngrams: set[str]
    length: int


class DedupIndex:
    """Small in-memory duplicate index.

    It avoids the old O(N) path that recomputed every existing fingerprint and
    full SequenceMatcher ratio on each completed sample.  Exact duplicates are
    O(1); near-duplicates first pass through cheap length + 5-gram Jaccard
    filters, and only plausible candidates pay the SequenceMatcher cost.
    """

    def __init__(self, rows: Iterable[Tuple[str, str]] = ()):
        self.records: List[DedupRecord] = []
        self.by_fingerprint: dict[str, str] = {}
        self._cursor = 0
        for sample_id, text in rows:
            self.add(sample_id, text)

    @staticmethod
    def _record(sample_id: str, text: str) -> DedupRecord:
        canonical = canonicalize_text(text)
        return DedupRecord(
            sample_id=sample_id,
            canonical=canonical,
            fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            ngrams=char_ngrams(canonical),
            length=len(canonical),
        )

    def add(self, sample_id: str, text: str) -> None:
        rec = self._record(sample_id, text)
        self.records.append(rec)
        self.by_fingerprint.setdefault(rec.fingerprint, sample_id)

    def find(self, text: str, threshold: float = 0.88) -> Tuple[bool, str | None, float]:
        rec = self._record("__new__", text)
        exact_id = self.by_fingerprint.get(rec.fingerprint)
        if exact_id:
            return True, exact_id, 1.0
        if not rec.canonical:
            return False, None, 0.0

        best_id = None
        best_score = 0.0
        # If the index is huge, do an inexpensive rolling sample after all
        # strong candidates.  This keeps the main ingest loop predictable.
        max_full_scan = int(max(1000, min(20000, float(os.getenv("DEDUP_MAX_SCAN", "5000") or 5000))))
        total = len(self.records)
        if total <= max_full_scan:
            candidates = self.records
        else:
            # Always check latest records plus a rotating window from history.
            latest = self.records[-max_full_scan // 2 :]
            span = max_full_scan - len(latest)
            start = self._cursor % max(1, total - len(latest))
            hist = self.records[: -len(latest)]
            candidates = latest + (hist[start : start + span] if start + span <= len(hist) else hist[start:] + hist[: (start + span) % len(hist)])
            self._cursor = (self._cursor + span) % max(1, len(hist))

        for old in candidates:
            if not old.canonical:
                continue
            len_ratio = min(rec.length, old.length) / max(1, max(rec.length, old.length))
            if len_ratio < 0.55:
                continue
            jac = jaccard(rec.ngrams, old.ngrams)
            # A final score >= 0.88 is practically impossible if both cheap
            # signals are low, so skip expensive SequenceMatcher.
            if jac < 0.28 and len_ratio < 0.82:
                continue
            seq = SequenceMatcher(None, rec.canonical[:5000], old.canonical[:5000]).ratio() * 0.92
            score = max(jac, seq)
            if score > best_score:
                best_id, best_score = old.sample_id, score
                if best_score >= 0.995:
                    break
        return best_score >= threshold, best_id, best_score


def conversation_text(messages: List[dict]) -> str:
    parts = []
    for msg in messages or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
