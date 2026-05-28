"""Lightweight duplicate detection for generated conversations.

不引入重量级 embedding 依赖，优先用字符 n-gram + SequenceMatcher。
数据量到几万条以后，可以把这里替换成 MinHash/向量索引。
"""
from __future__ import annotations

import hashlib
import re
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


def conversation_text(messages: List[dict]) -> str:
    parts = []
    for msg in messages or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)
