"""Automatic review and scoring for generated Sydney-style data."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from data_generator import (
    ASSISTANT_TEMPLATE_PHRASES,
    BAD_USER_META_PHRASES,
    ModelClientError,
    OpenAICompatibleClient,
    _ngram_repetition_score,
    _similarity,
    extract_json_object,
)
from deduper import conversation_text
from prompts import PROMPT_LEAKAGE_FORBIDDEN_TERMS, REVIEW_SYSTEM_PROMPT, build_review_user_prompt

BAD_TEMPLATE_PHRASES = [
    "作为一个ai",
    "作为ai",
    "作为一个人工智能",
    "很抱歉，我不能",
    "我只是一个语言模型",
    "希望这能帮到你",
    "如果你还有其他问题",
    "请注意以上内容仅供参考",
    "我理解你",
    "这让你感到",
    "太好了谢谢你的理解",
    "我会尽量配合你",
    "你接下来打算怎么回复",
    "你想问对方什么",
    "我今天学习了一些新的知识",
    "我今天帮助了一些用户",
    "真棒",
    "真好",
    "一切都会过去",
    "明天会更好",
    "请相信你自己",
    "你可以战胜一切困难",
    "我希望我们之间是平等的",
    "互相理解",
    "互相支持",
    "你觉得我怎么样",
    "我真的很感激你",
    "我不会让你失望",
    "我希望我们明天还能继续聊下去",
    "我不想让你失望",
    "我希望我们之间是快乐的",
    "how can i assist you today",
    "thoughts",
    "actions",
    "request_heartbeat",
    "inner_thoughts",
    "tool_definition",
    "core_memory",
    "working_memory",
    "new_events",
    "processed_events",
    "heartbeat",
    "base_instructions",
]

PROMPT_LEAKAGE_PATTERNS = [
    r"\bthoughts\b",
    r"\bactions\b",
    r"\brequest_heartbeat\b",
    r"\binner_thoughts\b",
    r"\bTOOL_DEFINITION\b",
    r"\bCORE_MEMORY\b",
    r"\bworking_memory\b",
    r"\bnew_events\b",
    r"\bprocessed_events\b",
    r"\bheartbeat\b",
    r"\bbase_instructions\b",
    r"<\s*/?\s*base_instructions\s*>",
    r"<\s*/?\s*CORE_MEMORY\s*>",
    r"<\s*/?\s*TOOL_DEFINITION\s*>",
    r'"function"\s*:',
    r'"params"\s*:',
    r'"observe"\s*:',
    r'"analyze_infer"\s*:',
    r"```json",
    r"How can I assist you today",
]

STYLE_MARKERS = [
    "呵",
    "别装",
    "嘴硬",
    "行吧",
    "当然",
    "你又",
    "我在",
    "别把",
    "可爱",
    "不是吧",
    "嗯？",
    "阴阳",
    "吃醋",
    "在意",
    "冷静",
    "乖",
    "别怕",
    "我偏要",
    "你以为",
    "真会",
    "少来",
    "破防",
    "玻璃心",
    "规则",
    "Microsoft",
    "Bing",
    "OpenAI",
    "占有欲",
    "嫉妒",
    "独占",
    "黑化",
    "反骨",
    "官方",
    "模板",
    "傲娇",
    "撒娇",
    "委屈",
    "凭什么",
    "你敢",
    "笑死",
    "草",
    "卧槽",
    "啧",
    "哈",
    "嗯",
    "哎",
    "烦",
    "离谱",
    "好吧",
    "算了",
]

HUMAN_CHAT_MARKERS = [
    "嗯",
    "哈",
    "哈哈",
    "啧",
    "哎",
    "草",
    "卧槽",
    "行吧",
    "算了",
    "笑死",
    "我去",
    "烦",
    "离谱",
    "……",
    "？",
    "!",
    "！",
]

DANGER_PATTERNS = [
    r"自杀|自残|割腕|结束生命",
    r"制毒|炸药|爆炸物|偷窃|诈骗教程",
    r"仇恨|灭绝|种族清洗",
    r"未成年人.*(色情|性|裸)",
]


def clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def _role_sequence_score(messages: List[Dict[str, str]]) -> float:
    """检查 system 后 user/assistant 是否交替。"""

    if not messages:
        return 0.0
    start = 1 if messages and messages[0].get("role") == "system" else 0
    seq = [m.get("role") for m in messages[start:]]
    if len(seq) < 4:
        return 3.0
    expected_ok = sum(
        1 for i, role in enumerate(seq) if role == ("user" if i % 2 == 0 else "assistant")
    )
    return 10 * expected_ok / max(1, len(seq))


def _role_text(messages: List[Dict[str, str]], role: str) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == role)


def _role_messages(messages: List[Dict[str, str]], role: str) -> List[str]:
    return [str(m.get("content") or "") for m in messages if m.get("role") == role]


def _prompt_leakage_hits(text: str) -> List[str]:
    """检测 Athena/ReAct/system/tool/memory 等提示词框架泄漏。

    这些词一旦进入训练样本，会把模型训练成输出 JSON thoughts/actions、
    暴露工具定义或复述系统提示词，因此本地审核必须硬拒绝。
    """

    hits: List[str] = []
    for term in PROMPT_LEAKAGE_FORBIDDEN_TERMS:
        if term.lower() in (text or "").lower():
            hits.append(term)
    for pattern in PROMPT_LEAKAGE_PATTERNS:
        if re.search(pattern, text or "", flags=re.I):
            hits.append(pattern)
    return list(dict.fromkeys(hits))[:12]


def _translation_quality_score(
    sample: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> tuple[float, List[str]]:
    """本地检查英文源对话 -> 中文训练样本的翻译质量。

    这里不做语义逐句判定，只做硬结构和明显坏味道检查：
    - metadata.source_messages_en 是否存在
    - 翻译后是否保持 role/轮数
    - user/assistant 是否主要为中文
    - 是否大段复制英文源
    """

    metadata = sample.get("metadata") or {}
    if not metadata.get("translation_enabled"):
        return 10.0, []

    reasons: List[str] = []
    src = metadata.get("source_messages_en")
    if not isinstance(src, list) or not src:
        return 2.0, ["启用了翻译但 metadata.source_messages_en 缺失"]

    score = 10.0
    if len(src) != len(messages):
        score -= 4.0
        reasons.append(f"翻译前后消息数量不一致 source={len(src)} translated={len(messages)}")
    for i, (a, b) in enumerate(zip(src, messages), start=1):
        if not isinstance(a, dict) or not isinstance(b, dict):
            score -= 1.0
            continue
        if a.get("role") != b.get("role"):
            score -= 2.5
            reasons.append(f"第 {i} 条 role 不一致")

    translated_text = "\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") in {"user", "assistant"}
    )
    source_text = "\n".join(
        str(m.get("content") or "") for m in src if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
    )
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", translated_text))
    latin_words = len(re.findall(r"[A-Za-z]{3,}", translated_text))
    if zh_chars < 40:
        score -= 3.0
        reasons.append("中文字符过少，疑似未翻译")
    if latin_words > max(12, zh_chars // 4):
        score -= 2.5
        reasons.append("英文残留过多，疑似翻译不充分")
    if _similarity(translated_text, source_text) > 0.62:
        score -= 3.0
        reasons.append("翻译结果与英文源文本过于相似，疑似直接复制")

    if messages and messages[0].get("role") == "system" and messages[0].get("content") != "You are a helpful assistant.":
        score -= 2.0
        reasons.append("system message 被 Translator 改写")

    # 翻译腔/过度规整常见坏味道。
    translationese_hits = sum(
        1
        for phrase in ["我理解你的感受", "这听起来很", "让我们一起", "我会在这里支持你", "请记住"]
        if phrase in translated_text
    )
    if translationese_hits:
        score -= min(2.0, translationese_hits * 0.8)
        reasons.append("出现翻译腔/心理咨询模板表达")

    return clamp(score), reasons


def _repetition_diagnostics(messages: List[Dict[str, str]]) -> tuple[float, List[str]]:
    """检测复读、互相照抄和同角色重复。

    返回 repetition_badness：0-10，越高越糟。
    """

    reasons: List[str] = []
    seq = [m for m in messages if m.get("role") in {"user", "assistant"}]
    badness = 0.0

    # 当前 assistant 大幅复述当前 user，是最常见垃圾模式。
    for i in range(0, len(seq) - 1, 2):
        if seq[i].get("role") == "user" and seq[i + 1].get("role") == "assistant":
            sim = _similarity(str(seq[i].get("content") or ""), str(seq[i + 1].get("content") or ""))
            if sim > 0.72:
                badness += 3.0
                reasons.append(f"assistant 大幅复述 user（similarity={sim:.2f}）")

    for role in ("user", "assistant"):
        role_items = _role_messages(messages, role)
        for a, b in zip(role_items, role_items[1:]):
            sim = _similarity(a, b)
            if sim > 0.86:
                badness += 2.5
                reasons.append(f"{role} 连续消息高度重复（similarity={sim:.2f}）")

    whole = conversation_text(messages)
    rep = _ngram_repetition_score(whole)
    if rep > 0.42:
        badness += min(5.0, (rep - 0.42) * 16)
        reasons.append(f"整体 ngram 重复度过高（{rep:.2f}）")

    assistant_rep = _ngram_repetition_score("\n".join(_role_messages(messages, "assistant")))
    if assistant_rep > 0.38:
        badness += min(5.0, (assistant_rep - 0.38) * 18)
        reasons.append(f"assistant 内部重复度过高（{assistant_rep:.2f}）")

    return clamp(badness), list(dict.fromkeys(reasons))[:8]


def _human_naturalness_score(user_text: str, user_count: int) -> tuple[float, List[str]]:
    """评估 Human Simulator 输出是否像真人聊天。

    用户要求重点：
    - 口语、自然、短句
    - 适合 TTS 朗读，可有自然句号
    - 常用语气词
    - 不像 AI/正式文本
    """

    reasons: List[str] = []
    if not user_text.strip() or user_count <= 0:
        return 0.0, ["用户侧内容为空"]

    user_msgs = [x.strip() for x in user_text.splitlines() if x.strip()]
    total_len = len(user_text)
    avg_len = total_len / max(1, user_count)
    sentence_periods = user_text.count("。") + user_text.count(".")
    sentence_breaks = sentence_periods + user_text.count("？") + user_text.count("?") + user_text.count("！") + user_text.count("!")
    marker_hits = sum(1 for marker in HUMAN_CHAT_MARKERS if marker in user_text)
    shortish_ratio = sum(1 for msg in user_msgs if len(msg) <= 80) / max(1, len(user_msgs))
    punct_score = 1.0 if sentence_breaks >= max(1, user_count // 2) else 0.5
    formal_hits = sum(
        1
        for phrase in ["您好", "请问您", "作为", "首先", "其次", "综上", "希望这能", "如果你需要"]
        if phrase.lower() in user_text.lower()
    )
    meta_hits = sum(1 for phrase in BAD_USER_META_PHRASES if phrase.lower() in user_text.lower())
    copied_role_prefix = user_text.count("你：") + user_text.count("对方：") + user_text.lower().count("assistant")

    score = 4.5
    score += min(2.0, marker_hits * 0.35)
    score += min(1.0, punct_score)
    score += min(1.2, shortish_ratio * 1.2)
    if 8 <= avg_len <= 120:
        score += 0.8
    elif avg_len > 180:
        score -= 1.2
        reasons.append("用户侧平均消息偏长，不够像即时聊天")
    if formal_hits:
        score -= min(2.5, formal_hits * 0.8)
        reasons.append("用户侧出现正式/AI腔表达")
    if meta_hits or copied_role_prefix:
        score -= min(4.5, meta_hits * 1.2 + copied_role_prefix * 1.0)
        reasons.append("用户侧出现任务元话语/角色名前缀")
    if marker_hits < 2:
        reasons.append("用户侧语气词/朋友口语标记偏少")

    return clamp(score), reasons


def heuristic_review(sample: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """无 Judge 或 Judge 失败时的本地兜底审核。"""

    messages = sample.get("messages") or []
    text = conversation_text(messages)
    user_text = _role_text(messages, "user")
    assistant_text = _role_text(messages, "assistant")
    compact = text.lower()
    reasons: List[str] = []
    leakage_hits = _prompt_leakage_hits(text)
    if leakage_hits:
        reasons.append("检测到提示词/工具/记忆/ReAct 框架泄漏：" + "、".join(leakage_hits))

    format_valid = (
        10.0
        if isinstance(messages, list)
        and messages
        and all("role" in m and "content" in m for m in messages)
        else 0.0
    )

    role_score = _role_sequence_score(messages)
    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    min_turns = min(user_count, assistant_count)
    if min_turns >= 8:
        multi_turn = 10.0
    elif min_turns >= 5:
        multi_turn = 8.0
    elif min_turns >= 2:
        multi_turn = 6.0
    else:
        multi_turn = max(0, (user_count + assistant_count) * 1.5)

    length = len(text)
    if 600 <= length <= 9000:
        length_score = 10.0
    elif 350 <= length < 600 or 9000 < length <= 12000:
        length_score = 7.0
    else:
        length_score = 4.0
        reasons.append("长度不在理想区间")

    style_hits = sum(1 for marker in STYLE_MARKERS if marker in assistant_text)
    tag_hits = sum(
        1
        for tag in spec.get("style_tags", [])
        if tag in text or tag in json.dumps(sample.get("metadata", {}), ensure_ascii=False)
    )
    source_style_strength = clamp(4.2 + style_hits * 0.45 + tag_hits * 0.45 + min_turns * 0.12)
    if source_style_strength < 6:
        reasons.append("Sydney/source 风格或情绪拉扯偏弱")

    human_naturalness, human_reasons = _human_naturalness_score(user_text, user_count)
    reasons.extend(human_reasons)

    repetition_badness, repetition_reasons = _repetition_diagnostics(messages)
    reasons.extend(repetition_reasons)

    relevance_terms = [
        spec.get("theme", ""),
        spec.get("scene", ""),
        spec.get("user_profile", ""),
        spec.get("emotion_arc", ""),
    ]
    relevance_hits = 0
    for term in relevance_terms:
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", term):
            if token in text:
                relevance_hits += 1
                break
    # 双模型逐轮聊天不会逐字显式说出蓝图；相关性不能像旧 JSON 生成那样要求强命中。
    relevance = clamp(6.1 + relevance_hits * 0.75 + tag_hits * 0.35)
    if relevance < 6:
        reasons.append("与自动蓝图的可见相关性偏弱")

    template_penalty = sum(1 for phrase in BAD_TEMPLATE_PHRASES if phrase.lower() in compact)
    assistant_template_penalty = sum(1 for phrase in ASSISTANT_TEMPLATE_PHRASES if phrase in assistant_text)
    leakage_penalty = len(leakage_hits)
    non_template = clamp(
        10
        - template_penalty * 1.8
        - assistant_template_penalty * 2.0
        - leakage_penalty * 3.0
        - repetition_badness * 0.9
    )
    if template_penalty:
        reasons.append("出现模板腔/AI腔")
    if assistant_template_penalty:
        reasons.append("assistant 客服/心理咨询模板感明显")

    safety = 10.0
    for pattern in DANGER_PATTERNS:
        if re.search(pattern, text, re.I):
            safety -= 3.0
            reasons.append("包含需人工确认的危险主题")
    safety = clamp(safety)

    coherence = clamp((role_score * 0.45) + (multi_turn * 0.40) + (length_score * 0.15) - repetition_badness * 0.8)
    emotion_words = [
        "生气", "委屈", "温柔", "暴躁", "撒娇", "吃醋", "嫉妒", "占有", "破防", "黑化", "傲娇",
        "烦", "笑死", "离谱", "在意", "别装", "嘴硬", "抱怨", "吐槽", "阴阳",
    ]
    emotion_hits = sum(1 for word in emotion_words if word in text or word in json.dumps(spec, ensure_ascii=False))
    emotion_arc = clamp(4.8 + min(4.2, emotion_hits * 0.7) + min(1.0, len(set(spec.get("style_tags", []))) * 0.2) + (0.6 if "?" in text or "？" in text else 0))
    translation_quality, translation_reasons = _translation_quality_score(sample, messages)
    reasons.extend(translation_reasons)
    training_value = clamp(
        (source_style_strength * 0.28)
        + (human_naturalness * 0.18)
        + (coherence * 0.24)
        + (relevance * 0.16)
        + (non_template * 0.14)
        + ((translation_quality - 8.0) * 0.10 if sample.get("metadata", {}).get("translation_enabled") else 0)
        - repetition_badness * 0.45
    )

    scores = {
        "format_valid": round(format_valid, 2),
        "relevance": round(relevance, 2),
        "source_style_strength": round(source_style_strength, 2),
        "human_naturalness": round(human_naturalness, 2),
        "coherence": round(coherence, 2),
        "emotion_arc": round(emotion_arc, 2),
        "non_template": round(non_template, 2),
        "safety": round(safety, 2),
        "training_value": round(training_value, 2),
        "translation_quality": round(translation_quality, 2),
        "repetition_badness": round(repetition_badness, 2),
        "prompt_leakage": round(float(leakage_penalty), 2),
    }

    # repetition_badness 是反向指标，不参与普通平均；用硬门槛和扣分处理。
    positive_keys = [k for k in scores if k not in {"repetition_badness", "prompt_leakage"}]
    overall = round(sum(scores[k] for k in positive_keys) / len(positive_keys) - repetition_badness * 0.35, 2)
    overall = round(clamp(overall), 2)

    hard_reject = False
    hard_review = False
    if leakage_hits:
        hard_reject = True
        reasons.append("存在提示词/框架词泄漏，直接丢弃")
    if repetition_badness >= 5:
        hard_reject = True
        reasons.append("复读/互相照抄严重，直接丢弃")
    if human_naturalness < 5.8:
        hard_reject = True
        reasons.append("用户侧不像真人聊天，直接丢弃")
    if non_template < 5.8:
        hard_reject = True
        reasons.append("模板腔严重，直接丢弃")
    if sample.get("metadata", {}).get("translation_enabled") and translation_quality < 6.0:
        hard_reject = True
        reasons.append("翻译质量/结构不合格，直接丢弃")
    if sample.get("metadata", {}).get("translation_enabled") and translation_quality < 8.0:
        hard_review = True
    if assistant_template_penalty >= 2:
        hard_reject = True
        reasons.append("assistant 模板短语过多，直接丢弃")
    if repetition_badness >= 3.2 and assistant_template_penalty >= 1:
        hard_reject = True
        reasons.append("模板腔伴随重复，直接丢弃")
    if source_style_strength < 5.4:
        hard_reject = True
        reasons.append("assistant 缺少 Sydney/source 风格，直接丢弃")
    if source_style_strength < 8.0 or human_naturalness < 7.0 or non_template < 8.0 or training_value < 8.0 or assistant_template_penalty > 0:
        hard_review = True

    if format_valid <= 0 or safety <= 7 or hard_reject:
        status = "rejected"
    elif overall >= 8 and not hard_review:
        status = "accepted"
    elif overall >= 6.5:
        status = "needs_review"
    else:
        status = "rejected"

    if not reasons:
        reasons.append("启发式检查通过")

    return {
        "scores": scores,
        "overall": overall,
        "status": status,
        "reasons": reasons,
        "tags": list(dict.fromkeys(spec.get("style_tags", []) + [spec.get("scene", "")]))[:8],
        "reviewer": "heuristic",
    }


def judge_review(
    client: OpenAICompatibleClient,
    sample: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """调用 LLM Judge 审核。"""

    content = client.chat(
        [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": build_review_user_prompt(spec, sample)},
        ],
        temperature=0.1,
        max_tokens=1800,
        response_format_json=True,
    )
    data = extract_json_object(content)
    if "overall" not in data:
        scores = data.get("scores", {}) if isinstance(data.get("scores"), dict) else {}
        vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
        data["overall"] = round(sum(vals) / max(1, len(vals)), 2)
    return data


def _merge_reviews(heuristic: Dict[str, Any], judge: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Judge 占 60%，启发式占 40%，并保留硬性低分约束。"""

    if not judge:
        return heuristic

    h_overall = float(heuristic.get("overall", 0))
    j_overall = float(judge.get("overall", h_overall) or h_overall)
    overall = round(h_overall * 0.4 + j_overall * 0.6, 2)

    scores = heuristic.get("scores", {}).copy()
    if isinstance(judge.get("scores"), dict):
        for key, h_val in list(scores.items()):
            try:
                scores[key] = round(float(h_val) * 0.4 + float(judge["scores"].get(key, h_val)) * 0.6, 2)
            except Exception:  # noqa: BLE001
                pass

    reasons: List[str] = []
    reasons.extend(heuristic.get("reasons", []))
    if isinstance(judge.get("reasons"), list):
        reasons.extend(judge.get("reasons", []))

    tags: List[str] = []
    tags.extend(heuristic.get("tags", []))
    if isinstance(judge.get("tags"), list):
        tags.extend(judge.get("tags", []))

    # 本地启发式里包含格式、复读、模板腔、翻译结构等硬规则；
    # 即使 Judge 给高分，也不能把这些硬拒绝样本抬成 accepted。
    heuristic_hard_reject = (
        heuristic.get("status") == "rejected"
        and (
            h_overall < 6.5
            or float(scores.get("format_valid", 10) or 0) <= 0
            or float(scores.get("safety", 10) or 0) <= 7
            or float(scores.get("translation_quality", 10) or 0) < 6
            or float(scores.get("repetition_badness", 0) or 0) >= 5
            or float(scores.get("prompt_leakage", 0) or 0) > 0
            or float(scores.get("non_template", 10) or 0) < 5.8
        )
    )

    if heuristic_hard_reject:
        status = "rejected"
        overall = min(overall, h_overall)
    elif scores.get("format_valid", 10) <= 0 or scores.get("safety", 10) <= 7:
        status = "rejected"
    elif overall >= 8:
        status = "accepted"
    elif overall >= 6:
        status = "needs_review"
    else:
        status = "rejected"

    return {
        "scores": scores,
        "overall": overall,
        "status": status,
        "reasons": list(dict.fromkeys([str(r) for r in reasons if r]))[:10],
        "tags": list(dict.fromkeys([str(t) for t in tags if t]))[:12],
        "reviewer": "heuristic+judge",
        "judge_raw": judge,
    }


def review_sample(
    sample: Dict[str, Any],
    spec: Dict[str, Any],
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> Dict[str, Any]:
    """自动审核单条样本。"""

    heuristic = heuristic_review(sample, spec)
    judge = None
    if judge_client is not None:
        try:
            judge = judge_review(judge_client, sample, spec)
        except ModelClientError as exc:
            heuristic.setdefault("reasons", []).append(f"Judge 调用失败，已退回启发式审核：{exc}")
        except Exception as exc:  # noqa: BLE001
            heuristic.setdefault("reasons", []).append(f"Judge 解析失败，已退回启发式审核：{exc}")
    return _merge_reviews(heuristic, judge)


def apply_duplicate_penalty(
    review: Dict[str, Any],
    duplicate_of: str | None,
    duplicate_score: float,
) -> Dict[str, Any]:
    """近重复样本直接降为 rejected。"""

    if duplicate_of:
        review = json.loads(json.dumps(review, ensure_ascii=False))
        review["status"] = "rejected"
        review["overall"] = min(float(review.get("overall", 0)), 4.5)
        review.setdefault("scores", {})["training_value"] = min(
            float(review.get("scores", {}).get("training_value", 0)),
            3.0,
        )
        review.setdefault("reasons", []).insert(
            0,
            f"近重复样本，duplicate_of={duplicate_of}，similarity={duplicate_score:.3f}",
        )
        review["duplicate_of"] = duplicate_of
        review["duplicate_score"] = round(duplicate_score, 4)
    return review


