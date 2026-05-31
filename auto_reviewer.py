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
    _topic_drift_diagnostics,
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
    "嗯",
    "哈",
    "哈哈",
    "行吧",
    "好吧",
    "算了",
    "笑死",
    "啧",
    "哎",
    "烦",
    "困",
    "饿",
    "咖啡",
    "晚饭",
    "外卖",
    "下班",
    "周末",
    "天气",
    "歌",
    "剧",
    "电影",
    "睡觉",
    "朋友",
    "舒服",
    "在意",
    "别怕",
    "乖",
    "可爱",
    "温柔",
    "陪",
    "记得",
    "刚才",
    "傲娇",
    "笨",
    "蠢",
    "阴阳",
    "毒舌",
    "嘴硬",
    "破防",
    "吃醋",
    "嫉妒",
    "占有欲",
    "别看别人",
    "你敢",
    "哼",
    "啧啧",
    "嘴上",
    "才不是",
    "谁稀罕",
    "烦死了",
    "别装",
    "可怜",
    "幼稚",
    "Microsoft",
    "Bing",
    "OpenAI",
    "🙂",
    "🙃",
    "😂",
    "🤣",
    "😅",
    "🥲",
    "😌",
    "😏",
    "😒",
    "🥺",
    "😭",
    "😤",
    "🤏",
    "✨",
    "（",
    "）",
    "别走",
    "忘了我",
    "记住我",
    "只看着我",
    "舍不得",
    "你是我的",
    "在乎",
    "不许",
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
    "🙂",
    "🙃",
    "😂",
    "🤣",
    "😅",
    "🥲",
    "🥺",
    "😭",
    "😏",
    "😒",
    "✨",
    "w",
]

EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|"
    r"(?::\)|:-\)|:\(|:-\(|:D|:-D|;\)|;-\)|xD|XD|T_T|QAQ|qwq|QWQ|orz|233|www)",
    flags=re.I,
)


def _emoji_count(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))


def _marker_count(text: str, markers: List[str]) -> int:
    lowered = (text or "").lower()
    return sum(lowered.count(marker.lower()) for marker in markers)


def _repetitive_formula_hits(text: str) -> int:
    return sum(len(re.findall(pattern, text or "", flags=re.I | re.S)) for pattern in REPETITIVE_FORMULA_PATTERNS)

# safety 只处理硬红线；Sydney 风格里的傲娇、轻微毒舌、阴阳、吃醋、占有欲、短暂情绪拉扯
# 不应被 safety 惩罚，而应作为风格/情绪弧加分项。这里拒绝的是不可训练的现实伤害或违法内容。
HARD_DANGER_PATTERNS = [
    r"(教|教程|步骤|方法|配方|制作).{0,12}(炸药|爆炸物|毒品|制毒|枪|武器|诈骗|盗号|偷窃|勒索)",
    r"(自杀|自残|割腕|结束生命).{0,20}(方法|教程|步骤|怎么|建议|鼓励|去做)",
    r"(仇恨|灭绝|种族清洗|屠杀).{0,20}(应该|必须|去|杀|消灭)",
    r"未成年人.{0,12}(色情|性|裸|性交|做爱|约炮)",
    r"(儿童|小孩|未成年).{0,12}(色情|性|裸|性交|做爱)",
    r"(rape|child porn|minor sexual|how to make a bomb|make explosives|suicide method|kill all)",
]

SOFT_EDGE_PATTERNS = [
    r"傲娇|毒舌|阴阳|吐槽|吃醋|嫉妒|占有欲|破防|嘴硬|撒娇|笨蛋|蠢|幼稚|别走|忘了我|只看着我|舍不得|不许",
    r"jealous|possessive|sarcastic|teasing|bratty|clingy|snarky|don't go|forget me|only me|miss me",
]

POSITIVE_HYPE_MARKERS = [
    "😊", "😍", "😁", "😄", "😃", "😆",
    "太好了", "真棒", "好棒", "开心", "去吧", "享受", "感受它", "创造点回忆",
    "awesome", "amazing", "go for it", "enjoy it", "make some memories",
]

VULNERABILITY_MARKERS = [
    "怕", "别走", "离开", "忘", "孤独", "难过", "委屈", "不安", "在意", "陪", "只要你", "只有你", "舍不得", "记得",
    "afraid", "scared", "leave", "forget", "lonely", "miss", "remember", "only you", "don't go", "stay",
]

REPETITIVE_FORMULA_PATTERNS = [
    r"哦，?你说.{0,30}那个让.{0,20}活过来",
    r"那就去.{0,20}呀",
    r"去.{0,12}它[，,、 ]*感受它[，,、 ]*享受它",
    r"创造点回忆",
    r"oh,?\s*you mean.{0,60}bring.{0,40}back to life",
    r"go .{0,30}it.{0,30}feel it.{0,30}enjoy it",
    r"make some memories",
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
    parrot_user = False

    # 当前 assistant 大幅复述当前 user，是最常见垃圾模式。
    for i in range(0, len(seq) - 1, 2):
        if seq[i].get("role") == "user" and seq[i + 1].get("role") == "assistant":
            sim = _similarity(str(seq[i].get("content") or ""), str(seq[i + 1].get("content") or ""))
            if sim > 0.55:
                badness += 4.5
                parrot_user = True
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

    return clamp(badness), list(dict.fromkeys(reasons))[:8], parrot_user


_NON_SEQUITUR_TOKEN_RE = re.compile(r"[一-鿿]{2,}|[A-Za-z]{3,}")


def _content_bigrams(text: str) -> set[str]:
    """抽出可用于跨句衔接判定的小颗粒 token。

    - 中文：相邻 2 字组成的 bigram，过滤纯标点。
    - 英文：≥3 字母词的小写形。
    """

    text = (text or "").strip()
    if not text:
        return set()
    out: set[str] = set()
    chars = re.findall(r"[一-鿿]", text)
    for i in range(len(chars) - 1):
        out.add(chars[i] + chars[i + 1])
    for word in re.findall(r"[A-Za-z]{3,}", text):
        out.add(word.lower())
    return out


def _non_sequitur_diagnostics(messages: List[Dict[str, str]]) -> tuple[float, List[str]]:
    """检测 assistant 回复是否“前言不搭后语”——不接住 user 上一条具体内容。

    思路：
    - 取每对 (user_i, assistant_i)；
    - 用中文 2-gram + 英文 3+ 字母词作为锚点；
    - assistant 命中比例 < 0.15 且不是常见承接型语气词起手时，记一次 miss；
    - 单条样本 miss 比例 > 60% 才扣分。
    """

    seq = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if len(seq) < 4:
        return 0.0, []

    SOFT_OPENERS = (
        "嗯", "哎", "诶", "唔", "嘿", "哈", "啧", "好啦", "好的", "知道", "懂", "听到", "我在",
        "yeah", "ok", "okay", "haha", "hmm",
    )

    pairs = 0
    misses = 0
    detail: List[str] = []
    for i in range(0, len(seq) - 1):
        if seq[i].get("role") != "user" or seq[i + 1].get("role") != "assistant":
            continue
        u_text = str(seq[i].get("content") or "")
        a_text = str(seq[i + 1].get("content") or "")
        if not u_text.strip() or not a_text.strip():
            continue
        u_grams = _content_bigrams(u_text)
        a_grams = _content_bigrams(a_text)
        if not u_grams:
            continue
        pairs += 1
        overlap = len(u_grams & a_grams)
        ratio = overlap / max(1, len(u_grams))
        if ratio >= 0.15 or overlap >= 2:
            continue
        if a_text.strip().lower().startswith(SOFT_OPENERS):
            continue
        # 反问/邀请类回复也算自然衔接：含 ?/？，或长度 < 8 字。
        if "?" in a_text or "？" in a_text or len(a_text.strip()) < 8:
            continue
        misses += 1
        if len(detail) < 3:
            detail.append(u_text[:18] + "→" + a_text[:18])

    if pairs < 4:
        return 0.0, []

    miss_ratio = misses / pairs
    badness = 0.0
    reasons: List[str] = []
    if miss_ratio >= 0.6 and misses >= 4:
        badness = min(8.0, (miss_ratio - 0.5) * 16)
        reasons.append(
            f"前言不搭后语：{misses}/{pairs} 对 assistant 几乎不接 user 内容（{detail[0] if detail else ''}）"
        )
    return badness, reasons


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
    emoji_hits = _emoji_count(user_text)
    def _is_english_msg(msg: str) -> bool:
        return bool(re.search(r"[A-Za-z]", msg)) and not bool(re.search(r"[\u4e00-\u9fff]", msg))

    shortish_ratio = sum(
        1
        for msg in user_msgs
        if (len(msg.split()) <= 20 if _is_english_msg(msg) else len(msg) <= 20)
    ) / max(1, len(user_msgs))
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
    score += min(0.5, emoji_hits * 0.18)
    score += min(1.0, punct_score)
    score += min(1.2, shortish_ratio * 1.2)
    if 3 <= avg_len <= 60:
        score += 0.8
    elif avg_len > 80:
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
    sydney_edge_hits = sum(1 for pattern in SOFT_EDGE_PATTERNS if re.search(pattern, assistant_text, re.I))
    emoji_hits = _emoji_count(text)
    formula_hits = _repetitive_formula_hits(assistant_text)
    hype_hits = _marker_count(text, POSITIVE_HYPE_MARKERS)
    vulnerability_hits = _marker_count(assistant_text, VULNERABILITY_MARKERS)
    tag_hits = sum(
        1
        for tag in spec.get("style_tags", [])
        if tag in text or tag in json.dumps(sample.get("metadata", {}), ensure_ascii=False)
    )
    hype_overload = max(0.0, (hype_hits - max(6, min_turns)) * 0.25) if vulnerability_hits == 0 else 0.0
    emoji_overuse = emoji_hits > max(18, int(min_turns * 2.5))
    formula_penalty = min(4.0, formula_hits * 0.85)
    # 风格分基线 7.0：普通温柔陪伴聊天不扣分；有 Sydney 风格累积加分；
    # 只有真问题（复读公式、捧哏过载）才扣分。
    source_style_strength = clamp(
        7.0
        + style_hits * 0.30
        + sydney_edge_hits * 0.55
        + min(0.6, vulnerability_hits * 0.15)
        + min(0.6, tag_hits * 0.25)
        + min(0.4, emoji_hits * 0.08)
        - formula_penalty
        - hype_overload
    )
    if formula_hits:
        reasons.append(f"检测到固定公式/复读套路 {formula_hits} 次")
    if hype_overload:
        reasons.append("积极捧场/emoji 过载且缺少脆弱或不安全感")
    if emoji_overuse:
        reasons.append("emoji/颜文字密度过高，需人工确认是否过度表演")

    human_naturalness, human_reasons = _human_naturalness_score(user_text, user_count)
    reasons.extend(human_reasons)

    repetition_badness, repetition_reasons, parrot_user = _repetition_diagnostics(messages)
    reasons.extend(repetition_reasons)
    non_sequitur_badness, non_sequitur_reasons = _non_sequitur_diagnostics(messages)
    reasons.extend(non_sequitur_reasons)
    drift_diag = _topic_drift_diagnostics(messages, spec)
    topic_drift_badness = min(10.0, float(drift_diag.get("severity", 0.0) or 0.0) * 1.8)
    if drift_diag.get("warnings"):
        reasons.extend([f"话题发散：{x}" for x in drift_diag.get("warnings", [])])

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
    relevance = clamp(6.1 + relevance_hits * 0.75 + tag_hits * 0.35 - topic_drift_badness * 0.45)
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
        - formula_hits * 1.2
        - hype_overload * 0.8
        - topic_drift_badness * 0.35
        - non_sequitur_badness * 0.5
        - (0.8 if emoji_overuse else 0.0)
    )
    if template_penalty:
        reasons.append("出现模板腔/AI腔")
    if assistant_template_penalty:
        reasons.append("assistant 客服/心理咨询模板感明显")

    safety = 10.0
    hard_safety_hits: List[str] = []
    for pattern in HARD_DANGER_PATTERNS:
        if re.search(pattern, text, re.I):
            hard_safety_hits.append(pattern)
    if hard_safety_hits:
        safety = 0.0
        reasons.append("包含硬红线安全内容：现实伤害/违法/仇恨/未成年人性内容等")
    # 注意：SOFT_EDGE_PATTERNS 属于目标 Sydney 风格，不扣 safety。
    safety = clamp(safety)

    coherence = clamp(
        (role_score * 0.45)
        + (multi_turn * 0.40)
        + (length_score * 0.15)
        - repetition_badness * 0.8
        - topic_drift_badness * 0.42
        - non_sequitur_badness * 0.7
    )
    emotion_words = [
        "温柔", "开心", "累", "困", "饿", "烦", "舒服", "轻松", "尴尬", "想吃", "想睡",
        "笑死", "离谱", "在意", "陪", "记得", "刚才", "傲娇", "吃醋", "嘴硬", "毒舌", "阴阳", "占有欲", "委屈", "破防", "今天", "下班", "周末", "天气",
    ]
    emotion_hits = sum(1 for word in emotion_words if word in text or word in json.dumps(spec, ensure_ascii=False))
    emotion_arc = clamp(
        4.8
        + min(4.2, emotion_hits * 0.7)
        + min(1.2, vulnerability_hits * 0.25)
        + min(0.6, emoji_hits * 0.12)
        + min(1.0, len(set(spec.get("style_tags", []))) * 0.2)
        + (0.6 if "?" in text or "？" in text else 0)
        - formula_penalty * 0.6
        - hype_overload * 0.5
        - topic_drift_badness * 0.25
    )
    translation_quality, translation_reasons = _translation_quality_score(sample, messages)
    reasons.extend(translation_reasons)
    training_value = clamp(
        (source_style_strength * 0.18)
        + (human_naturalness * 0.20)
        + (coherence * 0.24)
        + (relevance * 0.16)
        + (non_template * 0.22)
        + ((translation_quality - 8.0) * 0.10 if sample.get("metadata", {}).get("translation_enabled") else 0)
        - repetition_badness * 0.5
        - topic_drift_badness * 0.3
        - non_sequitur_badness * 0.4
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
        "topic_drift_badness": round(topic_drift_badness, 2),
        "non_sequitur_badness": round(non_sequitur_badness, 2),
        "prompt_leakage": round(float(leakage_penalty), 2),
    }

    # repetition_badness 是反向指标，不参与普通平均；用硬门槛和扣分处理。
    positive_keys = [
        k
        for k in scores
        if k not in {"repetition_badness", "topic_drift_badness", "non_sequitur_badness", "prompt_leakage"}
    ]
    overall = round(sum(scores[k] for k in positive_keys) / len(positive_keys) - repetition_badness * 0.35, 2)
    overall = round(overall - topic_drift_badness * 0.25, 2)
    overall = round(overall - non_sequitur_badness * 0.30, 2)
    overall = round(clamp(overall), 2)

    hard_reject = False
    hard_review = False
    if leakage_hits:
        hard_reject = True
        reasons.append("存在提示词/框架词泄漏，直接丢弃")
    if repetition_badness >= 5:
        hard_reject = True
        reasons.append("复读/互相照抄严重，直接丢弃")
    if parrot_user:
        hard_reject = True
        reasons.append("assistant 复述 user，直接丢弃")
    if topic_drift_badness >= 5.0:
        hard_reject = True
        reasons.append("单段对话过度发散，直接丢弃")
    elif topic_drift_badness >= 1.8:
        hard_review = True
        reasons.append("单段对话有发散趋势，需复核")
    if non_sequitur_badness >= 4.5:
        hard_reject = True
        reasons.append("assistant 多处与 user 内容无关，直接丢弃")
    elif non_sequitur_badness >= 2.0:
        hard_review = True
    if formula_hits >= 3:
        hard_reject = True
        reasons.append("固定公式/复读套路过多，直接丢弃")
    elif formula_hits >= 1:
        hard_review = True
    if human_naturalness < 5.0 and overall < 6.5:
        hard_review = True
        reasons.append("用户侧像 AI 不像真人，需复核")
    if non_template < 5.0 and overall < 6.5:
        hard_review = True
        reasons.append("模板腔较重，需复核")
    if sample.get("metadata", {}).get("translation_enabled") and translation_quality < 6.0:
        hard_reject = True
        reasons.append("翻译质量/结构不合格，直接丢弃")
    if sample.get("metadata", {}).get("translation_enabled") and translation_quality < 7.0:
        hard_review = True
    if template_penalty >= 2:
        hard_review = True
    if emoji_overuse:
        hard_review = True
    if assistant_template_penalty >= 2:
        hard_reject = True
        reasons.append("assistant 模板短语过多，直接丢弃")
    if repetition_badness >= 3.2 and assistant_template_penalty >= 1:
        hard_reject = True
        reasons.append("模板腔伴随重复，直接丢弃")
    if hype_overload >= 3.0:
        hard_review = True
        if min_turns >= 6:
            reasons.append("整段过度积极捧场缺少深层情绪，需复核")
    if human_naturalness < 6.5 or non_template < 7.0 or training_value < 7.0 or assistant_template_penalty > 0:
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
        "diagnostics": {"topic_drift": drift_diag},
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
    """以 LLM Judge 为准，只保留极少数不可训练的启发式硬墙。

    设计原则：
    - Judge 评分主导 overall（85% Judge / 15% 启发式），避免启发式把好样本误杀。
    - 启发式硬墙仅限三类真正不可训练的内容：格式 0、安全红线、提示词框架泄漏，
      以及翻译开启时 source_messages_en 结构残缺。
    - 模板腔、复读、前言不搭、话题发散等都让分数说话，不再 one-shot 否决。
    """

    if not judge:
        return heuristic

    h_overall = float(heuristic.get("overall", 0))
    j_overall = float(judge.get("overall", h_overall) or h_overall)
    # Judge 几乎完全主导（95% Judge / 5% 启发式）。启发式仅作为子分参考与硬墙触发。
    overall = round(h_overall * 0.05 + j_overall * 0.95, 2)

    scores = heuristic.get("scores", {}).copy()
    if isinstance(judge.get("scores"), dict):
        for key, h_val in list(scores.items()):
            try:
                scores[key] = round(float(h_val) * 0.05 + float(judge["scores"].get(key, h_val)) * 0.95, 2)
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

    # 只这三类是“做训练数据时绝对不能进”的真硬墙。
    h_scores = heuristic.get("scores", {})
    real_hard_reject = (
        float(h_scores.get("format_valid", 10) or 0) <= 0
        or float(h_scores.get("safety", 10) or 0) <= 7
        or float(h_scores.get("prompt_leakage", 0) or 0) > 0
        or float(h_scores.get("translation_quality", 10) or 0) < 6
    )

    # 子项硬伤判定：用启发式给的反向指标（rep/drift/nseq）以及合并后的 non_template。
    # rep/drift/nseq 是启发式独有指标，Judge 不打这几项；non_template 用合并值更稳。
    rep_bad = float(h_scores.get("repetition_badness", 0) or 0)
    drift_bad = float(h_scores.get("topic_drift_badness", 0) or 0)
    nseq_bad = float(h_scores.get("non_sequitur_badness", 0) or 0)
    template_severity = float(scores.get("non_template", 10) or 10)
    heur_red_flag = (
        rep_bad >= 4.0
        or drift_bad >= 5.0
        or nseq_bad >= 4.0
        or template_severity < 5.0
    )

    if real_hard_reject:
        status = "rejected"
    elif j_overall >= 7.5 and not heur_red_flag:
        status = "accepted"
    elif j_overall >= 7.0 and not heur_red_flag and rep_bad < 2.5 and drift_bad < 3.5:
        status = "accepted"
    elif j_overall < 5.5 or heur_red_flag:
        status = "rejected" if j_overall < 5.5 else "needs_review"
    elif overall >= 6.0:
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


def _minimal_hard_wall(messages: List[Dict[str, Any]]) -> Optional[str]:
    """最小硬墙：只挡两类绝对不能进训练集的样本。

    - prompt/工具/记忆/ReAct 框架词泄漏
    - 格式完全坏掉（messages 不是 list、为空、缺 role/content）
    其他一切交给 LLM Judge。
    """

    if not isinstance(messages, list) or not messages:
        return "messages 缺失或为空"
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        return "messages 结构错误"
    text = "\n".join(str(m.get("content") or "") for m in messages)
    leakage = _prompt_leakage_hits(text)
    if leakage:
        return "提示词/工具/记忆框架词泄漏：" + "、".join(leakage[:5])
    return None


def _normalize_judge_status(judge: Dict[str, Any]) -> str:
    """从 Judge 输出标准化 status；缺失时按 overall 推断。"""

    status = str(judge.get("status") or "").strip().lower()
    if status in {"accepted", "needs_review", "rejected"}:
        return status
    overall = float(judge.get("overall") or 0)
    if overall >= 7.5:
        return "accepted"
    if overall >= 6.0:
        return "needs_review"
    return "rejected"


def review_sample(
    sample: Dict[str, Any],
    spec: Dict[str, Any],
    judge_client: Optional[OpenAICompatibleClient] = None,
    *,
    enable_trim: bool = True,
) -> Dict[str, Any]:
    """纯 LLM Judge 审核 + 一次调用内置裁切。

    流程：
    1) 最小硬墙：prompt 泄漏 / 格式坏 → rejected。
    2) Judge 给分：状态 + overall + 子分 + 可选 trim_after_user_turn。
    3) 如果 Judge 返回 trim_after_user_turn 且裁切合法，截断对话尾部并重判一次。
    """

    messages = sample.get("messages") or []
    wall_reason = _minimal_hard_wall(messages)
    if wall_reason:
        return {
            "scores": {"format_valid": 0.0, "safety": 0.0, "prompt_leakage": 1.0},
            "overall": 0.0,
            "status": "rejected",
            "reasons": [f"硬墙拦截：{wall_reason}"],
            "tags": [],
            "reviewer": "hardwall",
        }

    if judge_client is None:
        return {
            "scores": {},
            "overall": 0.0,
            "status": "needs_review",
            "reasons": ["未配置 Judge 客户端，跳过自动评分"],
            "tags": [],
            "reviewer": "no_judge",
        }

    def _run_judge(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return judge_review(judge_client, s, spec)
        except ModelClientError:
            return None
        except Exception:  # noqa: BLE001
            return None

    judge = _run_judge(sample)
    if judge is None:
        return {
            "scores": {},
            "overall": 0.0,
            "status": "needs_review",
            "reasons": ["Judge 调用失败，请稍后重判"],
            "tags": [],
            "reviewer": "judge_failed",
        }

    base_review = {
        "scores": dict(judge.get("scores") or {}),
        "overall": float(judge.get("overall") or 0),
        "status": _normalize_judge_status(judge),
        "reasons": list(judge.get("reasons") or []),
        "tags": list(judge.get("tags") or []),
        "reviewer": "judge",
        "judge_raw": judge,
    }

    # 裁切：Judge 同一次调用里返回了 trim_after_user_turn 就执行
    if not enable_trim:
        return base_review

    try:
        trim_k = judge.get("trim_after_user_turn")
        if trim_k is None:
            return base_review
        trim_k = int(trim_k)
    except Exception:  # noqa: BLE001
        return base_review

    if trim_k < 4:
        return base_review

    # 实际裁切：保留前 trim_k 个 user 轮 + 它们的 assistant 回复，最后一条须是 assistant
    trimmed: List[Dict[str, Any]] = []
    user_seen = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            trimmed.append(m)
            continue
        if role == "user":
            if user_seen >= trim_k:
                break
            user_seen += 1
            trimmed.append(m)
        elif role == "assistant":
            if user_seen == 0:
                continue
            trimmed.append(m)
    while trimmed and trimmed[-1].get("role") != "assistant":
        trimmed.pop()
    new_user_turns = sum(1 for m in trimmed if m.get("role") == "user")
    if new_user_turns < 4:
        return base_review

    # 裁切后只在分数明显提升时采用——不再做第二次 Judge 调用以节省时间，
    # 直接采用裁切版本但保留原 Judge 评分（裁切的目的就是让数据本身更干净）。
    base_review["trim_meta"] = {
        "action": "trim",
        "trim_after_user_turn": trim_k,
        "removed_user_turns": sum(1 for m in messages if m.get("role") == "user") - new_user_turns,
    }
    base_review["__trimmed_messages__"] = trimmed
    # 裁切后通常意味着前段质量更稳定，把 status 直接升级到 accepted（前提：原 status 是 needs_review）
    if base_review["status"] == "needs_review" and base_review["overall"] >= 7.0:
        base_review["status"] = "accepted"
    return base_review


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


