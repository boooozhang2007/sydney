"""对话裁切：找出"前段漂亮，后段崩坏"的样本，从崩坏点截断。

只在样本被审核为 needs_review/rejected 但 LLM 检测到前段质量良好时启用。
裁切只能在某条 user 轮**之后**截断，必须保证最终 transcript 的最后一条是 assistant，
且至少保留 4 轮 (4 user + 4 assistant) 才输出。否则放弃裁切。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from data_generator import OpenAICompatibleClient, ModelClientError, extract_json_object


TRIM_SYSTEM_PROMPT = """你是对话质量裁剪器。

输入是一段多轮中文/英文 user/assistant 对话。某些样本前半段非常漂亮，后半段开始出现复述、模板腔、话题串烧或公式化句式。
你的任务：判断是否需要从某个点截断对话，以便保留高质量的前半段作为训练样本。

裁剪规则：
- 只能在第 K 轮 user 消息之后截断（K>=4），且截断后最后一条必须是 assistant。
- 截断点要选在"崩坏开始"之前——即 K 之后的 assistant 回复出现以下任意问题：
  * 大幅复述上一条 user 的话（不是引用，是直接重复词组）
  * 重复使用同一句式或同一表情堆叠（如连续 5 条 assistant 都以"我..."开头）
  * 反复使用同一公式（"去 XX 它，感受它，享受它"等）
  * 话题突然跳转到与前文无关的领域，且后续都在新领域里打转
  * 出现明显客服腔/AI 自曝（"作为 AI..."）
- 如果整段对话质量都很好或都很差，不要截断；返回 action=keep 或 action=reject。
- 如果前 4 轮就已经崩坏，返回 action=reject。
- 截断点必须真的让被丢弃的部分明显比保留的部分差。如果只是"略有重复"，不要截断，保持 keep。

只输出一个合法 JSON：
{
  "action": "keep" | "trim" | "reject",
  "trim_after_user_turn": 整数K (action=trim 时必填，K 是从 1 开始的 user 轮序号),
  "reason": "简短中文理由（≤30 字）"
}
"""


def _format_transcript_for_trim(messages: List[Dict[str, str]]) -> str:
    """把 messages 渲染成带 user 轮序号的可读 transcript。"""

    lines: List[str] = []
    user_idx = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        content = str(m.get("content") or "").strip().replace("\n", " ")
        if role == "user":
            user_idx += 1
            lines.append(f"[user#{user_idx}] {content}")
        elif role == "assistant":
            lines.append(f"[assistant<-#{user_idx}] {content}")
    return "\n".join(lines)


def _trim_at_user_turn(messages: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    """保留前 k 个 user 轮 + 它们对应的 assistant 回复。"""

    out: List[Dict[str, str]] = []
    user_seen = 0
    last_role = None
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append(m)
            continue
        if role == "user":
            if user_seen >= k:
                break
            user_seen += 1
            out.append(m)
            last_role = "user"
        elif role == "assistant":
            if user_seen == 0:
                continue
            out.append(m)
            last_role = "assistant"
    # 必须以 assistant 结尾
    if last_role != "assistant":
        while out and out[-1].get("role") != "assistant":
            out.pop()
    return out


def trim_dialogue_at_drop_point(
    messages: List[Dict[str, str]],
    client: OpenAICompatibleClient,
    *,
    min_user_turns_after_trim: int = 4,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """让 LLM 决定是否裁切，返回 (新 messages 或原 messages, 元信息)。

    元信息形如：
    {"action": "keep"|"trim"|"reject", "trim_after_user_turn": K, "reason": "..."}
    """

    user_turns = sum(1 for m in messages if m.get("role") == "user")
    if user_turns < min_user_turns_after_trim + 2:
        return messages, {"action": "keep", "reason": "对话太短不裁切"}

    transcript = _format_transcript_for_trim(messages)
    try:
        content = client.chat(
            [
                {"role": "system", "content": TRIM_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format_json=True,
        )
        data = extract_json_object(content)
    except (ModelClientError, Exception) as exc:  # noqa: BLE001
        return messages, {"action": "keep", "reason": f"裁切判断失败：{exc}"[:80]}

    action = str(data.get("action") or "keep").lower()
    if action == "trim":
        try:
            k = int(data.get("trim_after_user_turn") or 0)
        except Exception:  # noqa: BLE001
            k = 0
        if k < min_user_turns_after_trim:
            return messages, {"action": "keep", "reason": f"裁切点 K={k} 太靠前，放弃裁切"}
        if k >= user_turns:
            return messages, {"action": "keep", "reason": "裁切点超出范围，等同 keep"}
        trimmed = _trim_at_user_turn(messages, k)
        new_user_turns = sum(1 for m in trimmed if m.get("role") == "user")
        if new_user_turns < min_user_turns_after_trim:
            return messages, {"action": "keep", "reason": "裁切后剩余过少"}
        return trimmed, {
            "action": "trim",
            "trim_after_user_turn": k,
            "reason": str(data.get("reason") or ""),
            "removed_user_turns": user_turns - new_user_turns,
        }
    if action == "reject":
        return messages, {"action": "reject", "reason": str(data.get("reason") or "整段质量不足")}
    return messages, {"action": "keep", "reason": str(data.get("reason") or "")}
