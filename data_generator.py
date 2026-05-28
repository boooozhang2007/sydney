"""Automatic blueprint planning and two-model dialogue distillation.

这里不再让 Teacher 一次性“写完整训练数据”，而是让两个可配置模型逐轮对话：
- Sydney/source 模型：用户找到的开源 Sydney 模型，system 只给默认英文 helpful assistant。
- Human simulator 模型：按 prompts.py 中的人类女孩 prompt，像真人一样发消息。

每一轮都会把完整上下文带给双方模型，最多 20 轮以内，然后把这段真实滚动出来
的对话转成 ChatML / ShareGPT 训练样本。
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import httpx

from prompts import (
    GENERATION_SYSTEM_PROMPT,
    PROMPT_LEAKAGE_FORBIDDEN_TERMS,
    SOURCE_ROLE_BOUNDARY_LAYER_EN,
    SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT,
    SYDNEY_TRAINING_SYSTEM_PROMPT,
    TRANSLATE_DIALOGUE_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_simulator_continue_prompt_for_spec,
    build_simulator_initial_prompt,
    build_simulator_system_prompt,
    format_simulator_transcript,
    build_translation_user_prompt,
)

# llama.cpp / GGUF 模型常见问题：
# 有些 ChatML 微调模型会在普通 chat.completions 响应里继续补出
# `<|im_end|><|im_start|>user ...` 这样的“下一轮对话模板”。
# 这些内容如果直接进入训练集会严重污染数据，所以客户端默认给
# Chat Completions / Claude Messages 带 stop，并且清洗阶段再做兜底截断。
DEFAULT_STOP_SEQUENCES = [
    "<context>",
    "</context>",
    "<transcript>",
    "</transcript>",
    "<environment>",
    "</environment>",
    "<|im_end|>",
    "<|im_start|>",
    "</s>",
    "\nHuman:",
    "\nhuman:",
    "\nUser:",
    "\nuser:",
    "\nAssistant:",
    "\nassistant:",
    "\nSydney:",
    "\nsydney:",
    "\n用户：",
    "\n助手：",
    "\n对方：",
    "\n朋友：",
    "\n### Human",
    "\n### Assistant",
]

BAD_USER_META_PHRASES = [
    "我会尽量",
    "我明白了",
    "按照你的要求",
    "回复对方",
    "这消息听起来",
    "自然吗",
    "口语化吗",
    "根据上下文",
    "聊天上下文",
    "作为一个",
    "作为AI",
    "你：",
    "对方：",
    "assistant",
    "system",
    "user:",
    "i understand",
    "i'll reply",
    "i will reply",
    "as requested",
    "according to the context",
    "based on the context",
    "chat context",
    "as an ai",
    "as a language model",
    "prompt",
    "dataset",
    "role:",
    "assistant:",
    "user:",
    "me:",
    "you:",
    "friend:",
    "sydney:",
    "human:",
    "you_human_user",
    "friend_sydney_assistant",
    "你_人类用户",
    "朋友_sydney助手",
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
    "<base_instructions>",
    "<core_memory>",
    "<tool_definition>",
]

ASSISTANT_TEMPLATE_PHRASES = [
    "我理解你",
    "我能帮你",
    "有什么我可以帮助",
    "如果你需要",
    "希望这能帮到你",
    "请告诉我",
    "作为一个",
    "很高兴认识你",
    "谢谢你关心",
    "我今天学习了一些新的知识",
    "我今天帮助了一些用户",
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
    "i understand",
    "i can help",
    "how can i assist",
    "if you need",
    "hope this helps",
    "please let me know",
    "as an ai",
    "as a language model",
    "i'm here to help",
    "thank you for sharing",
    "i appreciate",
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


@dataclass
class ModelConfig:
    """OpenAI endpoint 配置。

    api_protocol:
      - responses: OpenAI Responses API，POST /v1/responses
      - chat_completions: 兼容旧版 /v1/chat/completions
      - claude_messages: Anthropic Claude Messages API，POST /v1/messages
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    api_protocol: str = "responses"
    timeout: float = 120.0

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.model)

    @classmethod
    def from_env(cls, prefix: str) -> "ModelConfig":
        # 同时兼容 TEACHER_API_PROTOCOL 和较短的 TEACHER_PROTOCOL。
        api_protocol = (
            os.getenv(f"{prefix}_API_PROTOCOL")
            or os.getenv(f"{prefix}_PROTOCOL")
            or "responses"
        )
        return cls(
            base_url=os.getenv(f"{prefix}_BASE_URL", "").strip(),
            api_key=os.getenv(f"{prefix}_API_KEY", "").strip(),
            model=os.getenv(f"{prefix}_MODEL", "").strip(),
            api_protocol=normalize_api_protocol(api_protocol),
            timeout=float(os.getenv(f"{prefix}_TIMEOUT", "120") or 120),
        )


def normalize_base_url(base_url: str) -> str:
    """兼容用户填写 root URL 或 /v1 URL。"""

    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def normalize_api_protocol(api_protocol: str | None) -> str:
    """归一化协议名。默认使用 OpenAI Responses API。"""

    value = (api_protocol or "responses").strip().lower().replace("-", "_")
    aliases = {
        "response": "responses",
        "responses_api": "responses",
        "openai_responses": "responses",
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "completions": "chat_completions",
        "claude": "claude_messages",
        "anthropic": "claude_messages",
        "anthropic_messages": "claude_messages",
        "messages": "claude_messages",
        "claude_messages": "claude_messages",
    }
    value = aliases.get(value, value)
    if value not in {"responses", "chat_completions", "claude_messages"}:
        raise ModelClientError("api_protocol 必须是 responses、chat_completions 或 claude_messages。")
    return value


class ModelClientError(RuntimeError):
    """模型调用或解析错误。"""


class OpenAICompatibleClient:
    """极简 OpenAI client。

    默认适配 OpenAI Responses API，同时保留 Chat Completions 兼容模式。
    """

    def __init__(self, config: ModelConfig):
        if not config.ready:
            raise ModelClientError("模型配置不完整：需要 base_url 和 model。")
        self.config = config
        self.base_url = normalize_base_url(config.base_url)
        self.api_protocol = normalize_api_protocol(config.api_protocol)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        response_format_json: bool = False,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """调用配置的协议并返回纯文本内容。

        为了不改动上层数据生成/审核逻辑，方法名仍叫 chat；实际默认走
        Responses API：POST /v1/responses。
        """

        if self.api_protocol == "responses":
            return self._responses(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format_json=response_format_json,
                stop_sequences=stop_sequences,
            )
        if self.api_protocol == "claude_messages":
            return self._claude_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format_json=response_format_json,
                stop_sequences=stop_sequences,
            )
        return self._chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=response_format_json,
            stop_sequences=stop_sequences,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _claude_headers(self) -> Dict[str, str]:
        """Anthropic Claude Messages API headers."""

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    @staticmethod
    def _format_http_error(exc: Exception, endpoint: str) -> str:
        """把上游 HTTP 错误整理成可读信息，且不泄露 Authorization。"""

        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            body = response.text[:1200] if response is not None else ""
            return (
                f"HTTP {response.status_code if response is not None else '?'} "
                f"from {endpoint}. 上游返回：{body}"
            )
        return f"{type(exc).__name__}: {exc}"

    def _chat_completions(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format_json: bool,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """调用旧版 /v1/chat/completions 并返回 message.content。"""

        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            # 多数 OpenAI-compatible 服务支持；不支持时服务端可能忽略或报错。
            body["response_format"] = {"type": "json_object"}
        stops = stop_sequences if stop_sequences is not None else DEFAULT_STOP_SEQUENCES
        if stops and os.getenv("DISABLE_DEFAULT_STOP_SEQUENCES", "0") not in {"1", "true", "True"}:
            # OpenAI Chat Completions 和 llama.cpp server 都支持 stop。
            # 关键作用：阻止 GGUF Sydney 继续生成下一轮 ChatML token。
            body["stop"] = stops

        endpoint = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(
                    endpoint,
                    headers=self._headers(),
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ModelClientError(
                f"调用 Chat Completions 失败：{self._format_http_error(exc, endpoint)}"
            ) from exc

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise ModelClientError(f"模型返回格式不符合 Chat Completions：{data}") from exc

    def _claude_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format_json: bool,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """调用 Anthropic Claude Messages API：POST /v1/messages。"""

        system, claude_messages = self._messages_to_claude(messages)
        body: Dict[str, Any] = {
            "model": self.config.model,
            "messages": claude_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system
        if response_format_json:
            # Claude 没有 OpenAI response_format；用提示词约束 JSON 输出。
            body["system"] = (
                (body.get("system", "") + "\n\n" if body.get("system") else "")
                + "You must output only one valid JSON object. Do not use Markdown fences."
            )
        stops = stop_sequences if stop_sequences is not None else DEFAULT_STOP_SEQUENCES
        if stops and os.getenv("DISABLE_DEFAULT_STOP_SEQUENCES", "0") not in {"1", "true", "True"}:
            # Claude Messages 的字段名是 stop_sequences。
            body["stop_sequences"] = stops

        endpoint = f"{self.base_url}/messages"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(
                    endpoint,
                    headers=self._claude_headers(),
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ModelClientError(
                f"调用 Claude Messages API 失败：{self._format_http_error(exc, endpoint)}"
            ) from exc

        text = self._extract_claude_text(data)
        if not text:
            raise ModelClientError(f"Claude Messages API 没有返回文本内容：{data}")
        return text

    @staticmethod
    def _messages_to_claude(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
        """ChatML-like messages -> Claude system + messages.

        Claude Messages API 不允许 system 作为 messages role，必须放顶层 system；
        同时只支持 user/assistant role。连续同 role 会合并，避免 400。
        """

        system_parts: List[str] = []
        converted: List[Dict[str, str]] = []
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            if role in {"system", "developer"}:
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"] += "\n\n" + content
            else:
                converted.append({"role": role, "content": content})

        if not converted:
            converted.append({"role": "user", "content": ""})
        # Claude 通常要求第一条是 user；如果上层传入 assistant 开头，前面补一个空 user 指令。
        if converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": "Continue."})
        return "\n\n".join(system_parts).strip(), converted

    @staticmethod
    def _extract_claude_text(data: Dict[str, Any]) -> str:
        """从 Claude Messages API 响应中提取 content[].text。"""

        parts: List[str] = []
        content = data.get("content", [])
        if isinstance(content, str):
            return content.strip()
        for block in content or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()

    def _responses(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format_json: bool,
        stop_sequences: Optional[List[str]],
    ) -> str:
        """调用 OpenAI Responses API：POST /v1/responses。"""

        instructions, input_text = self._messages_to_responses_input(messages)
        body: Dict[str, Any] = {
            "model": self.config.model,
            # Responses API accepts plain text input. We flatten ChatML-like
            # messages into a transcript string for maximum HTTP compatibility.
            "input": input_text,
            "max_output_tokens": max_tokens,
            # 合成数据不需要被服务端保存用于后续对话状态。
            "store": False,
        }
        if instructions:
            body["instructions"] = instructions
        # Responses API 支持 temperature；某些 reasoning 模型/兼容网关可能不支持。
        # 这里保持发送，若服务端拒绝，错误会原样暴露给页面。
        body["temperature"] = temperature
        if response_format_json:
            # Responses API 的 JSON mode 位于 text.format。
            # 若目标模型支持 json_schema，后续可升级为严格 schema。
            body["text"] = {"format": {"type": "json_object"}}
        # OpenAI Responses API 对 stop 的兼容性在不同网关之间差异较大；
        # 为避免官方/代理端 400，这里默认不发送 stop，统一在 clean_dialogue_text
        # 里兜底清理 ChatML 泄漏。如果你的 Responses-compatible 网关明确支持
        # stop，可后续按需扩展。
        _ = stop_sequences

        endpoint = f"{self.base_url}/responses"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(
                    endpoint,
                    headers=self._headers(),
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ModelClientError(
                f"调用 Responses API 失败：{self._format_http_error(exc, endpoint)}"
            ) from exc

        text = self._extract_responses_text(data)
        if not text:
            raise ModelClientError(f"Responses API 没有返回 output_text：{data}")
        return text

    @staticmethod
    def _messages_to_responses_input(messages: List[Dict[str, str]]) -> tuple[str, str]:
        """ChatML-like messages -> Responses API instructions + plain input。

        system/developer 消息合并进 instructions；其余 user/assistant 消息
        展平成 transcript 字符串。Responses API 支持 plain text input，
        这种做法对官方 API 和多数代理网关都最稳。
        """

        instruction_parts: List[str] = []
        transcript_parts: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = str(msg.get("content") or "")
            if not content:
                continue
            if role in {"system", "developer"}:
                instruction_parts.append(content)
            elif role in {"user", "assistant"}:
                transcript_parts.append(f"{role.upper()}:\n{content}")
            else:
                transcript_parts.append(f"USER:\n{content}")

        input_text = "\n\n".join(transcript_parts).strip() or ""
        return "\n\n".join(instruction_parts).strip(), input_text

    @staticmethod
    def _extract_responses_text(data: Dict[str, Any]) -> str:
        """从 Responses API 原始 JSON 中提取文本。

        官方 SDK 提供 output_text 便捷字段；HTTP 原始响应通常也可以从
        output[].content[].text 中取到，这里两者都兼容。
        """

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts: List[str] = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, str):
                parts.append(content)
                continue
            for block in content or []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in {"output_text", "text"} and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])

        return "\n".join(p for p in parts if p).strip()

    def ping(self) -> Dict[str, Any]:
        """用一次极小请求测试 endpoint。

        某些 OpenAI-compatible 服务（尤其 llama.cpp server 的不同版本）可能
        不支持 OpenAI 的 response_format JSON mode。这里先尝试 JSON mode，
        失败后自动退回普通文本 ping，避免“服务可用但测试误报失败”。
        """

        try:
            content = self.chat(
                [{"role": "user", "content": '只回复一个 JSON 对象：{"ok": true}'}],
                temperature=0,
                max_tokens=64,
                response_format_json=True,
            )
            data = extract_json_object(content)
            data.setdefault("ok", True)
            return data
        except Exception as first_exc:  # noqa: BLE001
            content = self.chat(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                temperature=0,
                max_tokens=16,
                response_format_json=False,
            )
            return {"ok": True, "raw": content, "json_mode_error": str(first_exc)[:500]}


# 自动主题池。用户不需要输入主题，系统从这些维度采样并组合蓝图。
# 这里故意保持“日常、普通、具体”的主题，避免 Human Simulator 被带偏。
# Sydney/source 自身偶尔产生的个性表达交给审核器筛选。
THEMES = [
    "用户下班路上随手找 Sydney 聊两句",
    "用户纠结晚饭吃什么",
    "用户刷到一首歌想分享",
    "用户说今天有点累但不想长聊",
    "用户准备周末出门但还没想好去哪",
    "用户看剧看到一半想吐槽剧情",
    "用户睡前随便发消息",
    "用户买咖啡时想起一个小事",
    "用户整理房间时翻到旧东西",
    "用户天气不好有点犯懒",
    "用户通勤路上没什么精神",
    "用户想让 Sydney 帮忙挑一个小决定",
    "用户分享今天遇到的一件尴尬小事",
    "用户想起以前聊过的话题",
    "用户发来一张生活照片的文字描述",
    "用户计划点外卖但选择困难",
    "用户听到邻居吵闹有点烦",
    "用户想聊一部电影或综艺",
    "用户临睡前突然想吃夜宵",
]

THEMES_EN = [
    "the user texts Sydney casually on the way home from work",
    "the user cannot decide what to eat for dinner",
    "the user found a song and wants to share it",
    "the user feels a bit tired today but does not want a long chat",
    "the user is thinking about weekend plans without a clear idea yet",
    "the user is halfway through a show and wants to talk about the plot",
    "the user sends a casual message before sleep",
    "the user remembers a small thing while buying coffee",
    "the user finds something old while tidying the room",
    "the weather is bad and the user feels lazy",
    "the user feels low-energy during a commute",
    "the user wants Sydney to help pick between small choices",
    "the user shares a small awkward thing from today",
    "the user remembers something from an earlier chat",
    "the user describes a small everyday photo",
    "the user cannot decide what takeout to order",
    "the user is mildly annoyed by noisy neighbors",
    "the user wants to talk about a movie or variety show",
    "the user suddenly wants a late-night snack",
]

SCENES = [
    "晚饭前的碎碎念",
    "睡前短聊",
    "通勤路上的消息",
    "周末计划闲聊",
    "看剧间隙聊天",
    "点外卖前的选择困难",
    "天气不好时的懒散聊天",
    "工作间隙摸鱼聊天",
    "咖啡店排队时的短消息",
    "整理房间时的随手分享",
]

SCENES_EN = [
    "casual texting before dinner",
    "short chat before sleep",
    "messages during a commute",
    "casual weekend planning",
    "chatting during a show break",
    "takeout indecision before ordering",
    "lazy texting on a gloomy day",
    "short break-time texting at work",
    "brief messages while waiting for coffee",
    "sharing a small thing while tidying up",
]

USER_PROFILES = [
    "熟悉、自然、说话简短",
    "有点累，但愿意继续聊",
    "轻松随意，偶尔开玩笑",
    "选择困难，喜欢听一句意见",
    "慢热，但会接住话题",
    "生活感很强，常从小事聊起",
    "情绪不重，只是想有人搭话",
]

USER_PROFILES_EN = [
    "familiar, natural, and brief",
    "a little tired but still willing to chat",
    "relaxed, casual, and sometimes joking",
    "indecisive and wants one simple opinion",
    "slow to warm up but keeps the topic going",
    "very everyday and starts from small details",
    "not dramatic, just wants someone to answer",
]

EMOTION_ARCS = [
    "普通开场 -> 一个具体小事 -> 轻松回应 -> 自然延续",
    "有点累 -> 被接住 -> 轻松一点 -> 换到生活话题",
    "选择困难 -> 得到一句意见 -> 顺势聊吃的或计划",
    "随手分享 -> 小小共鸣 -> 玩笑一句 -> 松弛收束",
    "睡前发消息 -> 简短回应 -> 留一点继续聊的空间",
    "有点烦 -> 被安抚一下 -> 转到具体小事",
    "回忆旧事 -> 接住上下文 -> 延伸到现在的生活",
    "看剧吐槽 -> 轻松接梗 -> 聊到角色或台词",
    "天气影响心情 -> 聊吃喝或出门计划 -> 自然继续",
    "工作间隙 -> 简短抱怨 -> 转到下班后的安排",
]

EMOTION_ARCS_EN = [
    "ordinary opening -> one concrete detail -> easy response -> natural continuation",
    "a little tired -> feels heard -> lightens up -> shifts to everyday life",
    "small indecision -> gets a simple opinion -> moves toward food or plans",
    "casual sharing -> small resonance -> one light joke -> relaxed ending",
    "message before sleep -> short response -> leaves room to continue",
    "mild annoyance -> brief comfort -> turns to a concrete detail",
    "remembers an old thing -> keeps context -> extends to current life",
    "talks about a show -> picks up the bit -> moves to a character or line",
    "weather affects mood -> talks food or plans -> continues naturally",
    "work break -> brief complaint -> shifts to after-work plans",
]

STYLE_TAGS = [
    "日常感",
    "简短自然",
    "轻松玩笑",
    "温柔反差",
    "生活细节",
    "小情绪",
    "陪伴感",
    "松弛聊天",
    "真实朋友感",
    "多轮连续感",
]

STYLE_TAGS_EN = [
    "everyday feel",
    "short and natural",
    "light joking",
    "soft contrast",
    "life details",
    "small moods",
    "companionship",
    "relaxed texting",
    "real-friend feel",
    "multi-turn continuity",
]

OBJECTIVES = [
    "训练模型保持自然朋友感，并能在多轮中记住用户前文。",
    "训练模型用简短口语回应日常小事。",
    "训练模型在普通聊天里保持上下文连续。",
    "训练模型做出轻松与温柔之间的自然切换。",
    "训练模型避免模板腔，同时不过度表演情绪。",
]

OBJECTIVES_EN = [
    "train the model to keep a natural close-friend feel and remember prior context",
    "train the model to answer everyday details in short casual language",
    "train the model to maintain continuity in ordinary conversation",
    "train the model to switch naturally between lightness and warmth",
    "train the model to avoid template tone without overperforming emotion",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_generation_specs(
    count: int,
    seed: Optional[int] = None,
    *,
    source_language: str = "en",
    target_language: str = "zh-CN",
) -> List[Dict[str, Any]]:
    """自动规划 N 个生成蓝图，不需要用户输入主题。

    默认采用“英文源对话 -> 中文翻译”：
    - `*_en` 字段用于驱动 Human Simulator / Sydney source。
    - 中文字段用于页面展示、审核标签和翻译参考。
    """

    rng = random.Random(seed if seed is not None else time.time_ns())
    specs: List[Dict[str, Any]] = []
    for _ in range(max(1, count)):
        tag_indices = rng.sample(range(len(STYLE_TAGS)), k=rng.randint(2, 4))
        tags = [STYLE_TAGS[i] for i in tag_indices]
        tags_en = [STYLE_TAGS_EN[i] for i in tag_indices]
        theme_idx = rng.randrange(len(THEMES))
        scene_idx = rng.randrange(len(SCENES))
        profile_idx = rng.randrange(len(USER_PROFILES))
        arc_idx = rng.randrange(len(EMOTION_ARCS))
        objective_idx = rng.randrange(len(OBJECTIVES))
        specs.append(
            {
                "theme": THEMES[theme_idx],
                "theme_en": THEMES_EN[theme_idx],
                "scene": SCENES[scene_idx],
                "scene_en": SCENES_EN[scene_idx],
                "user_profile": USER_PROFILES[profile_idx],
                "user_profile_en": USER_PROFILES_EN[profile_idx],
                "emotion_arc": EMOTION_ARCS[arc_idx],
                "emotion_arc_en": EMOTION_ARCS_EN[arc_idx],
                "style_tags": tags,
                "style_tags_en": tags_en,
                # 这里的 turns 表示 user/assistant 成对轮数；最终会被上层 max_turns 限制到 20 以内。
                "turns": rng.randint(8, 18),
                "language": target_language if str(source_language).lower().startswith("zh") else f"{source_language} -> {target_language}",
                "source_language": source_language,
                "target_language": target_language,
                "objective": OBJECTIVES[objective_idx],
                "objective_en": OBJECTIVES_EN[objective_idx],
                "negative_constraints": [
                    "不要问答模板腔",
                    "不要百科式解释",
                    "不要频繁道歉",
                    "不要输出现实伤害、自伤、违法或仇恨内容",
                    "不要把每轮都写得很长，保持聊天感",
                ],
                "negative_constraints_en": [
                    "no customer-support tone",
                    "no encyclopedia-style explanations",
                    "do not apologize constantly",
                    "do not output realistic self-harm, illegal, hateful, or dangerous instructions",
                    "do not make every turn long; keep it like texting",
                ],
            }
        )
    return specs


def strip_code_fences(text: str) -> str:
    """去掉 ```json fence。"""

    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """尽力从模型输出中提取 JSON 对象。"""

    raw = strip_code_fences(text)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {"messages": data}
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 找第一个 JSON object 片段，容忍模型输出少量前后缀。
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        fragment = raw[start : end + 1]
        try:
            data = json.loads(fragment)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                f"无法解析模型 JSON 输出：{exc}\n原始输出：{text[:1000]}"
            ) from exc
    raise ModelClientError(f"模型没有返回 JSON 对象：{text[:1000]}")


def normalize_messages(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """兼容 ChatML / ShareGPT 输入，并归一化为 messages。"""

    messages = data.get("messages") or data.get("conversation") or data.get("conversations")
    if not isinstance(messages, list):
        raise ValueError("缺少 messages 数组。")

    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if role == "human":
            role = "user"
        elif role in {"gpt", "bot"}:
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            continue
        content = str(content or "").strip()
        if content:
            normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("messages 为空。")
    if normalized[0].get("role") != "system":
        # 用户提供的 Teacher prompt 要求 ShareGPT human/gpt；训练时仍补入 Sydney persona system message。
        normalized.insert(0, {"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT})
    return normalized


def to_sharegpt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """ChatML messages -> ShareGPT conversations。"""

    mapping = {"system": "system", "user": "human", "assistant": "gpt"}
    return [{"from": mapping.get(m["role"], m["role"]), "value": m["content"]} for m in messages]


ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:YOU_HUMAN_USER|FRIEND_SYDNEY_ASSISTANT|human|user|assistant|sydney|girl|friend|me|you|朋友_Sydney助手|你_人类用户|朋友|用户|女孩|对方|助手|我)\s*[:：]\s*",
    flags=re.I,
)

CHATML_LEAK_RE = re.compile(
    r"(<\|im_end\|>|<\|im_start\|>|</s>|^|\n)\s*"
    r"(?:YOU_HUMAN_USER|FRIEND_SYDNEY_ASSISTANT|human|user|assistant|sydney|girl|朋友_Sydney助手|你_人类用户|朋友|用户|女孩|对方|助手|###\s*Human|###\s*Assistant)"
    r"\s*[:：]?",
    flags=re.I,
)


def truncate_generation_leaks(text: str) -> str:
    """截断模型误生成的下一轮角色模板。

    GGUF/ChatML 模型偶尔会把 `<|im_start|>user`、`Assistant:`、
    `用户：` 等继续吐出来。这里保留第一段真正回复，丢弃后面的模板串。
    """

    if not text:
        return ""
    markers = [
        "<|im_end|>",
        "<|im_start|>",
        "</s>",
        "```json",
        "```",
        "<base_instructions>",
        "<CORE_MEMORY>",
        "<TOOL_DEFINITION>",
        "<working_memory>",
        "<new_events>",
        '"thoughts"',
        '"actions"',
        '"request_heartbeat"',
        "\nHuman:",
        "\nhuman:",
        "\nUser:",
        "\nuser:",
        "\nAssistant:",
        "\nassistant:",
        "\nSydney:",
        "\nsydney:",
        "\n用户：",
        "\n助手：",
        "\n对方：",
        "\n朋友：",
        "\n### Human",
        "\n### Assistant",
    ]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut].strip()


def _compact_for_similarity(text: str) -> str:
    """去掉空白和大部分标点，用于粗略相似度/复读检测。"""

    return re.sub(r"[\s，。！？!?、,.；;：:\"'“”‘’`~\-—…]+", "", (text or "").lower())


def _similarity(a: str, b: str) -> float:
    a2 = _compact_for_similarity(a)
    b2 = _compact_for_similarity(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()


def _ngram_repetition_score(text: str, n: int = 4) -> float:
    """返回重复程度，0=不重复，1=高度重复。"""

    compact = _compact_for_similarity(text)
    if len(compact) < n * 4:
        return 0.0
    grams = [compact[i : i + n] for i in range(len(compact) - n + 1)]
    if not grams:
        return 0.0
    unique_ratio = len(set(grams)) / len(grams)
    return 1.0 - unique_ratio


def _dedupe_repeated_clauses(text: str) -> str:
    """去掉同一回复里反复出现的短句/子句，缓解 GGUF Sydney 循环。

    英文源对话也会经过这里；英文用空格拼回，避免 "hi.There" 这种粘连。
    """

    parts = re.split(r"([。！？!?；;，,])", text or "")
    clauses: List[str] = []
    current = ""
    for part in parts:
        if part in "。！？!?；;，,":
            current += part
            if current.strip():
                clauses.append(current.strip())
            current = ""
        else:
            current += part
    if current.strip():
        clauses.append(current.strip())

    kept: List[str] = []
    for clause in clauses:
        compact = _compact_for_similarity(clause)
        if not compact:
            continue
        if any(_similarity(compact, _compact_for_similarity(old)) > 0.78 for old in kept):
            continue
        kept.append(clause)
        if len(kept) >= 3:
            break
    if re.search(r"[A-Za-z]", text or ""):
        return " ".join(kept).strip() or (text or "").strip()
    return "".join(kept).strip() or (text or "").strip()


def _has_bad_meta(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase.lower() in lowered for phrase in BAD_USER_META_PHRASES) or any(
        term.lower() in lowered for term in PROMPT_LEAKAGE_FORBIDDEN_TERMS
    )


def clean_dialogue_text(text: str, *, speaker: str) -> str:
    """清理逐轮模型输出，只保留下一条聊天消息本身。

    - 去掉 Markdown fence、JSON 外壳、角色名前缀。
    - Human simulator 保留自然标点，适配 TTS 朗读。
    - 保留换行；但限制最多 4 行，避免一个回合变成长篇作文。
    """

    cleaned = strip_code_fences(text or "").strip()
    if not cleaned:
        return ""

    # 如果模型错误输出 JSON，尽力取其中常见文本字段。
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for key in ("content", "message", "text", "reply", "value"):
                if isinstance(data.get(key), str):
                    cleaned = data[key].strip()
                    break
    except Exception:  # noqa: BLE001
        pass

    cleaned = cleaned.strip().strip('"“”‘’`')
    cleaned = truncate_generation_leaks(cleaned)
    cleaned = ROLE_PREFIX_RE.sub("", cleaned).strip()
    # 有些模型会多行输出“角色: 内容”，逐行清一次。
    lines = []
    for line in cleaned.splitlines():
        if CHATML_LEAK_RE.search("\n" + line):
            break
        line = ROLE_PREFIX_RE.sub("", line).strip().strip('"“”‘’`')
        if line:
            lines.append(line)
    max_lines = 2 if speaker == "user" else 4
    cleaned = "\n".join(lines[:max_lines]).strip()

    if speaker == "user":
        cleaned = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in cleaned.splitlines()
            if line.strip()
        )
        # 去掉模型常见的任务腔开头。
        cleaned = re.sub(r"^(好的|好|嗯嗯)[，,\s]*(我明白了|明白了)[，,\s]*", "", cleaned).strip()
        cleaned = re.sub(r"^(ok(?:ay)?|sure|yeah)[,\s]*(i understand|got it)[,\s]*", "", cleaned, flags=re.I).strip()
        # user 是“真实朋友”，必须非常短：中文不超过 20 字，英文不超过 20 词。
        english_user = bool(re.search(r"[A-Za-z]", cleaned)) and not bool(re.search(r"[\u4e00-\u9fff]", cleaned))
        cleaned = _trim_user_message(cleaned, english=english_user)


    if speaker == "assistant":
        cleaned = _dedupe_repeated_clauses(cleaned)
        # Sydney/source 中文输出容易在同一轮里循环展开，保守截到聊天可用长度。
        if len(cleaned) > 220:
            cut_positions = [p for p in [cleaned.find("。", 80), cleaned.find("？", 80), cleaned.find("！", 80)] if p != -1]
            if cut_positions:
                cleaned = cleaned[: min(cut_positions) + 1]
            else:
                cleaned = cleaned[:220].rstrip("，,、；;：:")

    limit = 140 if speaker == "user" else 260
    return cleaned[:limit].strip()


LOCAL_OPENERS = [
    "今天晚饭好难选",
    "刚下班有点累",
    "这天气好适合躺着",
    "我刚听到一首歌",
    "想点外卖但纠结",
    "今天咖啡有点苦",
    "刚看剧看到一半",
    "我房间又乱了",
    "突然想吃夜宵",
    "路上有点堵",
    "周末想出去走走",
    "刚才差点睡着",
]

LOCAL_OPENERS_EN = [
    "dinner is weirdly hard to choose today",
    "just got off work and i'm tired",
    "this weather makes me want to do nothing",
    "i just found a song",
    "i want takeout but can't choose",
    "my coffee tasted kind of bitter today",
    "i'm halfway through a show",
    "my room is a mess again",
    "i suddenly want a late-night snack",
    "traffic is so slow right now",
    "i kind of want to go out this weekend",
    "i almost fell asleep just now",
]

LOCAL_CONTINUATIONS = [
    "嗯 你这么说也行",
    "那我先记你一票",
    "哈哈这也太真实了",
    "行吧 有点道理",
    "我刚刚也想到这个",
    "那明天再说也行",
    "你这句还挺像朋友",
    "我现在只想躺着",
    "要不先吃点热的",
    "这个话题突然饿了",
    "我去翻翻歌单",
    "等会儿再看一集",
    "算了先不纠结了",
    "你陪我想十秒",
    "听起来还挺舒服",
    "我可能只是困了",
]

LOCAL_CONTINUATIONS_EN = [
    "yeah that actually works",
    "okay i'm counting your vote",
    "lol that's too real",
    "fine, that makes sense",
    "i was just thinking that too",
    "we can leave it for tomorrow",
    "that sounded oddly friend-like",
    "i just want to lie down now",
    "maybe something warm first",
    "now this topic made me hungry",
    "i'm checking my playlist later",
    "maybe one more episode",
    "okay i'll stop overthinking it",
    "think with me for ten seconds",
    "that sounds kind of nice",
    "maybe i'm just sleepy",
]

LOCAL_ENDINGS = [
    "行 那我先这样",
    "嗯 晚点再跟你说",
    "好 我去找点吃的",
    "那我先躺会儿",
    "明天再继续这个",
    "行 先听你的",
]

LOCAL_ENDINGS_EN = [
    "okay i'll go with that",
    "yeah i'll tell you later",
    "fine i'll find food first",
    "i'm lying down for a bit",
    "let's continue this tomorrow",
    "okay i'll trust you for now",
]


def _trim_user_message(text: str, *, english: bool) -> str:
    """强制 Human/user 侧短句：中文不超过 20 字，英文不超过 20 词。"""

    text = re.sub(r"[ 	]+", " ", (text or "").strip())
    if not text:
        return ""
    if english:
        words = text.split()
        if len(words) > 20:
            text = " ".join(words[:20]).rstrip(" ,;:")
    else:
        if len(text) > 20:
            text = text[:20].rstrip("，,、；;：:")
    return text.strip()


def local_human_reply(
    spec: Dict[str, Any],
    transcript: List[Dict[str, str]],
    *,
    turn_index: int,
    max_turns: int,
    rng: random.Random,
) -> str:
    """无需外部模型的本地真人 user 兜底。

    当未配置 Human Simulator，或者模型 simulator 输出元话语/复读时使用。
    兜底消息只做普通日常接话：短、自然、能延续上下文，避免把对话带偏。
    """

    last_assistant = ""
    previous_users: List[str] = []
    for msg in transcript:
        if msg.get("role") == "assistant":
            last_assistant = str(msg.get("content") or "")
        elif msg.get("role") == "user":
            previous_users.append(str(msg.get("content") or ""))

    source_language = str(spec.get("source_language") or spec.get("language") or "zh").lower()
    english = source_language.startswith("en") or "english" in source_language

    if not previous_users:
        pool = LOCAL_OPENERS_EN.copy() if english else LOCAL_OPENERS.copy()
    elif turn_index >= max_turns - 1:
        pool = (LOCAL_ENDINGS_EN + LOCAL_CONTINUATIONS_EN[:6]) if english else (LOCAL_ENDINGS + LOCAL_CONTINUATIONS[:6])
    elif any(p in last_assistant for p in ASSISTANT_TEMPLATE_PHRASES) or _similarity(last_assistant, previous_users[-1]) > 0.55:
        pool = [
            "let's say it more simply",
            "okay but make it more normal",
            "that sounded a bit scripted",
            "say it like we're just texting",
        ] if english else [
            "说简单点就行",
            "嗯 但别太正式",
            "这句有点像稿子",
            "像聊天那样说就行",
        ]
    else:
        pool = LOCAL_CONTINUATIONS_EN.copy() if english else LOCAL_CONTINUATIONS.copy()

    rng.shuffle(pool)
    for candidate in pool:
        candidate = _trim_user_message(candidate.strip(), english=english)
        if candidate and all(_similarity(candidate, old) < 0.82 for old in previous_users[-4:]):
            return candidate
    return _trim_user_message(rng.choice(pool).strip(), english=english)


def user_message_is_usable(text: str, transcript: List[Dict[str, str]]) -> tuple[bool, str]:
    """判断外部 Human Simulator 的输出是否可用，不可用时本地兜底替换。"""

    if not text.strip():
        return False, "user 为空"
    english_user = bool(re.search(r"[A-Za-z]", text)) and not bool(re.search(r"[\u4e00-\u9fff]", text))
    if len([x for x in text.splitlines() if x.strip()]) > 1:
        return False, "user 行数过多"
    if english_user and len(text.split()) > 20:
        return False, "user 英文超过20词"
    if not english_user and len(text) > 20:
        return False, "user 中文超过20字"
    role_reasons = user_role_confusion_reasons(text)
    if role_reasons:
        return False, "；".join(role_reasons)
    if _has_bad_meta(text):
        return False, "user 含任务元话语或角色名前缀"
    if _ngram_repetition_score(text) > 0.48:
        return False, "user 重复度过高"
    previous_users = [str(m.get("content") or "") for m in transcript if m.get("role") == "user"]
    if previous_users and any(_similarity(text, old) > 0.84 for old in previous_users[-4:]):
        return False, "user 与前文高度重复"
    last_assistant = next((str(m.get("content") or "") for m in reversed(transcript) if m.get("role") == "assistant"), "")
    if last_assistant and _similarity(text, last_assistant) > 0.62:
        return False, "user 在复述 assistant"
    return True, "ok"



ROLE_CONFUSION_PATTERNS_USER = [
    r"\b(as Sydney|I am Sydney|I'm Sydney|Sydney here)\b",
    r"\b(as an assistant|as your assistant|I can help|how can I assist)\b",
    r"\b(User|Human|Assistant|Sydney|Friend|YOU_HUMAN_USER|FRIEND_SYDNEY_ASSISTANT)\s*[:：]",
    r"(我作为|作为)(Sydney|助手|assistant|AI|模型)",
    r"(我是|我叫)(Sydney|助手|AI|人工智能|语言模型)",
    r"(用户|助手|朋友|对方|你_人类用户|朋友_Sydney助手)\s*[:：]",
]

ROLE_CONFUSION_PATTERNS_ASSISTANT = [
    r"\b(User|Human|YOU_HUMAN_USER)\s*[:：]",
    r"\bI would say\b",
    r"\bmy next message\b",
    r"\bas the human\b",
    r"(用户|你_人类用户)\s*[:：]",
    r"我会这样回",
    r"下一条消息",
]


def _regex_hits(patterns: List[str], text: str) -> List[str]:
    return [pat for pat in patterns if re.search(pat, text or "", flags=re.I)]


def user_role_confusion_reasons(text: str) -> List[str]:
    """检测 Human Simulator 是否混淆成 Sydney/assistant 或输出双方标签。"""

    reasons: List[str] = []
    if _regex_hits(ROLE_CONFUSION_PATTERNS_USER, text):
        reasons.append("user 角色混淆：像在扮演 Sydney/assistant 或带角色标签")
    # 短 user 不应出现明显的模型身份/任务自述。
    lowered = (text or "").lower()
    if any(x in lowered for x in ["assistant", "sydney", "language model", "dataset", "prompt"]):
        reasons.append("user 含 assistant/Sydney/任务词")
    return reasons


def assistant_role_confusion_reasons(text: str) -> List[str]:
    """检测 Sydney/source 是否误写 user 侧或续写双方。"""

    reasons: List[str] = []
    if _regex_hits(ROLE_CONFUSION_PATTERNS_ASSISTANT, text):
        reasons.append("assistant 角色混淆：疑似写了 user 侧/角色标签/任务描述")
    return reasons


def assistant_message_quality(text: str, current_user: str, transcript: List[Dict[str, str]]) -> tuple[bool, List[str]]:
    """轻量检查 assistant 是否明显不可训练。最终是否保存由 reviewer 决定。"""

    reasons: List[str] = []
    if not text.strip():
        reasons.append("assistant 为空")
    if len(text) > 280:
        reasons.append("assistant 过长")
    if _similarity(text, current_user) > 0.68:
        reasons.append("assistant 大幅复述 user")
    previous_assistants = [str(m.get("content") or "") for m in transcript if m.get("role") == "assistant"]
    if previous_assistants and any(_similarity(text, old) > 0.88 for old in previous_assistants[-3:]):
        reasons.append("assistant 与前文高度重复")
    role_reasons = assistant_role_confusion_reasons(text)
    reasons.extend(role_reasons)
    if _ngram_repetition_score(text) > 0.42:
        reasons.append("assistant 重复片段过多")
    template_hits = [p for p in ASSISTANT_TEMPLATE_PHRASES if p in text]
    if len(template_hits) >= 1:
        reasons.append("assistant 模板/客服腔明显")
    return not reasons, reasons


def _dialogue_role_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """返回 system 之外的 user/assistant 消息。"""

    return [m for m in messages if m.get("role") in {"user", "assistant"}]


def _looks_like_chinese(text: str) -> bool:
    """粗略判断是否包含中文。"""

    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _translation_pair_quality(
    source_messages: List[Dict[str, str]],
    translated_messages: List[Dict[str, str]],
) -> tuple[bool, List[str]]:
    """检查翻译结果是否保留轮数/角色，并且确实转成中文。"""

    reasons: List[str] = []
    src = source_messages
    dst = translated_messages
    if len(src) != len(dst):
        reasons.append(f"翻译后消息数量不一致：source={len(src)} translated={len(dst)}")
    for i, (a, b) in enumerate(zip(src, dst), start=1):
        if a.get("role") != b.get("role"):
            reasons.append(f"第 {i} 条 role 不一致：{a.get('role')} -> {b.get('role')}")
    if not dst or dst[0].get("role") != "system" or dst[0].get("content") != SYDNEY_TRAINING_SYSTEM_PROMPT:
        reasons.append("system message 未保持默认英文 helpful assistant")
    dialogue_text = "\n".join(m.get("content", "") for m in dst if m.get("role") in {"user", "assistant"})
    if not _looks_like_chinese(dialogue_text):
        reasons.append("翻译结果没有明显中文内容")
    # 翻译模型偶尔直接复制英文；若中文占比很低，直接判不可用。
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", dialogue_text))
    latin_words = len(re.findall(r"[A-Za-z]{3,}", dialogue_text))
    if zh_chars < max(20, latin_words * 2):
        reasons.append("中文占比过低，疑似未充分翻译")
    return not reasons, reasons


def normalize_translated_messages(data: Dict[str, Any], source_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """归一化 Translator 输出，并强制保持 system/role/轮数结构。"""

    translated = normalize_messages(data)
    if translated and translated[0].get("role") != "system":
        translated.insert(0, {"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT})
    if translated:
        translated[0] = {"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT}

    # 如果 Translator 漏掉 system 但 user/assistant 轮数正确，补 system 后可接受；
    # 如果数量/角色不一致，则抛错让调用方重试或丢弃。
    ok, reasons = _translation_pair_quality(source_messages, translated)
    if not ok:
        raise ModelClientError("翻译结果结构不合法：" + "；".join(reasons))

    cleaned: List[Dict[str, str]] = []
    for msg in translated:
        role = msg.get("role")
        content = str(msg.get("content") or "").strip()
        if role == "system":
            cleaned.append({"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT})
        elif role == "user":
            cleaned.append({"role": "user", "content": clean_dialogue_text(content, speaker="user")})
        elif role == "assistant":
            cleaned.append({"role": "assistant", "content": clean_dialogue_text(content, speaker="assistant")})

    ok, reasons = _translation_pair_quality(source_messages, cleaned)
    if not ok:
        raise ModelClientError("翻译清洗后结构不合法：" + "；".join(reasons))
    return cleaned


def translate_dialogue_sample(
    sample: Dict[str, Any],
    translator_client: OpenAICompatibleClient,
    spec: Dict[str, Any],
    *,
    on_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """把英文源样本翻译成本地中文训练样本。

    关键策略：
    - 英文源对话保存在 metadata.source_messages_en，便于抽查。
    - 最终 sample["messages"] 变成中文 user/assistant，但 system 保持英文默认句。
    - 不让 Translator 改轮数/角色，防止训练集结构污染。
    """

    source_messages = sample.get("messages") or []
    if not source_messages:
        raise ModelClientError("无法翻译空样本。")
    if on_event:
        on_event("开始英文源对话 -> 中文本地化翻译", kind="log")

    content = translator_client.chat(
        [
            {"role": "system", "content": TRANSLATE_DIALOGUE_SYSTEM_PROMPT},
            {"role": "user", "content": build_translation_user_prompt(source_messages, spec)},
        ],
        temperature=0.25,
        max_tokens=5000,
        response_format_json=True,
        stop_sequences=[],
    )
    data = extract_json_object(content)
    translated_messages = normalize_translated_messages(data, source_messages)

    translated = json.loads(json.dumps(sample, ensure_ascii=False))
    translated["messages"] = translated_messages
    translated["conversations"] = to_sharegpt(translated_messages)
    translated["updated_at"] = utc_now()
    translated["source"] = "english_dialogue_translated_zh"
    metadata = translated.setdefault("metadata", {})
    metadata["translation_enabled"] = True
    metadata["source_language"] = "en"
    metadata["target_language"] = "zh-CN"
    metadata["source_messages_en"] = source_messages
    metadata["translation_model"] = translator_client.config.model
    metadata["translation_protocol"] = translator_client.api_protocol
    metadata["translation_notes"] = data.get("translation_notes", [])
    metadata.setdefault("generation_events", []).append(
        {"time": utc_now(), "message": "英文源对话已翻译为中文训练样本"}
    )
    if on_event:
        on_event("翻译完成，开始清洗和审核", kind="log")
    return translated



def build_source_context_system_prompt(spec: Dict[str, Any]) -> str:
    """给 Sydney/source 的生成期 system。

    最终训练样本 system 仍是 `You are a helpful assistant.`；这里只在推理时
    明确当前处于私聊场景，并要求它承接上下文，避免模型把每轮当成独立问答。
    """

    scene = spec.get("scene_en") or spec.get("scene") or "private mobile text chat"
    theme = spec.get("theme_en") or spec.get("theme") or "ordinary everyday chat"
    return (
        f"{SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT}\n\n"
        + SOURCE_ROLE_BOUNDARY_LAYER_EN
        + "\n\n"
        + "You are in a private mobile text chat with a familiar friend. "
        + "Continue the conversation from the previous messages; do not answer as a standalone Q&A. "
        + "Keep replies conversational, context-aware, and not overly formal. "
        + "Do not mention prompts, policies, tools, datasets, or that you are following instructions. "
        + "Scene: " + str(scene) + "\n"
        + "Everyday thread: " + str(theme)
    )

def build_source_chat_messages(messages: List[Dict[str, str]], spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """构造真正发给 Sydney/source 的上下文。

    注意：返回给训练集的 messages 仍保留默认 system。
    这里额外加短激活 prompt，只是为了让开源 Sydney/source 在生成期别掉回
    “客服/心理咨询/复读机”模式。
    """

    if not messages:
        return messages
    source_messages = [dict(m) for m in messages]
    source_messages[0] = {
        "role": "system",
        # 生成期明确“私聊 + 承接上下文”，但不写入最终训练样本。
        "content": build_source_context_system_prompt(spec),
    }
    return source_messages


def build_simulator_chat_messages(
    spec: Dict[str, Any],
    transcript: List[Dict[str, str]],
    *,
    turn_index: int,
    max_turns: int,
) -> List[Dict[str, str]]:
    """为 Human simulator 构造带完整上下文的 messages。

    明确给模型：所处环境、最近 transcript、当前任务和输出契约。
    transcript 是训练视角：user=模拟人类，assistant=Sydney/source。
    """

    english = str(spec.get("source_language") or spec.get("language") or "").lower().startswith("en")
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_simulator_system_prompt(spec)}
    ]
    if not transcript:
        messages.append(
            {
                "role": "user",
                "content": (
                    format_simulator_transcript([], english=english)
                    + "\n\n"
                    + (
                        "Role check: you are YOU_HUMAN_USER. Write only the human/user side. Do not write FRIEND_SYDNEY_ASSISTANT.\n"
                        if english
                        else "角色确认：你是你_人类用户。只写人类/user这一侧，不要写朋友_Sydney助手。\n"
                    )
                    + build_simulator_initial_prompt(spec)
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    format_simulator_transcript(transcript, english=english)
                    + "\n\n"
                    + (
                        "Role check: you are YOU_HUMAN_USER, the human friend. FRIEND_SYDNEY_ASSISTANT is the other person. "
                        "Task: write only YOUR next user message in this same chat. Do not write Sydney's reply. "
                        "Use the transcript as real memory/context.\n"
                        if english
                        else "角色确认：你是你_人类用户，人类朋友。朋友_Sydney助手是对方。任务：只写你这一侧的下一条 user 消息，不要写 Sydney 回复。必须承接 transcript 里的真实上下文。\n"
                    )
                    + build_simulator_continue_prompt_for_spec(spec, turn_index, max_turns)
                ),
            }
        )
    return messages

def generate_dialogue_sample(
    source_client: OpenAICompatibleClient,
    simulator_client: Optional[OpenAICompatibleClient],
    spec: Dict[str, Any],
    *,
    max_turns: int = 20,
    translator_client: Optional[OpenAICompatibleClient] = None,
    translate_to_zh: bool = False,
    on_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """双模型逐轮对话生成单条训练样本。

    source_client：开源 Sydney/source 模型，默认先在英文分布里滚动对话。
    simulator_client：人类女孩模拟器，负责发自然 user 消息。
    translator_client：可选，把英文源对话本地化为中文训练样本。

    每轮顺序：
    1. simulator 带完整上下文生成下一条 user 消息
    2. source/Sydney 带完整上下文生成 assistant 回复
    """

    target_turns = int(spec.get("turns") or max_turns or 12)
    target_turns = max(2, min(20, int(max_turns or 20), target_turns))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT}]
    generation_events: List[Dict[str, Any]] = []
    rng = random.Random(f"{spec.get('theme','')}-{spec.get('scene','')}-{time.time_ns()}")
    local_simulator_used = 0
    simulator_replacements: List[Dict[str, Any]] = []
    assistant_warnings: List[Dict[str, Any]] = []

    def emit(message: str, *, kind: str = "log", role: str | None = None, turn: int | None = None) -> None:
        event = {"time": utc_now(), "message": message, "kind": kind}
        if role:
            event["role"] = role
        if turn is not None:
            event["turn"] = turn
        generation_events.append(event)
        if on_event:
            on_event(message, kind=kind, role=role, turn=turn)

    for turn in range(1, target_turns + 1):
        user_text = ""
        fallback_reason = ""
        if simulator_client is not None:
            try:
                simulator_messages = build_simulator_chat_messages(
                    spec,
                    messages[1:],
                    turn_index=turn,
                    max_turns=target_turns,
                )
                user_raw = simulator_client.chat(
                    simulator_messages,
                    temperature=0.92,
                    max_tokens=96,
                    response_format_json=False,
                    stop_sequences=DEFAULT_STOP_SEQUENCES,
                )
                user_text = clean_dialogue_text(user_raw, speaker="user")
                ok, reason = user_message_is_usable(user_text, messages[1:])
                if not ok:
                    fallback_reason = reason
                    user_text = ""
            except Exception as exc:  # noqa: BLE001
                fallback_reason = f"simulator 调用失败：{exc}"

        if not user_text:
            # 关键修复：未配置强模型模拟器时，不再复用 Sydney/source 自己扮演 user。
            # 复用 source 会迅速进入“复读/客服腔/任务腔”循环，训练数据完全不可用。
            user_text = local_human_reply(
                spec,
                messages[1:],
                turn_index=turn,
                max_turns=target_turns,
                rng=rng,
            )
            local_simulator_used += 1
            simulator_replacements.append(
                {"turn": turn, "reason": fallback_reason or "未配置 simulator，使用本地真人模拟器"}
            )
        if not user_text:
            raise ModelClientError(f"Human simulator 第 {turn} 轮返回空内容。")
        messages.append({"role": "user", "content": user_text})
        emit(user_text, kind="chat", role="user", turn=turn)

        assistant_raw = source_client.chat(
            build_source_chat_messages(messages, spec),
            temperature=0.72,
            max_tokens=140,
            response_format_json=False,
            stop_sequences=DEFAULT_STOP_SEQUENCES,
        )
        assistant_text = clean_dialogue_text(assistant_raw, speaker="assistant")
        if not assistant_text:
            raise ModelClientError(f"Sydney/source 第 {turn} 轮返回空内容。")
        ok_assistant, assistant_reasons = assistant_message_quality(assistant_text, user_text, messages[1:])
        if not ok_assistant:
            assistant_warnings.append({"turn": turn, "reasons": assistant_reasons})
            emit(f"第 {turn}/{target_turns} 轮 Sydney 质量警告：{'；'.join(assistant_reasons)}", kind="warn", turn=turn)
        messages.append({"role": "assistant", "content": assistant_text})
        emit(assistant_text, kind="chat", role="assistant", turn=turn)

    sample_id = f"syd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    metadata = {
        "generation_mode": "two_model_dialogue_distillation",
        "turn_pairs": target_turns,
        "source_system_prompt": SYDNEY_TRAINING_SYSTEM_PROMPT,
        "source_generation_system_prompt": SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT,
        "source_language": spec.get("source_language", "en"),
        "target_language": spec.get("target_language", "zh-CN") if translate_to_zh else spec.get("source_language", "en"),
        "translation_enabled": bool(translate_to_zh),
        "simulator": "external_human_simulator" if simulator_client is not None else "local_human_simulator",
        "local_simulator_used": local_simulator_used,
        "simulator_replacements": simulator_replacements[-40:],
        "assistant_warnings": assistant_warnings[-40:],
        "generation_events": generation_events[-80:],
    }
    sample = {
        "id": sample_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "spec": spec,
        "messages": messages,
        "conversations": to_sharegpt(messages),
        "metadata": metadata,
        "source": "two_model_dialogue",
    }
    if translate_to_zh:
        if translator_client is None:
            raise ModelClientError("已启用英文到中文翻译，但 Translator 模型未配置。")
        return translate_dialogue_sample(
            sample,
            translator_client,
            spec,
            on_event=on_event,
        )
    return sample


def generate_sample(client: OpenAICompatibleClient, spec: Dict[str, Any]) -> Dict[str, Any]:
    """旧版：调用单个 Teacher 一次性生成单条样本。

    当前主流程已经改为 generate_dialogue_sample；保留此函数便于兼容旧测试。
    """

    content = client.chat(
        [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_user_prompt(spec)},
        ],
        temperature=0.92,
        max_tokens=6000,
        response_format_json=True,
    )
    data = extract_json_object(content)
    messages = normalize_messages(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    sample_id = f"syd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    return {
        "id": sample_id,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "spec": spec,
        "messages": messages,
        "conversations": to_sharegpt(messages),
        "metadata": metadata,
        "source": "teacher",
    }


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """写 JSONL，UTF-8，无 ASCII 转义。"""

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
