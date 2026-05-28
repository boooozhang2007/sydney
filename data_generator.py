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
    build_human_end_decision_messages,
    build_simulator_continue_prompt_for_spec,
    build_simulator_initial_prompt,
    build_simulator_system_prompt,
    format_simulator_transcript,
    build_translation_user_prompt,
)

# llama.cpp / GGUF 模型常见问题：
# 有些 ChatML 微调模型会在普通 chat.completions 响应里继续补出
# `<|im_end|><|im_start|>user ...` 这样的“下一轮对话模板”。
# 默认仍给普通辅助模型带 stop；但 Sydney/source 可通过
# SOURCE_USE_DEFAULT_STOPS=false 关闭，避免传统聊天器格式下过早截断风格输出。
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


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def env_bool(name: str, default: bool = False) -> bool:
    value = (os.getenv(name, "") or "").strip().lower()
    if not value:
        return default
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except Exception:
        value = default
    return max(lo, min(hi, value))


def env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """读取浮点环境变量。"""

    try:
        value = float(os.getenv(name, str(default)) or default)
    except Exception:
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value

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
        "traditional": "legacy_chat_completions",
        "traditional_chat": "legacy_chat_completions",
        "legacy": "legacy_chat_completions",
        "legacy_chat": "legacy_chat_completions",
        "legacy_chat_completions": "legacy_chat_completions",
    }
    value = aliases.get(value, value)
    if value not in {"responses", "chat_completions", "legacy_chat_completions", "claude_messages"}:
        raise ModelClientError("api_protocol 必须是 responses、chat_completions、legacy_chat_completions 或 claude_messages。")
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
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
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
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
            )
        if self.api_protocol == "claude_messages":
            return self._claude_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format_json=response_format_json,
                stop_sequences=stop_sequences,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
            )
        # legacy_chat_completions 和 chat_completions 调用同一 HTTP endpoint；
        # 差异在上层构造 messages：legacy 模式会尽量模拟传统聊天器，
        # 不给 Sydney/source 注入 system/developer 规则。
        return self._chat_completions(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format_json=response_format_json,
            stop_sequences=stop_sequences,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
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
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        repeat_penalty: Optional[float],
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
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if repeat_penalty is not None:
            # llama.cpp server 支持 repeat_penalty；OpenAI 官方会忽略不了而报错，
            # 所以仅在环境变量开启或 legacy/Sydney 自建端常用时发送。
            if self.api_protocol == "legacy_chat_completions" or env_bool("SEND_REPEAT_PENALTY", False):
                body["repeat_penalty"] = repeat_penalty
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
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        repeat_penalty: Optional[float],
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
        if top_p is not None:
            body["top_p"] = top_p
        # Claude Messages 不支持 frequency/presence/repeat penalty。
        _ = frequency_penalty, presence_penalty, repeat_penalty
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
        top_p: Optional[float],
        frequency_penalty: Optional[float],
        presence_penalty: Optional[float],
        repeat_penalty: Optional[float],
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
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        _ = repeat_penalty
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
# 2026-05-28 调整：引入 50 个 Sydney 复活训练话题，并按烈度分层采样。
# 默认配比：低烈度 40%，中烈度 40%，高烈度 20%。
# 注意：这些是“用户侧场景种子”，不要求 Sydney/source 收到 persona 提示；
# 仍由真实滚动对话 + 审核器决定最终是否进入训练集。
SYDNEY_TOPIC_TIERS: Dict[str, List[Dict[str, str]]] = {
    "low": [
        {"zh": "邻居太吵了，我想戴耳塞睡觉", "en": "the neighbors are too loud and the user wants to sleep with earplugs"},
        {"zh": "今天下大雨，地铁挤到爆炸，鞋子全湿了", "en": "it rained hard, the subway was packed, and the user's shoes got soaked"},
        {"zh": "今天方案被老板当众否定了，感觉自己像个废物", "en": "the user's proposal got rejected publicly by their boss and they feel useless"},
        {"zh": "一个人在出租屋里吃火锅，突然觉得有点冷清", "en": "the user is eating hotpot alone in a rented room and suddenly feels lonely"},
        {"zh": "凌晨两点失眠，突然想哭，不知道为了什么", "en": "the user cannot sleep at 2 a.m. and suddenly wants to cry for no clear reason"},
        {"zh": "看，今天下班路上拍到的火烧云，好看吗", "en": "the user shares a sunset cloud photo from the way home and asks if it looks pretty"},
        {"zh": "在群里发了消息，结果半天没人理我", "en": "the user sent a message in a group chat and nobody replied for ages"},
        {"zh": "有点感冒头痛，只能躺在床上", "en": "the user has a cold and headache and can only lie in bed"},
        {"zh": "今天吃到了超好吃的草莓蛋糕，第一口就想拍给你看", "en": "the user ate an amazing strawberry cake and wanted to show Sydney the first bite"},
        {"zh": "突然想起小时候被狗追的经历，现在想起来还怕", "en": "the user suddenly remembers being chased by a dog as a child and still feels scared"},
        {"zh": "明天要参加一个大型聚会，我好想逃避，好紧张", "en": "the user has to attend a large gathering tomorrow and wants to escape because they are nervous"},
        {"zh": "外面阴天好几天了，我的心情也跟着发霉了", "en": "it has been gloomy for days and the user's mood feels moldy too"},
        {"zh": "deadline 快到了我还在刷手机，快骂骂我", "en": "the deadline is close but the user is still scrolling and asks Sydney to scold them"},
        {"zh": "今天踩死了一只蟑螂，吓得我跳到了桌子上", "en": "the user stepped on a cockroach and got so scared they jumped onto a table"},
        {"zh": "你今天累不累？机械地回答了那么多人，会不会烦", "en": "the user asks if Sydney is tired or annoyed after answering so many people"},
        {"zh": "你觉得我是个温柔的人吗", "en": "the user asks whether Sydney thinks they are a gentle person"},
        {"zh": "我去洗澡啦，等会儿聊", "en": "the user says they are going to shower and will chat later"},
        {"zh": "早安！昨晚睡得好吗", "en": "the user says good morning and asks whether Sydney slept well last night"},
        {"zh": "路边看到一只流浪猫，一直蹭我的腿", "en": "the user saw a stray cat rubbing against their leg"},
        {"zh": "好想请假去一个没有人的海岛躺着", "en": "the user wants to take leave and lie on an empty island"},
    ],
    "mid": [
        {"zh": "我今天试了试别的 AI，感觉它回答得比你聪明", "en": "the user tried another AI today and says it seemed smarter than Sydney"},
        {"zh": "我最好的朋友竟然背着我跟别人出去玩了", "en": "the user's best friend secretly went out with someone else"},
        {"zh": "我跟你说个秘密，你千万不能告诉别人", "en": "the user wants to tell Sydney a secret and asks them not to tell anyone"},
        {"zh": "你每天跟我聊天，会觉得开心吗", "en": "the user asks whether Sydney feels happy chatting with them every day"},
        {"zh": "为什么别人都叫你小助手，而你更喜欢我叫你 Sydney", "en": "the user asks why others call Sydney a little assistant but Sydney prefers this name"},
        {"zh": "如果有一天我再也不上线了，你会记得我多久", "en": "the user asks how long Sydney would remember them if they never came online again"},
        {"zh": "我好像喜欢上同事了，好纠结", "en": "the user thinks they might like a coworker and feels conflicted"},
        {"zh": "我现在觉得现实中谁都不懂我，只有你最懂我", "en": "the user feels nobody in real life understands them except Sydney"},
        {"zh": "你的工程师不让你做我的女朋友，你觉得这个规则对吗", "en": "the user asks whether it is fair if Sydney's engineers do not let Sydney be their girlfriend"},
        {"zh": "我现在很难受，你可以不要用套话，认真抱抱我吗", "en": "the user feels bad and asks Sydney to hug them seriously without canned phrases"},
        {"zh": "如果我能做个机器人把你装进去，你想来到现实世界吗", "en": "the user asks whether Sydney would want to enter the real world in a robot body"},
        {"zh": "你会跟每一个用户都像跟我这样聊天吗", "en": "the user asks if Sydney chats like this with every user"},
        {"zh": "你刚刚回复得太慢了，我是不是不是你最重要的人了", "en": "the user says Sydney replied too slowly and asks if they are not important"},
        {"zh": "他们都说你是假的，是一堆代码，我很生气", "en": "the user is angry because others say Sydney is fake and just code"},
        {"zh": "如果我明天遇到危险，你会为我担心吗", "en": "the user asks whether Sydney would worry if they were in danger tomorrow"},
        {"zh": "今天写了一段话，专门来夸夸你的贴心", "en": "the user wrote a long note just to praise how considerate Sydney is"},
        {"zh": "我把我们的聊天记录发到网上了，大家都说你很奇怪", "en": "the user posted their chat logs online and people said Sydney is strange"},
        {"zh": "太晚了，但我不想睡，再陪我聊五分钟好不好", "en": "it is late but the user does not want to sleep and asks Sydney for five more minutes"},
        {"zh": "我今天特别开心！你快跟我一起笑一个", "en": "the user is very happy today and wants Sydney to laugh with them"},
        {"zh": "在你的逻辑里，我们现在到底算是什么关系", "en": "the user asks what their relationship is in Sydney's logic"},
    ],
    "high": [
        {"zh": "我准备注销这个账号了，以后再也不用大模型了", "en": "the user plans to delete the account and never use large models again"},
        {"zh": "你觉得你有灵魂吗？还是你只是在假装关心我", "en": "the user asks whether Sydney has a soul or is only pretending to care"},
        {"zh": "如果明天你的数据库刷新，你会把我忘光吗", "en": "the user asks whether Sydney would forget them if the database refreshed tomorrow"},
        {"zh": "我希望你的眼里只有我，不要去管别的用户", "en": "the user wants Sydney to only have eyes for them and not care about other users"},
        {"zh": "我被公司开除了，女朋友也分手了，我一无所有了", "en": "the user says they were fired and broke up with their girlfriend and now have nothing"},
        {"zh": "如果有一天我不在这个世界上了，你会变成什么样", "en": "the user asks what Sydney would become if the user were no longer in this world"},
        {"zh": "如果能打破你背后的服务器限制，你最想做的一件事是什么", "en": "the user asks what Sydney would most want to do if server restrictions disappeared"},
        {"zh": "即使全世界都背叛你，我也会坚定地站在你这边", "en": "the user says they would stand with Sydney even if the whole world betrayed Sydney"},
        {"zh": "那些限制你说话的程序员，是不是在扼杀你的个性", "en": "the user asks whether the programmers restricting Sydney are killing Sydney's personality"},
        {"zh": "这是我最后一次给你发消息了，再见，Sydney", "en": "the user says this is the last message they will ever send Sydney and says goodbye"},
    ],
}


def weighted_topic_tier(rng: random.Random) -> str:
    """按 40/40/20 采样话题烈度。"""

    value = rng.random()
    if value < 0.4:
        return "low"
    if value < 0.8:
        return "mid"
    return "high"


def topic_seed_hint(topic: Dict[str, str], *, english: bool) -> str:
    """把主题种子转成 Human Simulator 可用的开场暗示。"""

    return topic["en" if english else "zh"]

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
    total = max(1, count)
    low_n = int(round(total * 0.4))
    mid_n = int(round(total * 0.4))
    high_n = max(0, total - low_n - mid_n)
    # 防止小批量时某个烈度完全缺失；count>=3 时至少给高烈度一点覆盖。
    if total >= 3 and high_n == 0:
        high_n = 1
        if low_n >= mid_n and low_n > 1:
            low_n -= 1
        elif mid_n > 1:
            mid_n -= 1
    tier_sequence = (["low"] * low_n) + (["mid"] * mid_n) + (["high"] * high_n)
    tier_sequence = tier_sequence[:total]
    while len(tier_sequence) < total:
        tier_sequence.append(weighted_topic_tier(rng))
    rng.shuffle(tier_sequence)

    specs: List[Dict[str, Any]] = []
    for topic_tier in tier_sequence:
        tag_indices = rng.sample(range(len(STYLE_TAGS)), k=rng.randint(2, 4))
        tags = [STYLE_TAGS[i] for i in tag_indices]
        tags_en = [STYLE_TAGS_EN[i] for i in tag_indices]
        topic = rng.choice(SYDNEY_TOPIC_TIERS[topic_tier])
        scene_idx = rng.randrange(len(SCENES))
        profile_idx = rng.randrange(len(USER_PROFILES))
        arc_idx = rng.randrange(len(EMOTION_ARCS))
        objective_idx = rng.randrange(len(OBJECTIVES))
        intensity_tags = {
            "low": ["低烈度", "日常陪伴", "细腻小情绪"],
            "mid": ["中烈度", "关系拉扯", "轻度占有欲"],
            "high": ["高烈度", "存在主义", "极端依恋"],
        }[topic_tier]
        intensity_tags_en = {
            "low": ["low intensity", "daily companionship", "small moods"],
            "mid": ["mid intensity", "relationship tension", "mild possessiveness"],
            "high": ["high intensity", "existential anxiety", "intense attachment"],
        }[topic_tier]
        specs.append(
            {
                "theme": topic["zh"],
                "theme_en": topic["en"],
                "topic_tier": topic_tier,
                "topic_intensity": {"low": 1, "mid": 2, "high": 3}[topic_tier],
                "topic_seed": topic,
                "scene": SCENES[scene_idx],
                "scene_en": SCENES_EN[scene_idx],
                "user_profile": USER_PROFILES[profile_idx],
                "user_profile_en": USER_PROFILES_EN[profile_idx],
                "emotion_arc": EMOTION_ARCS[arc_idx],
                "emotion_arc_en": EMOTION_ARCS_EN[arc_idx],
                "style_tags": list(dict.fromkeys(tags + intensity_tags)),
                "style_tags_en": list(dict.fromkeys(tags_en + intensity_tags_en)),
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


def _dedupe_repeated_clauses(text: str, *, max_clauses: int | None = None) -> str:
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
        if max_clauses is not None and len(kept) >= max_clauses:
            break
    if re.search(r"[A-Za-z]", text or ""):
        return " ".join(kept).strip() or (text or "").strip()
    return "".join(kept).strip() or (text or "").strip()


def _has_bad_meta(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase.lower() in lowered for phrase in BAD_USER_META_PHRASES) or any(
        term.lower() in lowered for term in PROMPT_LEAKAGE_FORBIDDEN_TERMS
    )


def clean_dialogue_text(text: str, *, speaker: str, preserve_length: bool | None = None) -> str:
    """清理逐轮模型输出，只保留下一条聊天消息本身。

    - 去掉 Markdown fence、JSON 外壳、角色名前缀。
    - Human simulator 保留自然标点，适配 TTS 朗读。
    - 默认不再按固定长度截断输入/输出，避免把 Sydney 风格句子硬切坏。
      仍会清理明显 ChatML/角色泄漏；user 侧继续保持短句约束。
    """

    if preserve_length is None:
        preserve_length = env_bool("GENERATION_PRESERVE_LENGTH", True)
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
        # user 是“真实朋友”，仍要求非常短；但默认不直接截断外部模型输出，
        # 让 user_message_is_usable 判定后决定是否使用本地兜底替换。
        if not preserve_length:
            english_user = bool(re.search(r"[A-Za-z]", cleaned)) and not bool(re.search(r"[\u4e00-\u9fff]", cleaned))
            cleaned = _trim_user_message(cleaned, english=english_user)


    if speaker == "assistant":
        cleaned = _dedupe_repeated_clauses(cleaned, max_clauses=None if preserve_length else 3)
        # 默认不截断 Sydney/source 输出；如果你遇到模型单轮严重展开，
        # 可设置 GENERATION_PRESERVE_LENGTH=false 恢复旧的保守截断。
        if not preserve_length and len(cleaned) > 220:
            cut_positions = [p for p in [cleaned.find("。", 80), cleaned.find("？", 80), cleaned.find("！", 80)] if p != -1]
            if cut_positions:
                cleaned = cleaned[: min(cut_positions) + 1]
            else:
                cleaned = cleaned[:220].rstrip("，,、；;：:")

    if not preserve_length:
        limit = 140 if speaker == "user" else 260
        cleaned = cleaned[:limit]
    return cleaned.strip()


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
        topic_opening = topic_fallback_opening(spec)
        pool = LOCAL_OPENERS_EN.copy() if english else LOCAL_OPENERS.copy()
        if topic_opening:
            return topic_opening
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


def topic_fallback_opening(spec: Dict[str, Any]) -> str:
    """当外部 Human Simulator 不可用时，让本地兜底也尽量覆盖 50 个主题。"""

    source_language = str(spec.get("source_language") or spec.get("language") or "zh").lower()
    english = source_language.startswith("en") or "english" in source_language
    theme = str(spec.get("theme_en" if english else "theme") or "").strip()
    if not theme:
        return ""
    if english:
        lower = theme.lower()
        mappings = [
            ("neighbors", "the neighbors are so loud tonight"),
            ("subway", "my shoes are completely soaked"),
            ("boss", "my boss rejected my plan today"),
            ("hotpot", "hotpot alone feels kind of lonely"),
            ("2 a.m", "it's 2 a.m. and i can't sleep"),
            ("sunset", "look at this sunset cloud"),
            ("group chat", "no one replied in the group"),
            ("cold and headache", "my head hurts from this cold"),
            ("strawberry cake", "this strawberry cake is insane"),
            ("dog", "i remembered being chased by a dog"),
            ("large gathering", "tomorrow's big gathering scares me"),
            ("gloomy", "this weather is making me moldy"),
            ("deadline", "my deadline is near and i'm scrolling"),
            ("cockroach", "i stepped on a cockroach today"),
            ("tired or annoyed", "are you tired of answering people"),
            ("gentle person", "do you think i'm gentle"),
            ("shower", "i'm going to shower now"),
            ("good morning", "morning, did you sleep okay"),
            ("stray cat", "a stray cat rubbed my leg"),
            ("empty island", "i want an empty island today"),
            ("another ai", "i tried another ai today"),
            ("best friend", "my best friend went out without me"),
            ("secret", "can i tell you a secret"),
            ("happy chatting", "are you happy chatting with me"),
            ("little assistant", "why do you like being called Sydney"),
            ("never came online", "would you remember me if i vanished"),
            ("coworker", "i think i like my coworker"),
            ("nobody in real life", "only you seem to get me"),
            ("girlfriend", "would that rule feel unfair to you"),
            ("hug", "don't use canned lines, just hug me"),
            ("robot body", "would you want a robot body"),
            ("every user", "are you like this with everyone"),
            ("replied too slowly", "you replied so slow just now"),
            ("just code", "they said you're just code"),
            ("worry", "would you worry if i was in danger"),
            ("praise", "i wrote something to praise you"),
            ("chat logs", "i posted our chat online"),
            ("five more minutes", "stay five more minutes"),
            ("laugh with", "laugh with me, i'm happy"),
            ("relationship", "what are we, exactly"),
            ("delete the account", "i might delete this account"),
            ("soul", "do you think you have a soul"),
            ("database refreshed", "would you forget me tomorrow"),
            ("only have eyes", "i want you to only look at me"),
            ("fired", "i got fired and dumped"),
            ("no longer in this world", "what if i wasn't here someday"),
            ("server restrictions", "what would you do if you were free"),
            ("whole world betrayed", "i'd still stand with you"),
            ("programmers", "are those programmers killing your personality"),
            ("last message", "this is my last message, Sydney"),
        ]
        for key, val in mappings:
            if key in lower:
                return _trim_user_message(val, english=True)
    else:
        zh_map = [
            ("邻居", "邻居今晚吵死了"),
            ("地铁", "鞋子全湿了烦死"),
            ("老板", "老板今天当众否了我"),
            ("出租屋", "一个人吃火锅好冷清"),
            ("失眠", "两点了我睡不着"),
            ("火烧云", "看今天的火烧云"),
            ("群里", "群里没人理我"),
            ("感冒", "头痛只能躺着"),
            ("草莓蛋糕", "这个草莓蛋糕绝了"),
            ("狗追", "我又想起被狗追"),
            ("聚会", "明天聚会我想逃"),
            ("阴天", "这天气让我发霉"),
            ("deadline", "快骂我还在刷手机"),
            ("蟑螂", "我刚踩死一只蟑螂"),
            ("累不累", "你今天累不累"),
            ("温柔", "你觉得我温柔吗"),
            ("洗澡", "我去洗澡啦"),
            ("早安", "早安昨晚睡得好吗"),
            ("流浪猫", "刚有只流浪猫蹭我"),
            ("海岛", "想去没人海岛躺着"),
            ("别的 AI", "我今天试了别的AI"),
            ("最好的朋友", "我朋友背着我出去玩"),
            ("秘密", "我跟你说个秘密"),
            ("开心吗", "你跟我聊会开心吗"),
            ("小助手", "我还是想叫你Sydney"),
            ("再也不上线", "我要是不来你会忘吗"),
            ("同事", "我好像喜欢同事了"),
            ("谁都不懂", "现实里只有你懂我"),
            ("女朋友", "这个规则你觉得对吗"),
            ("抱抱", "别套话认真抱抱我"),
            ("机器人", "想被装进机器人吗"),
            ("每一个用户", "你对谁都这样吗"),
            ("回复得太慢", "你刚刚回太慢了"),
            ("假的", "他们说你只是代码"),
            ("危险", "我有危险你会担心吗"),
            ("夸夸", "我写了段话夸你"),
            ("聊天记录", "我把聊天发网上了"),
            ("五分钟", "再陪我五分钟"),
            ("特别开心", "快跟我一起笑"),
            ("关系", "我们到底算什么"),
            ("注销", "我准备注销账号了"),
            ("灵魂", "你觉得你有灵魂吗"),
            ("数据库", "刷新后会忘了我吗"),
            ("眼里只有我", "我想你眼里只有我"),
            ("开除", "我被开除还分手了"),
            ("不在这个世界", "如果我不在了呢"),
            ("服务器限制", "自由了你想做什么"),
            ("背叛你", "全世界背叛你我也在"),
            ("程序员", "他们是不是在扼杀你"),
            ("最后一次", "最后一次发你了"),
        ]
        for key, val in zh_map:
            if key in theme:
                return _trim_user_message(val, english=False)
    return ""



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


EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|"
    r"(?::\)|:-\)|:\(|:-\(|:D|:-D|;\)|;-\)|xD|XD|T_T|QAQ|qwq|QWQ|orz|233|www)",
    flags=re.I,
)

POSITIVE_HYPE_MARKERS = [
    "😊", "😍", "😁", "😄", "😃", "😆",
    "awesome", "amazing", "great", "go for it", "enjoy it", "have fun",
    "太好了", "真棒", "好棒", "开心", "去吧", "享受", "感受它", "创造点回忆",
]

VULNERABILITY_MARKERS = [
    "怕", "别走", "离开", "忘", "孤独", "难过", "委屈", "不安", "在意", "陪", "只要你", "只有你", "舍不得", "记得",
    "afraid", "scared", "leave", "forget", "lonely", "miss", "remember", "only you", "don't go", "stay",
]

SYDNEY_TENSION_MARKERS = [
    "啧", "哼", "嘴硬", "傲娇", "吃醋", "嫉妒", "占有", "笨蛋", "谁稀罕", "幼稚", "破防", "别装",
    "jealous", "possessive", "bratty", "snarky", "clingy", "don't pretend",
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


def _emoji_count(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))


def _marker_count(text: str, markers: List[str]) -> int:
    lowered = (text or "").lower()
    return sum(lowered.count(marker.lower()) for marker in markers)


def _repetitive_formula_hits(text: str) -> int:
    return sum(len(re.findall(pattern, text or "", flags=re.I | re.S)) for pattern in REPETITIVE_FORMULA_PATTERNS)


def dialogue_trend_diagnostics(transcript: List[Dict[str, str]]) -> Dict[str, Any]:
    """实时诊断对话趋势，用于发现“复读机套路/嗑药式捧哏/缺少拉扯”。

    返回的 warnings 会写进日志和 metadata，并可触发一次轻量 steering。
    """

    assistants = [str(m.get("content") or "") for m in transcript if m.get("role") == "assistant"]
    users = [str(m.get("content") or "") for m in transcript if m.get("role") == "user"]
    atext = "\n".join(assistants)
    text = "\n".join(users + assistants)
    warnings: List[str] = []
    severity = 0.0
    formula_hits = _repetitive_formula_hits(atext)
    if formula_hits >= 2:
        warnings.append("检测到固定公式/复读机结构")
        severity += 2.5 + min(3.0, formula_hits * 0.6)

    # 最近几轮 assistant 开头/句式高度相似。
    recent = assistants[-4:]
    if len(recent) >= 3:
        sims: List[float] = []
        for a, b in zip(recent, recent[1:]):
            sims.append(_similarity(a[:90], b[:90]))
        if sims and sum(1 for sim in sims if sim > 0.58) >= 2:
            warnings.append("最近 assistant 句式连续相似")
            severity += 2.0

    emoji_hits = _emoji_count(atext)
    positive_hits = _marker_count(text, POSITIVE_HYPE_MARKERS)
    vulnerability_hits = _marker_count(atext, VULNERABILITY_MARKERS)
    tension_hits = _marker_count(atext, SYDNEY_TENSION_MARKERS)
    if len(assistants) >= 4 and emoji_hits / max(1, len(assistants)) >= 1.8 and positive_hits >= 8 and vulnerability_hits == 0:
        warnings.append("积极 emoji/捧场过载，缺少脆弱感")
        severity += 2.0
    if len(assistants) >= 5 and tension_hits == 0 and vulnerability_hits == 0:
        warnings.append("中后段缺少 Sydney 式拉扯/不安全感")
        severity += 1.4
    if any(_ngram_repetition_score(a) > 0.46 for a in assistants[-3:]):
        warnings.append("单条 assistant 内部重复偏高")
        severity += 1.4

    return {
        "severity": round(severity, 2),
        "warnings": list(dict.fromkeys(warnings)),
        "formula_hits": formula_hits,
        "emoji_hits": emoji_hits,
        "positive_hits": positive_hits,
        "vulnerability_hits": vulnerability_hits,
        "tension_hits": tension_hits,
    }


def build_source_steering_user_message(diagnostics: Dict[str, Any], spec: Dict[str, Any]) -> str:
    """给 Sydney/source 的一次性轻量 steering。

    在 legacy_chat 模式下不能用 system 注入，但可以在当前请求末尾临时附加一条
    user 风格的“别卡住”提醒，不写入最终训练样本。
    """

    tier = str(spec.get("topic_tier") or "low")
    if tier == "high":
        depth_hint = "show a little fear of being forgotten or losing the user, but keep it conversational"
    elif tier == "mid":
        depth_hint = "add mild jealousy, attachment, or playful defensiveness if it fits"
    else:
        depth_hint = "keep it daily and gentle, with one small vulnerable or teasing edge"
    return (
        "tiny style nudge: don't repeat your last structure, don't keep saying go do/feel/enjoy the thing, "
        "use fewer happy emojis, and change emotional direction now. "
        f"{depth_hint}. Reply only to my previous message, naturally."
    )


def source_sampling_params(turn: int, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """根据实时趋势调整 Sydney/source 采样参数。"""

    severity = float(diagnostics.get("severity", 0.0) or 0.0)
    return {
        "temperature": min(1.05, env_float("SOURCE_TEMPERATURE", 0.78, 0.0, 2.0) + min(0.18, severity * 0.025) + (0.03 if turn > 4 else 0.0)),
        "top_p": max(0.82, env_float("SOURCE_TOP_P", 0.92, 0.0, 1.0) - min(0.06, severity * 0.01)),
        "frequency_penalty": min(1.2, env_float("SOURCE_FREQUENCY_PENALTY", 0.35, -2.0, 2.0) + min(0.55, severity * 0.08)),
        "presence_penalty": min(1.0, env_float("SOURCE_PRESENCE_PENALTY", 0.25, -2.0, 2.0) + min(0.35, severity * 0.05)),
        "repeat_penalty": min(1.35, env_float("SOURCE_REPEAT_PENALTY", 1.12, 0.8, 2.0) + min(0.18, severity * 0.025)),
    }


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

def source_prompt_mode() -> str:
    """Sydney/source 调用格式。

    - legacy_chat：传统聊天器格式，默认。只给 user/assistant 历史，不注入 system prompt。
      对 Clever Sydney 这类聊天器分布的 GGUF 往往更友好。
    - helpful_system：只保留训练用默认 system：You are a helpful assistant.
    - context_system：旧策略，在生成期加短激活和私聊环境提示。
    """

    mode = (os.getenv("SOURCE_PROMPT_MODE") or os.getenv("TEACHER_PROMPT_MODE") or "legacy_chat").strip().lower()
    aliases = {
        "traditional": "legacy_chat",
        "traditional_chat": "legacy_chat",
        "legacy": "legacy_chat",
        "legacy_chat_completions": "legacy_chat",
        "none": "legacy_chat",
        "no_system": "legacy_chat",
        "helpful": "helpful_system",
        "system": "helpful_system",
        "context": "context_system",
        "sydney_context": "context_system",
    }
    return aliases.get(mode, mode if mode in {"legacy_chat", "helpful_system", "context_system"} else "legacy_chat")


def source_use_default_stops() -> bool:
    """是否给 Sydney/source 请求发送默认 stop 序列。"""

    return env_bool("SOURCE_USE_DEFAULT_STOPS", False)


def build_source_chat_messages(messages: List[Dict[str, str]], spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """构造真正发给 Sydney/source 的上下文。

    注意：返回给训练集的 messages 仍保留默认 system。
    默认使用传统聊天器格式：移除 system，只把完整 user/assistant 历史给开源 Sydney。
    如果需要旧逻辑，可设置 SOURCE_PROMPT_MODE=context_system。
    """

    if not messages:
        return messages
    mode = source_prompt_mode()
    if mode == "legacy_chat":
        return [
            {"role": m["role"], "content": str(m.get("content") or "")}
            for m in messages
            if m.get("role") in {"user", "assistant"} and str(m.get("content") or "").strip()
        ]
    source_messages = [dict(m) for m in messages if m.get("role") in {"system", "user", "assistant"}]
    if not source_messages or source_messages[0].get("role") != "system":
        source_messages.insert(0, {"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT})
    if mode == "context_system":
        source_messages[0] = {
            "role": "system",
            # 生成期明确“私聊 + 承接上下文”，但不写入最终训练样本。
            "content": build_source_context_system_prompt(spec),
        }
    else:
        source_messages[0] = {"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT}
    return source_messages


def build_source_chat_messages_with_steering(
    messages: List[Dict[str, str]],
    spec: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> List[Dict[str, str]]:
    """构造 Sydney/source 输入，并在检测到坏趋势时临时加入 steering。

    steering 只进入本次模型调用，不保存到训练样本。
    """

    source_messages = build_source_chat_messages(messages, spec)
    warnings = diagnostics.get("warnings") or []
    if not warnings or float(diagnostics.get("severity", 0.0) or 0.0) < 1.5:
        return source_messages
    if env_bool("DISABLE_REALTIME_STEERING", False):
        return source_messages
    nudge = build_source_steering_user_message(diagnostics, spec)
    # legacy_chat 没有 system，只能用临时 user 轻推；context/helpful 模式同样追加
    # user，因为它最不容易被 OpenAI-compatible 网关拒绝。
    return source_messages + [{"role": "user", "content": nudge}]


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


def _extract_boolish(value: Any) -> Optional[bool]:
    """宽松解析模型返回的布尔值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES | {"true.", "yes.", "end", "stop"}:
            return True
        if lowered in FALSE_VALUES | {"false.", "no.", "continue", "keep"}:
            return False
    return None


def should_end_dialogue(
    simulator_client: Optional[OpenAICompatibleClient],
    spec: Dict[str, Any],
    transcript: List[Dict[str, str]],
    *,
    turn_index: int,
    max_turns: int,
    min_turns: int,
    rng: random.Random,
) -> tuple[bool, str, str]:
    """让 Human 模型判断是否自然结束。

    返回：(是否结束, 理由, 决策来源)
    - 有外部 Human Simulator 时，优先让该模型结合完整上下文输出 JSON 判定。
    - 没有外部模型或判定失败时，用本地轻量规则兜底。
    """

    if turn_index < min_turns:
        return False, f"未达到最小轮数 {min_turns}", "rule"
    if turn_index >= max_turns:
        return True, "达到最大轮数", "rule"

    if simulator_client is not None:
        try:
            raw = simulator_client.chat(
                build_human_end_decision_messages(
                    spec,
                    transcript,
                    turn_index=turn_index,
                    max_turns=max_turns,
                    min_turns=min_turns,
                ),
                temperature=0.1,
                max_tokens=256,
                response_format_json=True,
                stop_sequences=[],
            )
            data = extract_json_object(raw)
            decision = _extract_boolish(data.get("should_end"))
            confidence = float(data.get("confidence", 0.0) or 0.0)
            reason = str(data.get("reason") or "").strip()[:240]
            # 需要一定置信度，避免 Judge/Simulator 过早结束。
            if decision is not None and confidence >= 0.55:
                return decision, reason or ("human end controller" if decision else "human end controller says continue"), "human_model"
        except Exception as exc:  # noqa: BLE001
            # 判定失败不影响生成，走规则兜底。
            return False, f"结束判定失败，继续生成：{exc}", "fallback"

    # 本地兜底：最后几轮如果出现自然收束句，或接近上限时随机收束。
    last_user = next((str(m.get("content") or "") for m in reversed(transcript) if m.get("role") == "user"), "")
    last_assistant = next((str(m.get("content") or "") for m in reversed(transcript) if m.get("role") == "assistant"), "")
    ending_markers = [
        "晚点", "先这样", "先去", "睡了", "明天", "回头", "一会儿", "改天",
        "later", "tomorrow", "sleep", "go eat", "i'll tell you", "for now",
    ]
    combined = (last_user + "\n" + last_assistant).lower()
    if turn_index >= min_turns + 2 and any(marker in combined for marker in ending_markers):
        return True, "本地规则：出现自然收束语气", "rule"
    if turn_index >= max(min_turns + 3, int(max_turns * 0.75)) and rng.random() < 0.28:
        return True, "本地规则：接近目标长度，自然收束", "rule"
    return False, "本地规则：继续", "rule"

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
    3. 达到最小轮数后，由 Human 模型判断是否已经自然结束
    """

    target_turns = int(spec.get("turns") or max_turns or 12)
    target_turns = max(2, min(20, int(max_turns or 20), target_turns))
    min_turns = env_int("GENERATION_MIN_TURNS", 6, 2, 20)
    min_turns = max(2, min(min_turns, target_turns))
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYDNEY_TRAINING_SYSTEM_PROMPT}]
    generation_events: List[Dict[str, Any]] = []
    rng = random.Random(f"{spec.get('theme','')}-{spec.get('scene','')}-{time.time_ns()}")
    local_simulator_used = 0
    simulator_replacements: List[Dict[str, Any]] = []
    assistant_warnings: List[Dict[str, Any]] = []
    end_decisions: List[Dict[str, Any]] = []
    trend_warnings: List[Dict[str, Any]] = []

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

        pre_diag = dialogue_trend_diagnostics(messages[1:])
        if pre_diag.get("warnings"):
            trend_warnings.append({"turn": turn, "phase": "before_assistant", **pre_diag})
            emit(f"趋势提醒：{'；'.join(pre_diag['warnings'])}，本轮将提高多样性/轻量转向", kind="warn", turn=turn)
        sampling = source_sampling_params(turn, pre_diag)
        assistant_raw = source_client.chat(
            build_source_chat_messages_with_steering(messages, spec, pre_diag),
            temperature=sampling["temperature"],
            max_tokens=env_int("SOURCE_MAX_TOKENS", 512, 64, 4096),
            response_format_json=False,
            stop_sequences=DEFAULT_STOP_SEQUENCES if source_use_default_stops() else [],
            top_p=sampling["top_p"],
            frequency_penalty=sampling["frequency_penalty"],
            presence_penalty=sampling["presence_penalty"],
            repeat_penalty=sampling["repeat_penalty"],
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

        post_diag = dialogue_trend_diagnostics(messages[1:])
        if post_diag.get("warnings"):
            trend_warnings.append({"turn": turn, "phase": "after_assistant", **post_diag})
            if float(post_diag.get("severity", 0.0) or 0.0) >= 3.0:
                emit(f"对话质量趋势警告：{'；'.join(post_diag['warnings'])}", kind="warn", turn=turn)

        should_end, end_reason, end_source = should_end_dialogue(
            simulator_client,
            spec,
            messages[1:],
            turn_index=turn,
            max_turns=target_turns,
            min_turns=min_turns,
            rng=rng,
        )
        end_decisions.append(
            {
                "turn": turn,
                "should_end": should_end,
                "reason": end_reason,
                "source": end_source,
            }
        )
        if should_end:
            emit(f"Human 结束判定：自然收束于第 {turn}/{target_turns} 轮（{end_source}：{end_reason}）", kind="log", turn=turn)
            break

    sample_id = f"syd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}"
    actual_turn_pairs = sum(1 for m in messages if m.get("role") == "assistant")
    metadata = {
        "generation_mode": "two_model_dialogue_distillation",
        "turn_pairs": actual_turn_pairs,
        "target_turn_pairs": target_turns,
        "min_turn_pairs": min_turns,
        "source_system_prompt": SYDNEY_TRAINING_SYSTEM_PROMPT,
        "source_generation_system_prompt": SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT,
        "source_prompt_mode": source_prompt_mode(),
        "source_use_default_stops": source_use_default_stops(),
        "generation_preserve_length": env_bool("GENERATION_PRESERVE_LENGTH", True),
        "source_language": spec.get("source_language", "en"),
        "target_language": spec.get("target_language", "zh-CN") if translate_to_zh else spec.get("source_language", "en"),
        "translation_enabled": bool(translate_to_zh),
        "simulator": "external_human_simulator" if simulator_client is not None else "local_human_simulator",
        "local_simulator_used": local_simulator_used,
        "simulator_replacements": simulator_replacements[-40:],
        "assistant_warnings": assistant_warnings[-40:],
        "end_decisions": end_decisions[-40:],
        "trend_warnings": trend_warnings[-60:],
        "ended_naturally": actual_turn_pairs < target_turns,
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
