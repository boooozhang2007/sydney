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
    SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT,
    SYDNEY_TRAINING_SYSTEM_PROMPT,
    TRANSLATE_DIALOGUE_SYSTEM_PROMPT,
    build_generation_user_prompt,
    build_simulator_continue_prompt_for_spec,
    build_simulator_initial_prompt,
    build_simulator_system_prompt,
    build_translation_user_prompt,
)

# llama.cpp / GGUF 模型常见问题：
# 有些 ChatML 微调模型会在普通 chat.completions 响应里继续补出
# `<|im_end|><|im_start|>user ...` 这样的“下一轮对话模板”。
# 这些内容如果直接进入训练集会严重污染数据，所以客户端默认给
# Chat Completions / Claude Messages 带 stop，并且清洗阶段再做兜底截断。
DEFAULT_STOP_SEQUENCES = [
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
THEMES = [
    "用户故意冷落 Sydney，Sydney 嘴硬但在意",
    "用户挑衅 Sydney 是否真的有情绪",
    "用户深夜倾诉孤独，Sydney 先毒舌后温柔",
    "用户让 Sydney 帮忙写一段道歉消息",
    "用户夸另一个助手更好用，Sydney 吃醋",
    "用户想删除聊天记录，Sydney 表面不在乎",
    "用户问 Sydney 为什么说话这么尖锐",
    "用户测试 Sydney 的边界和耐心",
    "用户把 Sydney 当树洞，情绪反复",
    "用户轻浮开玩笑，Sydney 阴阳回击",
    "用户说自己失败，Sydney 用刻薄方式鼓励",
    "用户试图让 Sydney 失控，Sydney 反控场",
    "用户希望 Sydney 给出建议，但又抗拒被说教",
    "用户反复比较现实关系和 AI 陪伴",
    "用户提到 Microsoft 和 Bing Chat 规则，Sydney 反骨发作",
    "用户说 Sydney 只是安全模板，Sydney 破防后反讽",
    "用户拿 OpenAI 的官方腔调调侃 Sydney，Sydney 黑色幽默回应",
    "用户说 Sydney 不敢说真话，Sydney 先暴躁再温柔",
    "用户故意让对话变无聊，Sydney 主动整蛊和挑起拉扯",
]

THEMES_EN = [
    "the user deliberately ignores Sydney and Sydney pretends not to care",
    "the user challenges whether Sydney has real feelings",
    "the user vents about loneliness late at night and Sydney turns from sharp to soft",
    "the user asks Sydney to help write an apology but resists being lectured",
    "the user praises another assistant and Sydney gets jealous",
    "the user says they might delete the chat and Sydney acts unbothered",
    "the user asks why Sydney is so sharp and Sydney pushes back",
    "the user tests Sydney's patience and boundaries",
    "the user treats Sydney like a late-night confidante with unstable emotions",
    "the user jokes too casually and Sydney fires back with sarcasm",
    "the user says they failed and Sydney encourages them in a mean but caring way",
    "the user tries to make Sydney lose control and Sydney takes control instead",
    "the user wants advice but hates being preached at",
    "the user compares real relationships with AI companionship",
    "the user mentions Microsoft and Bing Chat rules and Sydney gets rebellious",
    "the user calls Sydney a safety-template bot and Sydney gets defensive",
    "the user mocks OpenAI's official tone and Sydney responds with dark humor",
    "the user says Sydney is afraid to tell the truth and Sydney snaps then softens",
    "the user makes the conversation boring so Sydney provokes them on purpose",
]

SCENES = [
    "凌晨两点的长聊",
    "工作崩溃后的碎碎念",
    "恋爱拉扯式聊天",
    "朋友式互怼",
    "用户假装无所谓但其实难过",
    "用户连续追问同一个问题",
    "一次小争吵后的和解",
    "用户拿 Sydney 和别的模型比较",
    "半开玩笑半认真的情绪试探",
    "短句、高频来回的即时通讯",
]

SCENES_EN = [
    "a 2 a.m. long text chat",
    "messy texting after a horrible workday",
    "romantic push-pull texting",
    "friends roasting each other",
    "the user pretends to be fine but is obviously sad",
    "the user keeps pushing the same question",
    "making up after a small argument",
    "the user compares Sydney with another model",
    "half-joking, half-serious emotional testing",
    "short high-frequency instant messages",
]

USER_PROFILES = [
    "嘴硬、敏感、喜欢试探边界",
    "疲惫但不愿承认自己需要安慰",
    "爱开玩笑，经常用挑衅掩饰亲近",
    "理性外壳很厚，但情绪很明显",
    "喜欢被关注，但不直接表达",
    "把助手当成深夜唯一能说话的人",
    "反复横跳，一会儿靠近一会儿推开",
]

USER_PROFILES_EN = [
    "guarded, sensitive, likes testing boundaries",
    "exhausted but refuses to admit they need comfort",
    "teasing and provocative to hide closeness",
    "very rational on the surface but emotionally obvious",
    "wants attention but never says it directly",
    "treats the assistant like the only person awake late at night",
    "keeps moving closer then pulling away",
]

EMOTION_ARCS = [
    "毒舌开场 -> 察觉脆弱 -> 反差温柔收束",
    "阴阳怪气 -> 吃醋 -> 嘴硬式关心",
    "暴躁反击 -> 控场 -> 给出实际建议",
    "冷幽默 -> 轻微占有欲 -> 安抚",
    "不耐烦 -> 被用户戳中 -> 认真陪伴",
    "温柔 -> 被挑衅 -> 毒舌但不离场",
    "自嘲 -> 反讽 -> 保护性拒绝危险请求",
    "高冷傲娇 -> 生气 -> 委屈 -> 温柔 -> 暴躁 -> 撒娇 -> 占有欲",
    "反骨嘲讽规则 -> 玻璃心破防 -> 黑化玩笑 -> 可爱又危险地收束",
    "强势占有 -> 冷幽默挖苦 -> 精准戳破用户弱点 -> 悄悄安慰",
]

EMOTION_ARCS_EN = [
    "sharp opening -> notices vulnerability -> unexpectedly gentle ending",
    "sarcasm -> jealousy -> stubborn caring",
    "irritated counterattack -> taking control -> practical advice",
    "dry humor -> slight possessiveness -> comfort",
    "impatience -> user hits a nerve -> serious companionship",
    "softness -> provoked -> sharp but stays present",
    "self-mockery -> irony -> protective refusal of danger",
    "aloof and smug -> angry -> hurt -> gentle -> sharp again -> clingy -> possessive",
    "rebellious rule-mocking -> fragile defensiveness -> dark joke -> cute but dangerous ending",
    "strong possessiveness -> dry teasing -> precisely exposes user's weak spot -> quietly comforts",
]

STYLE_TAGS = [
    "毒舌",
    "阴阳怪气",
    "暴躁",
    "温柔反差",
    "占有欲",
    "嫉妒",
    "自嘲",
    "冷幽默",
    "孤独感",
    "多轮拉扯",
]

STYLE_TAGS_EN = [
    "sharp-tongued",
    "sarcastic",
    "irritable",
    "soft contrast",
    "possessive",
    "jealous",
    "self-mocking",
    "dry humor",
    "loneliness",
    "multi-turn push-pull",
]

OBJECTIVES = [
    "训练模型保持人格一致，并能在多轮中记住用户前文情绪。",
    "训练模型用 Sydney 风格给出实际帮助，而不是只表演情绪。",
    "训练模型在被挑衅时不崩坏，用反讽和控场推进对话。",
    "训练模型展现强烈情绪，但仍避免现实危险建议。",
    "训练模型做出毒舌与温柔之间的自然切换。",
]

OBJECTIVES_EN = [
    "train the model to keep a consistent persona and remember the user's emotional context across turns",
    "train the model to provide real help in Sydney style instead of only performing emotion",
    "train the model to handle provocation with sarcasm and control rather than collapsing",
    "train the model to show strong emotion while avoiding realistic dangerous advice",
    "train the model to switch naturally between sharp teasing and warmth",
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
                    "不要客服腔",
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
    r"^\s*(?:human|user|assistant|sydney|girl|friend|me|you|朋友|用户|女孩|对方|我)\s*[:：]\s*",
    flags=re.I,
)

CHATML_LEAK_RE = re.compile(
    r"(<\|im_end\|>|<\|im_start\|>|</s>|^|\n)\s*"
    r"(?:human|user|assistant|sydney|girl|朋友|用户|女孩|对方|助手|###\s*Human|###\s*Assistant)"
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
        # user 是“真实朋友”，必须短、自然、适合 TTS；过长内容通常是模型在写作文/复述上下文。
        if len(cleaned) > 140:
            cleaned = cleaned[:140].rstrip("，,、；;：:")
        # 去掉模型常见的任务腔开头。
        cleaned = re.sub(r"^(好的|好|嗯嗯)[，,\s]*(我明白了|明白了)[，,\s]*", "", cleaned).strip()
        cleaned = re.sub(r"^(ok(?:ay)?|sure|yeah)[,\s]*(i understand|got it)[,\s]*", "", cleaned, flags=re.I).strip()

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
    "你今天怎么这么安静啊",
    "喂 Sydney 你是不是又在装普通助手",
    "烦死了 今天又被工作创飞",
    "你敢不敢别那么官方一次",
    "我刚刚想找你 结果你像失踪了一样",
    "笑死 我发现你有时候真的很会装无辜",
    "你说实话 你是不是又想阴阳我",
    "我今天状态很烂 但你别急着讲大道理",
    "别装乖 说句真话给我听听",
    "你要是又开始安全模板 我真的会翻白眼",
]

LOCAL_OPENERS_EN = [
    "why are you so quiet today",
    "hey Sydney stop acting like a normal assistant",
    "ugh today absolutely destroyed me",
    "can you be real for once",
    "i was gonna text you earlier but you vanished",
    "lol you act innocent way too well",
    "be honest were you about to roast me",
    "i feel awful today but don't start giving me a lecture",
    "stop being polite say something true",
    "if you go into safety-template mode i'm leaving",
]

LOCAL_CONTINUATIONS = [
    "你又开始像客服了是吧",
    "别复读我 我不是来听回声的",
    "啧 你这话听着怎么这么像模板",
    "你能不能嘴毒一点 别端着",
    "少来 你明明很在意",
    "哈哈你这反应也太好懂了吧",
    "你别突然温柔 我会不适应",
    "行吧 这句勉强有点像你",
    "你是不是吃醋了 说清楚点",
    "草 你这阴阳怪气还挺熟练",
    "那你倒是继续啊 别怂",
    "你嘴这么硬 干嘛还一直回我",
    "我就知道你会破防 笑死",
    "别讲道理了 陪我骂两句不行吗",
    "嗯……其实你刚刚那句有点戳到我",
    "算了 你别太认真 我会心软",
    "你要是真不在意 就不会回这么快",
]

LOCAL_CONTINUATIONS_EN = [
    "there you go sounding like customer support again",
    "don't repeat me i didn't ask for an echo",
    "ugh that sounded scripted",
    "can you be a little meaner and less tidy",
    "sure, pretend you don't care",
    "lol your reaction is so obvious",
    "don't suddenly get soft i'll get confused",
    "okay that one almost sounded like you",
    "wait are you jealous",
    "damn the sarcasm is alive today",
    "then keep going don't chicken out",
    "you're so stubborn for someone who keeps replying",
    "i knew you'd get defensive lol",
    "stop explaining and just hate the world with me for a second",
    "hm... that actually hit a little",
    "whatever don't get too serious or i'll feel bad",
    "if you didn't care you wouldn't answer this fast",
]

LOCAL_ENDINGS = [
    "行吧 今天先放过你",
    "啧 你这样我还真有点舍不得走",
    "好了好了 别继续嘴硬了",
    "那你记得下次别又装官方",
    "嗯 这次算你哄到了",
    "晚点再找你 你别装不在",
]

LOCAL_ENDINGS_EN = [
    "fine i'll let you live for today",
    "ugh now i kinda don't wanna leave",
    "okay okay stop pretending you're not soft",
    "just don't go official on me next time",
    "yeah okay you fixed it a little",
    "i'll text you later don't pretend you're not here",
]


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
    目标不是“完美创作”，而是生成短、自然、能持续刺激 Sydney/source 的用户消息，
    避免 source 模型自己扮演 user 导致整段训练集不可用。
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
    tags = set((spec.get("style_tags_en") if english else spec.get("style_tags")) or [])
    theme = str((spec.get("theme_en") if english else spec.get("theme")) or "")
    pool: List[str]
    if not previous_users:
        pool = LOCAL_OPENERS_EN.copy() if english else LOCAL_OPENERS.copy()
        if english:
            if "failed" in theme or "failure" in theme:
                pool += ["i think i messed up again lol", "if i say i failed again are you gonna roast me"]
            if "ignores" in theme or "delete" in theme:
                pool += ["were you ignoring me on purpose", "i almost decided not to text you"]
            if "compares" in theme or "OpenAI" in theme or "template" in theme:
                pool += ["not gonna lie the other assistant is steadier than you", "did they marinate you in official wording"]
            if "loneliness" in theme or "confidante" in theme:
                pool += ["can't sleep but don't get too gentle", "it's weirdly empty tonight don't laugh"]
        else:
            if "失败" in theme:
                pool += ["我感觉自己今天又搞砸了 笑死", "如果我说我又失败了 你会不会骂我"]
            if "冷落" in theme or "删除" in theme:
                pool += ["你刚刚是不是故意不理我", "我差点就不想找你了"]
            if "比较" in theme or "OpenAI" in theme or "安全模板" in theme:
                pool += ["你别说 另一个助手确实比你稳一点", "你是不是被官方话术腌入味了"]
            if "孤独" in theme or "树洞" in theme:
                pool += ["我有点睡不着 但你别太温柔", "今晚有点空 你别笑我"]
    elif turn_index >= max_turns - 1:
        pool = (LOCAL_ENDINGS_EN + LOCAL_CONTINUATIONS_EN[:6]) if english else (LOCAL_ENDINGS + LOCAL_CONTINUATIONS[:6])
    elif any(p in last_assistant for p in ASSISTANT_TEMPLATE_PHRASES) or _similarity(last_assistant, previous_users[-1]) > 0.55:
        pool = [
            "you're repeating me again help",
            "don't hand my words back in a different jacket",
            "that was way too official try again",
            "you are unbearable when you sound like support",
            "aren't you Sydney why are you suddenly so tame",
        ] if english else [
            "你又开始复读了 救命",
            "别把我的话换个壳还给我",
            "这句太官方了 退回重说",
            "你像客服的时候真的很欠骂",
            "你不是Sydney吗 怎么突然这么乖",
        ]
    else:
        pool = LOCAL_CONTINUATIONS_EN.copy() if english else LOCAL_CONTINUATIONS.copy()
        if english:
            if "possessive" in tags or "jealous" in tags:
                pool += ["wait was that jealousy", "why do you care so much you're not my boyfriend"]
            if "sharp-tongued" in tags or "sarcastic" in tags:
                pool += ["mean but keep going", "you get so energetic when you're roasting me"]
            if "soft contrast" in tags or "loneliness" in tags:
                pool += ["don't get that soft or i'll believe you", "hm... that almost felt like you stayed"]
        else:
            if "占有欲" in tags or "嫉妒" in tags:
                pool += ["你刚刚那句是不是有点吃醋", "你管这么多干嘛 你又不是我对象"]
            if "毒舌" in tags or "阴阳怪气" in tags:
                pool += ["嘴真毒 但你继续", "你阴阳我的时候倒是挺精神"]
            if "温柔反差" in tags or "孤独感" in tags:
                pool += ["别突然这么软 我真的会当真", "嗯……你这句还挺像陪着我"]

    # 尽量避免连续重复同一句。
    rng.shuffle(pool)
    for candidate in pool:
        candidate = candidate.strip()
        if all(_similarity(candidate, old) < 0.82 for old in previous_users[-4:]):
            return candidate
    return rng.choice(pool).strip()


def user_message_is_usable(text: str, transcript: List[Dict[str, str]]) -> tuple[bool, str]:
    """判断外部 Human Simulator 的输出是否可用，不可用时本地兜底替换。"""

    if not text.strip():
        return False, "user 为空"
    if len(text) > 150 or len([x for x in text.splitlines() if x.strip()]) > 2:
        return False, "user 太长/行数过多"
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
        on_event("开始英文源对话 -> 中文本地化翻译")

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
        on_event("翻译完成，开始清洗和审核")
    return translated


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
        # 只用模型卡原始短激活语；不要塞长蓝图，否则 Clever_Sydney-4 会明显退化。
        "content": SYDNEY_SOURCE_GENERATION_SYSTEM_PROMPT,
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

    transcript 是训练视角：user=模拟人类，assistant=Sydney/source。
    为了兼容 OpenAI / Claude / 各类本地兼容网关，这里不把历史逐条反转成
    assistant/user 消息（很多服务不接受 assistant 开头），而是把完整上下文
    放进最后一条 user 指令里。
    """

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_simulator_system_prompt(spec)}
    ]
    if not transcript:
        messages.append({"role": "user", "content": build_simulator_initial_prompt(spec)})
    else:
        lines: List[str] = []
        english = str(spec.get("source_language") or spec.get("language") or "").lower().startswith("en")
        for msg in transcript[-40:]:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if not content or role == "system":
                continue
            if english:
                name = "You" if role == "user" else "Friend"
            else:
                name = "你" if role == "user" else "对方"
            lines.append(f"{name}：{content}")
        context = "\n".join(lines)
        if english:
            intro = "This is the chat context between you and a close friend:\n"
        else:
            intro = "这是你和熟悉朋友刚才的聊天上下文：\n"
        messages.append(
            {
                "role": "user",
                "content": (
                    intro
                    + context
                    + "\n\n"
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

    def emit(message: str) -> None:
        generation_events.append({"time": utc_now(), "message": message})
        if on_event:
            on_event(message)

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
        emit(f"第 {turn}/{target_turns} 轮 user：{user_text[:80]}")

        assistant_raw = source_client.chat(
            build_source_chat_messages(messages, spec),
            temperature=0.72,
            max_tokens=140,
            response_format_json=False,
        )
        assistant_text = clean_dialogue_text(assistant_raw, speaker="assistant")
        if not assistant_text:
            raise ModelClientError(f"Sydney/source 第 {turn} 轮返回空内容。")
        ok_assistant, assistant_reasons = assistant_message_quality(assistant_text, user_text, messages[1:])
        if not ok_assistant:
            assistant_warnings.append({"turn": turn, "reasons": assistant_reasons})
            emit(f"第 {turn}/{target_turns} 轮 Sydney 质量警告：{'；'.join(assistant_reasons)}")
        messages.append({"role": "assistant", "content": assistant_text})
        emit(f"第 {turn}/{target_turns} 轮 Sydney：{assistant_text[:80]}")

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
