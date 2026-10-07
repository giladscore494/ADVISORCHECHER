"""Provider-agnostic chat client.

The rest of the app only uses `LLMClient.chat()` and the `ChatResponse` /
`ToolCall` types. Provider specifics (base URL, model name, API key variable,
extra request parameters) live in `PROVIDERS`, so switching from Kimi to GLM is
a configuration change (`LLM_PROVIDER=glm`), not a rewrite.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from config import get_int, get_setting

PROVIDERS: dict[str, dict[str, str]] = {
    "kimi": {
        "api_key_env": "KIMI_API_KEY",
        "base_url_env": "KIMI_BASE_URL",
        "base_url": "https://api.moonshot.ai/v1",
        "model_env": "KIMI_MODEL",
        "model": "kimi-k3",
        "effort_env": "KIMI_REASONING_EFFORT",
        "effort": "max",
    },
    "glm": {
        "api_key_env": "GLM_API_KEY",
        "base_url_env": "GLM_BASE_URL",
        "base_url": "https://api.z.ai/api/paas/v4",
        "model_env": "GLM_MODEL",
        "model": "glm-5.3",
        "effort_env": "GLM_REASONING_EFFORT",
        "effort": "",
    },
}


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as returned by the model


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    # Assistant message to append to the conversation history (includes any
    # provider-specific fields such as `reasoning_content` that must be echoed
    # back on tool-call turns). Never shown to the user.
    message: dict[str, Any]
    usage: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0


def provider_name() -> str:
    return (get_setting("LLM_PROVIDER", "kimi") or "kimi").lower()


def api_key_env_name() -> str:
    return PROVIDERS.get(provider_name(), PROVIDERS["kimi"])["api_key_env"]


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or provider_name()).lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'. Options: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        api_key = get_setting(cfg["api_key_env"])
        if not api_key:
            raise LLMError(f"Missing {cfg['api_key_env']}. Set it in the environment, .env or Streamlit secrets.")
        self.model = get_setting(cfg["model_env"], cfg["model"])
        self.reasoning_effort = get_setting(cfg["effort_env"], cfg["effort"]) or None
        self.client = OpenAI(
            api_key=api_key,
            base_url=get_setting(cfg["base_url_env"], cfg["base_url"]),
            timeout=float(get_int("LLM_TIMEOUT", 600)),
            max_retries=2,
        )
        # Optional parameters the endpoint rejected; we stop sending them.
        self.dropped_params: set[str] = set()
        self.warnings: list[str] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False) -> ChatResponse:
        optional: dict[str, Any] = {}
        if self.reasoning_effort:
            optional["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            optional["response_format"] = {"type": "json_object"}

        while True:
            extra = {k: v for k, v in optional.items() if k not in self.dropped_params}
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if extra:
                # Sent via extra_body so the SDK does not validate provider-specific values.
                kwargs["extra_body"] = extra
            started = time.monotonic()
            try:
                completion = self.client.chat.completions.create(**kwargs)
                break
            except openai.BadRequestError as exc:
                # If the endpoint rejects an optional parameter, retry once without it.
                rejected = next((k for k in extra if k in str(exc)), None)
                if rejected is None:
                    raise LLMError(f"Model request rejected: {exc}") from exc
                self.dropped_params.add(rejected)
                self.warnings.append(f"Endpoint rejected '{rejected}'; continuing without it.")
            except openai.APITimeoutError as exc:
                raise LLMError("Model request timed out.") from exc
            except openai.APIError as exc:
                raise LLMError(f"Model API error: {exc}") from exc

        duration = time.monotonic() - started
        if not completion.choices:
            raise LLMError("Model returned no choices.")
        choice = completion.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        history_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            history_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in tool_calls
            ]
        reasoning = (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
        if reasoning:
            history_msg["reasoning_content"] = reasoning

        usage = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens or 0,
                "completion_tokens": completion.usage.completion_tokens or 0,
            }
        return ChatResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            message=history_msg,
            usage=usage,
            duration_s=round(duration, 2),
        )


def parse_tool_arguments(raw: str) -> dict:
    """Parse a tool call's JSON arguments; raises ValueError on malformed input."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return args
