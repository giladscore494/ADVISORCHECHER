"""Provider-agnostic chat client.

The rest of the app only uses `LLMClient.chat()` and the `ChatResponse` /
`ToolCall` types. Provider specifics (base URL, model name, API key variable,
extra request parameters, API style) live in `PROVIDERS`, so switching between
Kimi, GLM and OpenAI is a configuration change (`LLM_PROVIDER=glm`), not a rewrite.

Kimi and GLM use the OpenAI-compatible Chat Completions API. OpenAI uses the
Responses API with function calling (`api: "responses"`): the same tool schemas
and the same internal chat-style history are converted on every call, and no
OpenAI built-in tools (such as web search) are ever enabled, so search stays on
Serper and every tool runs in this application. Requests are stateless
(`store=false`); reasoning items are kept as encrypted content in the history so a
run can be checkpointed and resumed.
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
    "openai": {
        "api": "responses",
        "label": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "model_env": "OPENAI_MODEL",
        "model": "gpt-6.1-sol",
        "effort_env": "OPENAI_REASONING_EFFORT",
        "effort": "high",
    },
}
PROVIDER_LABELS = {"kimi": "Kimi (Moonshot)", "glm": "GLM (Z.ai)", "openai": "OpenAI"}
# Keys in history messages that are internal to this app and never sent to Chat Completions endpoints.
INTERNAL_MESSAGE_KEYS = ("_responses_output",)


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


def api_key_env_name(provider: str | None = None) -> str:
    return PROVIDERS.get((provider or provider_name()).lower(), PROVIDERS["kimi"])["api_key_env"]


def configured_providers() -> list[str]:
    """Providers whose API key is set (server-side: environment, .env or Streamlit secrets)."""
    return [p for p, cfg in PROVIDERS.items() if get_setting(cfg["api_key_env"])]


def provider_model(provider: str) -> str:
    cfg = PROVIDERS[provider]
    return get_setting(cfg["model_env"], cfg["model"]) or cfg["model"]


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = (provider or provider_name()).lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM_PROVIDER '{self.provider}'. Options: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self.api = cfg.get("api", "chat")
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
        if self.api == "responses":
            return self._responses_chat(messages, tools, json_mode)
        messages = [{k: v for k, v in m.items() if k not in INTERNAL_MESSAGE_KEYS} for m in messages]
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


    # ------------------------------------------------------------ Responses API
    def _responses_chat(self, messages: list[dict], tools: list[dict] | None, json_mode: bool) -> ChatResponse:
        optional: dict[str, Any] = {}
        if self.reasoning_effort:
            optional["reasoning"] = {"effort": self.reasoning_effort}
        if json_mode:
            optional["text"] = {"format": {"type": "json_object"}}
        include_reasoning = True
        while True:
            extra = {k: v for k, v in optional.items() if k not in self.dropped_params}
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": to_responses_input(messages, include_reasoning=include_reasoning),
                "store": False,
                "include": ["reasoning.encrypted_content"],
                **extra,
            }
            if tools:
                kwargs["tools"] = to_responses_tools(tools)
                kwargs["tool_choice"] = "auto"
            started = time.monotonic()
            try:
                response = self.client.responses.create(**kwargs)
                break
            except openai.BadRequestError as exc:
                text = str(exc)
                if include_reasoning and ("encrypted_content" in text or "rs_" in text or "Item with id" in text):
                    # Replayed reasoning items were not accepted: continue from the visible history only.
                    include_reasoning = False
                    self.warnings.append("Endpoint rejected replayed reasoning items; continuing without them.")
                    continue
                rejected = next((k for k in extra if k in text), None)
                if rejected is not None:
                    self.dropped_params.add(rejected)
                    self.warnings.append(f"Endpoint rejected '{rejected}'; continuing without it.")
                    continue
                raise LLMError(f"Model request rejected: {exc}") from exc
            except openai.APITimeoutError as exc:
                raise LLMError("Model request timed out.") from exc
            except openai.APIError as exc:
                raise LLMError(f"Model API error: {exc}") from exc
        return parse_responses_output(response, time.monotonic() - started)


def to_responses_tools(tools: list[dict]) -> list[dict]:
    """Chat Completions function schemas -> Responses API function tools (no built-in tools)."""
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({"type": "function", "name": fn["name"], "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}), "strict": False})
    return out


_REPLAY_DROP = ("status",)


def to_responses_input(messages: list[dict], include_reasoning: bool = True) -> list[dict]:
    """Internal chat-style history -> Responses API input items."""
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            items.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            replay = m.get("_responses_output")
            if replay:
                for it in replay:
                    if it.get("type") == "reasoning" and (not include_reasoning or not it.get("encrypted_content")):
                        continue
                    items.append({k: v for k, v in it.items() if k not in _REPLAY_DROP})
                continue
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                items.append({"type": "function_call", "call_id": tc["id"], "name": fn.get("name", ""),
                              "arguments": fn.get("arguments") or "{}"})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": m.get("content") or ""})
    return items


def parse_responses_output(response, duration: float) -> ChatResponse:
    output = list(getattr(response, "output", None) or [])
    texts, tool_calls, replay = [], [], []
    for item in output:
        data = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        kind = data.get("type")
        if kind == "message":
            for part in data.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
            text = "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "output_text")
            if text:
                replay.append({"role": "assistant", "content": text})
        elif kind == "function_call":
            tool_calls.append(ToolCall(id=data["call_id"], name=data["name"], arguments=data.get("arguments") or "{}"))
            replay.append({"type": "function_call", "call_id": data["call_id"], "name": data["name"],
                           "arguments": data.get("arguments") or "{}"})
        elif kind == "reasoning":
            replay.append({k: v for k, v in data.items() if k in ("type", "id", "summary", "encrypted_content")})
    status = getattr(response, "status", None)
    if status == "failed":
        err = getattr(response, "error", None)
        raise LLMError(f"Model response failed: {getattr(err, 'message', err)}")
    if not output:
        raise LLMError("Model returned no output.")
    content = "".join(texts)
    history_msg: dict[str, Any] = {"role": "assistant", "content": content, "_responses_output": replay}
    if tool_calls:
        history_msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]
    usage = {}
    u = getattr(response, "usage", None)
    if u is not None:
        usage = {"prompt_tokens": getattr(u, "input_tokens", 0) or 0,
                 "completion_tokens": getattr(u, "output_tokens", 0) or 0}
        details = getattr(u, "output_tokens_details", None)
        if details is not None and getattr(details, "reasoning_tokens", None):
            usage["reasoning_tokens"] = details.reasoning_tokens
    if tool_calls:
        finish = "tool_calls"
    elif status == "incomplete":
        finish = "length"
    else:
        finish = "stop"
    return ChatResponse(content=content, tool_calls=tool_calls, finish_reason=finish, message=history_msg,
                        usage=usage, duration_s=round(duration, 2))


def parse_tool_arguments(raw: str) -> dict:
    """Parse a tool call's JSON arguments; raises ValueError on malformed input."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool arguments are not valid JSON: {exc}") from exc
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return args
