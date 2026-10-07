try:
    import httpx2 as httpx  # openai>=3 depends on httpx2
except ImportError:
    import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

import llm


def completion(message: dict, finish="stop") -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "kimi-k3",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    return llm.LLMClient()


def test_defaults(client):
    assert client.model == "kimi-k3"
    assert client.reasoning_effort == "max"
    assert str(client.client.base_url).startswith("https://api.moonshot.ai/v1")


def test_missing_key(monkeypatch):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "get_setting", lambda name, default=None: default)
    with pytest.raises(llm.LLMError, match="KIMI_API_KEY"):
        llm.LLMClient()


def test_tool_calls_and_reasoning_content_preserved(client, monkeypatch):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return completion({
            "role": "assistant", "content": None, "reasoning_content": "private thoughts",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search_web", "arguments": '{"query":"פטור"}'}}],
        }, finish="tool_calls")

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    resp = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert sent["extra_body"] == {"reasoning_effort": "max"}
    assert "temperature" not in sent
    assert resp.tool_calls[0].name == "search_web"
    assert resp.message["reasoning_content"] == "private thoughts"
    assert resp.message["tool_calls"][0]["id"] == "c1"
    assert resp.usage["prompt_tokens"] == 10


def test_rejected_optional_param_is_dropped_and_retried(client, monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs.get("extra_body", {}):
            req = httpx.Request("POST", "https://api.moonshot.ai/v1/chat/completions")
            raise openai.BadRequestError("response_format not supported", response=httpx.Response(400, request=req), body=None)
        return completion({"role": "assistant", "content": "{}"})

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    resp = client.chat([{"role": "user", "content": "x"}], json_mode=True)
    assert resp.content == "{}"
    assert len(calls) == 2 and "response_format" in client.dropped_params
    assert client.warnings


def test_glm_provider_switch(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "glm")
    monkeypatch.setenv("GLM_API_KEY", "k")
    c = llm.LLMClient()
    assert c.model == "glm-5.3" and c.reasoning_effort is None


def test_parse_tool_arguments():
    assert llm.parse_tool_arguments('{"a": 1}') == {"a": 1}
    for bad in ("{x", "[1]"):
        with pytest.raises(ValueError):
            llm.parse_tool_arguments(bad)
