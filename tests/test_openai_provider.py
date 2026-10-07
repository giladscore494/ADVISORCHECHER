"""OpenAI gpt-6.1-sol via the Responses API + function calling, routed through the app's own tools."""

import json

import pytest
from openai.types.responses import Response

import agent
import llm
import search
from govdata_fixtures import TARIFF_RID, build_govdata
from helpers import EXCERPT, fake_fetch, valid_report
from local_data import GovernmentData

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"


def response(*items, status="completed"):
    return Response.model_validate({
        "id": "resp_1", "created_at": 0, "model": "gpt-6.1-sol", "object": "response", "output": list(items),
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "status": status,
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 7}},
    })


def reasoning(i=1):
    return {"type": "reasoning", "id": f"rs_{i}", "summary": [], "encrypted_content": f"ENC{i}"}


def call(call_id, name, args):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": json.dumps(args, ensure_ascii=False), "status": "completed"}


def message(text):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for k in ("OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    return llm.LLMClient("openai")


def test_openai_defaults_and_kimi_glm_preserved(client, monkeypatch):
    assert client.model == "gpt-6.1-sol" and client.reasoning_effort == "high" and client.api == "responses"
    assert str(client.client.base_url).startswith("https://api.openai.com/v1")
    assert set(llm.PROVIDERS) == {"kimi", "glm", "openai"}
    assert llm.PROVIDERS["kimi"].get("api", "chat") == "chat" and llm.PROVIDERS["glm"].get("api", "chat") == "chat"
    monkeypatch.setenv("KIMI_API_KEY", "k")
    assert "kimi" in llm.configured_providers() and "openai" in llm.configured_providers()
    assert llm.api_key_env_name("openai") == "OPENAI_API_KEY"


def test_function_call_request_and_parsing(client, monkeypatch):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return response(reasoning(), call("call_1", "search_web", {"query": "פטור רישיון"}))

    monkeypatch.setattr(client.client.responses, "create", create)
    resp = client.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}], tools=agent.TOOLS)
    assert sent["model"] == "gpt-6.1-sol" and sent["reasoning"] == {"effort": "high"}
    assert sent["store"] is False and sent["include"] == ["reasoning.encrypted_content"]
    # Only the app's own function tools: no OpenAI built-in web search.
    assert all(t["type"] == "function" for t in sent["tools"])
    assert {t["name"] for t in sent["tools"]} == {t["function"]["name"] for t in agent.TOOLS}
    assert "web_search" not in json.dumps(sent["tools"]) and sent["tool_choice"] == "auto"
    assert sent["input"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    assert resp.tool_calls[0].id == "call_1" and resp.tool_calls[0].name == "search_web"
    assert json.loads(resp.tool_calls[0].arguments) == {"query": "פטור רישיון"}
    assert resp.finish_reason == "tool_calls"
    assert resp.usage == {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 7}
    assert resp.message["tool_calls"][0]["id"] == "call_1"
    assert resp.message["_responses_output"][0] == {"type": "reasoning", "id": "rs_1", "summary": [],
                                                    "encrypted_content": "ENC1"}


def test_history_conversion_replays_reasoning_and_tool_outputs():
    history = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
         "_responses_output": [reasoning(), {"type": "function_call", "call_id": "call_1", "name": "search_web",
                                             "arguments": "{}"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"results": []}'},
        # A message produced by a chat-completions provider (e.g. after resuming with another provider).
        {"role": "assistant", "content": "thinking", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "fetch_url", "arguments": '{"url": "x"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "page"},
    ]
    items = llm.to_responses_input(history)
    assert items[2] == reasoning()
    assert items[3] == {"type": "function_call", "call_id": "call_1", "name": "search_web", "arguments": "{}"}
    assert items[4] == {"type": "function_call_output", "call_id": "call_1", "output": '{"results": []}'}
    assert items[5] == {"role": "assistant", "content": "thinking"}
    assert items[6]["type"] == "function_call" and items[7]["call_id"] == "c2"
    assert all(i.get("type") != "reasoning" for i in llm.to_responses_input(history, include_reasoning=False))


def test_json_mode_and_rejected_reasoning_replay_fallback(client, monkeypatch):
    import httpx2 as httpx
    import openai

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            req = httpx.Request("POST", "https://api.openai.com/v1/responses")
            raise openai.BadRequestError("Item with id 'rs_1' not found (reasoning)", response=httpx.Response(
                400, request=req), body=None)
        return response(message('{"opportunities": []}'))

    monkeypatch.setattr(client.client.responses, "create", create)
    history = [{"role": "user", "content": "u"},
               {"role": "assistant", "content": "", "_responses_output": [reasoning()]}]
    resp = client.chat(history, json_mode=True)
    assert calls[0]["text"] == {"format": {"type": "json_object"}} and "tools" not in calls[0]
    assert any(i.get("type") == "reasoning" for i in calls[0]["input"])
    assert not any(i.get("type") == "reasoning" for i in calls[1]["input"])
    assert resp.content == '{"opportunities": []}' and resp.finish_reason == "stop"
    assert client.warnings


def test_failed_response_raises(client, monkeypatch):
    monkeypatch.setattr(client.client.responses, "create",
                        lambda **k: response(message("x"), status="failed"))
    with pytest.raises(llm.LLMError):
        client.chat([{"role": "user", "content": "u"}])


def test_chat_providers_never_receive_internal_keys(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    c = llm.LLMClient("kimi")
    sent = {}
    from openai.types.chat import ChatCompletion

    def create(**kwargs):
        sent.update(kwargs)
        return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": "k",
                                              "choices": [{"index": 0, "finish_reason": "stop",
                                                           "message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(c.client.chat.completions, "create", create)
    c.chat([{"role": "assistant", "content": "a", "_responses_output": [reasoning()]}])
    assert "_responses_output" not in sent["messages"][0]


class FakeSerper:
    def __init__(self):
        self.requests = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        body = {"organic": [{"title": "תקנות X", "link": URL, "snippet": "...", "position": 1}]}
        return type("R", (), {"status_code": 200, "json": lambda self: body, "text": ""})()


def test_agent_end_to_end_through_openai_adapter_uses_serper_and_local_data(client, monkeypatch, tmp_path):
    """The OpenAI model drives the existing tools: Serper search, fetch_url, local datasets, final JSON."""
    serper = FakeSerper()
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setattr(search.requests, "post", serper)
    gov = GovernmentData(data_dir=build_govdata(tmp_path), cache_dir=tmp_path / "cache")
    report = valid_report(URL)
    report["opportunities"][0]["primary_sources"].append(
        {"title": "Customs tariff", "url": "https://data.gov.il/dataset/customs_tariff/resource/" + TARIFF_RID,
         "resource_id": TARIFF_RID, "excerpt": "GoodsDescription: רכב חשמלי לנוסעים"})
    script = [
        response(reasoning(1), call("call_1", "search_web", {"query": "פטור רישיון השכרה", "phase": "searching"})),
        response(reasoning(2), call("call_2", "fetch_url", {"url": URL, "phase": "reading"}),
                 call("call_3", "search_local_government_records", {"dataset": "customs_tariff", "query": "8703233000"})),
        response(message("DONE")),
        response(message(json.dumps(report, ensure_ascii=False))),
    ]
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return script.pop(0)

    monkeypatch.setattr(client.client.responses, "create", create)
    a = agent.ResearchAgent(client, search_fn=search.search_web, fetch_fn=fake_fetch, gov_data=gov)
    run = a.run("electric vehicle import")
    assert run.status == "completed", run.error
    # Serper received the model's query.
    assert serper.requests[0]["url"] == search.SERPER_URL
    assert serper.requests[0]["headers"]["X-API-KEY"] == "serper-key"
    assert serper.requests[0]["json"]["q"] == "פטור רישיון השכרה"
    # Tool results went back to the model as function_call_output items with matching call ids.
    second_input = sent[1]["input"]
    outputs = [i for i in second_input if i.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "call_1" and URL in outputs[0]["output"]
    assert any(i.get("type") == "reasoning" and i["encrypted_content"] == "ENC1" for i in second_input)
    third = [i for i in sent[2]["input"] if i.get("type") == "function_call_output"]
    local = json.loads(next(i["output"] for i in third if i["call_id"] == "call_3"))
    assert local["records"][0]["record_id"] == "3" and local["records"][0]["match"].startswith("exact_code")
    assert sent[3]["text"] == {"format": {"type": "json_object"}}
    opp = run.result.opportunities[0]
    assert opp.verification_status == "verified"
    tariff = next(s for s in opp.primary_sources if s.resource_id == TARIFF_RID)
    assert tariff.excerpt_verified and "local official snapshot 20261007T030000Z" in tariff.verification_note
    assert run.trace["token_usage"]["prompt_tokens"] == 400 and run.trace["token_usage"]["reasoning_tokens"] == 28
    assert EXCERPT in a.retrieved_text[agent.normalize_url(URL)]
