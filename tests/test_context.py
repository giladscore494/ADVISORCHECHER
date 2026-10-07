"""Context compaction, evidence ledger, deduplication, token accounting and provider compatibility."""

import json

import pytest

import agent
import context
import llm
from fetcher import FetchResult
from helpers import EXCERPT, FakeLLM, legal_checks, text_response, tool_response, valid_report

GOV = "https://www.gov.il/he/departments/legalInfo/"


def big_fetch(url):
    """Every page is long (like a law), and contains the shared excerpt once."""
    body = "\n\n".join(f"סעיף {i}. {'הוראה מפורטת לעניין יבוא טובין ותנאי שחרורם ' * 12}" for i in range(40))
    return FetchResult(url=url, final_url=url, ok=True, source_type="html", title=f"Law {url[-2:]}",
                       text=f"{body}\n\n{EXCERPT}, בתחנת בדיקה מורשית.")


def search(query, num_results=10):
    return [{"title": f"{query} result {i}", "url": f"{GOV}law-{i}", "snippet": "תקנות " * 40, "position": i,
             "primary_source": True} for i in range(1, 9)]


def research_script(n_rounds=10):
    turns = []
    for i in range(n_rounds):
        turns.append(tool_response(("search_web", {"query": f"תקנות יבוא {i}", "phase": "searching"}),
                                   ("fetch_url", {"url": f"{GOV}doc-{i:02d}", "phase": "reading"})))
    return turns


def run(responses, **limits):
    fake = FakeLLM(responses)
    a = agent.ResearchAgent(fake, limits=agent.Limits(max_steps=30, **limits), search_fn=search, fetch_fn=big_fetch)
    return a, a.run("import of screws"), fake


def sent_tokens(fake):
    return [context.estimate_tokens(c["messages"]) for c in fake.calls]


def test_compaction_bounds_context_and_keeps_evidence_retrievable():
    url = f"{GOV}doc-00"
    report = valid_report(url)
    findings = tool_response(("record_findings", {"findings": [
        {"statement": "Yearly inspection required", "source_url": url, "excerpt": EXCERPT,
         "claim_type": "legal_conclusion"}]}))
    responses = research_script(1) + [findings] + research_script(9)[1:] + [
        tool_response(("get_evidence", {"evidence_id": "E2"})), text_response("DONE"), text_response(json.dumps(report))]
    a, result, fake = run(responses, context_budget_tokens=6000)
    tokens = sent_tokens(fake)
    assert a.compactions >= 1 and result.trace["compactions"]
    # Without compaction the history grows every round; with it, later calls stay within the budget plus the
    # recent rounds kept in full and the digest.
    assert max(tokens[-4:]) < 2 * 6000
    # Compacted messages are stubs that point to the ledger; the ledger keeps the full original.
    compacted = [m for m in a.messages if m.get("_compacted")]
    assert compacted and all("get_evidence" in m["content"] for m in compacted)
    assert a.ledger["E2"]["tool"] == "fetch_url" and a.ledger["E2"]["result"]["document_id"]
    restored = [json.loads(m["content"]) for m in fake.calls[-2]["messages"] if m["role"] == "tool"][-1]
    assert restored["evidence_id"] == "E2" and restored["result"]["text"]
    # The digest preserves the finding and its verbatim quote even though the original tool result was compacted.
    digest = fake.calls[-1]["messages"][-2]["content"]
    assert digest.startswith("RESEARCH DIGEST") and EXCERPT in digest and "F1 [source VERIFIED" in digest
    assert "## Evidence index" in digest and "## Documents" in digest
    # Legal verification still works after compaction: the excerpt is checked against the stored document.
    assert result.result.opportunities[0].verification_status == "verified"


def test_compaction_reduces_tokens_versus_no_compaction():
    script = research_script(10) + [text_response("DONE"), text_response(json.dumps({"opportunities": []}))]
    _, _, full = run(list(script), context_budget_tokens=10_000_000)
    _, _, compact = run(list(script), context_budget_tokens=6000)
    full_total, compact_total = sum(sent_tokens(full)), sum(sent_tokens(compact))
    assert compact_total < 0.6 * full_total


def test_search_results_and_local_records_are_deduplicated(tmp_path):
    from govdata_fixtures import build_govdata
    from local_data import GovernmentData

    gov = GovernmentData(data_dir=build_govdata(tmp_path), cache_dir=tmp_path / "cache")
    fake = FakeLLM([
        tool_response(("search_web", {"query": "a"}), ("search_local_government_records",
                                                         {"dataset": "customs_tariff", "query": "8703"})),
        tool_response(("search_web", {"query": "b"}), ("search_local_government_records",
                                                         {"dataset": "all", "query": "8703"})),
        text_response("DONE"), text_response("{}"),
    ])
    a = agent.ResearchAgent(fake, search_fn=search, fetch_fn=big_fetch, gov_data=gov)
    a.run("x")
    tools = [json.loads(m["content"]) for m in fake.calls[2]["messages"] if m["role"] == "tool"]
    first_search, first_local, second_search, second_local = tools
    assert all("seen_in" not in r for r in first_search["results"])
    assert all(r["seen_in"] == "E1" and "snippet" not in r for r in second_search["results"])
    assert "fields" not in first_local["records"][0]  # record_line only: fields duplicated the same values
    repeated = [r for r in second_local["records"] if "already_returned_in" in r]
    assert repeated and all(r["already_returned_in"] == "E2" for r in repeated)
    assert second_local["provenance"]["customs_tariff"]["full_provenance_in"] == "E2"
    # The ledger keeps everything; the evidence used for verification is unaffected by deduplication.
    assert a.dataset_evidence and a.ledger["E4"]["result"]["records"]


def test_near_duplicate_search_without_new_results_is_rejected():
    fake = FakeLLM([
        tool_response(("search_web", {"query": "פטור ברגים יבוא"})),
        tool_response(("search_web", {"query": "פטור ברגים יבוא"})),  # exact duplicate
        tool_response(("search_web", {"query": "יבוא ברגים פטור"})),  # reordered: same results -> nothing new
        tool_response(("search_web", {"query": "ברגים פטור יבוא"})),
        text_response("DONE"), text_response("{}"),
    ])
    a = agent.ResearchAgent(fake, search_fn=search, fetch_fn=big_fetch)
    run_ = a.run("x")
    outcomes = [t["outcome"] for t in run_.trace["tool_calls"]]
    assert outcomes[0] == "ok" and "Duplicate search" in outcomes[1]
    # The reordered query is not an exact duplicate, so it runs, but finds nothing new ...
    assert outcomes[2] == "ok" and a.search_outcomes[1]["new_urls"] == 0
    # ... and a near-duplicate of a query that produced nothing new is rejected: change the strategy.
    assert "Near-duplicate" in outcomes[3]


def test_token_accounting_by_call_and_phase():
    responses = [tool_response(("search_web", {"query": "a", "phase": "searching"})),
                 tool_response(("fetch_url", {"url": f"{GOV}doc-01", "phase": "reading"})),
                 text_response("DONE"), text_response("{}")]
    for r in responses:
        r.usage = {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 600, "reasoning_tokens": 10}
    a, result, _ = run(responses)
    tu = result.trace["token_usage"]
    assert tu["model_calls"] == 4 and tu["prompt_tokens"] == 4000 and tu["cached_tokens"] == 2400
    assert tu["estimated_input_tokens"] > 0
    by_phase = result.trace["token_usage_by_phase"]
    assert by_phase["mapping"]["model_calls"] == 1 and by_phase["searching"]["model_calls"] == 1
    assert by_phase["reading"]["model_calls"] == 1 and by_phase["reading"]["cached_tokens"] == 600
    assert by_phase["final_report"]["model_calls"] == 1
    calls = result.trace["model_calls"]
    assert [c["phase"] for c in calls[:3]] == ["mapping", "searching", "reading"]
    assert all(c["estimated_input_tokens"] > 0 for c in calls)


def test_state_stores_tool_results_once_and_restores_identical_messages():
    a, _, _ = run(research_script(3) + [text_response("DONE"), text_response("{}")], context_budget_tokens=10**9)
    state = json.loads(json.dumps(a.export_state()))
    stored_tools = [m for m in state["messages"] if m["role"] == "tool"]
    assert stored_tools and all(m["content"] == "" and m["_from_ledger"] for m in stored_tools)
    b = agent.ResearchAgent(FakeLLM([text_response("{}")]), fetch_fn=big_fetch)
    b.restore_state(state)
    assert b.messages == a.messages


def test_version_1_checkpoint_can_still_be_restored():
    a, _, _ = run(research_script(1) + [text_response("DONE"), text_response("{}")])
    state = a.export_state()
    state["state_version"] = 1
    for key in ("ledger", "url_docs", "documents", "checklist", "search_outcomes"):
        state.pop(key)
    state["messages"] = [{k: v for k, v in m.items() if k not in ("_evidence_id", "_from_ledger")} | (
        {"content": "{}"} if m.get("_from_ledger") else {}) for m in state["messages"]]
    b = agent.ResearchAgent(FakeLLM([text_response("{}")]))
    b.restore_state(state)
    assert b.messages and b.ledger == {} and b.documents == {}


# --------------------------------------------------- provider compatibility
def _compacted_history():
    a, _, _ = run(research_script(6) + [text_response("DONE"), text_response("{}")], context_budget_tokens=3000)
    return a


def test_chat_providers_receive_compacted_history_without_internal_keys(monkeypatch):
    a = _compacted_history()
    msgs = a._context_for_call()
    for provider, env in (("kimi", "KIMI_API_KEY"), ("glm", "GLM_API_KEY")):
        monkeypatch.setenv(env, "k-test")
        client = llm.LLMClient(provider)
        sent = {}

        class Completion:
            choices = [type("C", (), {"message": type("M", (), {"content": "DONE", "tool_calls": None,
                                                                 "model_extra": {}})(),
                                      "finish_reason": "stop"})()]
            usage = None

        monkeypatch.setattr(client.client.chat.completions, "create", lambda **kw: sent.update(kw) or Completion())
        client.chat(msgs, tools=agent.TOOLS)
        assert all(not any(k.startswith("_") for k in m) for m in sent["messages"])
        tool_ids = {m["tool_call_id"] for m in sent["messages"] if m["role"] == "tool"}
        call_ids = {tc["id"] for m in sent["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
        assert tool_ids == call_ids  # every tool call still has its (possibly compacted) result
        assert {t["function"]["name"] for t in sent["tools"]} >= {"search_document", "read_document_range",
                                                                    "get_document_status", "get_evidence"}


def test_old_reasoning_is_dropped_but_function_calls_survive_for_responses_api():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(4):
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "search_web", "arguments": "{}"}}],
            "reasoning_content": "r" * 2000,
            "_responses_output": [{"type": "reasoning", "id": f"rs_{i}", "encrypted_content": "E" * 4000},
                                  {"type": "function_call", "call_id": f"c{i}", "name": "search_web", "arguments": "{}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"results": ["x" * 3000]}),
                         "_evidence_id": f"E{i + 1}"})
    ledger = {f"E{i + 1}": {"id": f"E{i + 1}", "tool": "search_web", "args": {"query": "q"},
                            "result": {"results": [{"title": "t", "url": "https://www.gov.il/a"}]}} for i in range(4)}
    before = context.estimate_tokens(messages)
    assert context.compact(messages, ledger, keep_rounds=1) > 0
    assert context.estimate_tokens(messages) < before / 3
    items = llm.to_responses_input(messages)
    reasoning = [it for it in items if it.get("type") == "reasoning"]
    assert [r["id"] for r in reasoning] == ["rs_3"]  # only the latest round keeps its reasoning
    assert [it["call_id"] for it in items if it.get("type") == "function_call"] == ["c0", "c1", "c2", "c3"]
    outputs = [it for it in items if it.get("type") == "function_call_output"]
    assert len(outputs) == 4 and json.loads(outputs[0]["output"])["compacted"]
    # Chat providers keep a shortened reasoning_content on older tool-call turns.
    assert messages[2]["reasoning_content"].endswith("compacted]") and messages[-2]["reasoning_content"] == "r" * 2000


def test_openai_adapter_reports_cached_tokens(monkeypatch):
    from openai.types.responses import Response

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = llm.LLMClient("openai")
    resp = Response.model_validate({
        "id": "r", "created_at": 0, "model": "m", "object": "response", "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [], "status": "completed",
        "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE", "annotations": []}]}],
        "usage": {"input_tokens": 900, "output_tokens": 10, "total_tokens": 910,
                  "input_tokens_details": {"cached_tokens": 512, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 3}}})
    monkeypatch.setattr(client.client.responses, "create", lambda **kw: resp)
    out = client.chat([{"role": "user", "content": "x"}])
    assert out.usage == {"prompt_tokens": 900, "completion_tokens": 10, "reasoning_tokens": 3, "cached_tokens": 512}


@pytest.mark.parametrize("budget", [0])
def test_zero_budget_disables_compaction(budget):
    a, _, fake = run(research_script(4) + [text_response("DONE"), text_response("{}")], context_budget_tokens=budget)
    assert a.compactions == 0 and not any(m.get("_compacted") for m in a.messages)
