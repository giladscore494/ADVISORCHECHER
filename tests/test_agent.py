import json

from agent import Limits, ResearchAgent, normalize_url
from helpers import FakeLLM, fake_fetch, fake_search, text_response, tool_response, valid_report
from llm import LLMError
from search import SearchError

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"


def make_agent(responses, **limits):
    llm = FakeLLM(responses)
    events = []
    a = ResearchAgent(llm, limits=Limits(**limits), on_event=events.append, search_fn=fake_search, fetch_fn=fake_fetch)
    return a, llm, events


def test_full_run_produces_validated_result():
    a, llm, events = make_agent([
        tool_response(("search_web", {"query": "פטור רישיון השכרה", "phase": "searching", "purpose": "Searching rental exemptions"})),
        tool_response(("fetch_url", {"url": URL, "phase": "reading"}),
                      ("update_candidates", {"candidates": [{"name": "X", "status": "surviving"}, {"name": "Y", "status": "rejected", "reason": "needs licence"}]})),
        text_response("DONE"),
        text_response(json.dumps(valid_report(URL))),
    ])
    run = a.run("equipment rental")
    assert run.error == "" and run.stop_reason == "model finished research"
    assert len(run.result.opportunities) == 1
    assert run.result.opportunities[0].unread_primary_sources == []
    assert llm.calls[-1]["json_mode"] is True
    assert run.trace["searches"][0]["query"] == "פטור רישיון השכרה"
    assert run.trace["fetches"][0]["ok"] and run.trace["fetches"][0]["primary"]
    assert run.trace["candidates"]["Y"]["status"] == "rejected"
    assert any(e["message"] == "Searching rental exemptions" for e in events)
    # every tool call got a tool response with the matching id
    msgs = llm.calls[-1]["messages"]
    assert [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == ["call_0", "call_0", "call_1"]


def test_duplicate_search_and_fetch_rejected_then_loop_stops():
    dup = tool_response(("search_web", {"query": "same  QUERY"}))
    a, llm, _ = make_agent([
        tool_response(("search_web", {"query": "same query"}), ("fetch_url", {"url": URL + "/"})),
        tool_response(("fetch_url", {"url": URL + "#frag"})),  # duplicate URL: no progress (1)
        dup,  # duplicate query: no progress (2)
        dup,  # no progress (3) -> loop stops
        text_response(json.dumps({"opportunities": []})),
    ], max_steps=20)
    run = a.run("x")
    assert a.search_count == 1 and a.fetch_count == 1
    assert "repeated duplicate" in run.stop_reason
    outcomes = [t["outcome"] for t in run.trace["tool_calls"]]
    assert any("Duplicate fetch" in o for o in outcomes)
    assert any("Duplicate search" in o for o in outcomes)
    assert run.result.opportunities == []
    assert run.result.no_opportunity_reason == "No sufficiently strong opportunity found."


def test_step_and_search_limits():
    responses = [tool_response(("search_web", {"query": f"q{i}"})) for i in range(10)]
    responses.append(text_response(json.dumps({"opportunities": []})))
    a, llm, _ = make_agent(responses, max_steps=4, max_searches=2)
    run = a.run("x")
    assert a.search_count == 2
    assert run.stop_reason == "step limit reached"
    assert len([c for c in llm.calls if c["tools"]]) == 4


def test_malformed_tool_args_and_unknown_tool_do_not_crash():
    bad = tool_response(("search_web", {"query": "a"}))
    bad.tool_calls[0].arguments = "{not json"
    a, _, _ = make_agent([
        bad,
        tool_response(("delete_everything", {})),
        text_response("DONE"),
        text_response(json.dumps({"opportunities": []})),
    ])
    run = a.run("x")
    assert run.error == ""
    assert "not valid JSON" in run.trace["tool_calls"][0]["outcome"]
    assert "Unknown tool" in run.trace["tool_calls"][1]["outcome"]


def test_failed_search_and_fetch_do_not_kill_run():
    def broken_search(q, num_results=10):
        raise SearchError("Serper returned HTTP 500")

    from fetcher import FetchResult
    a, _, _ = make_agent([
        tool_response(("search_web", {"query": "a"}), ("fetch_url", {"url": "https://example.com/x"})),
        text_response("DONE"),
        text_response(json.dumps({"opportunities": []})),
    ])
    a.search_fn = broken_search
    a.fetch_fn = lambda url: FetchResult(url=url, final_url=url, ok=False, source_type="unknown", error="HTTP 403")
    run = a.run("x")
    assert run.error == "" and run.result is not None
    assert run.trace["searches"][0]["error"] and not run.trace["fetches"][0]["ok"]


def test_malformed_final_json_is_repaired_once_then_fails():
    a, llm, _ = make_agent([text_response("DONE"), text_response("not json"), text_response(json.dumps(valid_report(URL)))])
    run = a.run("x")
    assert run.result is not None and len(run.result.opportunities) == 1
    # The source was never fetched in this run, so it is flagged.
    assert run.result.opportunities[0].unread_primary_sources == [URL]

    a, _, _ = make_agent([text_response("DONE"), text_response("still not json")])
    run = a.run("x")
    assert run.result is None and "did not return a valid report" in run.error


def test_model_error_on_first_step_reported():
    class Broken:
        warnings = []

        def chat(self, *a, **k):
            raise LLMError("Model request timed out.")

    run = ResearchAgent(Broken(), search_fn=fake_search, fetch_fn=fake_fetch).run("x")
    assert run.result is None and "timed out" in run.error


def test_normalize_url():
    assert normalize_url("HTTPS://WWW.Gov.il/a/#x") == normalize_url("https://www.gov.il/a")
