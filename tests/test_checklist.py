"""Configurable critical-evidence checklist and the bounded completion review."""

import json

import pytest

import agent
import checklist
from helpers import EXCERPT, FakeLLM, fake_fetch, text_response, tool_response

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"
ITEMS = [{"id": "legal_text", "question": "Was the governing legal text retrieved?"},
         {"id": "classification", "question": "Is the customs classification confirmed?"}]


def search(query, num_results=10):
    return [{"title": f"r {query}", "url": f"https://www.gov.il/x/{abs(hash(query)) % 1000}", "snippet": "",
             "position": 1, "primary_source": True}]


def make(responses, items=ITEMS, **limits):
    fake = FakeLLM(responses)
    a = agent.ResearchAgent(fake, limits=agent.Limits(**limits), search_fn=search, fetch_fn=fake_fetch,
                            checklist=items)
    return a, fake


def test_default_and_configured_checklists(monkeypatch, tmp_path):
    monkeypatch.delenv("CRITICAL_EVIDENCE_CHECKLIST", raising=False)
    assert [i["id"] for i in checklist.load()] == [i["id"] for i in checklist.DEFAULT_CHECKLIST]
    monkeypatch.setenv("CRITICAL_EVIDENCE_CHECKLIST", json.dumps([{"id": "Standards Q", "question": "Which standard?"},
                                                                  "Is it in force?"]))
    assert checklist.load() == [{"id": "standards_q", "question": "Which standard?"},
                                {"id": "q2", "question": "Is it in force?"}]
    path = tmp_path / "checklist.json"
    path.write_text(json.dumps(ITEMS), "utf-8")
    monkeypatch.setenv("CRITICAL_EVIDENCE_CHECKLIST", str(path))
    assert checklist.load() == ITEMS
    monkeypatch.setenv("CRITICAL_EVIDENCE_CHECKLIST", "[]")
    assert checklist.load() == [] and not checklist.Checklist(checklist.load()).enabled


def test_completion_review_runs_once_then_marks_unresolved():
    a, fake = make([
        tool_response(("search_web", {"query": "צו יבוא חופשי ברגים"}), ("fetch_url", {"url": URL})),
        text_response("DONE"),  # -> completion review
        tool_response(("update_checklist", {"items": [
            {"id": "legal_text", "status": "resolved", "evidence": ["E2"], "note": "regulation text read"},
            {"id": "classification", "status": "unresolved", "note": "tariff item not confirmed in any official "
                                                                     "source", "strategy_attempted": "local tariff"}]})),
        text_response("DONE"),
        text_response(json.dumps({"opportunities": []})),
    ])
    run = a.run("screws")
    review = fake.calls[2]["messages"][-1]["content"]
    assert review.startswith("Before you finish") and "classification" in review and "legal_text" in review
    assert "צו יבוא חופשי ברגים" in review and "doc-" in review  # strategies already tried + stored documents
    assert len(run.trace["completion_reviews"]) == 1
    assert run.stop_reason == "model finished after the completion review"
    status = {i["id"]: i["status"] for i in run.result.checklist}
    assert status == {"legal_text": "resolved", "classification": "unresolved"}
    assert run.result.unresolved_questions == [
        "UNRESOLVED: Is the customs classification confirmed? (tariff item not confirmed in any official source)"]
    assert len(fake.calls) == 5  # no loop: one review, then the final report


def test_open_items_are_closed_by_the_system_when_the_model_stops():
    a, fake = make([text_response("DONE"), text_response("DONE"), text_response("{}")])
    run = a.run("x")
    assert all(i["status"] == "unresolved" and i["updated_by"] == "system" for i in run.result.checklist)
    assert "no sufficient evidence" in run.result.checklist[0]["note"]
    assert len(run.trace["completion_reviews"]) == 1 and len(fake.calls) == 3


def test_completion_review_is_bounded_by_its_step_budget():
    work = [tool_response(("search_web", {"query": f"query number {i}"})) for i in range(10)]
    a, fake = make([text_response("DONE")] + work + [text_response("{}")], completion_steps=3, max_steps=20)
    run = a.run("x")
    assert run.stop_reason == "completion review finished (step budget used)"
    assert a.search_count == 3 and len(run.trace["completion_reviews"]) == 1


def test_completion_review_stops_when_nothing_new_is_found():
    same = lambda q, num_results=10: [{"title": "t", "url": "https://www.gov.il/same", "snippet": "", "position": 1,
                                       "primary_source": True}]
    fake = FakeLLM([tool_response(("search_web", {"query": "first query"})), text_response("DONE")]
                   + [tool_response(("search_web", {"query": f"another wording {i}"})) for i in range(6)]
                   + [text_response("{}")])
    a = agent.ResearchAgent(fake, limits=agent.Limits(completion_steps=6, max_steps=20), search_fn=same,
                            fetch_fn=fake_fetch, checklist=ITEMS)
    run = a.run("x")
    assert run.stop_reason == "completion review stopped: further searches produced no new evidence"
    assert a.search_count == 1 + agent.COMPLETION_IDLE_ROUNDS


def test_no_review_without_budget_or_when_disabled():
    a, fake = make([text_response("DONE"), text_response("{}")], max_searches=0, max_fetches=0,
                   max_local_queries=0, max_document_queries=0)
    run = a.run("x")
    assert run.trace["completion_reviews"] == [] and "exhausted" in run.stop_reason
    a, fake = make([text_response("DONE"), text_response("{}")], items=[])
    run = a.run("x")
    assert run.trace["completion_reviews"] == [] and run.result.checklist == []
    a, fake = make([text_response("DONE"), text_response("{}")], max_completion_rounds=0)
    assert a.run("x").trace["completion_reviews"] == []


def test_update_checklist_requires_evidence_and_notes():
    a, fake = make([
        tool_response(("fetch_url", {"url": URL})),
        tool_response(("update_checklist", {"items": [
            {"id": "legal_text", "status": "resolved"},
            {"id": "legal_text", "status": "resolved", "evidence": ["E99", "https://www.gov.il/never-read"]},
            {"id": "classification", "status": "unresolved"},
            {"id": "nope", "status": "resolved"},
            {"id": "classification", "status": "resolved", "evidence": [f"{URL}"]}]})),
        tool_response(("record_findings", {"findings": [{"statement": "s", "source_url": URL, "excerpt": EXCERPT}]}),
                      ("update_checklist", {"items": [{"id": "legal_text", "status": "resolved", "evidence": ["F1"]}]})),
        text_response("DONE"), text_response("DONE"), text_response("{}"),
    ])
    a.run("x")
    out = [json.loads(m["content"]) for m in fake.calls[2]["messages"] if m["role"] == "tool"][-1]["results"]
    assert "cite at least one" in out[0]["error"] and "cite at least one" in out[1]["error"]
    assert "note explaining" in out[2]["error"] and "Unknown checklist id" in out[3]["error"]
    assert out[4]["status"] == "resolved"  # a URL retrieved in this run is valid evidence
    assert a.checklist.items["legal_text"]["status"] == "resolved" and a.checklist.items["legal_text"]["evidence"] == ["F1"]


def test_checklist_survives_checkpoint_and_resume():
    a, _ = make([tool_response(("fetch_url", {"url": URL})), text_response("DONE"), text_response("DONE"),
                 text_response("{}")])
    a.run("x")
    state = json.loads(json.dumps(a.export_state()))
    b = agent.ResearchAgent(FakeLLM([text_response("{}")]), fetch_fn=fake_fetch)
    b.restore_state(state)
    assert b.checklist.to_state() == a.checklist.to_state() and b.completion_rounds == 1


@pytest.mark.parametrize("a,b,dup", [("פטור ברגים יבוא", "יבוא ברגים פטור", True),
                                     ("ISO 4032 Israel", "ISO 7089 Israel", False),
                                     ("", "x", False)])
def test_near_duplicate(a, b, dup):
    assert checklist.near_duplicate(a, b) is dup
