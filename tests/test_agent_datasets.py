"""Agent integration with the data.gov.il tools (CKAN API mocked)."""

import json

import pytest

import datagov
import fetcher
from agent import Limits, ResearchAgent
from ckan_fixtures import (
    PRESERVATION_RESOURCE_ID, STD_DATASET_ID, STD_RECORD_LINE, STD_RESOURCE_ID, CkanSession, Resp, standard_session,
)
from helpers import EXCERPT, FakeLLM, fake_fetch, text_response, tool_response, valid_report

LAW_URL = "https://www.gov.il/BlobFolder/legalinfo/standards-order/he/order.pdf"
STD_PAGE = f"https://data.gov.il/dataset/official-standards/resource/{STD_RESOURCE_ID}"
QUERY = "תקן רשמי ברגים"


@pytest.fixture(autouse=True)
def public_hosts(monkeypatch):
    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)


def make_agent(responses, session=None, **limits):
    llm = FakeLLM(responses)
    events = []
    ckan = datagov.CkanClient(session=session or standard_session(), min_interval=0,
                              max_calls=limits.pop("max_api_calls", 40))
    agent = ResearchAgent(llm, limits=Limits(**limits), on_event=events.append,
                          search_fn=lambda q, num_results=10: [], fetch_fn=fake_fetch, ckan_client=ckan)
    return agent, llm, events


def tool_results(llm):
    return [json.loads(m["content"]) for m in llm.calls[-1]["messages"] if m["role"] == "tool"]


def dataset_source(resource_id=STD_RESOURCE_ID, url=STD_PAGE, excerpt=STD_RECORD_LINE, **extra):
    return {"title": "Official standards", "url": url, "dataset_id": STD_DATASET_ID, "resource_id": resource_id,
            "excerpt": excerpt, "support": "Standard 1234 is official", **extra}


def legal_source():
    return {"title": "Standards order", "url": LAW_URL, "section": "2", "support": "Order", "excerpt": EXCERPT}


def report(*sources, cls="A"):
    r = valid_report(LAW_URL, cls=cls)
    r["opportunities"][0]["primary_sources"] = list(sources)
    return r


def full_flow(final_report):
    return [
        tool_response(("search_government_datasets", {"query": QUERY})),
        tool_response(("inspect_government_dataset", {"dataset_id": STD_DATASET_ID, "query": QUERY})),
        tool_response(("read_government_resource", {"resource_id": STD_RESOURCE_ID, "query": QUERY}),
                      ("fetch_url", {"url": LAW_URL})),
        text_response("DONE"),
        text_response(json.dumps(final_report)),
    ]


def test_discover_inspect_read_and_verify_class_a():
    agent, llm, events = make_agent(full_flow(report(legal_source(), dataset_source())))
    run = agent.run("industrial fasteners import standards")
    search, inspect, read, _law = tool_results(llm)

    assert [d["likely_relevant"] for d in search["results"]] == [True, False]
    assert "not evidence" in search["note"]
    assert inspect["status"] == "relevant" and inspect["publisher"] == "משרד הכלכלה והתעשייה"
    assert {r["datastore_active"] for r in inspect["resources"]} == {True, False}
    assert read["status"] == "ok" and STD_RECORD_LINE in read["records"]
    assert read["provenance"]["resource_id"] == STD_RESOURCE_ID

    opp = run.result.opportunities[0]
    assert opp.classification == "A" and opp.verification_status == "verified"
    law, ds = opp.primary_sources
    assert law.kind == "legal_document" and law.verified and law.excerpt_verified
    assert ds.kind == "dataset" and ds.verified and ds.excerpt_verified
    assert ds.dataset_title == "רשימת תקנים רשמיים" and ds.publisher == "משרד הכלכלה והתעשייה"
    assert ds.last_updated == "2026-09-15T09:00:00" and ds.retrieval_status == "ok"
    assert any("not by itself proof" in n for n in opp.verification_notes)

    messages = [e["message"] for e in events]
    assert any(m.startswith("Searching official Israeli government datasets") for m in messages)
    assert any(m.startswith("Inspecting dataset metadata") for m in messages)
    assert any(m.startswith("Read official dataset records: רשימת תקנים רשמיים") for m in messages)
    trace = run.trace
    assert trace["dataset_searches"][0]["query"] == QUERY
    assert trace["dataset_inspections"][0]["status"] == "relevant"
    assert trace["dataset_reads"][0]["status"] == "ok" and trace["dataset_reads"][0]["source_url"] == STD_PAGE
    assert [c["action"] for c in trace["api_calls"]] == [
        "package_search", "package_show", "resource_show", "package_show", "datastore_search"]
    assert trace["api_calls"][3]["cached"]  # package_show reused from the inspection


def test_unrelated_dataset_rejected_and_cannot_support_a_claim():
    responses = [
        tool_response(("read_government_resource", {"resource_id": PRESERVATION_RESOURCE_ID, "query": QUERY})),
        text_response("DONE"),
        text_response(json.dumps(report(dataset_source(
            resource_id=PRESERVATION_RESOURCE_ID, url="https://data.gov.il/dataset/preservation-buildings",
            excerpt="בית העם | רחוב הרצל 1 | standards")))),
    ]
    agent, llm, events = make_agent(responses)
    run = agent.run("industrial fasteners import standards")
    read = tool_results(llm)[0]
    assert read["status"] == "rejected" and "records" not in read
    assert any(e["message"].startswith("Dataset rejected: unrelated to research topic — מבנים לשימור")
               for e in events)
    opp = run.result.opportunities[0]
    src = opp.primary_sources[0]
    assert src.retrieval_status == "rejected" and not src.verified
    assert "unrelated to the research topic" in src.verification_note
    assert opp.classification == "B" and opp.verification_status == "unverified"


def test_inspect_rejects_unrelated_dataset():
    agent, llm, events = make_agent([
        tool_response(("inspect_government_dataset",
                       {"dataset_id": "c3d4e5f6-4444-4c5d-9e0f-000000000004", "query": QUERY})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []}))])
    run = agent.run("industrial fasteners import standards")
    result = tool_results(llm)[0]
    assert result["status"] == "rejected" and "resources" not in result
    assert run.trace["dataset_inspections"][0]["status"] == "rejected"


def test_dataset_only_evidence_cannot_be_class_a():
    flow = full_flow(report(dataset_source()))
    agent, _, _ = make_agent(flow)
    opp = agent.run("industrial fasteners import standards").result.opportunities[0]
    assert opp.verification_status == "verified"  # the dataset evidence itself checks out ...
    assert opp.classification == "B" and opp.downgraded_from == "A"  # ... but it is not legal proof
    assert "supported only by dataset evidence" in opp.verification_notes[0]


def test_excerpt_not_in_retrieved_records_is_unverified():
    flow = full_flow(report(legal_source(), dataset_source(excerpt="9999 | תקן מומצא שלא הוחזר")))
    agent, _, _ = make_agent(flow)
    opp = agent.run("industrial fasteners import standards").result.opportunities[0]
    ds = opp.primary_sources[1]
    assert ds.retrieval_status == "ok" and ds.excerpt_verified is False and not ds.verified
    assert opp.verification_status == "partially_verified" and opp.classification == "B"


def test_government_domain_alone_verifies_nothing():
    no_quote = dict(legal_source(), excerpt="")
    agent, _, _ = make_agent([tool_response(("fetch_url", {"url": LAW_URL})), text_response("DONE"),
                              text_response(json.dumps(report(no_quote)))])
    opp = agent.run("x").result.opportunities[0]
    assert opp.primary_sources[0].official and not opp.primary_sources[0].verified
    assert "no supporting excerpt" in opp.primary_sources[0].verification_note
    assert opp.classification == "B"


def test_api_403_reported_unverified_and_run_continues():
    session = CkanSession().add("/package_search", Resp(403, b"Forbidden", "text/html"))
    agent, llm, events = make_agent([
        tool_response(("search_government_datasets", {"query": QUERY})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []}))], session=session)
    run = agent.run("x")
    result = tool_results(llm)[0]
    assert result["ok"] is False and result["error_kind"] == "blocked" and "Do NOT try to bypass" in result["guidance"]
    assert any(e["message"].startswith("Official source unavailable — evidence not verified") for e in events)
    assert run.error == "" and run.result is not None
    failed = run.trace["api_calls"][0]
    assert failed["http_status"] == 403 and not failed["ok"] and failed["url"].startswith(datagov.DEFAULT_BASE_URL)
    assert len(session.requests) == 1


def test_duplicate_dataset_requests_and_api_budget():
    read = ("read_government_resource", {"resource_id": STD_RESOURCE_ID, "query": QUERY})
    agent, llm, _ = make_agent([
        tool_response(read), tool_response(read),
        tool_response(("search_government_datasets", {"query": "אחר"})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []}))], max_api_calls=3)
    run = agent.run("x")
    first, dup, over_budget = tool_results(llm)
    assert first["status"] == "ok"
    assert "Duplicate read rejected" in dup["error"]
    assert over_budget["error_kind"] == "limit"
    assert agent.ckan.calls == 3


def test_model_cannot_forge_dataset_provenance():
    forged = dataset_source(retrieval_status="ok", dataset_title="Forged", publisher="Forged", verified=True,
                            excerpt_verified=True)
    agent, _, _ = make_agent([text_response("DONE"), text_response(json.dumps(report(forged)))])
    src = agent.run("x").result.opportunities[0].primary_sources[0]
    assert src.retrieval_status == "not_retrieved" and src.dataset_title == "" and src.verified is False
