"""The agent's local government snapshot tools and findings tool."""

import json

import pytest

import agent
from govdata_fixtures import FIO_RID, TARIFF_RID, build_govdata
from helpers import FakeLLM, fake_fetch, fake_search, text_response, tool_response, valid_report
from local_data import GovernmentData

URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"
TARIFF_PAGE = f"https://data.gov.il/dataset/customs_tariff/resource/{TARIFF_RID}"


@pytest.fixture
def gov(tmp_path):
    return GovernmentData(data_dir=build_govdata(tmp_path), cache_dir=tmp_path / "cache")


def make(gov, responses, **limits):
    llm = FakeLLM(responses)
    a = agent.ResearchAgent(llm, limits=agent.Limits(**limits), search_fn=fake_search, fetch_fn=fake_fetch,
                            gov_data=gov)
    return a, llm


def tool_outputs(llm, call_index):
    return [json.loads(m["content"]) for m in llm.calls[call_index]["messages"] if m["role"] == "tool"]


def test_local_tools_return_focused_records_with_provenance(gov):
    a, llm = make(gov, [
        tool_response(("list_local_government_datasets", {}), ("get_government_snapshot_status", {})),
        tool_response(("search_local_government_records", {"dataset": "all", "query": "0407110000/4"}),
                      ("get_local_government_record", {"dataset": "customs_tariff", "record_id": "3"})),
        text_response("DONE"),
        text_response(json.dumps({"opportunities": []})),
    ])
    run = a.run("egg imports")
    listed, status = tool_outputs(llm, 1)
    assert {d["dataset"] for d in listed["datasets"]} == {"customs_tariff", "free_import_order", "mandatory_standards"}
    assert "rows" not in json.dumps(listed) or all("records" not in d for d in listed["datasets"])
    assert status["datasets"][0]["snapshot_version"] == "20261007T030000Z"
    search, record = tool_outputs(llm, 2)[2:]
    assert search["records"][0]["dataset"] == "free_import_order" and search["records"][0]["record_id"] == "1"
    assert search["provenance"]["free_import_order"]["resource_id"] == FIO_RID
    assert "NOT legally binding" in search["evidence_note"] and search["local_queries_left"] == 58
    assert record["fields"]["CustomsTariff"] == "פטור"
    assert [q["tool"] for q in run.trace["local_queries"]] == [
        "list_local_government_datasets", "search_local_government_records", "get_local_government_record"]
    assert run.trace["local_queries"][1]["record_ids"] == ["free_import_order:1"]
    assert run.trace["local_queries"][1]["snapshot_versions"] == ["20261007T030000Z"]


def test_zero_matches_note_and_no_evidence(gov):
    a, llm = make(gov, [
        tool_response(("search_local_government_records", {"dataset": "customs_tariff", "query": "zzznothing"})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []})),
    ])
    a.run("x")
    out = tool_outputs(llm, 1)[0]
    assert out["total_matches"] == 0 and "NOT proof of a legal exemption" in out["note"]
    assert TARIFF_RID not in a.dataset_evidence


def test_local_record_can_support_a_claim_but_not_class_a_alone(gov):
    report = valid_report(TARIFF_PAGE)
    report["opportunities"][0]["primary_sources"] = [
        {"title": "Customs tariff", "url": TARIFF_PAGE, "resource_id": TARIFF_RID,
         "excerpt": "CustomsItemFullClassification: 8703233000/0"}]
    a, _ = make(gov, [
        tool_response(("search_local_government_records", {"dataset": "customs_tariff", "query": "8703233000"})),
        text_response("DONE"), text_response(json.dumps(report)),
    ])
    run = a.run("electric cars")
    opp = run.result.opportunities[0]
    src = opp.primary_sources[0]
    assert src.kind == "dataset" and src.excerpt_verified and src.verified
    assert "local official snapshot" in src.verification_note
    # Dataset evidence alone never yields class A.
    assert opp.classification == "B" and opp.downgraded_from == "A"
    assert any("not by itself proof" in n for n in opp.verification_notes)


def test_forged_local_excerpt_is_unverified(gov):
    report = valid_report(TARIFF_PAGE)
    report["opportunities"][0]["primary_sources"] = [
        {"title": "Customs tariff", "url": TARIFF_PAGE, "resource_id": TARIFF_RID,
         "excerpt": "CustomsTariff: פטור מלא לכל כלי הרכב"}]
    a, _ = make(gov, [
        tool_response(("search_local_government_records", {"dataset": "customs_tariff", "query": "8703"})),
        text_response("DONE"), text_response(json.dumps(report)),
    ])
    opp = a.run("x").result.opportunities[0]
    assert not opp.primary_sources[0].excerpt_verified and opp.verification_status == "unverified"


def test_local_query_budget_duplicates_and_bad_dataset(gov):
    a, llm = make(gov, [
        tool_response(("search_local_government_records", {"dataset": "nope", "query": "x"}),
                      ("search_local_government_records", {"dataset": "customs_tariff", "query": "8703"}),
                      ("search_local_government_records", {"dataset": "customs_tariff", "query": "8703"}),
                      ("search_local_government_records", {"dataset": "customs_tariff", "query": "9503"})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []})),
    ], max_local_queries=2)
    a.run("x")
    bad, ok, dup, over = tool_outputs(llm, 1)
    assert "Unknown dataset" in bad["error"] and ok["records"]
    assert "Duplicate" in dup["error"] and "limit reached" in over["error"]


def test_large_local_results_are_trimmed_for_context(gov, monkeypatch):
    monkeypatch.setattr(agent, "MAX_LOCAL_RESULT_CHARS", 1500)
    a, llm = make(gov, [
        tool_response(("search_local_government_records", {"dataset": "all", "query": "8703", "limit": 25})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []})),
    ])
    a.run("x")
    out = tool_outputs(llm, 1)[0]
    assert out["truncated"] and out["returned"] == len(out["records"]) < 4


def test_record_findings_verifies_excerpts(gov):
    from helpers import EXCERPT

    a, llm = make(gov, [
        tool_response(("fetch_url", {"url": URL})),
        tool_response(("record_findings", {"findings": [
            {"statement": "Yearly inspection", "source_url": URL, "excerpt": EXCERPT},
            {"statement": "Made up", "source_url": URL, "excerpt": "this sentence is not in the regulation"},
            {"statement": "Bad url", "source_url": "ftp://x", "excerpt": "x"}],
            "open_questions": ["Q1", "Q1", "Q2"]})),
        text_response("DONE"), text_response(json.dumps({"opportunities": []})),
    ])
    run = a.run("x")
    out = tool_outputs(llm, 2)[-1]
    assert [r["verified"] for r in out["results"]] == [True, False, False]
    assert [f["statement"] for f in a.findings] == ["Yearly inspection", "Made up"]
    assert a.open_questions == ["Q1", "Q2"] and run.trace["findings"] == a.findings
