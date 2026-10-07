"""Failed official-source retrieval, alternative search on 403, and evidence verification / Class A gate."""

import json

from agent import Limits, ResearchAgent
from fetcher import FetchResult
from helpers import FakeLLM, text_response, tool_response, valid_report

BLOCKED = "https://www.gov.il/he/departments/legalInfo/trailer-regulations"
ALT_PDF = "https://www.gov.il/BlobFolder/legalinfo/trailer-regulations/he/takanot.pdf"
LAW_FIRM = "https://www.example-lawfirm.co.il/articles/trailers"
CKAN_RES = "https://data.gov.il/api/3/action/resource_show?id=053cea08"


def fetch_by_url(url):
    if url == BLOCKED:
        return FetchResult(url=url, final_url=url, ok=False, source_type="unknown", http_status=403,
                           error="HTTP 403 (access denied; not retried or bypassed)")
    if url == CKAN_RES:
        return FetchResult(url=url, final_url=url, ok=True, source_type="ckan_resource", title="רשימת נגררים",
                           text="CKAN resource metadata: ...", resource_url="https://data.gov.il/x/trailers.csv",
                           metadata={"format": "CSV"})
    return FetchResult(url=url, final_url=url, ok=True, source_type="pdf" if url.endswith(".pdf") else "html",
                       title="doc", text="§4 ...")


class SearchSpy:
    def __init__(self):
        self.queries = []

    def __call__(self, query, num_results=10):
        self.queries.append(query)
        return [
            {"title": "תקנות התעבורה נגררים", "url": BLOCKED, "snippet": "", "position": 1, "primary_source": True},
            {"title": "takanot.pdf", "url": ALT_PDF, "snippet": "", "position": 2, "primary_source": True},
        ]


def run_agent(actions, report, **limits):
    """actions: list of tool_response turns; then DONE; then the final report JSON."""
    llm = FakeLLM(actions + [text_response("DONE"), text_response(json.dumps(report))])
    search = SearchSpy()
    agent = ResearchAgent(llm, limits=Limits(**limits), search_fn=search, fetch_fn=fetch_by_url)
    run = agent.run("trailer rental")
    return run, agent, llm, search


def tool_results(llm):
    msgs = llm.calls[-1]["messages"]
    return [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]


def report_citing(*urls, cls="A", extra=None):
    report = valid_report(urls[0], cls=cls)
    opp = report["opportunities"][0]
    opp["primary_sources"] = [{"title": f"src {i}", "url": u, "section": "4", "support": "x"} for i, u in enumerate(urls)]
    opp.update(extra or {})
    return report


def test_verified_official_evidence_keeps_class_a():
    run, *_ = run_agent([tool_response(("fetch_url", {"url": ALT_PDF}))], report_citing(ALT_PDF))
    opp = run.result.opportunities[0]
    assert opp.classification == "A" and opp.downgraded_from == ""
    assert opp.verification_status == "verified"
    assert opp.primary_sources[0].verified and opp.primary_sources[0].official
    assert opp.unread_primary_sources == []


def test_gov_il_403_triggers_guidance_and_one_alternative_search():
    run, agent, llm, search = run_agent([
        tool_response(("search_web", {"query": "תקנות נגררים"})),
        tool_response(("fetch_url", {"url": BLOCKED})),
    ], report_citing(BLOCKED))
    blocked_result = tool_results(llm)[1]
    assert blocked_result["ok"] is False and blocked_result["http_status"] == 403
    assert "Do NOT try to bypass" in blocked_result["guidance"]
    alt = blocked_result["alternative_search"]
    # Query built from the page title seen in earlier search results, restricted to official PDFs.
    assert alt["query"] == "תקנות התעבורה נגררים filetype:pdf site:gov.il"
    assert [r["url"] for r in alt["results"]] == [ALT_PDF]  # the blocked page itself is filtered out
    assert search.queries == ["תקנות נגררים", alt["query"]]
    assert agent.search_count == 2 and agent.fetch_count == 1  # counted against the normal budget
    assert any("blocked" in w for w in run.trace["warnings"])
    assert run.trace["fetches"][0]["http_status"] == 403


def test_class_a_citing_blocked_source_is_downgraded_and_unverified():
    run, *_ = run_agent([tool_response(("fetch_url", {"url": BLOCKED}))], report_citing(BLOCKED))
    opp = run.result.opportunities[0]
    assert opp.classification == "B" and opp.downgraded_from == "A"
    assert opp.verification_status == "unverified"
    assert opp.primary_sources[0].verified is False
    assert "HTTP 403" in opp.primary_sources[0].verification_note
    assert opp.verification_notes[0].startswith("Downgraded from A to B")
    assert any("Legal finding UNVERIFIED" in n for n in opp.verification_notes)
    assert any("downgraded from A to B" in w for w in run.trace["warnings"])


def test_partially_verified_class_a_is_downgraded():
    run, *_ = run_agent(
        [tool_response(("fetch_url", {"url": BLOCKED}), ("fetch_url", {"url": ALT_PDF}))],
        report_citing(ALT_PDF, BLOCKED),
    )
    opp = run.result.opportunities[0]
    assert opp.verification_status == "partially_verified"
    assert opp.classification == "B" and opp.downgraded_from == "A"
    assert [s.verified for s in opp.primary_sources] == [True, False]


def test_alternative_official_copy_restores_class_a():
    run, *_ = run_agent(
        [tool_response(("fetch_url", {"url": BLOCKED})), tool_response(("fetch_url", {"url": ALT_PDF}))],
        report_citing(ALT_PDF),
    )
    opp = run.result.opportunities[0]
    assert opp.classification == "A" and opp.verification_status == "verified"


def test_never_fetched_source_is_unverified():
    run, *_ = run_agent([], report_citing(ALT_PDF))
    opp = run.result.opportunities[0]
    assert opp.verification_status == "unverified" and opp.classification == "B"
    assert opp.primary_sources[0].verification_note == "Not retrieved during this run."


def test_non_official_primary_source_cannot_support_class_a():
    run, *_ = run_agent([tool_response(("fetch_url", {"url": LAW_FIRM}))], report_citing(LAW_FIRM))
    opp = run.result.opportunities[0]
    assert opp.primary_sources[0].verified and opp.primary_sources[0].official is False
    assert opp.verification_status == "unverified" and opp.classification == "B"
    assert "not an official source" in opp.primary_sources[0].verification_note


def test_class_b_is_not_changed_by_verification():
    run, *_ = run_agent([tool_response(("fetch_url", {"url": BLOCKED}))], report_citing(BLOCKED, cls="B"))
    opp = run.result.opportunities[0]
    assert opp.classification == "B" and opp.downgraded_from == "" and opp.verification_status == "unverified"


def test_model_cannot_claim_verification():
    report = report_citing(BLOCKED, extra={"verification_status": "verified", "downgraded_from": "", "verification_notes": ["trust me"]})
    report["opportunities"][0]["primary_sources"][0].update(verified=True, official=True, verification_note="read it")
    run, *_ = run_agent([tool_response(("fetch_url", {"url": BLOCKED}))], report)
    opp = run.result.opportunities[0]
    assert opp.verification_status == "unverified" and opp.classification == "B"
    assert opp.primary_sources[0].verified is False and "trust me" not in opp.verification_notes


def test_no_alternative_search_when_budget_exhausted():
    run, agent, llm, search = run_agent([
        tool_response(("search_web", {"query": "תקנות נגררים"})),
        tool_response(("fetch_url", {"url": BLOCKED})),
    ], report_citing(BLOCKED), max_searches=1)
    blocked_result = tool_results(llm)[1]
    assert "guidance" in blocked_result
    assert "budget exhausted" in blocked_result["alternative_search"]["note"]
    assert agent.search_count == 1 and len(search.queries) == 1


def test_slug_query_used_without_title_and_skipped_for_numeric_slugs():
    agent = ResearchAgent(FakeLLM([text_response("x")]))
    assert agent._alternative_query(BLOCKED) == "trailer regulations filetype:pdf site:gov.il"
    assert agent._alternative_query("https://www.gov.il/he/departments/publications/12345") == ""
    assert agent._alternative_query("https://www.gov.il/") == ""


def test_403_on_non_official_site_gets_no_special_handling():
    def fetch(url):
        return FetchResult(url=url, final_url=url, ok=False, source_type="unknown", http_status=403, error="HTTP 403")

    llm = FakeLLM([tool_response(("fetch_url", {"url": LAW_FIRM})), text_response("DONE"),
                   text_response(json.dumps({"opportunities": []}))])
    search = SearchSpy()
    agent = ResearchAgent(llm, search_fn=search, fetch_fn=fetch)
    agent.run("x")
    result = tool_results(llm)[0]
    assert "guidance" not in result and "alternative_search" not in result and search.queries == []


def test_ckan_resource_url_passed_to_model_and_counts_as_verified():
    run, agent, llm, _ = run_agent([tool_response(("fetch_url", {"url": CKAN_RES}))], report_citing(CKAN_RES))
    result = tool_results(llm)[0]
    assert result["source_type"] == "ckan_resource"
    assert result["resource_url"] == "https://data.gov.il/x/trailers.csv" and result["metadata"] == {"format": "CSV"}
    assert run.trace["fetches"][0]["resource_url"] == "https://data.gov.il/x/trailers.csv"
    assert run.result.opportunities[0].verification_status == "verified"


def test_verified_ranks_above_unverified_within_class():
    report = valid_report(ALT_PDF, n=2, cls="B")
    report["opportunities"][0].update(business_score=95)
    report["opportunities"][0]["primary_sources"] = [{"title": "blocked", "url": BLOCKED}]  # opps share one list
    report["opportunities"][1].update(business_score=60)
    run, *_ = run_agent(
        [tool_response(("fetch_url", {"url": BLOCKED}), ("fetch_url", {"url": ALT_PDF}))], report)
    assert [o.verification_status for o in run.result.opportunities] == ["verified", "unverified"]
