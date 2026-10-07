"""Source, legal-applicability and business-advantage verification are separate and conservative."""

import json

import agent
import partial_report
import verification
from fetcher import FetchResult
from helpers import EXCERPT, SOURCE_TEXT, FakeLLM, legal_checks, text_response, tool_response, valid_report

LAW = "https://www.gov.il/he/departments/legalInfo/regulation-x"
SCOPE = "הוראות תקנה זו יחולו על יבואן של ברגים מפלדה בלבד"
EXCEPTION = "למעט ברגים המיועדים לכלי טיס"
MARKET = "https://www.example-market.co.il/screws-importers"
MARKET_TEXT = "כל היבואנים הגדולים כבר מייבאים ברגים בפטור זה ללא הגבלה"


def fetch(url):
    if url == MARKET:
        return FetchResult(url=url, final_url=url, ok=True, source_type="html", title="market", text=MARKET_TEXT)
    return FetchResult(url=url, final_url=url, ok=True, source_type="html", title="Regulation X",
                       text=f"{SOURCE_TEXT}\n{SCOPE}.\n{EXCEPTION}.")


def run(actions, report=None, **limits):
    fake = FakeLLM(actions + [text_response("DONE"), text_response(json.dumps(report or {"opportunities": []}))])
    a = agent.ResearchAgent(fake, limits=agent.Limits(**limits), search_fn=lambda q, num_results=10: [],
                            fetch_fn=fetch)
    return a, a.run("screws import"), fake


def findings_result(fake, call=2):
    return [json.loads(m["content"]) for m in fake.calls[call]["messages"] if m["role"] == "tool"][-1]


def test_source_verified_but_legally_unverified_finding():
    a, run_, fake = run([
        tool_response(("fetch_url", {"url": LAW})),
        tool_response(("record_findings", {"findings": [
            {"statement": "Owners must inspect yearly", "source_url": LAW, "excerpt": EXCERPT,
             "claim_type": "legal_conclusion"},
            {"statement": "Regulation X mentions yearly inspection", "source_url": LAW, "excerpt": EXCERPT}]})),
    ])
    legal, fact = a.findings
    assert legal["source_status"] == fact["source_status"] == "verified" and legal["verified"]
    # The quote is real, but scope, validity, exceptions and classification were never checked.
    assert legal["legal_status"] == "partially_verified"
    assert any("not checked" in n for n in legal["legal_notes"])
    assert "legal_status" not in fact  # a plain fact is not a legal conclusion
    counts = run_.trace["verification_counters"]
    assert counts["source_verified"] == 2 and counts["legal_conclusions"] == 1 and counts["legal_verified"] == 0
    summary = a._summary("x")
    assert summary["verified_findings"] == 2 and summary["verification"]["legal_verified"] == 0
    assert "does not verify a legal conclusion" in findings_result(fake)["note"]


def test_fully_checked_legal_conclusion_is_verified():
    checks = {
        "provision": {"status": "checked", "source_url": LAW, "excerpt": EXCERPT},
        "scope": {"status": "checked", "source_url": LAW, "excerpt": SCOPE},
        "validity": {"status": "checked", "source_url": LAW, "excerpt": EXCERPT},
        "exceptions": {"status": "checked", "source_url": LAW, "excerpt": EXCEPTION},
        "product_classification": {"status": "checked", "source_url": LAW, "excerpt": SCOPE},
    }
    a, *_ = run([tool_response(("fetch_url", {"url": LAW})),
                 tool_response(("record_findings", {"findings": [
                     {"statement": "Steel screw importers must inspect yearly", "source_url": LAW, "excerpt": EXCERPT,
                      "claim_type": "legal_conclusion", "legal_checks": checks}]}))])
    assert a.findings[0]["legal_status"] == "verified"
    # A check whose quote is invented is not verified, whatever the model says.
    checks["exceptions"]["excerpt"] = "אין חריגים כלל לתקנה זו בשום מקרה"
    a, *_ = run([tool_response(("fetch_url", {"url": LAW})),
                 tool_response(("record_findings", {"findings": [
                     {"statement": "x", "source_url": LAW, "excerpt": EXCERPT, "claim_type": "legal_conclusion",
                      "legal_checks": checks}]}))])
    assert a.findings[0]["legal_status"] == "partially_verified"
    assert a.findings[0]["legal_checks"]["exceptions"]["verified"] is False


def test_verified_quote_without_legal_checks_downgrades_class_a():
    report = valid_report(LAW, legal=False)
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}))], report)
    opp = run_.result.opportunities[0]
    assert opp.verification_status == "verified"  # SOURCE dimension
    assert opp.legal_verification == "unverified" and opp.business_verification == "unverified"
    assert opp.classification == "B" and opp.downgraded_from == "A"
    assert "a verified quotation is not a verified legal conclusion" in opp.verification_notes[0]


def test_unverified_negative_claim_is_never_an_established_exemption():
    report = valid_report(LAW, legal=False)
    opp = report["opportunities"][0]
    opp.update(name="Import screws without a standard approval",
               regulatory_mechanism="Screws under 7318 are exempt (פטור) from the Free Import Order: zero dataset "
                                    "records were found.")
    a, run_, _ = run([
        tool_response(("fetch_url", {"url": LAW})),
        tool_response(("record_findings", {"findings": [
            {"statement": "No Free Import Order requirement applies to 7318 (פטור)", "source_url": LAW,
             "excerpt": EXCERPT}]})),
    ], report)
    finding = a.findings[0]
    assert finding["negative_claim"] and finding["legal_status"] == "unverified"
    assert any("NOT established" in n for n in finding["legal_notes"])
    result = run_.result.opportunities[0]
    assert result.negative_claim and result.legal_verification == "unverified" and result.classification == "B"
    assert any("NOT established" in n for n in result.verification_notes)
    assert run_.trace["verification_counters"]["negative_unresolved"] == 1
    md = partial_report.to_markdown(partial_report.build(a.export_state()))
    assert "UNRESOLVED exemption / no-requirement claims" in md and "No Free Import Order requirement" in md


def test_contradicted_legal_conclusion_ranks_last():
    good = valid_report(LAW)["opportunities"][0]
    bad = dict(valid_report(LAW)["opportunities"][0], name="Contradicted idea")
    bad["legal_checks"] = legal_checks(LAW)
    bad["legal_checks"]["exceptions"] = {"status": "contradicted", "finding": "excluded",
                                         "evidence": [{"url": LAW, "excerpt": EXCEPTION}]}
    report = {"opportunities": [bad, good], "research_summary": "", "no_opportunity_reason": ""}
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}))], report)
    first, second = run_.result.opportunities
    assert first.name.startswith("Inspection") and first.legal_verification == "verified" and first.classification == "A"
    assert second.legal_verification == "contradicted" and second.classification == "B"


def test_business_advantage_requires_differentiator_and_competitor_evidence():
    report = valid_report(LAW)
    report["opportunities"][0]["business_advantage"] = {
        "claim": "Cheaper than competitors", "generally_available": True, "evidence": [{"url": LAW, "excerpt": EXCERPT}]}
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}))], report)
    opp = run_.result.opportunities[0]
    assert opp.legal_verification == "verified" and opp.business_verification == "unverified"
    assert "generally available" in opp.dimension_notes["business"][0]

    report["opportunities"][0]["business_advantage"] = {
        "claim": "Only steel-screw importers with a yearly inspection can use it", "generally_available": False,
        "differentiator": "inspection capacity", "evidence": [{"url": LAW, "excerpt": SCOPE}],
        "competitor_evidence": [{"url": MARKET, "excerpt": MARKET_TEXT}]}
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}), ("fetch_url", {"url": MARKET}))], report)
    assert run_.result.opportunities[0].business_verification == "verified"

    report["opportunities"][0]["business_advantage"]["status"] = "contradicted"
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}), ("fetch_url", {"url": MARKET}))], report)
    assert run_.result.opportunities[0].business_verification == "contradicted"


def test_business_finding_linked_to_legal_finding():
    a, *_ = run([
        tool_response(("fetch_url", {"url": LAW}), ("fetch_url", {"url": MARKET})),
        tool_response(("record_findings", {"findings": [
            {"statement": "Yearly inspection duty", "source_url": LAW, "excerpt": EXCERPT,
             "claim_type": "legal_conclusion"},
            {"statement": "Advantage for inspected importers", "source_url": LAW, "excerpt": SCOPE,
             "claim_type": "business_advantage",
             "business": {"generally_available": False, "differentiator": "x", "based_on": "F1",
                          "competitor_source_url": MARKET, "competitor_excerpt": MARKET_TEXT}}]})),
    ])
    legal, business = a.findings
    # Evidence is in place, but the underlying legal conclusion is only partially verified.
    assert business["business_status"] == "partially_verified"
    assert any("not fully verified" in n for n in business["business_notes"])


def test_model_cannot_set_dimension_fields():
    report = valid_report(LAW, legal=False)
    report["opportunities"][0].update(legal_verification="verified", business_verification="verified",
                                      negative_claim=False, dimension_notes={"legal": ["trust me"]})
    report["opportunities"][0]["legal_checks"] = {"provision": {"status": "checked", "verified": True,
                                                                "evidence": [{"url": LAW, "excerpt": "invented quote here",
                                                                              "excerpt_verified": True}]}}
    _, run_, _ = run([tool_response(("fetch_url", {"url": LAW}))], report)
    opp = run_.result.opportunities[0]
    assert opp.legal_verification == "unverified" and opp.business_verification == "unverified"
    assert opp.legal_checks.provision.verified is False


def test_dimension_logic_units():
    from models import BusinessAdvantage, LegalChecks

    assert verification.is_negative_claim("The import is exempt") and verification.is_negative_claim("אינו טעון רישיון")
    assert not verification.is_negative_claim("Importers must obtain approval")
    status, notes = verification.legal_dimension(LegalChecks(), negative=True)
    assert status == "unverified" and "NOT established" in notes[0]
    status, _ = verification.business_dimension(BusinessAdvantage(), "verified")
    assert status == "unverified"
    assert verification.counters([]) == {k: 0 for k in verification.counters([])}
