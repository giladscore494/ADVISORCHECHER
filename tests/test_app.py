"""Drives the Streamlit UI end to end with the model, search and fetch mocked."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

import agent
import llm
from helpers import FakeLLM, fake_fetch, fake_search, text_response, tool_response, valid_report

APP = str(Path(__file__).resolve().parent.parent / "app.py")
URL = "https://www.gov.il/he/departments/legalInfo/regulation-x"


def test_missing_keys_disable_button(monkeypatch):
    monkeypatch.setattr("config.get_setting", lambda name, default=None: default)
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert at.title[0].value == "Regulatory Opportunity Hunter"
    assert any("does not provide legal advice" in w.value for w in at.warning)
    assert any("Missing configuration" in e.value for e in at.error)
    assert at.button[0].disabled


def _run_ui(monkeypatch, responses, domain="equipment rental", instructions=None):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setenv("RESEARCH_RUN_MODE", "inline")
    fake = FakeLLM(responses)
    monkeypatch.setattr(llm, "LLMClient", lambda provider=None: fake)
    real = agent.ResearchAgent
    monkeypatch.setattr(agent, "ResearchAgent", lambda client, **kw: real(client, search_fn=fake_search, fetch_fn=fake_fetch, **kw))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input(domain).run()
    if instructions is not None:
        at.text_area[0].input(instructions).run()
    at.button[0].click().run()
    assert not at.exception
    at.fake_llm = fake
    return at


def test_full_research_flow_renders_results(monkeypatch):
    at = _run_ui(monkeypatch, [
        tool_response(("search_web", {"query": "פטור השכרה", "purpose": "Searching rental exemptions"})),
        tool_response(("fetch_url", {"url": URL})),
        text_response("DONE"),
        text_response(json.dumps(valid_report(URL))),
    ])
    labels = [e.label for e in at.expander]
    assert any(l.startswith("Inspection equipment rental 0") and "Class A" in l for l in labels)
    assert "Research Trace" in labels
    md = " ".join(m.value for m in at.markdown)
    assert URL in md and "פטור השכרה" in md


def test_no_opportunity_flow(monkeypatch):
    at = _run_ui(monkeypatch, [
        text_response("DONE"),
        text_response(json.dumps({"opportunities": [], "no_opportunity_reason": "Everything needed a licence."})),
    ])
    assert any("No sufficiently strong opportunity found." in i.value for i in at.info)


def test_custom_instructions_field_is_optional(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    at = AppTest.from_file(APP).run()
    assert at.text_area[0].label == "Custom Research Instructions"
    assert at.text_area[0].value in ("", None)
    at.text_input[0].input("equipment rental").run()
    assert not at.button[0].disabled  # domain alone is enough


def test_custom_instructions_sent_to_agent_and_shown_in_trace(monkeypatch):
    text = ("Prioritize opportunities requiring less than ₪30,000 in startup capital.\n"
            "עדיפות לעסק שניתן לנהל לצד עבודה במשרה מלאה.")
    at = _run_ui(monkeypatch, [
        text_response("DONE"),
        text_response(json.dumps({"opportunities": [], "no_opportunity_reason": "none"})),
    ], domain="Equipment rental in Israel", instructions=text)
    user_msg = at.fake_llm.calls[0]["messages"][1]["content"]
    assert "Research domain: Equipment rental in Israel" in user_msg
    assert f"<custom_instructions>\n{text}\n</custom_instructions>" in user_msg
    md = " ".join(m.value for m in at.markdown)
    assert "Custom research instructions" in md
    assert "₪30,000" in md and "עדיפות לעסק" in md


def test_trace_shows_none_when_no_instructions(monkeypatch):
    at = _run_ui(monkeypatch, [
        text_response("DONE"),
        text_response(json.dumps({"opportunities": []})),
    ])
    assert "custom_instructions" not in at.fake_llm.calls[0]["messages"][1]["content"]
    assert any(m.value == "_None provided._" for m in at.markdown)


def test_verified_opportunity_shows_verified_badge(monkeypatch):
    at = _run_ui(monkeypatch, [
        tool_response(("fetch_url", {"url": URL})),
        text_response("DONE"),
        text_response(json.dumps(valid_report(URL))),
    ])
    label = next(e.label for e in at.expander if e.label.startswith("Inspection"))
    assert "Class A" in label and "Source ✅" in label and "Legal ✅" in label
    # No business-advantage evidence was given: that dimension stays unverified and is shown separately.
    assert "Advantage ❌" in label
    assert any("Source evidence verified" in s.value for s in at.success)


def test_unverified_opportunity_is_flagged_and_downgraded(monkeypatch):
    blocked = "https://www.gov.il/he/departments/legalInfo/never-read"
    at = _run_ui(monkeypatch, [
        text_response("DONE"),
        text_response(json.dumps(valid_report(blocked))),
    ])
    label = next(e.label for e in at.expander if e.label.startswith("Inspection"))
    assert "Class B" in label and "Source ❌" in label and "Legal ❌" in label
    errors = " ".join(e.value for e in at.error)
    assert "Legal finding UNVERIFIED" in errors and "Downgraded from A to B" in errors
    md = " ".join(m.value for m in at.markdown)
    assert f"❌ [Regulation X]({blocked})" in md and "Not retrieved during this run" in md


def test_dataset_trace_and_provenance_rendered(monkeypatch):
    import datagov
    import fetcher
    from ckan_fixtures import PRESERVATION_RESOURCE_ID, STD_RECORD_LINE, STD_RESOURCE_ID, standard_session
    from helpers import EXCERPT

    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setenv("RESEARCH_RUN_MODE", "inline")
    page = f"https://data.gov.il/dataset/official-standards/resource/{STD_RESOURCE_ID}"
    report = valid_report(URL)
    report["opportunities"][0]["primary_sources"] = [
        {"title": "Order", "url": URL, "excerpt": EXCERPT},
        {"title": "Standards", "url": page, "resource_id": STD_RESOURCE_ID, "excerpt": STD_RECORD_LINE},
    ]
    fake = FakeLLM([
        tool_response(("read_government_resource", {"resource_id": PRESERVATION_RESOURCE_ID, "query": "תקן רשמי ברגים"})),
        tool_response(("read_government_resource", {"resource_id": STD_RESOURCE_ID, "query": "תקן רשמי ברגים"}),
                      ("fetch_url", {"url": URL})),
        text_response("DONE"),
        text_response(json.dumps(report)),
    ])
    monkeypatch.setattr(llm, "LLMClient", lambda provider=None: fake)
    real = agent.ResearchAgent
    ckan = datagov.CkanClient(session=standard_session(), min_interval=0)
    monkeypatch.setattr(agent, "ResearchAgent", lambda client, **kw: real(
        client, search_fn=fake_search, fetch_fn=fake_fetch, ckan_client=ckan, **kw))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("industrial fasteners import standards").run()
    at.button[0].click().run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Official government datasets (data.gov.il)" in md
    assert "⛔" in md and "rejected: unrelated to research topic" in md  # the preservation dataset
    assert f"]({page})" in md and "records read" in md
    assert "Publisher: משרד הכלכלה והתעשייה" in md and "(not a legal date)" in md
    label = next(e.label for e in at.expander if e.label.startswith("Inspection"))
    assert "Class A" in label and "Source ✅" in label


# ------------------------------------------------------------ recovery UI
import research_runner  # noqa: E402
import research_store  # noqa: E402
from helpers import EXCERPT  # noqa: E402

RESEARCH = [
    tool_response(("search_web", {"query": "פטור השכרה"})),
    tool_response(("fetch_url", {"url": URL}),
                  ("update_candidates", {"candidates": [{"name": "Equipment rental", "status": "surviving"}]})),
    tool_response(("record_findings", {"findings": [{"statement": "Yearly inspection required", "source_url": URL,
                                                     "excerpt": EXCERPT}]})),
    text_response("DONE"),
]


class FinalCrash(FakeLLM):
    def chat(self, messages, tools=None, json_mode=False):
        if json_mode:
            raise RuntimeError("worker lost while generating the final JSON")
        return super().chat(messages, tools, json_mode)


def _failed_run():
    store = research_store.get_store()
    run_id, run = research_runner.start(store, "equipment rental", "", agent.Limits(), "kimi", background=False,
                                        llm_factory=lambda p: FinalCrash(list(RESEARCH)),
                                        agent_kwargs={"search_fn": fake_search, "fetch_fn": fake_fetch})
    return store, run_id


def _labels(at, kind):
    return [e.proto.label for e in at.get(kind)]


def test_failed_run_recovered_from_url_shows_partial_report_and_resume(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setenv("RESEARCH_RUN_MODE", "inline")
    store, run_id = _failed_run()
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["run"] = run_id
    at.run()
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert f"**Run ID:** `{run_id}`" in md and "❌ failed" in md
    assert "VERIFIED findings" in md and "Yearly inspection required" in md and "WORKING HYPOTHESES" in md
    assert any("NOT a validated final report" in w.value for w in at.warning)
    assert any("worker lost" in e.value for e in at.error)
    downloads = _labels(at, "download_button")
    assert {"Download partial report (Markdown)", "Download partial report (JSON)",
            "Download research trace (JSON)"} <= set(downloads)
    assert any("Last successful checkpoint" in c.value for c in at.caption)

    # Resume the run from the UI: finalization only, then the validated report is shown.
    monkeypatch.setattr(llm, "LLMClient", lambda provider=None: FakeLLM([text_response(json.dumps(valid_report(URL)))]))
    real = agent.ResearchAgent
    monkeypatch.setattr(agent, "ResearchAgent", lambda client, **kw: real(client, search_fn=fake_search,
                                                                          fetch_fn=fake_fetch, **kw))
    resume = next(b for b in at.button if b.label == "Resume research from last checkpoint")
    resume.click().run()
    assert not at.exception
    assert store.load(run_id)["status"] == "completed"
    assert any(e.label.startswith("Inspection equipment rental 0") for e in at.expander)
    md = " ".join(m.value for m in at.markdown)
    assert "✅ completed" in md


def test_interrupted_run_detected_and_recovered_by_id(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    store = research_store.get_store()
    store.create_run("20261007-000000-deadbeefdeadbeef", "imports", {"provider": "kimi", "limits": {}})
    store.save_checkpoint("20261007-000000-deadbeefdeadbeef", "tool search_web (step 1)", "searching",
                          {"state_version": 1, "domain": "imports", "search_count": 1,
                           "trace": {"searches": [{"query": "יבוא אישי", "results": [], "error": ""}]}},
                          {"step": 1, "searches": 1})
    old = research_store.ts(research_store.utc_now().replace(year=2020))
    store._execute([("UPDATE research_runs SET heartbeat_at = ?", (old,))])
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input(key="recover_run_id").input("20261007-000000-deadbeefdeadbeef").run()
    next(b for b in at.button if b.label == "Load run").click().run()
    assert not at.exception
    assert any("was interrupted" in e.value for e in at.error)
    md = " ".join(m.value for m in at.markdown)
    assert "⚠️ interrupted" in md and "יבוא אישי" in md
    assert any(b.label == "Resume research from last checkpoint" for b in at.button)


def test_unknown_run_id(monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["run"] = "nope"
    at.run()
    assert any("was not found" in e.value for e in at.error)


def test_background_run_completes_and_renders(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setenv("RESEARCH_RUN_MODE", "background")
    fake = FakeLLM([tool_response(("search_web", {"query": "פטור"})), tool_response(("fetch_url", {"url": URL})),
                    text_response("DONE"), text_response(json.dumps(valid_report(URL)))])
    monkeypatch.setattr(llm, "LLMClient", lambda provider=None: fake)
    real = agent.ResearchAgent
    monkeypatch.setattr(agent, "ResearchAgent", lambda client, **kw: real(client, search_fn=fake_search,
                                                                          fetch_fn=fake_fetch, **kw))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input("equipment rental").run()
    at.button[0].click().run()
    assert not at.exception
    run_id = at.session_state["run_id"]
    assert research_runner.wait(run_id, timeout=30)
    at.run()
    assert any(e.label.startswith("Inspection equipment rental 0") for e in at.expander)
    assert research_store.get_store().load(run_id)["status"] == "completed"


def test_snapshot_status_and_provider_choice(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("Local official datasets: 5 validated snapshots" in e.label for e in at.expander)
    box = at.selectbox[0]
    assert box.options[2].startswith("OpenAI · gpt-6.1-sol") and "(API key not configured)" in box.options[2]
    box.select("openai").run()
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
    assert at.button[0].disabled


def test_separate_verification_counters_and_unresolved_claims(monkeypatch):
    """10/11 source-verified quotes must never read as 10/11 verified legal conclusions."""
    from helpers import EXCERPT

    report = valid_report(URL, legal=False)
    report["opportunities"][0]["regulatory_mechanism"] = "Rental equipment is exempt (פטור) from licensing."
    at = _run_ui(monkeypatch, [
        tool_response(("fetch_url", {"url": URL})),
        tool_response(("record_findings", {"findings": [
            {"statement": "Yearly inspection duty applies", "source_url": URL, "excerpt": EXCERPT,
             "claim_type": "legal_conclusion"},
            {"statement": "Regulation X text mentions inspections", "source_url": URL, "excerpt": EXCERPT}]})),
        text_response("DONE"),
        text_response(json.dumps(report)),
    ])
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Sources verified"] == "2/2"
    assert metrics["Legal conclusions verified"] == "0/1"
    assert metrics["Business advantages verified"] == "0/0"
    label = next(e.label for e in at.expander if e.label.startswith("Inspection"))
    assert "Class B" in label and "Source ✅" in label and "Legal ❌" in label
    errors = " ".join(e.value for e in at.error)
    assert "Exemption / no-requirement claim NOT established" in errors
    md = " ".join(m.value for m in at.markdown)
    assert "legal conclusions verified 0/1" in md
    assert any("Opportunities: source verified 1/1 · legal applicability verified 0/1" in c.value for c in at.caption)
