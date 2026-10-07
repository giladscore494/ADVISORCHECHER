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


def _run_ui(monkeypatch, responses, domain="equipment rental"):
    monkeypatch.setenv("KIMI_API_KEY", "k")
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(llm, "LLMClient", lambda: FakeLLM(responses))
    real = agent.ResearchAgent
    monkeypatch.setattr(agent, "ResearchAgent", lambda client, **kw: real(client, search_fn=fake_search, fetch_fn=fake_fetch, **kw))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input(domain).run()
    at.button[0].click().run()
    assert not at.exception
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
