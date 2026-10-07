"""Serper web search (kept as the independent legal-source discovery tool)."""

import pytest
import requests

import agent
import search


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_search_web_calls_serper(monkeypatch):
    sent = {}

    def post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, json=json, timeout=timeout)
        return Resp(200, {"organic": [
            {"title": "חוק רישוי עסקים", "link": "https://www.gov.il/he/x", "snippet": "s", "position": 1},
            {"title": "Blog", "link": "https://example.com/y", "snippet": "s"}]})

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setattr(search.requests, "post", post)
    results = search.search_web("רישוי עסקים", num_results=50)
    assert sent["url"] == "https://google.serper.dev/search" and sent["headers"]["X-API-KEY"] == "serper-key"
    assert sent["json"] == {"q": "רישוי עסקים", "num": 20, "gl": "il"}
    assert [r["primary_source"] for r in results] == [True, False]


def test_search_errors(monkeypatch):
    monkeypatch.setattr(search, "get_setting", lambda name, default=None: default)
    with pytest.raises(search.SearchError, match="SERPER_API_KEY"):
        search.search_web("x")
    monkeypatch.setattr(search, "get_setting", lambda name, default=None: "k")
    monkeypatch.setattr(search.requests, "post", lambda *a, **k: Resp(500, "boom"))
    with pytest.raises(search.SearchError, match="HTTP 500"):
        search.search_web("x")

    def fail(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(search.requests, "post", fail)
    with pytest.raises(search.SearchError, match="request failed"):
        search.search_web("x")


def test_agent_defaults_keep_serper_and_fetch():
    a = agent.ResearchAgent(object())
    assert a.search_fn is search.search_web
    names = {t["function"]["name"] for t in agent.TOOLS}
    assert {"search_web", "fetch_url", "search_government_datasets", "inspect_government_dataset",
            "read_government_resource", "search_local_government_records", "record_findings"} <= names
