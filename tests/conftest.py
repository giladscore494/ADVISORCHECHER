import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetcher  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_network(monkeypatch, request):
    """No real sleeping in retry backoff; robots.txt checks off unless a test opts in."""
    monkeypatch.setattr(fetcher, "_sleep", lambda s: None)
    fetcher._robots_cache.clear()
    if "robots" not in request.keywords:
        monkeypatch.setattr(fetcher, "robots_allowed", lambda url, session: True)


@pytest.fixture(autouse=True)
def _isolated_research_store(monkeypatch, tmp_path):
    """Every test gets its own durable SQLite research store and document cache; the local index is not
    pre-warmed. The critical-evidence checklist (and its completion review, which adds a model step) is off
    unless a test enables it, so scripted model conversations stay deterministic."""
    import documents
    import research_store

    monkeypatch.setenv("RESEARCH_STORE_URL", f"sqlite:///{tmp_path / 'runs.sqlite'}")
    monkeypatch.setenv("GOVDATA_WARMUP", "0")
    monkeypatch.setenv("DOCUMENT_CACHE_DIR", str(tmp_path / "documents"))
    monkeypatch.setenv("CRITICAL_EVIDENCE_CHECKLIST", "[]")
    research_store.reset_default_store()
    documents.reset_default()
    yield
    research_store.reset_default_store()
    documents.reset_default()
