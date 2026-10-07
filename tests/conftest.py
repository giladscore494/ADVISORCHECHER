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
