"""Web search via the Serper (Google) API."""

from urllib.parse import urlparse

import requests

from config import get_setting

SERPER_URL = "https://google.serper.dev/search"

# Domains treated as Israeli primary/official sources. Subdomains match too
# (e.g. main.knesset.gov.il, www.gov.il, taxes.gov.il).
PRIMARY_DOMAINS = (
    "gov.il",
    "knesset.gov.il",
    "court.gov.il",
    "boi.org.il",  # Bank of Israel
    "sii.org.il",  # Standards Institution of Israel
)


class SearchError(Exception):
    pass


def is_primary_source(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in PRIMARY_DOMAINS)


def search_web(query: str, num_results: int = 10) -> list[dict]:
    """Search the web; returns [{title, url, snippet, position, primary_source}]."""
    api_key = get_setting("SERPER_API_KEY")
    if not api_key:
        raise SearchError("Missing SERPER_API_KEY.")
    num_results = max(1, min(int(num_results or 10), 20))
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results, "gl": "il"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise SearchError(f"Serper request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SearchError(f"Serper returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise SearchError("Serper returned invalid JSON.") from exc
    return parse_serper_results(data)[:num_results]


def parse_serper_results(data: dict) -> list[dict]:
    results = []
    for i, item in enumerate(data.get("organic", []) or [], start=1):
        url = item.get("link")
        if not url:
            continue
        results.append(
            {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", ""),
                "position": item.get("position", i),
                "primary_source": is_primary_source(url),
            }
        )
    return results
