import fetcher
from fetcher import extract_html_text, extract_pdf_text, fetch_url
from search import is_primary_source, parse_serper_results
from helpers import make_pdf


def test_html_extraction_strips_boilerplate():
    html = """<html><head><title>תקנות</title><style>.x{}</style></head><body>
    <nav>Menu Home About</nav><header>Site header</header>
    <main><h1>תקנות הרישוי</h1><p>פטור מרישיון   עד 3.5 טון</p><script>alert(1)</script></main>
    <footer>Copyright</footer></body></html>"""
    title, text = extract_html_text(html)
    assert title == "תקנות"
    assert "פטור מרישיון עד 3.5 טון" in text
    for junk in ("Menu", "Site header", "alert", "Copyright", ".x{}"):
        assert junk not in text


def test_pdf_extraction():
    title, text = extract_pdf_text(make_pdf("Exemption threshold 3500 kg"))
    assert "Exemption threshold 3500 kg" in text
    assert text.startswith("[page 1]")


def test_pdf_garbage_reports_error():
    try:
        extract_pdf_text(b"%PDF-1.4 not really a pdf")
    except ValueError as exc:
        assert "PDF" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class FakeResp:
    def __init__(self, status=200, body=b"", ctype="text/html", headers=None, url="https://www.gov.il/x"):
        self.status_code = status
        self.headers = {"Content-Type": ctype, **(headers or {})}
        self.url = url
        self.encoding = "utf-8"
        self._body = body
        self.is_redirect = status in (301, 302) and "Location" in self.headers

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)

    def get(self, url, **kwargs):
        resp = self.responses.pop(0)
        resp.url = url
        return resp


def _public(monkeypatch):
    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)


def test_fetch_html_ok(monkeypatch):
    _public(monkeypatch)
    res = fetch_url("https://www.gov.il/x", session=FakeSession([FakeResp(body=b"<html><body><p>hello law</p></body></html>")]))
    assert res.ok and res.source_type == "html" and "hello law" in res.text


def test_fetch_pdf_detected_by_magic_and_follows_redirect(monkeypatch):
    _public(monkeypatch)
    session = FakeSession([
        FakeResp(status=302, headers={"Location": "/file.bin"}),
        FakeResp(body=make_pdf("Section 12 exemption"), ctype="application/octet-stream"),
    ])
    res = fetch_url("https://www.gov.il/doc", session=session)
    assert res.ok and res.source_type == "pdf" and "Section 12 exemption" in res.text
    assert res.final_url == "https://www.gov.il/file.bin"


def test_fetch_404_and_truncation(monkeypatch):
    _public(monkeypatch)
    res = fetch_url("https://www.gov.il/x", session=FakeSession([FakeResp(status=404)]))
    assert not res.ok and res.http_status == 404
    long_html = b"<html><body><p>" + b"a" * 5000 + b"</p></body></html>"
    res = fetch_url("https://www.gov.il/y", max_chars=100, session=FakeSession([FakeResp(body=long_html)]))
    assert res.ok and res.truncated and len(res.text) < 200


def test_fetch_rejects_bad_scheme_and_private_hosts():
    assert not fetch_url("file:///etc/passwd").ok
    res = fetch_url("http://127.0.0.1:8501/")
    assert not res.ok and "private" in res.error


def test_fetch_size_limit(monkeypatch):
    _public(monkeypatch)
    monkeypatch.setattr(fetcher, "MAX_DOWNLOAD_BYTES", 10)
    res = fetch_url("https://www.gov.il/x", session=FakeSession([FakeResp(body=b"<p>" + b"x" * 100)]))
    assert not res.ok and "limit" in res.error


def test_primary_source_and_serper_parsing():
    assert is_primary_source("https://main.knesset.gov.il/x")
    assert is_primary_source("https://www.gov.il/he/departments")
    assert not is_primary_source("https://notgov.il.example.com/")
    results = parse_serper_results({"organic": [
        {"title": "A", "link": "https://www.gov.il/a", "snippet": "s", "position": 1},
        {"title": "no link"},
    ]})
    assert results == [{"title": "A", "url": "https://www.gov.il/a", "snippet": "s", "position": 1, "primary_source": True}]
