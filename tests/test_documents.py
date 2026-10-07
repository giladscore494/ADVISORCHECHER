"""Complete, page-indexed document extraction, search and exact page retrieval (incl. Hebrew PDFs)."""

import hashlib
import json

import pytest

import agent
import documents
import fetcher
from fetcher import FetchResult, fetch_url
from helpers import FakeLLM, legal_checks, make_text_pdf, text_response, tool_response, valid_report

ORDER_URL = "https://www.gov.il/BlobFolder/legalinfo/free-import-order/he/free_import_order.pdf"
SCHEDULE_PAGE = 120
SCHEDULE_QUOTE = "פרט 7318.15 - ברגים ולולבים אחרים: אישור תקן לפי ת\"י 1347"


def order_pages(n=140, schedule_page=SCHEDULE_PAGE):
    """A long Hebrew order whose relevant schedule is far beyond the old 60-page / 15,000-character limits."""
    pages = []
    for i in range(1, n + 1):
        if i == schedule_page:
            pages.append("תוספת שנייה\n(סעיף 2(א))\n" + SCHEDULE_QUOTE + "\nISO 4032 ו-ISO 7089 אינם תקנים רשמיים")
        else:
            pages.append(f'צו יבוא חופשי, התשע"ד-2014\nעמוד {i}: הוראות כלליות לעניין יבוא טובין\nסעיף {i} - הגדרות\n'
                         f"({i}א) יבואן יגיש לממונה את המסמכים הנדרשים לפי התוספת הרלוונטית לפני שחרור הטובין\n"
                         f"({i}ב) הממונה רשאי לדרוש אישור נוסף מהרשות המוסמכת לעניין טובין כאמור")
    return pages


@pytest.fixture(scope="module")
def order_pdf():
    return make_text_pdf(order_pages())


@pytest.fixture
def store(tmp_path):
    return documents.DocumentStore(tmp_path / "docs")


# ------------------------------------------------------------- extraction
def test_pdf_over_100_pages_is_extracted_completely(order_pdf):
    doc = documents.extract_pdf(order_pdf)
    assert len(doc.pages) == doc.expected_pages == 140
    assert doc.status == "complete" and doc.issues == []
    total = sum(len(p) for p in doc.pages)
    assert total > fetcher.MAX_TEXT_CHARS  # beyond the old 15,000-character cut
    assert SCHEDULE_QUOTE in doc.pages[SCHEDULE_PAGE - 1]  # page 120: beyond the old 60-page cut


def test_hebrew_pdf_text_in_logical_order_with_numbers_and_standards():
    lines = ['אישור תקן לפי ת"י 1347 בלבד', "עמוד 121: הוראות כלליות",
             "פרט 7318.15 - ברגים (סעיף 2(א)) ו-ISO 4032", "English line 12345"]
    doc = documents.extract_pdf(make_text_pdf(["\n".join(lines)]))
    assert doc.pages[0].split("\n") == lines
    assert doc.methods == ["pdfplumber"]  # pypdf alone drops Hebrew next to numbers


def test_pypdf_alone_loses_hebrew_which_is_why_pdfplumber_is_used():
    from pypdf import PdfReader
    import io

    pdf = make_text_pdf(['אישור תקן לפי ת"י 1347 בלבד'])
    plain = PdfReader(io.BytesIO(pdf)).pages[0].extract_text()
    assert "אישור" not in plain  # documents the pypdf behaviour the extractor works around


def test_reversed_hebrew_is_detected_and_corrected():
    logical = "הוראות כלליות לעניין יבוא טובין ומוצרים אחרים\nתוספת שנייה לצו ברגים ואומים מפלדה"
    reversed_text = "\n".join(line[::-1] for line in logical.split("\n"))
    assert documents.looks_reversed(reversed_text) and not documents.looks_reversed(logical)
    fixed, changed = documents.fix_hebrew_order(reversed_text)
    assert changed and fixed == logical


def test_scanned_pages_make_the_document_partial(tmp_path):
    doc = documents.extract_pdf(make_text_pdf(order_pages(20), blank_pages=(7, 8)))
    assert doc.status == "partial"
    assert any("pages 7-8" in i and "OCR is not supported" in i for i in doc.issues)
    st = documents.DocumentStore(tmp_path)
    meta = st.put("https://www.gov.il/x.pdf", doc)
    status = st.status(meta["document_id"])
    assert not status["complete"] and status["pages_without_text"] == "7-8" and status["pages_with_text"] == 18
    miss = st.search(meta["document_id"], "פטור מיוחד")
    assert miss["total_matching_passages"] == 0 and "INCOMPLETE" in miss["note"]


def test_garbled_font_encoding_is_reported():
    doc = documents.extract_pdf(make_text_pdf(["  abc"]))
    assert doc.status == "partial" and any("garbled" in i for i in doc.issues)


def test_page_limit_is_reported_not_silent(order_pdf):
    doc = documents.extract_pdf(order_pdf, max_pages=50)
    assert len(doc.pages) == 50 and doc.status == "partial"
    assert any("first 50 of 140 pages" in i for i in doc.issues)


def test_no_text_at_all_fails_loudly():
    with pytest.raises(ValueError, match="no extractable text"):
        documents.extract_pdf(make_text_pdf(["x", "y"], blank_pages=(1, 2)))


def test_html_is_split_into_sections_without_losing_text():
    text = "\n\n".join(f"סעיף {i}. " + "טקסט חוקי ארוך " * 40 for i in range(60))
    doc = documents.extract_sections(text, "html", "law")
    assert len(doc.pages) > 5 and all(len(p) <= documents.SECTION_CHARS for p in doc.pages)
    assert documents.normalize_text(" ".join(doc.pages)) == documents.normalize_text(text)


# ------------------------------------------------------------ store / search
def test_stable_content_addressed_ids_and_persistence(tmp_path, order_pdf):
    doc = documents.extract_pdf(order_pdf)
    a = documents.DocumentStore(tmp_path)
    m1 = a.put(ORDER_URL, doc, content=order_pdf, raw=order_pdf)
    m2 = a.put(ORDER_URL + "?copy", doc, content=order_pdf, raw=order_pdf)
    assert m1["document_id"] == m2["document_id"] and m1["document_id"].startswith("doc-")
    other = documents.extract_pdf(make_text_pdf(order_pages(3)))
    assert a.put("https://www.gov.il/y.pdf", other)["document_id"] != m1["document_id"]
    reopened = documents.DocumentStore(tmp_path)  # persistent across processes/restarts on this disk
    assert reopened.doc_for_url(ORDER_URL) == m1["document_id"]
    assert reopened.meta(m1["document_id"])["page_count"] == 140


def test_search_finds_schedule_beyond_old_limits_with_page_numbers(store, order_pdf):
    doc_id = store.put(ORDER_URL, documents.extract_pdf(order_pdf), content=order_pdf)["document_id"]
    out = store.search(doc_id, "תוספת שנייה ברגים")
    assert out["pages_with_matches"] == [SCHEDULE_PAGE]
    hit = out["results"][0]
    assert hit["page"] == SCHEDULE_PAGE and SCHEDULE_QUOTE in hit["text"]
    # Numbers match exactly (7318.15 -> tokens 7318, 15), never as substrings of other numbers.
    assert store.search(doc_id, "7318.15")["pages_with_matches"] == [SCHEDULE_PAGE]
    assert store.search(doc_id, "731")["total_matching_passages"] == 0
    assert store.search(doc_id, "סעיף 75")["pages_with_matches"][0] == 75
    restricted = store.search(doc_id, "הגדרות", start_page=10, end_page=12)
    assert set(restricted["pages_with_matches"]) == {10, 11, 12}


def test_read_exact_page_range_and_continuation(store, order_pdf):
    doc_id = store.put(ORDER_URL, documents.extract_pdf(order_pdf), content=order_pdf)["document_id"]
    out = store.read_range(doc_id, SCHEDULE_PAGE, SCHEDULE_PAGE + 1)
    assert [p["page"] for p in out["pages"]] == [120, 121] and out["complete_range"]
    assert SCHEDULE_QUOTE in out["pages"][0]["text"] and out["source_url"] == ORDER_URL
    capped = store.read_range(doc_id, 1, 40)
    assert len(capped["pages"]) == documents.READ_MAX_PAGES and capped["next"]["start_page"] == 11
    with pytest.raises(documents.DocumentError):
        store.read_range(doc_id, 141)


def test_long_page_is_returned_in_parts_with_char_offset(store):
    long_page = "\n".join(f"שורה {i}: טקסט של תקנה ארוכה מאוד" for i in range(1200))
    doc_id = store.put("https://www.gov.il/long.html", documents.extract_sections(long_page, "text"))["document_id"]
    first = store.read_range(doc_id, 1, max_chars=2000)
    assert first["pages"][0]["partial_page"] and not first["complete_range"]
    nxt = first["next"]
    second = store.read_range(doc_id, nxt["start_page"], char_offset=nxt["char_offset"], max_chars=2000)
    joined = first["pages"][0]["text"] + second["pages"][0]["text"]
    page_text = store.pages(doc_id)[0][2]
    assert page_text.startswith(joined)


def test_tables_extracted_on_demand_in_rtl_column_order(store):
    pdf = make_text_pdf(["תוספת שנייה - טבלה"], tables={
        1: [["פרט מכס", "תיאור הטובין", "דרישה"], ["7318.15", "ברגים ולולבים", 'ת"י 1347']]})
    doc_id = store.put("https://www.gov.il/t.pdf", documents.extract_pdf(pdf), content=pdf, raw=pdf)["document_id"]
    tables = store.read_range(doc_id, 1, tables=True)["tables"]
    assert tables[0]["rows"] == ["פרט מכס | תיאור הטובין | דרישה", '7318.15 | ברגים ולולבים | ת"י 1347']


def test_locate_excerpt_reports_pages(store, order_pdf):
    doc_id = store.put(ORDER_URL, documents.extract_pdf(order_pdf), content=order_pdf)["document_id"]
    assert store.locate_excerpt(doc_id, SCHEDULE_QUOTE) == [SCHEDULE_PAGE]
    assert store.locate_excerpt(doc_id, "פטור מכל דרישה לפי צו זה לכל הברגים") == []


# ------------------------------------------------------------------ fetcher
class _Resp:
    def __init__(self, body, ctype="application/pdf", status=200):
        self.status_code, self.headers, self.url, self.encoding = status, {"Content-Type": ctype}, ORDER_URL, "utf-8"
        self._body, self.is_redirect = body, False

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        pass


class _Session:
    def __init__(self, resp):
        self.resp, self.urls = resp, []

    def get(self, url, **kw):
        self.urls.append(url)
        return self.resp


def test_fetch_url_returns_all_pages_without_truncation(monkeypatch, order_pdf):
    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)
    res = fetch_url(ORDER_URL, session=_Session(_Resp(order_pdf)))
    assert res.ok and not res.truncated and res.source_type == "pdf"
    assert len(res.document.pages) == 140 and res.raw == order_pdf
    assert SCHEDULE_QUOTE in res.text and len(res.text) > fetcher.MAX_TEXT_CHARS
    assert "document" not in res.to_dict() and "raw" not in res.to_dict()


def test_fetch_403_is_not_stored_or_bypassed(monkeypatch):
    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)
    session = _Session(_Resp(b"denied", "text/html", status=403))
    res = fetch_url(ORDER_URL, session=session)
    assert not res.ok and res.access_restricted and res.document is None and len(session.urls) == 1


# -------------------------------------------------------------- agent tools
_EXTRACTED: dict[bytes, object] = {}


def _extracted(pdf: bytes):
    if pdf not in _EXTRACTED:
        _EXTRACTED[pdf] = documents.extract_pdf(pdf)
    return _EXTRACTED[pdf]


def pdf_fetch(pdf):
    calls = []
    doc = _extracted(pdf)

    def fetch(url):
        calls.append(url)
        return FetchResult(url=url, final_url=url, ok=True, source_type="pdf", title="צו יבוא חופשי",
                           text=fetcher.pages_text(doc), document=doc, raw=pdf)

    fetch.calls = calls
    return fetch


def doc_id_of(pdf: bytes) -> str:
    return "doc-" + hashlib.sha256(pdf).hexdigest()[:16]  # content-addressed: stable for the same file


def run_agent(responses, fetch, store=None, **limits):
    llm = FakeLLM(responses)
    a = agent.ResearchAgent(llm, limits=agent.Limits(**limits), search_fn=lambda q, num_results=10: [],
                            fetch_fn=fetch, document_store=store)
    return a, a.run("import of screws HS 7318"), llm


def tool_outputs(llm, call):
    return [json.loads(m["content"]) for m in llm.calls[call]["messages"] if m["role"] == "tool"]


def test_agent_fetch_returns_preview_and_document_tools_reach_page_120(order_pdf):
    report = valid_report(ORDER_URL)
    report["opportunities"][0]["primary_sources"] = [
        {"title": "Free Import Order", "url": ORDER_URL, "excerpt": SCHEDULE_QUOTE, "page": "120"}]
    report["opportunities"][0]["legal_checks"] = legal_checks(ORDER_URL, SCHEDULE_QUOTE)
    doc_id = doc_id_of(order_pdf)
    a, run, llm = run_agent([
        tool_response(("fetch_url", {"url": ORDER_URL})),
        tool_response(("search_document", {"document_id": doc_id, "query": "תוספת שנייה ברגים"}),
                      ("get_document_status", {"document_id": doc_id})),
        tool_response(("read_document_range", {"document_id": doc_id, "start_page": 120, "end_page": 120})),
        text_response("DONE"), text_response(json.dumps(report)),
    ], pdf_fetch(order_pdf))
    fetched = tool_outputs(llm, 1)[0]
    assert fetched["document_id"] == doc_id and fetched["document"]["page_count"] == 140
    assert not fetched["text_is_whole_document"] and len(fetched["text"]) <= documents.PREVIEW_CHARS + 100
    assert SCHEDULE_QUOTE not in fetched["text"]  # the large PDF is NOT pushed into the context
    search, status = tool_outputs(llm, 2)[1:]
    assert search["results"][0]["page"] == 120 and status["complete"]
    page = tool_outputs(llm, 3)[-1]
    assert page["pages"][0]["page"] == 120 and SCHEDULE_QUOTE in page["pages"][0]["text"]
    src = run.result.opportunities[0].primary_sources[0]
    assert src.excerpt_verified and src.matched_pages == [120] and src.page_mismatch is False
    assert run.trace["fetches"][0]["pages"] == 140 and run.trace["fetches"][0]["document_status"] == "complete"
    assert [q["tool"] for q in run.trace["document_queries"]] == ["search_document", "get_document_status",
                                                                 "read_document_range"]


def test_excerpt_on_a_page_never_shown_still_verifies_and_wrong_page_is_flagged(order_pdf):
    report = valid_report(ORDER_URL)
    report["opportunities"][0]["primary_sources"] = [
        {"title": "Free Import Order", "url": ORDER_URL, "excerpt": SCHEDULE_QUOTE, "page": "12"}]
    _, run, _ = run_agent([tool_response(("fetch_url", {"url": ORDER_URL})), text_response("DONE"),
                           text_response(json.dumps(report))], pdf_fetch(order_pdf))
    src = run.result.opportunities[0].primary_sources[0]
    assert src.excerpt_verified and src.matched_pages == [120]
    assert src.page_mismatch and "cited page (12) is wrong" in src.verification_note


def test_unknown_or_duplicate_document_requests_are_rejected(order_pdf):
    a, run, llm = run_agent([
        tool_response(("fetch_url", {"url": ORDER_URL})),
        tool_response(("search_document", {"document_id": "doc-nope", "query": "x"})),
        text_response("DONE"), text_response("{}"),
    ], pdf_fetch(order_pdf))
    assert "Unknown document_id" in tool_outputs(llm, 2)[-1]["error"]
    doc_id = doc_id_of(order_pdf)
    a, run, llm = run_agent([
        tool_response(("fetch_url", {"url": ORDER_URL})),
        tool_response(("search_document", {"document_id": doc_id, "query": "ברגים"})),
        tool_response(("search_document", {"document_id": doc_id, "query": "ברגים"}),
                      ("fetch_url", {"url": ORDER_URL})),
        text_response("DONE"), text_response("{}"),
    ], pdf_fetch(order_pdf))
    dup_search, dup_fetch = tool_outputs(llm, 3)[-2:]
    assert "Duplicate document request" in dup_search["error"]
    assert dup_fetch["document_id"] == doc_id and "Duplicate fetch" in dup_fetch["error"]


def test_cached_document_is_reused_across_runs_without_network(tmp_path, order_pdf):
    store = documents.DocumentStore(tmp_path / "shared")
    first = pdf_fetch(order_pdf)
    run_agent([tool_response(("fetch_url", {"url": ORDER_URL})), text_response("DONE"), text_response("{}")],
              first, store)
    second = pdf_fetch(order_pdf)
    a, run, llm = run_agent([tool_response(("fetch_url", {"url": ORDER_URL})), text_response("DONE"),
                             text_response("{}")], second, store)
    assert len(first.calls) == 1 and second.calls == []
    out = tool_outputs(llm, 1)[0]
    assert out["served_from_cache"] and out["document"]["page_count"] == 140
    assert run.trace["fetches"][0]["from_cache"]


def test_document_missing_after_resume_is_redownloaded_for_verification(tmp_path, order_pdf):
    fetch = pdf_fetch(order_pdf)
    a, _, _ = run_agent([tool_response(("fetch_url", {"url": ORDER_URL})), text_response("DONE"),
                         text_response("{}")], fetch, documents.DocumentStore(tmp_path / "a"))
    state = a.export_state()
    report = valid_report(ORDER_URL)
    report["opportunities"][0]["primary_sources"] = [{"title": "o", "url": ORDER_URL, "excerpt": SCHEDULE_QUOTE}]
    b = agent.ResearchAgent(FakeLLM([text_response(json.dumps(report))]), fetch_fn=fetch,
                            document_store=documents.DocumentStore(tmp_path / "b"))  # another machine: empty cache
    state["loop_done"] = True
    run = b.resume(state)
    src = run.result.opportunities[0].primary_sources[0]
    assert src.excerpt_verified and src.matched_pages == [120] and len(fetch.calls) == 2
    assert any("re-downloaded" in w for w in run.trace["warnings"])
