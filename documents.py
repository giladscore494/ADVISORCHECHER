"""Persistent, page-indexed store for retrieved documents (laws, regulations, orders, gov.il PDFs).

fetch_url downloads a document once, extracts ALL of its pages (no silent character or page cap) and
stores them here under a stable, content-addressed document id. The model never receives a whole large
document: it gets a short preview plus the document id, then locates evidence with search_document and
reads exact pages with read_document_range. Every passage keeps its original page number and source URL,
so quotes can be cited and verified against the complete source text.

Extraction quality is checked and reported (get_document_status): page-limit overruns, pages without
extractable text (scanned images; OCR is not supported), page-level parser failures, garbled font
encodings and Hebrew text in reversed (visual) order. A document is "complete" only when none of these
was detected; otherwise it is "partial" and the problems are listed, so a missing passage is never
mistaken for an absent provision.

Hebrew PDFs: pypdf drops right-to-left text when a line switches direction (e.g. Hebrew next to a section
or tariff number), so pages containing Hebrew are rebuilt character by character from their positions
(PDFium via pypdfium2; pdfplumber as a fallback) and converted from visual to logical order. Pages without
right-to-left text keep pypdf's extraction. Tables are rendered on demand (read_document_range tables=true).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from config import get_int, get_setting
from evidence import STOPWORDS, _english_match, _hebrew_match, excerpt_found, normalize_text, strip_controls

try:  # fast character positions for Hebrew / right-to-left pages (installed with pdfplumber)
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_raw
except ImportError:  # pragma: no cover - listed in requirements.txt
    pdfium = pdfium_raw = None
try:  # fallback character-level extraction, and tables
    import pdfplumber
except ImportError:  # pragma: no cover - listed in requirements.txt
    pdfplumber = None

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".document_cache"
MAX_PAGES = get_int("DOCUMENT_MAX_PAGES", 3000)
SECTION_CHARS = 4000  # non-PDF documents are split into sections of about this size
PREVIEW_CHARS = get_int("DOCUMENT_PREVIEW_CHARS", 3500)
READ_MAX_CHARS = get_int("DOCUMENT_READ_MAX_CHARS", 12000)
READ_MAX_PAGES = 10
SEARCH_MAX_RESULTS = 12
SNIPPET_CHARS = 700
WINDOW_LINES = 4
EMPTY_PAGE_CHARS = 1  # a page with fewer non-space characters has no extractable text
MEMORY_DOCS = 8  # documents whose normalized lines are kept in memory for search

RTL_RE = re.compile(r"[֐-׿יִ-ﭏ]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Left-to-right runs inside a right-to-left line (numbers, Latin words, codes such as 7318.15 or ISO 4032).
LTR_RUN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9 .,:/%\-+_]*[A-Za-z0-9])?")
FINALS = set("ךםןףץ")
GARBLED_RE = re.compile(r"[-�]|\(cid:\d+\)")


class DocumentError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- Hebrew order
def is_rtl_line(text: str) -> bool:
    return len(RTL_RE.findall(text)) > len(LATIN_RE.findall(text))


def visual_to_logical(line: str) -> str:
    """Visual-order (left-to-right glyph order) right-to-left line -> logical reading order.
    The line is reversed, then left-to-right runs (numbers, Latin words, codes) are restored."""
    return LTR_RUN_RE.sub(lambda m: m.group(0)[::-1], line[::-1])


_OPEN, _CLOSE = "([{", ")]}"
_SWAP = str.maketrans("()[]{}", ")(][}{")


def _bracket_depth_ok(line: str) -> bool:
    depth = 0
    for ch in line:
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
            if depth < 0:
                return False
    return True


def fix_mirrored_brackets(line: str) -> str:
    """Some extractors mirror brackets in right-to-left runs ("(סעיף 2(א))" -> ")סעיף 2)א(("). Swap them back
    when that makes the line well-formed."""
    if not _bracket_depth_ok(line):
        swapped = line.translate(_SWAP)
        if _bracket_depth_ok(swapped):
            return swapped
    return line


def reversed_hebrew_score(text: str) -> tuple[int, int]:
    """(words starting with a Hebrew final letter, words ending with one). Final letters never start a word,
    so many of the former mean the text is in reversed order."""
    starts = ends = 0
    for word in re.findall(r"[א-ת]{2,}", text):
        starts += word[0] in FINALS
        ends += word[-1] in FINALS
    return starts, ends


def looks_reversed(text: str) -> bool:
    starts, ends = reversed_hebrew_score(text)
    return starts >= 3 and starts > ends


def fix_hebrew_order(text: str) -> tuple[str, bool]:
    """Repair a page whose Hebrew lines are reversed. Returns (text, corrected)."""
    if not looks_reversed(text):
        return text, False
    fixed = "\n".join(visual_to_logical(line) if RTL_RE.search(line) else line for line in text.split("\n"))
    if reversed_hebrew_score(fixed)[0] < reversed_hebrew_score(text)[0]:
        return fixed, True
    return text, False


# --------------------------------------------------------------- extraction
class Extracted:
    """Pages of one document plus extraction diagnostics."""

    def __init__(self, source_type: str, title: str = "", unit: str = "page"):
        self.source_type = source_type
        self.title = title
        self.unit = unit  # page (PDF) | section (HTML, text, tables)
        self.pages: list[str] = []
        self.labels: list[str] = []  # printed page labels (PDF), when the file defines them
        self.methods: list[str] = []
        self.issues: list[str] = []
        self.expected_pages: int | None = None
        self.partial = False

    def problem(self, message: str, partial: bool = True) -> None:
        self.issues.append(message)
        self.partial = self.partial or partial

    @property
    def status(self) -> str:
        return "partial" if self.partial else "complete"


def _ranges(numbers: list[int]) -> str:
    out, start, prev = [], None, None
    for n in numbers:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(out)


def _line_text(chars: list[tuple[float, float, float, str]]) -> str:
    """chars: (x0, x1, size, text) on one line -> logical-order text (gap-based spaces, visual->logical for RTL)."""
    chars = sorted(chars, key=lambda c: c[0])
    text, prev = "", None
    for x0, x1, size, ch in chars:
        if prev is not None and x0 - prev[1] > (size or 10) * 0.15 and not text.endswith(" ") and ch != " ":
            text += " "
        text += ch
        prev = (x0, x1)
    text = re.sub(r" {2,}", " ", text).strip()
    return fix_mirrored_brackets(visual_to_logical(text)) if is_rtl_line(text) else text


def _pdfium_text(page) -> str:
    """Line reconstruction from PDFium character boxes (fast; C library)."""
    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        full = tp.get_text_range(0, n) if n else ""
        if len(full) != n:
            full = "".join(tp.get_text_range(i, 1) for i in range(n))
        chars = []
        for i, ch in enumerate(full):
            if ch in "\r\n\x00" or pdfium_raw.FPDFText_IsGenerated(tp.raw, i) == 1:
                continue
            x0, bottom, x1, top = tp.get_charbox(i, loose=True)
            chars.append(((bottom + top) / 2, top - bottom, x0, x1, ch))
    finally:
        tp.close()
    lines: list[list] = []
    for c in sorted(chars, key=lambda c: -c[0]):
        if lines and abs(lines[-1][0] - c[0]) <= max(1.0, c[1] * 0.3):
            lines[-1][1].append((c[2], c[3], c[1], c[4]))
        else:
            lines.append([c[0], [(c[2], c[3], c[1], c[4])]])
    return "\n".join(t for t in (_line_text(cs) for _, cs in lines) if t)


def _pdfium_probe(page) -> str:
    tp = page.get_textpage()
    try:
        return tp.get_text_range()
    finally:
        tp.close()


def _plumber_text(page) -> str:
    """Fallback character-level line reconstruction with pdfplumber (pdfminer)."""
    return "\n".join(_line_text([(c["x0"], c["x1"], c.get("size") or 10, c["text"]) for c in line["chars"]])
                     for line in page.extract_text_lines(return_chars=True, strip=True))


def extract_pdf(data: bytes, max_pages: int | None = None) -> Extracted:
    """All pages of a PDF, page by page. Raises ValueError when no text at all can be extracted."""
    from pypdf import PdfReader

    max_pages = max_pages or MAX_PAGES
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"PDF is encrypted and cannot be read: {exc}") from exc
        total = len(reader.pages)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"PDF could not be parsed: {exc}") from exc
    title = ""
    try:
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception:  # noqa: BLE001
        pass
    try:
        labels = list(reader.page_labels)
    except Exception:  # noqa: BLE001
        labels = []
    doc = Extracted("pdf", strip_controls(title), "page")
    doc.expected_pages = total
    n = min(total, max_pages)
    if total > max_pages:
        doc.problem(f"Only the first {max_pages} of {total} pages were extracted (DOCUMENT_MAX_PAGES).")
    plumber, errors, rtl_fallback, corrected, reversed_pages, garbled = None, [], [], [], [], []
    fast = None
    if pdfium is not None:
        try:
            fast = pdfium.PdfDocument(data)
        except Exception:  # noqa: BLE001 - fall back to pypdf / pdfplumber
            fast = None
    try:
        for i in range(n):
            text, method = "", "failed"
            probe = None
            if fast is not None:
                try:
                    probe = _pdfium_probe(fast[i])
                except Exception:  # noqa: BLE001
                    probe = None
            frags: list[str] = []
            plain = None
            if probe is None or not RTL_RE.search(probe):
                try:
                    plain = reader.pages[i].extract_text(visitor_text=lambda t, *a: frags.append(t)) or ""
                except Exception:  # noqa: BLE001
                    plain = None
            rtl = bool(probe and RTL_RE.search(probe)) or any(RTL_RE.search(f) for f in frags if f) or bool(
                plain and RTL_RE.search(plain))
            if plain is not None and not rtl:
                text, method = plain, "pypdf"
            else:
                if fast is not None:
                    try:
                        text, method = _pdfium_text(fast[i]), "pdfium"
                    except Exception:  # noqa: BLE001
                        method = "failed"
                if method == "failed" and pdfplumber is not None:
                    try:
                        plumber = plumber or pdfplumber.open(io.BytesIO(data))
                        text, method = _plumber_text(plumber.pages[i]), "pdfplumber"
                    except Exception:  # noqa: BLE001
                        method = "failed"
                if method == "failed" and plain is not None:
                    text, method = plain, "pypdf"
                    if rtl:
                        rtl_fallback.append(i + 1)
                if method == "failed":
                    errors.append(i + 1)
            text = strip_controls(text).strip()
            if RTL_RE.search(text):
                text, fixed = fix_hebrew_order(text)
                if fixed:
                    corrected.append(i + 1)
                elif looks_reversed(text):
                    reversed_pages.append(i + 1)
            body = re.sub(r"\s", "", text)
            if body and len(GARBLED_RE.findall(text)) > 0.05 * len(body):
                garbled.append(i + 1)
            doc.pages.append(text)
            doc.methods.append(method)
            doc.labels.append(str(labels[i]) if i < len(labels) else str(i + 1))
    finally:
        if plumber is not None:
            plumber.close()
        if fast is not None:
            fast.close()

    empty = [i + 1 for i, t in enumerate(doc.pages) if len(re.sub(r"\s", "", t)) < EMPTY_PAGE_CHARS]
    if len(empty) == len(doc.pages):
        raise ValueError("PDF contains no extractable text (possibly scanned images; OCR is not supported).")
    if errors:
        doc.problem(f"Text extraction failed on pages {_ranges(errors)}.")
    if empty:
        # A single blank first/last page is normal; anything else may be a scanned page or table.
        benign = len(empty) == 1 and empty[0] in (1, len(doc.pages))
        doc.problem(f"No extractable text on pages {_ranges(empty)} (possibly scanned images or graphics; "
                    "OCR is not supported). Content on those pages was NOT searched.", partial=not benign)
    if rtl_fallback:
        doc.problem(f"Pages {_ranges(rtl_fallback)} contain Hebrew but could only be read with pypdf; text order and "
                    "completeness are unreliable.")
    if corrected:
        doc.problem(f"Hebrew text order was reversed and has been corrected on pages {_ranges(corrected)}.",
                    partial=False)
    if reversed_pages:
        doc.problem(f"Hebrew text on pages {_ranges(reversed_pages)} appears to be in reversed order.")
    if garbled:
        doc.problem(f"Text on pages {_ranges(garbled)} looks garbled (unmapped font encoding).")
    return doc


def split_sections(text: str, target: int = SECTION_CHARS) -> list[str]:
    """Split long text into ~target-sized sections at paragraph, then line, boundaries (never mid-word)."""
    text = (text or "").strip()
    if len(text) <= target:
        return [text] if text else []
    sections, current = [], ""
    for block in re.split(r"(\n\s*\n)", text):
        pieces = [block] if len(block) <= target else re.split(r"(?<=\n)", block)
        for piece in pieces:
            while len(piece) > target:  # one enormous line: cut at the last space before the limit
                cut = piece.rfind(" ", 0, target)
                cut = cut if cut > target // 2 else target
                if current:
                    sections.append(current)
                    current = ""
                sections.append(piece[:cut])
                piece = piece[cut:]
            if len(current) + len(piece) > target and current.strip():
                sections.append(current)
                current = ""
            current += piece
    if current.strip():
        sections.append(current)
    return [s.strip() for s in sections if s.strip()]


def extract_sections(text: str, source_type: str, title: str = "") -> Extracted:
    doc = Extracted(source_type, strip_controls(title), "section")
    doc.pages = split_sections(strip_controls(text))
    doc.labels = [str(i + 1) for i in range(len(doc.pages))]
    doc.methods = [source_type] * len(doc.pages)
    doc.expected_pages = len(doc.pages)
    return doc


# --------------------------------------------------------------- matching
def query_terms(query: str) -> tuple[list[str], str]:
    """Search terms (words without stopwords + every number) and the normalized phrase."""
    phrase = normalize_text(query)
    terms = []
    for tok in phrase.split():
        if tok in STOPWORDS or tok in terms:
            continue
        if tok.isdigit() or len(tok) >= 2:
            terms.append(tok)
    return terms, phrase


def _term_hit(term: str, tokens: set[str]) -> bool:
    if term in tokens:
        return True
    if term.isdigit():
        return False  # numbers (sections, tariff items, standards) match exactly
    hebrew = any("֐" <= c <= "׿" for c in term)
    for tok in tokens:
        if (hebrew and _hebrew_match(term, tok)) or (not hebrew and _english_match(term, tok)):
            return True
    return False


# ------------------------------------------------------------------- store
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, url TEXT, final_url TEXT, source_type TEXT, title TEXT, unit TEXT,
    content_sha256 TEXT, content_bytes INTEGER, page_count INTEGER, expected_pages INTEGER, chars INTEGER,
    status TEXT, issues TEXT, methods TEXT, http_status INTEGER, fetched_at TEXT, raw_path TEXT);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL, page_no INTEGER NOT NULL, label TEXT, text TEXT NOT NULL,
    PRIMARY KEY (doc_id, page_no));
CREATE TABLE IF NOT EXISTS urls (url_key TEXT PRIMARY KEY, doc_id TEXT NOT NULL, fetched_at TEXT);
"""


def url_key(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    p = urlparse((url or "").strip())
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", "", p.query, ""))


class DocumentStore:
    """SQLite-backed cache of extracted documents, shared by all runs in this process (and across restarts
    on the same disk). Documents are immutable: a changed source gets a new content-addressed id."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir or get_setting("DOCUMENT_CACHE_DIR") or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / "documents.sqlite"
        self._lock = threading.Lock()
        self._memory: OrderedDict[str, list[tuple[int, str, list[str], list[set[str]]]]] = OrderedDict()
        con = self._connect()
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        return con

    # ----------------------------------------------------------- writing
    def put(self, url: str, doc: Extracted, *, final_url: str = "", content: bytes | None = None,
            raw: bytes | None = None, http_status: int | None = None) -> dict:
        """Store an extracted document; returns its metadata. Identical content -> the same document id."""
        joined = "\f".join(doc.pages).encode("utf-8")
        sha = hashlib.sha256(content if content is not None else joined).hexdigest()
        doc_id = "doc-" + sha[:16]
        now = _now()
        raw_path = ""
        if raw is not None and doc.source_type == "pdf":
            raw_dir = self.cache_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            target = raw_dir / f"{doc_id}.pdf"
            if not target.exists():
                tmp = target.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_bytes(raw)
                os.replace(tmp, target)
            raw_path = str(target)
        with self._lock:
            con = self._connect()
            try:
                exists = con.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
                if not exists:
                    con.execute(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (doc_id, url, final_url or url, doc.source_type, doc.title, doc.unit, sha,
                         len(content) if content is not None else len(joined), len(doc.pages), doc.expected_pages,
                         sum(len(p) for p in doc.pages), doc.status, json.dumps(doc.issues, ensure_ascii=False),
                         json.dumps(doc.methods), http_status, now, raw_path))
                    con.executemany("INSERT INTO pages VALUES (?,?,?,?)",
                                    [(doc_id, i + 1, doc.labels[i] if i < len(doc.labels) else str(i + 1), text)
                                     for i, text in enumerate(doc.pages)])
                for u in {url_key(url), url_key(final_url or url)}:
                    con.execute("INSERT OR REPLACE INTO urls VALUES (?,?,?)", (u, doc_id, now))
                con.commit()
            finally:
                con.close()
        return self.meta(doc_id)

    # ----------------------------------------------------------- reading
    def meta(self, doc_id: str) -> dict | None:
        con = self._connect()
        try:
            row = con.execute("SELECT doc_id, url, final_url, source_type, title, unit, content_sha256, "
                              "content_bytes, page_count, expected_pages, chars, status, issues, methods, "
                              "http_status, fetched_at, raw_path FROM documents WHERE doc_id = ?",
                              (str(doc_id).strip(),)).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        keys = ("document_id", "url", "final_url", "source_type", "title", "unit", "content_sha256", "content_bytes",
                "page_count", "expected_pages", "chars", "status", "issues", "methods", "http_status", "fetched_at",
                "raw_path")
        meta = dict(zip(keys, row))
        meta["issues"] = json.loads(meta["issues"] or "[]")
        meta["methods"] = json.loads(meta["methods"] or "[]")
        return meta

    def has(self, doc_id: str) -> bool:
        return self.meta(doc_id) is not None

    def doc_for_url(self, url: str) -> str | None:
        con = self._connect()
        try:
            row = con.execute("SELECT doc_id FROM urls WHERE url_key = ?", (url_key(url),)).fetchone()
        finally:
            con.close()
        return row[0] if row else None

    def _require(self, doc_id: str) -> dict:
        meta = self.meta(doc_id)
        if meta is None:
            raise DocumentError(f"Unknown document_id '{doc_id}'. Use the document_id returned by fetch_url.")
        return meta

    def pages(self, doc_id: str) -> list[tuple[int, str, str]]:
        """[(page_no, label, text)] in page order."""
        con = self._connect()
        try:
            return con.execute("SELECT page_no, label, text FROM pages WHERE doc_id = ? ORDER BY page_no",
                               (doc_id,)).fetchall()
        finally:
            con.close()

    def _indexed(self, doc_id: str):
        with self._lock:
            if doc_id in self._memory:
                self._memory.move_to_end(doc_id)
                return self._memory[doc_id]
        out = []
        for page_no, label, text in self.pages(doc_id):
            lines = [ln for ln in text.split("\n")]
            out.append((page_no, label, lines, [set(normalize_text(ln).split()) for ln in lines]))
        with self._lock:
            self._memory[doc_id] = out
            while len(self._memory) > MEMORY_DOCS:
                self._memory.popitem(last=False)
        return out

    def status(self, doc_id: str) -> dict:
        meta = self._require(doc_id)
        pages = self.pages(doc_id)
        empty = [p for p, _, t in pages if len(re.sub(r"\s", "", t)) < EMPTY_PAGE_CHARS]
        methods = meta.pop("methods")
        meta.pop("raw_path", None)
        counts: dict[str, int] = {}
        for m in methods:
            counts[m] = counts.get(m, 0) + 1
        return {
            **meta,
            "complete": meta["status"] == "complete",
            "pages_with_text": len(pages) - len(empty),
            "pages_without_text": _ranges(empty),
            "extraction_methods": counts,
            "tables_available": bool(self.meta(doc_id).get("raw_path")),
            "note": ("Complete extraction: every page was extracted and indexed." if meta["status"] == "complete" else
                     "INCOMPLETE extraction: see issues. A provision not found in this document may be on a page "
                     "that could not be extracted; do not treat its absence as proof."),
        }

    # ------------------------------------------------------------ search
    def search(self, doc_id: str, query: str, max_results: int = 8, start_page: int | None = None,
               end_page: int | None = None) -> dict:
        meta = self._require(doc_id)
        terms, phrase = query_terms(query)
        if not terms:
            raise DocumentError("query must contain at least one meaningful word or number")
        max_results = max(1, min(int(max_results or 8), SEARCH_MAX_RESULTS))
        need = len(terms) if len(terms) <= 2 else max(2, int(len(terms) * 0.6 + 0.999))
        hits = []
        for page_no, label, lines, tokens in self._indexed(doc_id):
            if (start_page and page_no < start_page) or (end_page and page_no > end_page):
                continue
            page_tokens = set().union(*tokens) if tokens else set()
            if sum(_term_hit(t, page_tokens) for t in terms) < need:
                continue
            step = max(1, WINDOW_LINES // 2)
            for i in range(0, max(1, len(lines) - WINDOW_LINES + step), step):
                win_tokens = set().union(*tokens[i:i + WINDOW_LINES])
                got = [t for t in terms if _term_hit(t, win_tokens)]
                if len(got) < need:
                    continue
                norm = " ".join(normalize_text(ln) for ln in lines[i:i + WINDOW_LINES])
                exact = len(phrase.split()) > 1 and phrase in norm
                score = round(10 * len(got) / len(terms) + (5 if exact else 0), 2)
                hits.append({"page": page_no, "page_label": label, "first_line": i + 1,
                             "last_line": min(len(lines), i + WINDOW_LINES), "score": score,
                             "matched_terms": got, "exact_phrase": exact,
                             "text": "\n".join(lines[i:i + WINDOW_LINES]).strip()[:SNIPPET_CHARS]})
        # Keep the best non-overlapping window per region of a page.
        hits.sort(key=lambda h: (-h["score"], h["page"], h["first_line"]))
        chosen: list[dict] = []
        for h in hits:
            if any(c["page"] == h["page"] and c["first_line"] <= h["last_line"] and h["first_line"] <= c["last_line"]
                   for c in chosen):
                continue
            chosen.append(h)
        pages_hit = sorted({h["page"] for h in chosen})
        out = {
            "document_id": doc_id, "query": query, "terms": terms, "unit": meta["unit"],
            "page_count": meta["page_count"], "document_status": meta["status"],
            "total_matching_passages": len(chosen), "pages_with_matches": pages_hit[:60],
            "results": chosen[:max_results],
        }
        if len(chosen) > max_results:
            out["more"] = (f"{len(chosen) - max_results} more passages; narrow the query or restrict "
                           "start_page/end_page.")
        if not chosen:
            out["note"] = (f"No passage in this document matched {terms}. "
                           + ("The whole document was extracted and searched, " if meta["status"] == "complete"
                              else "Extraction was INCOMPLETE (see get_document_status), ")
                           + "but absence here is not proof that a provision does not exist: wording may differ; "
                             "try synonyms, the section number, or Hebrew/English variants.")
        return out

    # -------------------------------------------------------------- read
    def read_range(self, doc_id: str, start_page: int, end_page: int | None = None, max_chars: int | None = None,
                   char_offset: int = 0, tables: bool = False) -> dict:
        meta = self._require(doc_id)
        total = meta["page_count"]
        start = int(start_page or 1)
        end = int(end_page or start)
        if start < 1 or start > total:
            raise DocumentError(f"start_page must be between 1 and {total}")
        if end < start:
            raise DocumentError("end_page must be >= start_page")
        end = min(end, total, start + READ_MAX_PAGES - 1)
        max_chars = max(500, min(int(max_chars or READ_MAX_CHARS), READ_MAX_CHARS))
        offset = max(0, int(char_offset or 0))
        pages = {p: (label, text) for p, label, text in self.pages(doc_id) if start <= p <= end}
        out_pages, used, nxt = [], 0, None
        for p in range(start, end + 1):
            label, text = pages[p]
            body = text[offset:] if p == start else text
            room = max_chars - used
            if len(body) > room:
                if not out_pages:  # the page alone exceeds the limit: return part of it and say where to continue
                    begin = offset if p == start else 0
                    cut = body.rfind("\n", 0, room)
                    cut = cut if cut > room // 2 else room
                    out_pages.append({"page": p, "page_label": label, "char_offset": begin,
                                      "text": body[:cut], "page_chars": len(text), "partial_page": True})
                    nxt = {"start_page": p, "char_offset": begin + cut}
                else:
                    nxt = {"start_page": p, "char_offset": 0}
                break
            entry = {"page": p, "page_label": label, "text": body, "page_chars": len(text)}
            if p == start and offset:
                entry.update(char_offset=offset, partial_page=True)
            out_pages.append(entry)
            used += len(body)
        if nxt is None and end < int(end_page or start) and end < total:
            nxt = {"start_page": end + 1, "char_offset": 0}
        out = {"document_id": doc_id, "source_url": meta["final_url"] or meta["url"], "title": meta["title"],
               "unit": meta["unit"], "page_count": total, "document_status": meta["status"],
               "pages": out_pages, "complete_range": nxt is None}
        if nxt:
            out["next"] = nxt
            out["note"] = "Not all requested text was returned (size limit). Continue with read_document_range(next)."
        if tables:
            out["tables"] = self.tables(doc_id, [e["page"] for e in out_pages])
        return out

    def tables(self, doc_id: str, page_numbers: list[int]) -> list[dict] | str:
        meta = self._require(doc_id)
        if meta["source_type"] != "pdf" or not meta.get("raw_path") or not Path(meta["raw_path"]).is_file():
            return "Table extraction is only available for PDF documents stored in this cache."
        if pdfplumber is None:
            return "Table extraction requires pdfplumber."
        out = []
        with pdfplumber.open(meta["raw_path"]) as pdf:
            for p in page_numbers:
                try:
                    found = pdf.pages[p - 1].extract_tables()
                except Exception as exc:  # noqa: BLE001
                    out.append({"page": p, "error": f"table extraction failed: {exc}"})
                    continue
                for t_index, table in enumerate(found, start=1):
                    rows = []
                    rtl = sum(is_rtl_line(c or "") for r in table for c in r) > len(table) * len(table[0] or [1]) / 4
                    for r in table:
                        cells = [re.sub(r"\s+", " ", strip_controls(c or "")).strip() for c in r]
                        cells = [fix_mirrored_brackets(visual_to_logical(c)) if is_rtl_line(c) else c for c in cells]
                        if rtl:
                            cells = cells[::-1]  # right-to-left table: first column is on the right
                        rows.append(" | ".join(cells))
                    out.append({"page": p, "table": t_index, "rows": rows[:200],
                                **({"rows_truncated": len(rows) - 200} if len(rows) > 200 else {})})
        return out

    # ------------------------------------------------------- verification
    def locate_excerpt(self, doc_id: str, excerpt: str) -> list[int]:
        """Pages on which a quoted excerpt is found (also across a page break)."""
        pages = self.pages(doc_id)
        found = [p for p, _, text in pages if excerpt_found(excerpt, text)]
        if found:
            return found
        for (p1, _, t1), (_, _, t2) in zip(pages, pages[1:]):
            if excerpt_found(excerpt, t1[-2000:] + "\n" + t2[:2000]):
                return [p1, p1 + 1]
        return []

    def preview(self, doc_id: str, chars: int | None = None) -> tuple[str, bool]:
        """The beginning of the document (with page markers) and whether it is the whole document."""
        chars = chars or PREVIEW_CHARS
        parts, used, unit = [], 0, (self.meta(doc_id) or {}).get("unit", "page")
        pages = self.pages(doc_id)
        for p, label, text in pages:
            marker = f"[{unit} {p}" + (f" (printed {label})" if label and label != str(p) else "") + "]\n"
            if used + len(marker) + len(text) > chars:
                room = chars - used - len(marker)
                if room > 200:
                    parts.append(marker + text[:room] + "\n[... preview ends ...]")
                return "\n\n".join(parts), False
            parts.append(marker + text)
            used += len(marker) + len(text) + 2
        return "\n\n".join(parts), True


_default: DocumentStore | None = None
_default_lock = threading.Lock()


def get_default() -> DocumentStore:
    global _default
    with _default_lock:
        if _default is None:
            _default = DocumentStore()
        return _default


def reset_default() -> None:
    global _default
    with _default_lock:
        _default = None
