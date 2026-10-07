"""Fetch a URL and extract readable text from HTML, PDF, JSON (incl. CKAN APIs), CSV or XLSX."""

import csv
import io
import ipaddress
import json
import re
import socket
import zipfile
from dataclasses import asdict, dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 RegulatoryOpportunityHunter/0.1"
)
TIMEOUT = (10, 30)  # connect, read
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MAX_TEXT_CHARS = 15000
MAX_PDF_PAGES = 60
MAX_REDIRECTS = 5
MAX_TABLE_ROWS = 200  # rows rendered per CSV file / XLSX sheet
MAX_XLSX_SHEETS = 5
MAX_XLSX_UNCOMPRESSED = 100 * 1024 * 1024  # zip-bomb guard
MAX_CELL_CHARS = 200

XLSX_TYPES = ("spreadsheetml", "ms-excel", "officedocument.spreadsheet")
CKAN_RESOURCE_FIELDS = (
    "name", "description", "format", "mimetype", "url", "package_id", "id",
    "created", "last_modified", "metadata_modified", "size", "datastore_active",
)

STRIP_TAGS = ["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "button"]


@dataclass
class FetchResult:
    url: str
    final_url: str
    ok: bool
    source_type: str  # html | pdf | json | ckan_resource | ckan_dataset | ckan_datastore | csv | xlsx | text | unknown
    title: str = ""
    text: str = ""
    error: str = ""
    http_status: int | None = None
    truncated: bool = False
    # Downloadable file behind a CKAN resource (data.gov.il resource_show); fetch it to read the data.
    resource_url: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_public_host(host: str) -> bool:
    """Reject localhost / private network targets (the model chooses URLs)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _check_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "Only http(s) URLs are supported."
    if not _is_public_host(parsed.hostname):
        return "Host is not publicly resolvable or is a private address."
    return None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n[... truncated ...]", True


def extract_html_text(html: str | bytes) -> tuple[str, str]:
    """Return (title, readable_text) from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def extract_pdf_text(data: bytes) -> tuple[str, str]:
    """Return (title, text) from PDF bytes. Raises ValueError if no text can be extracted."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"[page {i}]\n{page_text}")
        title = ""
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title)
    except Exception as exc:
        raise ValueError(f"PDF could not be parsed: {exc}") from exc
    if not pages:
        raise ValueError("PDF contains no extractable text (possibly scanned; OCR is not supported).")
    return title, "\n\n".join(pages)


def _cell(value) -> str:
    text = "" if value is None else str(value).strip().replace("\n", " ")
    return text[:MAX_CELL_CHARS]


def _render_rows(header: list, rows: list[list], total: int | None) -> str:
    lines = ["Columns: " + " | ".join(_cell(h) for h in header)] if header else []
    lines += [" | ".join(_cell(c) for c in row) for row in rows]
    shown = len(rows)
    if total is not None and total > shown:
        lines.append(f"[showing first {shown} of {total} data rows]")
    return "\n".join(lines)


def decode_text(data: bytes) -> str:
    """Decode bytes as UTF-8, then Windows-1255 (common in Israeli government files), then Latin-1."""
    for encoding in ("utf-8-sig", "cp1255"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def extract_csv_text(data: bytes) -> str:
    text = decode_text(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    header = next(reader, [])
    rows, total = [], 0
    for row in reader:
        if not any(c.strip() for c in row):
            continue
        total += 1
        if len(rows) < MAX_TABLE_ROWS:
            rows.append(row)
    if not header and not rows:
        raise ValueError("CSV file is empty.")
    return _render_rows(header, rows, total)


def extract_xlsx_text(data: bytes) -> str:
    from openpyxl import load_workbook

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if sum(i.file_size for i in zf.infolist()) > MAX_XLSX_UNCOMPRESSED:
                raise ValueError("XLSX file is too large when uncompressed.")
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except zipfile.BadZipFile as exc:
        raise ValueError("Not a valid XLSX file (legacy .xls is not supported).") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"XLSX could not be parsed: {exc}") from exc
    parts = []
    try:
        for ws in wb.worksheets[:MAX_XLSX_SHEETS]:
            rows_iter = ws.iter_rows(values_only=True)
            header = list(next(rows_iter, None) or [])
            rows = []
            for row in rows_iter:
                if len(rows) >= MAX_TABLE_ROWS:
                    break
                if any(c not in (None, "") for c in row):
                    rows.append(list(row))
            total = (ws.max_row - 1) if ws.max_row else None
            parts.append(f"[sheet: {ws.title}]\n" + _render_rows(header, rows, total))
        if len(wb.worksheets) > MAX_XLSX_SHEETS:
            parts.append(f"[{len(wb.worksheets) - MAX_XLSX_SHEETS} more sheets not shown]")
    finally:
        wb.close()
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("XLSX contains no readable cells.")
    return text


def _ckan_text(result) -> tuple[str, str, str, str, dict]:
    """Render a CKAN action result. Returns (source_type, title, text, resource_url, metadata)."""
    if isinstance(result, dict) and "url" in result and ("package_id" in result or "format" in result):
        meta = {k: result.get(k) for k in CKAN_RESOURCE_FIELDS if result.get(k) not in (None, "")}
        lines = ["CKAN resource metadata:"] + [f"- {k}: {v}" for k, v in meta.items()]
        lines.append("To read the data itself, call fetch_url on the resource url above.")
        return "ckan_resource", str(result.get("name") or ""), "\n".join(lines), str(result.get("url") or ""), meta
    if isinstance(result, dict) and isinstance(result.get("resources"), list):
        org = (result.get("organization") or {}).get("title", "")
        lines = [f"CKAN dataset: {result.get('title', '')}", f"Publisher: {org}",
                 f"Last modified: {result.get('metadata_modified', '')}", f"Description: {result.get('notes', '')}",
                 "Resources:"]
        lines += [f"- {r.get('name', '')} [{r.get('format', '')}] id={r.get('id', '')} url={r.get('url', '')}"
                  for r in result["resources"][:50] if isinstance(r, dict)]
        meta = {"name": result.get("name"), "title": result.get("title"), "organization": org,
                "metadata_modified": result.get("metadata_modified")}
        return "ckan_dataset", str(result.get("title") or ""), "\n".join(lines), "", meta
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        fields = [f.get("id", "") for f in result.get("fields", []) if isinstance(f, dict)]
        records = result["records"][:MAX_TABLE_ROWS]
        header = fields or (list(records[0]) if records and isinstance(records[0], dict) else [])
        rows = [[r.get(h) for h in header] for r in records if isinstance(r, dict)]
        text = f"CKAN datastore records (resource {result.get('resource_id', '')}):\n" + _render_rows(header, rows, result.get("total"))
        return "ckan_datastore", "", text, "", {"resource_id": result.get("resource_id"), "total": result.get("total")}
    return "json", "", json.dumps(result, ensure_ascii=False, indent=1), "", {}


def extract_json(data: bytes) -> tuple[str, str, str, str, dict]:
    """Parse JSON; CKAN action API responses get a readable rendering.
    Returns (source_type, title, text, resource_url, metadata). Raises ValueError."""
    try:
        doc = json.loads(decode_text(data))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if isinstance(doc, dict) and "success" in doc and ("result" in doc or "error" in doc):
        if not doc.get("success"):
            err = doc.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise ValueError(f"CKAN API error: {msg or 'request failed'}")
        return _ckan_text(doc.get("result"))
    return "json", "", json.dumps(doc, ensure_ascii=False, indent=1), "", {}


def _download(url: str, session: requests.Session) -> tuple[requests.Response, bytes]:
    """GET with manual redirect handling (each hop is checked) and a size cap."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        problem = _check_url(current)
        if problem:
            raise ValueError(problem)
        resp = session.get(
            current,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "he,en;q=0.8"},
            timeout=TIMEOUT,
            stream=True,
            allow_redirects=False,
        )
        if resp.is_redirect and resp.headers.get("Location"):
            current = urljoin(current, resp.headers["Location"])
            resp.close()
            continue
        chunks, total = [], 0
        for chunk in resp.iter_content(64 * 1024):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                resp.close()
                raise ValueError(f"Download exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit.")
            chunks.append(chunk)
        return resp, b"".join(chunks)
    raise ValueError("Too many redirects.")


def fetch_url(url: str, max_chars: int = MAX_TEXT_CHARS, session: requests.Session | None = None) -> FetchResult:
    """Fetch a URL and return its readable text. Never raises: failures are reported in the result."""
    session = session or requests.Session()
    try:
        resp, data = _download(url, session)
    except (requests.RequestException, ValueError) as exc:
        return FetchResult(url=url, final_url=url, ok=False, source_type="unknown", error=str(exc))

    final_url = resp.url or url
    if resp.status_code >= 400:
        error = f"HTTP {resp.status_code}"
        if resp.status_code in (401, 403):
            error += " (access denied; not retried or bypassed)"
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type="unknown",
            http_status=resp.status_code, error=error,
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    path = urlparse(final_url).path.lower()
    head = data.lstrip()[:1]
    is_pdf = "pdf" in content_type or data[:5] == b"%PDF-" or path.endswith(".pdf")
    is_json = "json" in content_type or (head in (b"{", b"[") and "html" not in content_type)
    is_csv = "csv" in content_type or path.endswith(".csv")
    # CSV downloads are often labelled application/vnd.ms-excel, so a .csv path wins.
    is_xlsx = path.endswith((".xlsx", ".xlsm")) or (any(t in content_type for t in XLSX_TYPES) and not path.endswith(".csv"))
    resource_url, metadata = "", {}
    source_type = "pdf" if is_pdf else "json" if is_json else "xlsx" if is_xlsx else "csv" if is_csv else "html"
    try:
        if is_pdf:
            title, text = extract_pdf_text(data)
        elif is_json:
            source_type, title, text, resource_url, metadata = extract_json(data)
            if resource_url:
                resource_url = urljoin(final_url, resource_url)
        elif is_xlsx:
            title, text = "", extract_xlsx_text(data)
        elif is_csv:
            title, text = "", extract_csv_text(data)
        elif "html" in content_type or "xml" in content_type or data.lstrip()[:1] == b"<":
            source_type = "html"
            title, text = extract_html_text(data)
        elif content_type.startswith("text/"):
            source_type = "text"
            title, text = "", data.decode(resp.encoding or "utf-8", errors="replace")
        else:
            return FetchResult(
                url=url, final_url=final_url, ok=False, source_type="unknown",
                http_status=resp.status_code, error=f"Unsupported content type: {content_type or 'unknown'}",
            )
    except ValueError as exc:
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type=source_type,
            http_status=resp.status_code, error=str(exc),
        )
    except Exception as exc:  # parser failure on odd markup or files
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type=source_type,
            http_status=resp.status_code, error=f"Could not parse content: {exc}",
        )

    if not text.strip():
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type=source_type,
            http_status=resp.status_code, error="No readable text extracted (page may require JavaScript).",
        )
    text, truncated = _truncate(text, max_chars)
    return FetchResult(
        url=url, final_url=final_url, ok=True, source_type=source_type, title=title,
        text=text, http_status=resp.status_code, truncated=truncated,
        resource_url=resource_url, metadata=metadata,
    )
