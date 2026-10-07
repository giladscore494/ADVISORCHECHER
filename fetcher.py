"""Fetch a URL and extract readable text from HTML or PDF."""

import io
import ipaddress
import re
import socket
from dataclasses import asdict, dataclass
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

STRIP_TAGS = ["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "button"]


@dataclass
class FetchResult:
    url: str
    final_url: str
    ok: bool
    source_type: str  # html | pdf | text | unknown
    title: str = ""
    text: str = ""
    error: str = ""
    http_status: int | None = None
    truncated: bool = False

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
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type="unknown",
            http_status=resp.status_code, error=f"HTTP {resp.status_code}",
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    is_pdf = "pdf" in content_type or data[:5] == b"%PDF-" or urlparse(final_url).path.lower().endswith(".pdf")
    try:
        if is_pdf:
            source_type = "pdf"
            title, text = extract_pdf_text(data)
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
            url=url, final_url=final_url, ok=False, source_type="pdf" if is_pdf else "html",
            http_status=resp.status_code, error=str(exc),
        )
    except Exception as exc:  # parser failure on odd markup
        return FetchResult(
            url=url, final_url=final_url, ok=False, source_type="html",
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
    )
