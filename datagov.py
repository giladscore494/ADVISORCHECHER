"""data.gov.il CKAN API client and the dataset read pipeline.

Workflow enforced here:
  package_search (discover) -> package_show (dataset metadata) -> resource_show (resource metadata)
  -> relevance check on metadata -> datastore_search if datastore_active, else download the resource file
  -> relevance check on schema and records -> bounded, query-matched records/passages.

resource_show returns metadata only, never the data. Irrelevant datasets are rejected before their
contents are read. Retrieved content is untrusted data: control characters are stripped and cells are
truncated; it is evidence, never instructions.
"""

import json
import time
from urllib.parse import urlencode, urlparse

import requests

import fetcher
from config import get_setting
from evidence import extract_terms, matched_terms, strip_controls

DEFAULT_BASE_URL = "https://data.gov.il/api/3/action/"
DEFAULT_USER_AGENT = "RegulatoryOpportunityHunter/0.1 (research tool; +https://github.com/giladscore494/ADVISORCHECHER)"
MAX_SEARCH_ROWS = 20
MAX_RECORDS = 50  # records/passages returned per read
MAX_DATASTORE_LIMIT = 100
MAX_SCAN_ROWS = 20000  # rows scanned in a downloaded file
MAX_DATASET_BYTES = 10 * 1024 * 1024
MAX_API_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_RESOURCES_SHOWN = 30
MAX_FILTERS = 5
MIN_INTERVAL_S = 0.5  # polite spacing between API calls
SAMPLE_ROWS = 5
CELL_CHARS = 200


class CkanError(Exception):
    """kind: blocked | http | api | malformed | limit | network | invalid | unsupported"""

    def __init__(self, message: str, kind: str, http_status: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status
        self.attempts = attempts


def _clean(value, limit: int = CELL_CHARS) -> str:
    text = strip_controls("" if value is None else str(value)).replace("\n", " ").strip()
    return text[:limit]


def _as_bool(value) -> bool:
    return value is True or str(value).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------- client
class CkanClient:
    def __init__(
        self,
        base_url: str | None = None,
        session: requests.Session | None = None,
        max_calls: int = 40,
        min_interval: float = MIN_INTERVAL_S,
        user_agent: str | None = None,
    ):
        self.base_url = (base_url or get_setting("CKAN_BASE_URL", DEFAULT_BASE_URL)).rstrip("/") + "/"
        parsed = urlparse(self.base_url)
        self.site_url = f"{parsed.scheme}://{parsed.netloc}"
        self.session = session or requests.Session()
        self.max_calls = max_calls
        self.min_interval = min_interval
        self.user_agent = user_agent or get_setting("CKAN_USER_AGENT", DEFAULT_USER_AGENT)
        self.calls = 0
        self.cache: dict[str, dict] = {}
        self.log: list[dict] = []  # every request, including failures and cache hits (research trace)
        self._last_call = 0.0

    def api_url(self, action: str, params: dict) -> str:
        clean = {k: v for k, v in params.items() if v not in (None, "")}
        return self.base_url + action + ("?" + urlencode(clean) if clean else "")

    def dataset_page(self, dataset: str) -> str:
        return f"{self.site_url}/dataset/{dataset}"

    def resource_page(self, dataset: str, resource_id: str) -> str:
        return f"{self.site_url}/dataset/{dataset}/resource/{resource_id}"

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            fetcher._sleep(wait)
        self._last_call = time.monotonic()

    def call(self, action: str, params: dict, validate) -> dict:
        """GET an action, validate the envelope and result shape. Raises CkanError; never retries 401/403."""
        url = self.api_url(action, params)
        entry = {"action": action, "url": url, "ok": False, "cached": False, "http_status": None,
                 "attempts": 0, "error": ""}
        self.log.append(entry)
        if url in self.cache:
            entry.update(ok=True, cached=True)
            return self.cache[url]
        if self.calls >= self.max_calls:
            entry["error"] = "Per-run data.gov.il API call limit reached."
            raise CkanError(entry["error"], "limit")
        self.calls += 1
        self._throttle()
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        try:
            resp, data, attempts = fetcher.download_with_retries(
                url, self.session, max_bytes=MAX_API_RESPONSE_BYTES, headers=headers)
        except (requests.RequestException, ValueError) as exc:
            entry["error"] = f"Request failed: {exc}"
            raise CkanError(entry["error"], "network") from exc
        entry.update(http_status=resp.status_code, attempts=attempts)
        try:
            result = self._validate_envelope(resp.status_code, data, attempts)
            result = validate(result)
        except CkanError as exc:
            entry["error"] = str(exc)
            raise
        entry["ok"] = True
        self.cache[url] = result
        return result

    @staticmethod
    def _validate_envelope(status: int, data: bytes, attempts: int):
        if status in (401, 403):
            raise CkanError(f"HTTP {status}: access denied by the API (not retried or bypassed).", "blocked", status, attempts)
        try:
            doc = json.loads(fetcher.decode_text(data))
        except ValueError:
            doc = None
        if isinstance(doc, dict) and doc.get("success") is False:
            err = doc.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            raise CkanError(f"CKAN API error (HTTP {status}): {msg or 'request failed'}", "api", status, attempts)
        if status != 200:
            suffix = f" after {attempts} attempts" if status in fetcher.TRANSIENT_STATUSES else ""
            raise CkanError(f"HTTP {status}{suffix}.", "http", status, attempts)
        if not isinstance(doc, dict) or doc.get("success") is not True or "result" not in doc:
            raise CkanError("Malformed CKAN response: expected a JSON envelope with success=true and a result.",
                            "malformed", status, attempts)
        return doc["result"]

    # ------------------------------------------------------------- actions
    def package_search(self, query: str, rows: int = 10, start: int = 0) -> dict:
        rows = max(1, min(int(rows or 10), MAX_SEARCH_ROWS))
        start = max(0, int(start or 0))

        def validate(r):
            if not isinstance(r, dict) or not isinstance(r.get("results"), list):
                raise CkanError("Malformed package_search result: missing 'results' list.", "malformed")
            return r

        r = self.call("package_search", {"q": query, "rows": rows, "start": start}, validate)
        results = [dataset_summary(d) for d in r["results"] if isinstance(d, dict)]
        count = r.get("count") if isinstance(r.get("count"), int) else len(results)
        return {"count": count, "start": start, "rows": rows, "results": results,
                "next_start": start + rows if count > start + rows else None}

    def package_show(self, dataset_id: str) -> dict:
        def validate(r):
            if not isinstance(r, dict) or not r.get("id") or not isinstance(r.get("resources", []), list):
                raise CkanError("Malformed package_show result: expected a dataset with id and resources.", "malformed")
            return r

        return dataset_detail(self.call("package_show", {"id": dataset_id}, validate), self)

    def resource_show(self, resource_id: str) -> dict:
        def validate(r):
            if not isinstance(r, dict) or not r.get("id") or "url" not in r:
                raise CkanError("Malformed resource_show result: expected a resource with id and url.", "malformed")
            return r

        return resource_detail(self.call("resource_show", {"id": resource_id}, validate))

    def datastore_search(self, resource_id: str, q: str = "", filters: dict | None = None,
                         limit: int = 20, offset: int = 0) -> dict:
        limit = max(1, min(int(limit or 20), MAX_DATASTORE_LIMIT))
        offset = max(0, int(offset or 0))
        params = {"resource_id": resource_id, "limit": limit, "offset": offset, "q": q or None,
                  "filters": json.dumps(filters, ensure_ascii=False, sort_keys=True) if filters else None}

        def validate(r):
            if not isinstance(r, dict) or not isinstance(r.get("records"), list):
                raise CkanError("Malformed datastore_search result: missing 'records' list.", "malformed")
            return r

        r = self.call("datastore_search", params, validate)
        fields = [f.get("id", "") for f in r.get("fields", []) if isinstance(f, dict)]
        records = [rec for rec in r["records"] if isinstance(rec, dict)]
        if not fields and records:
            fields = list(records[0])
        total = r.get("total") if isinstance(r.get("total"), int) else None
        rows = [[_clean(rec.get(f)) for f in fields] for rec in records]
        more = total is not None and total > offset + len(rows)
        return {"fields": [_clean(f) for f in fields], "rows": rows, "total": total, "offset": offset,
                "limit": limit, "next_offset": offset + len(rows) if more else None,
                "api_url": self.api_url("datastore_search", params)}


# ------------------------------------------------------------------ summaries
def dataset_summary(d: dict) -> dict:
    resources = [r for r in d.get("resources", []) if isinstance(r, dict)]
    return {
        "id": _clean(d.get("id")),
        "name": _clean(d.get("name")),
        "title": _clean(d.get("title")),
        "publisher": _clean((d.get("organization") or {}).get("title") if isinstance(d.get("organization"), dict) else ""),
        "description": _clean(d.get("notes"), 300),
        "last_updated": _clean(d.get("metadata_modified")),
        "license": _clean(d.get("license_title") or d.get("license_id")),
        "num_resources": len(resources),
        "formats": sorted({_clean(r.get("format")).upper() for r in resources if r.get("format")}),
        "tags": [_clean(t.get("name") if isinstance(t, dict) else t) for t in (d.get("tags") or [])[:10]],
    }


def dataset_detail(d: dict, client: "CkanClient") -> dict:
    out = dataset_summary(d)
    out.update(
        description=_clean(d.get("notes"), 1500),
        created=_clean(d.get("metadata_created")),
        license_url=_clean(d.get("license_url")),
        dataset_url=client.dataset_page(out["name"] or out["id"]),
        resources=[
            {
                "id": _clean(r.get("id")),
                "name": _clean(r.get("name")),
                "description": _clean(r.get("description"), 200),
                "format": _clean(r.get("format")),
                "datastore_active": _as_bool(r.get("datastore_active")),
                "last_modified": _clean(r.get("last_modified") or r.get("metadata_modified")),
            }
            for r in d.get("resources", [])[:MAX_RESOURCES_SHOWN] if isinstance(r, dict)
        ],
    )
    return out


def resource_detail(r: dict) -> dict:
    return {
        "id": _clean(r.get("id")),
        "package_id": _clean(r.get("package_id")),
        "name": _clean(r.get("name")),
        "description": _clean(r.get("description"), 1000),
        "format": _clean(r.get("format")),
        "mimetype": _clean(r.get("mimetype")),
        "download_url": _clean(r.get("url"), 2000),
        "datastore_active": _as_bool(r.get("datastore_active")),
        "created": _clean(r.get("created")),
        "last_modified": _clean(r.get("last_modified") or r.get("metadata_modified")),
        "size": r.get("size") if isinstance(r.get("size"), int) else None,
        "license": _clean(r.get("license_title") or r.get("license_id")),
    }


# ------------------------------------------------------------------ relevance
def is_relevant_match(matched: set[str], terms: set[str]) -> bool:
    """A distinctive match is required: a term of 4+ chars, or 2+ terms. Short-only queries need 1 hit."""
    if not matched:
        return False
    return any(len(t) >= 4 for t in matched) or len(matched) >= 2 or all(len(t) < 4 for t in terms)


def assess_relevance(terms: set[str], fields: dict[str, str]) -> dict:
    hits = {name: sorted(matched_terms(terms, text)) for name, text in fields.items() if text}
    matched = set().union(*map(set, hits.values())) if hits else set()
    return {
        "relevant": bool(terms) and is_relevant_match(matched, terms),
        "matched_terms": sorted(matched),
        "matched_in": {k: v for k, v in hits.items() if v},
    }


def metadata_fields(dataset: dict, resource: dict | None = None) -> dict[str, str]:
    fields = {
        "dataset_title": dataset.get("title", ""),
        "dataset_description": dataset.get("description", ""),
        "publisher": dataset.get("publisher", ""),
        "tags": " ".join(dataset.get("tags", [])),
    }
    if resource:
        fields["resource_name"] = resource.get("name", "")
        fields["resource_description"] = resource.get("description", "")
    return fields


# --------------------------------------------------------------- file reading
def _row_score(terms: set[str], row: list, filters: dict | None, header: list) -> int:
    """0 if the row fails a column filter or matches no term; otherwise the number of matched terms."""
    for key, value in (filters or {}).items():
        if key in header:
            idx = header.index(key)
            if (row[idx] if idx < len(row) else "") != _clean(value):
                return 0
    if not terms:
        return 1
    return len(matched_terms(terms, " ".join(row)))


def _scan_rows(header, rows_iter, terms, filters, start_index=0):
    """Scan at most MAX_SCAN_ROWS rows. Returns (scored_matches, sample, scanned)."""
    scored, sample, scanned = [], [], 0
    for row in rows_iter:
        if scanned >= MAX_SCAN_ROWS:
            break
        scanned += 1
        row = [_clean(c) for c in row]
        if len(sample) < SAMPLE_ROWS:
            sample.append(row)
        score = _row_score(terms, row, filters, header)
        if score:
            scored.append((score, start_index + scanned, row))
    return scored, sample, scanned


def read_file(data: bytes, kind: str, terms: set[str], limit: int, offset: int, filters: dict | None) -> dict:
    """Bounded extraction from a downloaded resource: matching rows for tables, passages for documents."""
    if kind in ("csv", "xlsx", "json"):
        tables = []
        if kind == "csv":
            header, rows_iter = fetcher.iter_csv_rows(data)
            tables.append(("", header, rows_iter))
        elif kind == "xlsx":
            wb = fetcher.open_xlsx(data)
            try:
                for title, header, rows_iter, _ in fetcher.iter_xlsx_sheets(wb):
                    tables.append((title, header, list(_bounded(rows_iter))))
            finally:
                wb.close()
        else:
            try:
                doc = json.loads(fetcher.decode_text(data))
            except ValueError as exc:
                raise CkanError(f"Resource is not valid JSON: {exc}", "malformed") from exc
            records = fetcher.json_records(doc)
            if records is None:
                text = json.dumps(doc, ensure_ascii=False)[:200000]
                return _passages_result(text, terms, limit)
            header = []
            for rec in records[:200]:
                header += [k for k in rec if k not in header]
            tables.append(("", header, ([rec.get(h) for h in header] for rec in records)))
        header = [_clean(h) for h in tables[0][1]] if tables else []
        scored, sample, scanned = [], [], 0
        for _title, tbl_header, rows_iter in tables:
            tbl_header = [_clean(h) for h in tbl_header]
            m, smp, n = _scan_rows(tbl_header, rows_iter, terms, filters, start_index=scanned)
            scored += m
            sample = sample or smp
            scanned += n
            header = header or tbl_header
        # Best matches first (more query terms matched), then file order.
        scored.sort(key=lambda item: (-item[0], item[1]))
        rows = [row for _, _, row in scored[offset: offset + limit]]
        total = len(scored)
        more = total > offset + len(rows)
        return {"kind": "table", "fields": header, "rows": rows, "sample": sample, "scanned_rows": scanned,
                "total_matches": total, "next_offset": offset + len(rows) if more else None}
    if kind == "pdf":
        _, text = fetcher.extract_pdf_text(data)
    else:
        _, text = fetcher.extract_html_text(data)
    return _passages_result(text, terms, limit)


def _bounded(rows_iter):
    for i, row in enumerate(rows_iter):
        if i >= MAX_SCAN_ROWS:
            break
        yield row


def _passages_result(text: str, terms: set[str], limit: int) -> dict:
    paragraphs = [p.strip() for p in strip_controls(text).split("\n") if p.strip()]
    matching = [p[:600] for p in paragraphs if matched_terms(terms, p)]
    return {"kind": "document", "passages": matching[:limit], "sample": [p[:600] for p in paragraphs[:SAMPLE_ROWS]],
            "total_matches": len(matching), "next_offset": None}


def render_rows(fields: list[str], rows: list[list]) -> str:
    lines = ["Columns: " + " | ".join(fields)] if fields else []
    return "\n".join(lines + [" | ".join(r) for r in rows])


# ------------------------------------------------------------------- pipeline
def read_resource(
    client: CkanClient,
    resource_id: str,
    query: str,
    topic: str,
    limit: int = 20,
    offset: int = 0,
    filters: dict | None = None,
) -> dict:
    """Read a resource safely. Returns a dict with status:
    ok | no_matching_records | rejected | failed, plus provenance and bounded evidence.
    Raises CkanError only for metadata failures (resource_show / package_show)."""
    limit = max(1, min(int(limit or 20), MAX_RECORDS))
    offset = max(0, int(offset or 0))
    if filters is not None:
        if not isinstance(filters, dict) or len(filters) > MAX_FILTERS or not all(
                isinstance(v, (str, int, float, bool)) for v in filters.values()):
            raise CkanError(f"filters must be an object with at most {MAX_FILTERS} scalar values.", "invalid")

    resource = client.resource_show(resource_id)
    dataset = client.package_show(resource["package_id"]) if resource["package_id"] else {}
    dataset_ref = dataset.get("name") or resource["package_id"]
    provenance = {
        "dataset_id": dataset.get("id") or resource["package_id"],
        "dataset_name": dataset.get("name", ""),
        "dataset_title": dataset.get("title", ""),
        "publisher": dataset.get("publisher", ""),
        "license": dataset.get("license") or resource.get("license", ""),
        "dataset_last_updated": dataset.get("last_updated", ""),
        "resource_id": resource["id"],
        "resource_name": resource["name"],
        "resource_format": resource["format"],
        "resource_last_modified": resource["last_modified"],
        "datastore_active": resource["datastore_active"],
        "source_url": client.resource_page(dataset_ref, resource["id"]) if dataset_ref else "",
        "download_url": resource["download_url"],
        "metadata_api_url": client.api_url("resource_show", {"id": resource["id"]}),
        "data_api_url": "",
        "date_note": "Dataset/resource dates are publication or update dates, not legal effective dates.",
    }
    terms = extract_terms(query, topic)
    out = {"resource_id": resource["id"], "provenance": provenance}

    relevance = assess_relevance(terms, metadata_fields(dataset, resource))
    out["metadata_relevance"] = relevance
    if not relevance["relevant"]:
        out.update(status="rejected", reason=(
            "Dataset rejected: its title, description, publisher, tags and resource name do not match the "
            "research topic or query. Its contents were not read and must not be used as evidence."))
        return out

    if resource["datastore_active"]:
        try:
            ds = client.datastore_search(resource["id"], q=query, filters=filters, limit=limit, offset=offset)
            provenance["data_api_url"] = ds["api_url"]
            fields, rows, total, next_offset = ds["fields"], ds["rows"], ds["total"], ds["next_offset"]
            sample = []
            if not rows:
                sample_ds = client.datastore_search(resource["id"], limit=SAMPLE_ROWS)
                fields, sample = fields or sample_ds["fields"], sample_ds["rows"]
            content = {"kind": "table", "fields": fields, "rows": rows, "sample": sample,
                       "total_matches": total if rows else 0, "next_offset": next_offset}
            out["retrieval"] = "datastore_search"
        except CkanError as exc:
            out.update(status="failed", error=f"datastore_search failed: {exc}", http_status=exc.http_status)
            return out
    else:
        url = resource["download_url"]
        if not url.startswith(("http://", "https://")):
            out.update(status="failed", error="Resource has no downloadable http(s) URL.")
            return out
        if not fetcher.robots_allowed(url, client.session):
            out.update(status="failed", error="Download disallowed by the site's robots.txt; not fetched.",
                       access_restricted=True)
            return out
        try:
            resp, data, attempts = fetcher.download_with_retries(
                url, client.session, max_bytes=MAX_DATASET_BYTES, headers={"User-Agent": client.user_agent})
        except (requests.RequestException, ValueError) as exc:
            out.update(status="failed", error=f"Download failed: {exc}")
            return out
        if resp.status_code != 200:
            restricted = resp.status_code in (401, 403)
            out.update(status="failed", http_status=resp.status_code, access_restricted=restricted,
                       error=f"Download failed: HTTP {resp.status_code}" + (" (access denied; not bypassed)" if restricted else ""))
            return out
        kind = fetcher.detect_kind(resp.headers.get("Content-Type", ""), resp.url or url, data)
        try:
            content = read_file(data, kind, terms, limit, offset, filters)
        except (CkanError, ValueError) as exc:
            out.update(status="failed", error=f"Could not parse the {kind} resource: {exc}")
            return out
        out["retrieval"] = f"download ({kind})"
        provenance["data_api_url"] = url

    if content["kind"] == "table":
        schema_hits = sorted(matched_terms(terms, " ".join(content["fields"])))
        out.update(fields=content["fields"], total_matches=content["total_matches"],
                   next_offset=content["next_offset"], schema_matched_terms=schema_hits)
        if content.get("scanned_rows") is not None:
            out["scanned_rows"] = content["scanned_rows"]
        if content["rows"]:
            out.update(status="ok", records=render_rows(content["fields"], content["rows"]))
        else:
            out.update(status="no_matching_records", sample=render_rows(content["fields"], content["sample"]),
                       note="No records matched the query. The sample shows the schema only and is NOT evidence "
                            "for the research claim.")
    else:
        out.update(total_matches=content["total_matches"])
        if content["passages"]:
            out.update(status="ok", passages=content["passages"])
        else:
            out.update(status="no_matching_records", sample=content["sample"],
                       note="No passages matched the query; the sample is NOT evidence for the research claim.")
    return out
