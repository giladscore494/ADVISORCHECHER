"""Download COMPLETE official data.gov.il datasets into versioned, validated snapshots.

Usage:
    python scripts/sync_government_data.py                 # sync every configured resource
    python scripts/sync_government_data.py --only customs_tariff --force
    python scripts/sync_government_data.py --verify        # check manifest + files, no network

For every resource:
  1. resource_show + package_show: official metadata (name, publisher, license, dates).
  2. Discover the retrieval method: the CKAN DataStore (datastore_active) or the resource's official
     download URL.
  3. Download ALL records. DataStore: datastore_search with limit=0 gives the full table total
     (no q / filters, so it is NOT a search-result count); then page through every row ordered by _id
     until the total is reached, and re-read the total afterwards to detect concurrent changes.
  4. Validate: row count == total, unique _id values, stable schema on every page and record,
     non-empty, no unexplained shrink vs. the previous snapshot.
  5. Write UTF-8 JSONL (gzip, deterministic) + snapshot.json (provenance, checksums) to a temporary
     directory, re-read and re-validate it, then atomically rename it into place and atomically
     rewrite manifest.json. A failed or incomplete download never replaces the last complete snapshot.

Requests are read-only GETs, spaced by a minimum interval, retried with exponential backoff only for
transient failures (429 / 5xx / connection errors). HTTP 401/403 is never retried or bypassed.
Only the `requests` package is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "government"
RELEASE_STAGING_DIR = REPO_ROOT / "release-staging"
FORMAT_VERSION = 1

API_BASE = "https://data.gov.il/api/3/action/"
SITE_URL = "https://data.gov.il"
DEFAULT_USER_AGENT = (
    "datagov-external-client RegulatoryOpportunityHunter-sync/1.0 "
    "(read-only; +https://github.com/giladscore494/ADVISORCHECHER)"
)

# Verified data.gov.il resource ids (see README "Government dataset snapshots").
RESOURCES: list[dict[str, str]] = [
    {"key": "customs_tariff", "resource_id": "5536eaa1-2e51-406b-aff6-b9ca02801b7c",
     "label": "Customs tariff & purchase tax book (ספר סיווג טובין ביבוא - תעריף המכס ומס קניה)"},
    {"key": "free_import_order", "resource_id": "a36db570-09f2-4521-8e3d-0290eb839c68",
     "label": "Legal requirements - Free Import Order (דרישות חוקיות - צו יבוא חופשי)"},
    {"key": "mandatory_standards", "resource_id": "1a4d94e2-369a-488d-a223-eb1020612fbd",
     "label": "Official standards registry (מאגר תקנים רשמיים)"},
    {"key": "import_regulations", "resource_id": "d9750b40-c0b9-4e05-a08e-ae768a92e9ca",
     "label": "Legal requirements - additional orders (דרישות חוקיות - צוים נוספים)"},
    {"key": "standards_declarations", "resource_id": "d8611d0e-f5c8-4552-8615-da37e920f07b",
     "label": "Official standards declarations in Reshumot (אכרזת תקנים ברשומות)"},
]

PAGE_SIZE = 5000
MIN_INTERVAL_S = 1.0  # polite spacing between requests
TIMEOUT = (15, 180)  # connect, read
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0
MAX_BACKOFF_S = 120.0
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
MAX_FILE_BYTES = 2 * 1024 ** 3  # official file downloads (non-DataStore resources)
# GitHub rejects files > 100 MiB and warns above 50 MiB; larger snapshots go to a Release asset.
MAX_REPO_FILE_BYTES = 45 * 1024 ** 2
MAX_SHRINK_RATIO = 0.5  # reject a snapshot with < 50% of the previous row count (unless --allow-shrink)
FULL_REFRESH_DAYS = 7  # re-download even when metadata looks unchanged after this many days
KEEP_VERSIONS = 2
INTERNAL_FIELDS = {"_full_text"}  # CKAN search index column, not data


class SyncError(Exception):
    """kind: blocked | http | api | network | malformed | incomplete | invalid"""

    def __init__(self, message: str, kind: str = "error", http_status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def log(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------------- HTTP
class Http:
    """Read-only GET client: spacing, timeouts, retries with exponential backoff, request log."""

    def __init__(self, session: requests.Session | None = None, user_agent: str | None = None,
                 min_interval: float = MIN_INTERVAL_S, max_retries: int = MAX_RETRIES,
                 backoff_base: float = BACKOFF_BASE_S, sleep: Callable[[float], None] = time.sleep,
                 api_base: str = API_BASE):
        self.session = session or requests.Session()
        self.user_agent = user_agent or os.environ.get("DATAGOV_USER_AGENT") or DEFAULT_USER_AGENT
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.sleep = sleep
        self.api_base = api_base
        self.requests = 0
        self.retries = 0
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def _delay(self, attempt: int, resp: requests.Response | None) -> float:
        delay = self.backoff_base * (2 ** attempt)
        retry_after = resp.headers.get("Retry-After") if resp is not None else None
        if retry_after and str(retry_after).strip().isdigit():
            delay = max(delay, float(retry_after))
        return min(delay, MAX_BACKOFF_S)

    def get(self, url: str, accept: str = "application/json", stream: bool = False) -> requests.Response:
        headers = {"User-Agent": self.user_agent, "Accept": accept}
        attempt = 0
        while True:
            self._throttle()
            self.requests += 1
            try:
                resp = self.session.get(url, headers=headers, timeout=TIMEOUT, stream=stream)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise SyncError(f"Network error after {attempt + 1} attempts: {exc}", "network") from exc
                delay = self._delay(attempt, None)
                log(f"    transient network error ({exc.__class__.__name__}); retry in {delay:.0f}s")
                self.retries += 1
                self.sleep(delay)
                attempt += 1
                continue
            if resp.status_code in (401, 403):
                raise SyncError(f"HTTP {resp.status_code} access denied for {url} (not retried or bypassed).",
                                "blocked", resp.status_code)
            if resp.status_code in TRANSIENT_STATUSES and attempt < self.max_retries:
                delay = self._delay(attempt, resp)
                log(f"    HTTP {resp.status_code}; retry in {delay:.0f}s")
                resp.close()
                self.retries += 1
                self.sleep(delay)
                attempt += 1
                continue
            return resp

    def api_url(self, action: str, params: dict) -> str:
        clean = {k: v for k, v in params.items() if v is not None}
        return self.api_base + action + ("?" + urlencode(clean) if clean else "")

    def action(self, action: str, **params) -> Any:
        url = self.api_url(action, params)
        resp = self.get(url)
        try:
            doc = resp.json()
        except ValueError:
            doc = None
        if isinstance(doc, dict) and doc.get("success") is False:
            err = doc.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            raise SyncError(f"CKAN {action} error (HTTP {resp.status_code}): {msg}", "api", resp.status_code)
        if resp.status_code != 200:
            raise SyncError(f"CKAN {action}: HTTP {resp.status_code}", "http", resp.status_code)
        if not isinstance(doc, dict) or doc.get("success") is not True or "result" not in doc:
            raise SyncError(f"CKAN {action}: malformed response envelope", "malformed", resp.status_code)
        return doc["result"]


# --------------------------------------------------------------------- metadata
def fetch_metadata(http: Http, resource_id: str) -> dict:
    resource = http.action("resource_show", id=resource_id)
    if not isinstance(resource, dict) or resource.get("id") != resource_id:
        raise SyncError("resource_show returned a different or malformed resource", "malformed")
    dataset: dict = {}
    if resource.get("package_id"):
        dataset = http.action("package_show", id=resource["package_id"])
        if not isinstance(dataset, dict):
            raise SyncError("package_show returned a malformed dataset", "malformed")
    org = dataset.get("organization") if isinstance(dataset.get("organization"), dict) else {}
    datastore_active = resource.get("datastore_active") in (True, "true", "True", 1)
    dataset_ref = dataset.get("name") or resource.get("package_id") or ""
    return {
        "resource_id": resource_id,
        "resource_name": resource.get("name") or "",
        "resource_description": (resource.get("description") or "")[:2000],
        "resource_format": resource.get("format") or "",
        "resource_created": resource.get("created") or "",
        "resource_last_modified": resource.get("last_modified") or resource.get("metadata_modified") or "",
        "resource_metadata_modified": resource.get("metadata_modified") or "",
        "resource_size": resource.get("size"),
        "resource_hash": resource.get("hash") or "",
        "download_url": resource.get("url") or "",
        "datastore_active": datastore_active,
        "dataset_id": dataset.get("id") or resource.get("package_id") or "",
        "dataset_name": dataset.get("name") or "",
        "dataset_title": dataset.get("title") or "",
        "publisher": org.get("title") or org.get("name") or "",
        "license_id": dataset.get("license_id") or "",
        "license_title": dataset.get("license_title") or "",
        "license_url": dataset.get("license_url") or "",
        "dataset_metadata_modified": dataset.get("metadata_modified") or "",
        "source_url": f"{SITE_URL}/dataset/{dataset_ref}/resource/{resource_id}" if dataset_ref else "",
        "metadata_api_url": http.api_url("resource_show", {"id": resource_id}),
    }


def datastore_info(http: Http, resource_id: str) -> dict:
    """Full-table total and schema: limit=0, no q, no filters (not a search-result count)."""
    r = http.action("datastore_search", resource_id=resource_id, limit=0)
    if not isinstance(r, dict) or not isinstance(r.get("total"), int):
        raise SyncError("datastore_search did not return an integer total", "malformed")
    if r.get("total_was_estimated"):
        raise SyncError("DataStore total was estimated, not exact; cannot verify completeness", "incomplete")
    fields = [f for f in r.get("fields", []) if isinstance(f, dict) and f.get("id") not in INTERNAL_FIELDS]
    if not fields:
        raise SyncError("datastore_search returned no fields", "malformed")
    return {"total": r["total"], "fields": [{"id": f["id"], "type": f.get("type", "")} for f in fields]}


def metadata_fingerprint(meta: dict, ds: dict | None) -> str:
    basis = {k: meta.get(k) for k in ("resource_last_modified", "resource_metadata_modified", "resource_size",
                                      "resource_hash", "download_url", "datastore_active",
                                      "dataset_metadata_modified")}
    if ds:
        basis["total"] = ds["total"]
        basis["fields"] = ds["fields"]
    return hashlib.sha256(json.dumps(basis, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------- download
@dataclass
class Download:
    records: list[dict]
    fields: list[dict]
    method: str
    api_url: str
    expected_total: int | None
    pages: int = 0
    id_field: str = "_id"
    notes: list[str] = field(default_factory=list)


def download_datastore(http: Http, resource_id: str, page_size: int = PAGE_SIZE, info: dict | None = None) -> Download:
    info = info or datastore_info(http, resource_id)
    total, fields = info["total"], info["fields"]
    field_ids = [f["id"] for f in fields]
    if "_id" not in field_ids:
        raise SyncError("DataStore schema has no _id column; cannot verify uniqueness", "malformed")
    records: list[dict] = []
    offset, pages, last_id = 0, 0, None
    while offset < total:
        page = http.action("datastore_search", resource_id=resource_id, limit=page_size, offset=offset,
                           sort="_id asc")
        pages += 1
        recs = page.get("records") if isinstance(page, dict) else None
        if not isinstance(recs, list):
            raise SyncError(f"Page at offset {offset}: missing records list", "malformed")
        page_fields = [f.get("id") for f in page.get("fields", []) if isinstance(f, dict)
                       and f.get("id") not in INTERNAL_FIELDS]
        if page_fields and page_fields != field_ids:
            raise SyncError(f"Schema changed during download at offset {offset}", "incomplete")
        if not recs:
            raise SyncError(f"Empty page at offset {offset} before reaching total {total}", "incomplete")
        for rec in recs:
            if not isinstance(rec, dict):
                raise SyncError(f"Non-object record at offset {offset}", "malformed")
            for k in INTERNAL_FIELDS:
                rec.pop(k, None)
            # Pages are ordered by _id: ids must strictly increase, or offsets shifted mid-download.
            rid = rec.get("_id")
            if not isinstance(rid, int) or (last_id is not None and rid <= last_id):
                raise SyncError(f"Pagination overlap or out-of-order _id {rid!r} after {last_id!r} at offset "
                                f"{offset}; the table changed or paging is unstable", "incomplete")
            last_id = rid
        records.extend(recs)
        offset += len(recs)
        if pages % 10 == 0 or offset >= total:
            log(f"    {offset:,}/{total:,} records ({pages} pages)")
    final_total = datastore_info(http, resource_id)["total"]
    if final_total != total:
        raise SyncError(f"Table changed during download (total {total} -> {final_total})", "incomplete")
    api_url = http.api_url("datastore_search", {"resource_id": resource_id})
    return Download(records=records, fields=fields, method="datastore_search", api_url=api_url,
                    expected_total=total, pages=pages)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1255", "iso-8859-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SyncError("File is not valid UTF-8 / Windows-1255", "malformed")


def parse_file(data: bytes, fmt: str, url: str) -> tuple[list[str], list[dict], list[str]]:
    """Parse an official CSV / XLSX / JSON file into records with a synthetic _row id (1-based)."""
    kind = (fmt or "").lower()
    lower_url = url.lower().split("?")[0]
    notes: list[str] = []
    if kind in ("xlsx", "xls") or lower_url.endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook

        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0]
            if len(wb.worksheets) > 1:
                notes.append(f"Workbook has {len(wb.worksheets)} sheets; only the first was snapshotted.")
            rows = ws.iter_rows(values_only=True)
            header = [str(h) if h is not None else f"column_{i + 1}" for i, h in enumerate(next(rows, []))]
            table = [["" if v is None else str(v) for v in row] for row in rows]
        finally:
            wb.close()
    elif kind == "json" or lower_url.endswith(".json"):
        doc = json.loads(_decode(data))
        items = doc if isinstance(doc, list) else next((v for v in doc.values() if isinstance(v, list)), None) \
            if isinstance(doc, dict) else None
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise SyncError("JSON file is not a list of records", "malformed")
        header = []
        for rec in items:
            header += [k for k in rec if k not in header]
        return ["_row"] + header, [{"_row": i, **rec} for i, rec in enumerate(items, start=1)], notes
    else:
        reader = csv.reader(io.StringIO(_decode(data), newline=""))
        header = next(reader, [])
        table = [row for row in reader]
    records = []
    for i, row in enumerate(table, start=1):
        if not any(str(v).strip() for v in row):
            continue
        rec = {"_row": i}
        for j, name in enumerate(header):
            rec[name] = row[j] if j < len(row) else ""
        if len(row) > len(header):
            rec["_extra"] = row[len(header):]
        records.append(rec)
    return ["_row"] + header, records, notes


def download_file(http: Http, meta: dict) -> Download:
    url = meta["download_url"]
    if not url.startswith(("http://", "https://")):
        raise SyncError("Resource has no http(s) download URL", "invalid")
    resp = http.get(url, accept="*/*", stream=True)
    if resp.status_code != 200:
        raise SyncError(f"Download HTTP {resp.status_code}", "http", resp.status_code)
    chunks, size = [], 0
    for chunk in resp.iter_content(1024 * 1024):
        size += len(chunk)
        if size > MAX_FILE_BYTES:
            raise SyncError(f"File exceeds {MAX_FILE_BYTES:,} bytes", "invalid")
        chunks.append(chunk)
    data = b"".join(chunks)
    expected = resp.headers.get("Content-Length")
    if expected and expected.isdigit() and int(expected) != len(data) and not resp.headers.get("Content-Encoding"):
        raise SyncError(f"Truncated download: {len(data)} of {expected} bytes", "incomplete")
    notes = []
    if isinstance(meta.get("resource_size"), int) and meta["resource_size"] and meta["resource_size"] != len(data):
        notes.append(f"resource_show size {meta['resource_size']} differs from downloaded {len(data)} bytes.")
    header, records, parse_notes = parse_file(data, meta.get("resource_format", ""), url)
    fields = [{"id": h, "type": "text"} for h in header]
    d = Download(records=records, fields=fields, method="file_download", api_url=url, expected_total=None,
                 id_field="_row", notes=notes + parse_notes)
    d.notes.append(f"Original file sha256 {hashlib.sha256(data).hexdigest()} ({len(data)} bytes).")
    return d


# ------------------------------------------------------------------- validation
def validate(dl: Download, previous_rows: int | None, allow_shrink: bool = False) -> dict:
    """Raise SyncError if the download is not a complete, consistent export."""
    count = len(dl.records)
    if count == 0:
        raise SyncError("No records downloaded; refusing to publish an empty snapshot", "incomplete")
    if dl.expected_total is not None and count != dl.expected_total:
        raise SyncError(f"Row count {count} != official total {dl.expected_total}", "incomplete")
    field_ids = [f["id"] for f in dl.fields]
    if len(set(field_ids)) != len(field_ids):
        raise SyncError("Duplicate column names in schema", "malformed")
    ids = [r.get(dl.id_field) for r in dl.records]
    if any(i is None for i in ids):
        raise SyncError(f"Records without {dl.id_field}", "malformed")
    duplicates = count - len(set(ids))
    if duplicates:
        raise SyncError(f"{duplicates} duplicate {dl.id_field} values", "malformed")
    expected_keys = set(field_ids)
    bad_schema = sum(1 for r in dl.records if set(r) - {"_extra"} != expected_keys)
    if bad_schema:
        raise SyncError(f"{bad_schema} records do not match the schema", "malformed")
    if previous_rows and not allow_shrink and count < previous_rows * MAX_SHRINK_RATIO:
        raise SyncError(f"Row count dropped from {previous_rows} to {count} (> {int((1 - MAX_SHRINK_RATIO) * 100)}%"
                        " shrink); rejected as possibly incomplete. Re-run with --allow-shrink after review.",
                        "incomplete")
    int_ids = [i for i in ids if isinstance(i, int)]
    contiguous = len(int_ids) == count and min(int_ids) == 1 and max(int_ids) == count
    return {
        "row_count": count,
        "expected_total": dl.expected_total,
        "count_matches_total": dl.expected_total is None or count == dl.expected_total,
        "unique_ids": True,
        "duplicate_ids": 0,
        "id_field": dl.id_field,
        "ids_contiguous_1_to_n": contiguous,
        "schema_consistent": True,
        "column_count": len(field_ids),
        "previous_row_count": previous_rows,
    }


# ---------------------------------------------------------------------- storage
def serialize(records: Iterable[dict]) -> bytes:
    return b"".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                    for r in records)


def gzip_bytes(raw: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9, mtime=0) as gz:  # deterministic output
        gz.write(raw)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_snapshot_records(path: Path) -> tuple[bytes, list[dict]]:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    return raw, [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def atomic_write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _fsync_write(tmp, (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def load_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if path.exists():
        doc = json.loads(path.read_text("utf-8"))
        doc.setdefault("resources", {})
        return doc
    return {"format_version": FORMAT_VERSION, "resources": {}}


def repo_slug() -> str:
    return os.environ.get("GITHUB_REPOSITORY", "giladscore494/ADVISORCHECHER")


def write_snapshot(data_dir: Path, staging_dir: Path, key: str, version: str, meta: dict, dl: Download,
                   checks: dict, raw: bytes, gz: bytes, retrieved_at: str, fingerprint: str,
                   max_repo_bytes: int) -> dict:
    """Write to a temp dir, re-validate from disk, then atomically rename into place."""
    base = data_dir / "snapshots" / key
    base.mkdir(parents=True, exist_ok=True)
    final_dir = base / version
    tmp_dir = base / f".tmp-{version}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    file_name = "records.jsonl.gz"
    storage = "repo" if len(gz) <= max_repo_bytes else "release"
    snapshot = {
        "format_version": FORMAT_VERSION,
        "key": key,
        "version": version,
        **meta,
        "retrieved_at": retrieved_at,
        "retrieval_method": dl.method,
        "data_api_url": dl.api_url,
        "pages": dl.pages,
        "fields": dl.fields,
        "row_count": len(dl.records),
        "validation": checks,
        "notes": dl.notes,
        "records_sha256": sha256(raw),
        "uncompressed_bytes": len(raw),
        "file_name": file_name,
        "file_sha256": sha256(gz),
        "file_bytes": len(gz),
        "encoding": "utf-8",
        "compression": "gzip",
        "record_format": "jsonl (one JSON object per line, values exactly as served by the official API)",
        "metadata_fingerprint": fingerprint,
        "storage": storage,
        "path": (final_dir / file_name).relative_to(data_dir).as_posix() if storage == "repo" else "",
    }
    if storage == "release":
        tag = f"govdata-{key}-{version}"
        asset = f"{key}-{version}.jsonl.gz"
        snapshot["release"] = {"tag": tag, "asset_name": asset,
                               "url": f"https://github.com/{repo_slug()}/releases/download/{tag}/{asset}"}
        staging_dir.mkdir(parents=True, exist_ok=True)
        _fsync_write(staging_dir / asset, gz)
        written = (staging_dir / asset).read_bytes()
    else:
        _fsync_write(tmp_dir / file_name, gz)
        written = (tmp_dir / file_name).read_bytes()
    # Post-write validation: what is on disk must be byte-identical and decode to the same rows.
    if sha256(written) != snapshot["file_sha256"]:
        raise SyncError("Written file checksum mismatch", "invalid")
    raw_back = gzip.decompress(written)
    if sha256(raw_back) != snapshot["records_sha256"] or raw_back.count(b"\n") != len(dl.records):
        raise SyncError("Written file does not decode to the downloaded records", "invalid")
    atomic_write_json(tmp_dir / "snapshot.json", snapshot)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    return snapshot


def prune_versions(data_dir: Path, key: str, keep: int, protect: set[str]) -> list[str]:
    base = data_dir / "snapshots" / key
    if not base.exists():
        return []
    versions = sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(".tmp-"):
            shutil.rmtree(p, ignore_errors=True)
    removed = []
    for v in versions[:-keep] if keep > 0 else versions:
        if v not in protect:
            shutil.rmtree(base / v, ignore_errors=True)
            removed.append(v)
    return [v for v in versions if v not in removed]


# ------------------------------------------------------------------------- sync
def sync_resource(http: Http, cfg: dict, manifest: dict, data_dir: Path, staging_dir: Path, *,
                  force: bool = False, allow_shrink: bool = False, page_size: int = PAGE_SIZE,
                  max_repo_bytes: int = MAX_REPO_FILE_BYTES, keep: int = KEEP_VERSIONS,
                  now: datetime | None = None) -> dict:
    """Sync one resource. Updates `manifest` in memory; returns the attempt record."""
    key, resource_id = cfg["key"], cfg["resource_id"]
    now = now or utc_now()
    entry = manifest["resources"].setdefault(key, {})
    entry.update(resource_id=resource_id, label=cfg.get("label", key))
    current = entry.get("current")
    attempt: dict[str, Any] = {"at": iso(now), "status": "failed", "message": ""}
    started = time.monotonic()
    requests_before = http.requests
    try:
        log(f"[{key}] resource_show / package_show {resource_id}")
        meta = fetch_metadata(http, resource_id)
        ds = datastore_info(http, resource_id) if meta["datastore_active"] else None
        fingerprint = metadata_fingerprint(meta, ds)
        log(f"[{key}] {meta['dataset_title']} / {meta['resource_name']} | publisher: {meta['publisher']} | "
            f"datastore_active={meta['datastore_active']} | total={ds['total'] if ds else 'n/a'}")
        current_ok = bool(current) and snapshot_file_present(current, data_dir)
        verified_at = parse_iso(current.get("verified_at", "")) if current else None
        recent = verified_at is not None and now - verified_at < timedelta(days=FULL_REFRESH_DAYS)
        if (not force and current_ok and current.get("metadata_fingerprint") == fingerprint and recent):
            current["checked_at"] = iso(now)
            attempt.update(status="unchanged", message="Official metadata and row total unchanged; download skipped.")
            log(f"[{key}] unchanged (fingerprint match) - skipped")
            return attempt

        log(f"[{key}] downloading via {'datastore_search' if ds else 'official file download'}")
        dl = download_datastore(http, resource_id, page_size, ds) if ds else download_file(http, meta)
        previous_rows = current.get("row_count") if current else None
        checks = validate(dl, previous_rows, allow_shrink)
        raw = serialize(dl.records)
        records_sha = sha256(raw)
        if current_ok and current.get("records_sha256") == records_sha:
            current.update(verified_at=iso(now), checked_at=iso(now), metadata_fingerprint=fingerprint)
            attempt.update(status="unchanged", rows=len(dl.records),
                           message="Full download matched the current snapshot checksum; no new version.")
            log(f"[{key}] content unchanged ({len(dl.records):,} rows, sha256 {records_sha[:12]})")
            return attempt
        gz = gzip_bytes(raw)
        version = now.strftime("%Y%m%dT%H%M%SZ")
        snapshot = write_snapshot(data_dir, staging_dir, key, version, meta, dl, checks, raw, gz, iso(now),
                                  fingerprint, max_repo_bytes)
        summary = {k: snapshot[k] for k in (
            "version", "retrieved_at", "row_count", "records_sha256", "file_sha256", "file_bytes",
            "uncompressed_bytes", "storage", "path", "retrieval_method", "dataset_title", "dataset_id",
            "resource_name", "publisher", "license_title", "license_url", "source_url", "data_api_url",
            "resource_last_modified", "dataset_metadata_modified", "metadata_fingerprint")}
        if "release" in snapshot:
            summary["release"] = snapshot["release"]
        summary["snapshot_json"] = f"snapshots/{key}/{version}/snapshot.json"
        summary["verified_at"] = summary["checked_at"] = iso(now)
        entry["current"] = summary
        entry["versions"] = prune_versions(data_dir, key, keep, protect={version})
        attempt.update(status="updated", rows=len(dl.records), version=version, storage=snapshot["storage"],
                       file_bytes=len(gz), message=f"New validated snapshot {version}.")
        log(f"[{key}] UPDATED {len(dl.records):,} rows -> {version} ({len(gz):,} bytes gz, {snapshot['storage']})")
        return attempt
    except SyncError as exc:
        attempt.update(status="failed", error_kind=exc.kind, http_status=exc.http_status, message=str(exc))
        log(f"[{key}] FAILED ({exc.kind}): {exc} -- keeping last complete snapshot "
            f"{current.get('version') if current else '(none)'}")
        return attempt
    except Exception as exc:  # noqa: BLE001 - one bad resource must not abort the others
        attempt.update(status="failed", error_kind="unexpected", message=f"{exc.__class__.__name__}: {exc}")
        log(f"[{key}] FAILED (unexpected): {exc!r} -- keeping last complete snapshot")
        return attempt
    finally:
        attempt["duration_s"] = round(time.monotonic() - started, 1)
        attempt["requests"] = http.requests - requests_before
        entry["last_attempt"] = attempt
        if attempt["status"] in ("updated", "unchanged"):
            entry["last_success_at"] = attempt["at"]


def snapshot_file_present(current: dict, data_dir: Path) -> bool:
    if current.get("storage") == "release":
        return bool(current.get("release", {}).get("url"))
    return bool(current.get("path")) and (data_dir / current["path"]).is_file()


def verify_manifest(data_dir: Path, check_files: bool = True) -> list[str]:
    """Problems with the manifest's current snapshots (empty list = all good). No network."""
    problems = []
    manifest = load_manifest(data_dir)
    for key, entry in manifest["resources"].items():
        cur = entry.get("current")
        if not cur:
            continue
        if cur.get("storage") == "release":
            asset = RELEASE_STAGING_DIR / cur["release"]["asset_name"]
            if check_files and asset.exists() and sha256(asset.read_bytes()) != cur["file_sha256"]:
                problems.append(f"{key}: staged release asset checksum mismatch")
            continue
        path = data_dir / cur["path"]
        if not path.is_file():
            problems.append(f"{key}: missing snapshot file {cur['path']}")
            continue
        if not check_files:
            continue
        data = path.read_bytes()
        if sha256(data) != cur["file_sha256"]:
            problems.append(f"{key}: file checksum mismatch")
            continue
        raw = gzip.decompress(data)
        if sha256(raw) != cur["records_sha256"] or raw.count(b"\n") != cur["row_count"]:
            problems.append(f"{key}: decoded content does not match manifest")
    return problems


def write_summary(manifest: dict, attempts: dict[str, dict], http: Http) -> str:
    lines = ["## Government data sync", "",
             "| Resource | Status | Rows | Snapshot | Size (gz) | Storage | Message |",
             "|---|---|---:|---|---:|---|---|"]
    for key, att in attempts.items():
        cur = manifest["resources"].get(key, {}).get("current") or {}
        lines.append(f"| {key} | {att['status']} | {cur.get('row_count', '-')!s} | {cur.get('version', '-')} | "
                     f"{cur.get('file_bytes', '-')!s} | {cur.get('storage', '-')} | "
                     f"{att.get('message', '').replace('|', '/')[:160]} |")
    lines += ["", f"HTTP requests: {http.requests}, retries: {http.retries}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="*", help="Resource keys to sync (default: all)")
    p.add_argument("--force", action="store_true", help="Download even if metadata is unchanged")
    p.add_argument("--allow-shrink", action="store_true", help="Accept a large drop in row count")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--staging-dir", default=str(RELEASE_STAGING_DIR))
    p.add_argument("--page-size", type=int, default=PAGE_SIZE)
    p.add_argument("--max-repo-bytes", type=int, default=MAX_REPO_FILE_BYTES)
    p.add_argument("--verify", action="store_true", help="Only verify the manifest and snapshot files")
    args = p.parse_args(argv)
    data_dir, staging_dir = Path(args.data_dir), Path(args.staging_dir)

    if args.verify:
        problems = verify_manifest(data_dir)
        for prob in problems:
            log(f"PROBLEM: {prob}")
        log("Manifest verified." if not problems else f"{len(problems)} problem(s).")
        return 1 if problems else 0

    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(data_dir)
    http = Http()
    selected = [r for r in RESOURCES if not args.only or r["key"] in args.only]
    attempts: dict[str, dict] = {}
    run_started = utc_now()
    for cfg in selected:
        attempts[cfg["key"]] = sync_resource(http, cfg, manifest, data_dir, staging_dir, force=args.force,
                                             allow_shrink=args.allow_shrink, page_size=args.page_size,
                                             max_repo_bytes=args.max_repo_bytes)
        # Persist after every resource: a crash later in the run keeps earlier validated work.
        manifest["format_version"] = FORMAT_VERSION
        manifest["generated_at"] = iso(utc_now())
        manifest["source"] = {"portal": SITE_URL, "api": API_BASE,
                              "terms": "Data retrieved read-only from data.gov.il under each dataset's license."}
        atomic_write_json(data_dir / "manifest.json", manifest)
    manifest["last_run"] = {"started_at": iso(run_started), "finished_at": iso(utc_now()),
                            "requests": http.requests, "retries": http.retries,
                            "results": {k: a["status"] for k, a in attempts.items()},
                            "github_run_url": (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                                               f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                                               f"{os.environ['GITHUB_RUN_ID']}")
                            if os.environ.get("GITHUB_RUN_ID") else ""}
    atomic_write_json(data_dir / "manifest.json", manifest)
    summary = write_summary(manifest, attempts, http)
    log(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(summary + "\n")
    problems = verify_manifest(data_dir)
    for prob in problems:
        log(f"PROBLEM: {prob}")
    failed = [k for k, a in attempts.items() if a["status"] == "failed"]
    return 1 if problems else (2 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
