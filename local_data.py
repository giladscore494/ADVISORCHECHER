"""Indexed local queries over the validated data.gov.il snapshots in data/government/.

The sync workflow (scripts/sync_government_data.py) commits complete, validated snapshots plus a
manifest. This module builds a SQLite index (FTS5 full-text + customs-code tables) from them once per
snapshot version and answers focused queries. The model never receives a whole dataset: every call
returns at most MAX_LIMIT records, with exact field values, record ids, snapshot version and provenance.

Snapshots stored as GitHub Release assets (too large for git) are downloaded from the repository's
release URL on first use and checked against the manifest's SHA-256; no search or LLM provider is involved.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import get_setting
from evidence import HEBREW_PREFIX_LETTERS, extract_terms, matched_terms, normalize_text, strip_controls

DATA_DIR = Path(__file__).resolve().parent / "data" / "government"
INDEX_FORMAT = 3
MAX_LIMIT = 25
DEFAULT_LIMIT = 10
MAX_FIELD_CHARS = 1500
MAX_CANDIDATES = 2000
MAX_FILTERS = 5
FRESH_DAYS = 2
STALE_DAYS = 8
# A record's own classification fields. CustomsItem_2/4/6_Digits are its ancestors' prefixes; indexing
# them as codes would make every row under a heading an "exact" match for that heading.
CODE_FIELD_RE = re.compile(r"classification", re.IGNORECASE)
CODE_VALUE_RE = re.compile(r"^-?\d[\d.\- ]*(/\d)?$")
EVIDENCE_NOTE = (
    "Factual record from an official data.gov.il dataset snapshot (see provenance). Dataset records describe "
    "the dataset's contents; they are NOT legally binding text. Confirm legal obligations against the official "
    "legal source (law, regulation, order or Reshumot publication) before relying on them."
)
ZERO_RESULT_NOTE = (
    "No matching records in the local snapshot. This is NOT proof of a legal exemption or of the absence of a "
    "requirement: wording, classification or dataset coverage may differ. Try other terms or codes, check the "
    "live data.gov.il resource, and verify against official legal sources."
)


class LocalDataError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ codes
def is_code_field(name: str) -> bool:
    return bool(CODE_FIELD_RE.search(name))


# The field holding a record's own tariff item. Parent/child (hierarchy) matches use only this field;
# other classification fields (e.g. the item a legal requirement is attached to) match exactly.
OWN_CODE_FIELD = "CustomsItemFullClassification"


def normalize_code(value, numeric_field: bool = False) -> str:
    """Customs classification -> digits only (check digit after '/' dropped, '-' kept for special items).
    Numeric-typed classification values lost their leading zero in the DataStore (705100000 = 0705100000)."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        digits = str(int(value))
        return digits.zfill(10) if numeric_field and 7 <= len(digits) < 10 else digits
    text = str(value).strip()
    if not text or not CODE_VALUE_RE.match(text):
        return ""
    sign = "-" if text.startswith("-") else ""
    text = re.sub(r"/\d$", "", text)
    return sign + re.sub(r"\D", "", text)


def hierarchy_key(code: str) -> str:
    """Code trimmed of trailing zeros to the nearest HS level (2/4/6/8/10 digits): 6115000000 -> 6115."""
    sign = "-" if code.startswith("-") else ""
    digits = code.lstrip("-")
    if not digits:
        return ""
    trimmed = digits.rstrip("0")
    length = max(2, len(trimmed))
    length += length % 2
    return sign + digits[:min(length, len(digits))]


def code_query(query: str) -> str:
    """The query as a customs code if it looks like one (e.g. 8703.23.00.00/2, 87032300, 8703), else ''."""
    q = query.strip()
    if not CODE_VALUE_RE.match(q):
        return ""
    code = normalize_code(q)
    return code if len(code.lstrip("-")) >= 2 else ""


# ------------------------------------------------------------------- text
def _hebrew(token: str) -> bool:
    return any("֐" <= c <= "׿" for c in token)


@functools.lru_cache(maxsize=500_000)
def _variants(token: str) -> tuple[str, ...]:
    out = [token]
    if _hebrew(token):
        for i in range(1, 4):
            prefix, rest = token[:i], token[i:]
            if len(rest) < 3 or not all(c in HEBREW_PREFIX_LETTERS for c in prefix):
                break
            out.append(rest)
    return tuple(out)


def token_variants(token: str) -> set[str]:
    """A token plus forms without Hebrew proclitic prefixes (ו/ה/ב/ל/מ/ש/כ), so 'רכב' finds 'והרכב'."""
    return set(_variants(token))


def index_text(values) -> str:
    text = normalize_text(" \n ".join(str(v) for v in values if v is not None))
    return " ".join(dict.fromkeys(v for tok in text.split() for v in _variants(tok)))


def _fts_term(term: str) -> str:
    return '"' + term.replace('"', "") + '"*'


def fts_query(terms: list[str]) -> str:
    groups = []
    for t in terms:
        variants = sorted(token_variants(t))
        groups.append("(" + " OR ".join(_fts_term(v) for v in variants) + ")" if len(variants) > 1 else _fts_term(t))
    return " OR ".join(groups)


def _render_value(value) -> str | int | float | None:
    if isinstance(value, str):
        text = strip_controls(value)
        return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "…"
    return value


def record_line(record: dict) -> str:
    """A single-line rendering of a record that the model can quote verbatim as an excerpt."""
    return " | ".join(f"{k}: {_render_value(v)}" for k, v in record.items() if v not in (None, ""))


# ------------------------------------------------------------------- store
class GovernmentData:
    """Manifest-driven access to local snapshots with a lazily built SQLite index."""

    def __init__(self, data_dir: Path | str | None = None, cache_dir: Path | str | None = None,
                 session: requests.Session | None = None):
        self.data_dir = Path(data_dir or get_setting("GOVDATA_DIR") or DATA_DIR)
        self.cache_dir = Path(cache_dir or get_setting("GOVDATA_CACHE_DIR")
                              or Path(tempfile.gettempdir()) / "advisorchecher-govdata")
        self.session = session or requests.Session()
        self._lock = threading.Lock()
        self._index_path: Path | None = None
        self._manifest: dict | None = None
        self._snapshots: dict[str, dict] = {}

    # ---------------------------------------------------------- manifest
    def manifest(self) -> dict:
        if self._manifest is None:
            path = self.data_dir / "manifest.json"
            if not path.exists():
                self._manifest = {"resources": {}}
            else:
                self._manifest = json.loads(path.read_text("utf-8"))
                self._manifest.setdefault("resources", {})
        return self._manifest

    def snapshot_meta(self, key: str) -> dict:
        if key not in self._snapshots:
            cur = self._current(key)
            path = self.data_dir / cur["snapshot_json"]
            self._snapshots[key] = json.loads(path.read_text("utf-8"))
        return self._snapshots[key]

    def _current(self, key: str) -> dict:
        entry = self.manifest()["resources"].get(key)
        if not entry:
            raise LocalDataError(f"Unknown dataset '{key}'. Available: {', '.join(self.dataset_keys()) or 'none'}")
        if not entry.get("current"):
            raise LocalDataError(f"Dataset '{key}' has no validated snapshot yet.")
        return entry["current"]

    def dataset_keys(self) -> list[str]:
        return [k for k, e in self.manifest()["resources"].items() if e.get("current")]

    def provenance(self, key: str) -> dict:
        cur = self._current(key)
        return {
            "dataset": key,
            "dataset_title": cur.get("dataset_title", ""),
            "resource_name": cur.get("resource_name", ""),
            "resource_id": self.manifest()["resources"][key]["resource_id"],
            "dataset_id": cur.get("dataset_id", ""),
            "publisher": cur.get("publisher", ""),
            "license": cur.get("license_title", ""),
            "source_url": cur.get("source_url", ""),
            "data_api_url": cur.get("data_api_url", ""),
            "snapshot_version": cur.get("version", ""),
            "retrieved_at": cur.get("retrieved_at", ""),
            "last_verified_at": cur.get("verified_at", ""),
            "snapshot_records_sha256": cur.get("records_sha256", ""),
            "resource_last_modified": cur.get("resource_last_modified", ""),
            "date_note": "Snapshot and dataset dates are retrieval/update dates, not legal effective dates.",
        }

    def status(self, now: datetime | None = None) -> dict:
        now = now or _now()
        m = self.manifest()
        datasets = []
        for key, entry in m["resources"].items():
            cur = entry.get("current") or {}
            verified = _parse_time(cur.get("verified_at", ""))
            age_days = round((now - verified).total_seconds() / 86400, 1) if verified else None
            freshness = "missing" if not cur else (
                "fresh" if age_days is not None and age_days <= FRESH_DAYS else
                "aging" if age_days is not None and age_days <= STALE_DAYS else "stale")
            attempt = entry.get("last_attempt") or {}
            datasets.append({
                "dataset": key,
                "label": entry.get("label", key),
                "resource_id": entry.get("resource_id", ""),
                "available": bool(cur),
                "row_count": cur.get("row_count"),
                "snapshot_version": cur.get("version", ""),
                "retrieved_at": cur.get("retrieved_at", ""),
                "last_verified_at": cur.get("verified_at", ""),
                "age_days": age_days,
                "freshness": freshness,
                "storage": cur.get("storage", ""),
                "file_bytes": cur.get("file_bytes"),
                "last_attempt_status": attempt.get("status", ""),
                "last_attempt_at": attempt.get("at", ""),
                "last_attempt_message": attempt.get("message", "")[:300],
            })
        return {"datasets": datasets, "last_run": m.get("last_run", {}),
                "freshness_policy": f"fresh <= {FRESH_DAYS} days since last official verification, "
                                    f"aging <= {STALE_DAYS} days, stale beyond that"}

    # ------------------------------------------------------------- files
    def _records_file(self, key: str) -> Path:
        cur = self._current(key)
        if cur.get("storage") != "release":
            path = self.data_dir / cur["path"]
            if not path.is_file():
                raise LocalDataError(f"Snapshot file missing for '{key}': {cur['path']}")
            return path
        rel = cur["release"]
        path = self.cache_dir / "release-assets" / rel["asset_name"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cur["file_sha256"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                resp = self.session.get(rel["url"], timeout=(15, 300))
            except requests.RequestException as exc:
                raise LocalDataError(f"Could not download release asset for '{key}': {exc}") from exc
            if resp.status_code != 200:
                raise LocalDataError(f"Release asset for '{key}' returned HTTP {resp.status_code}")
            if hashlib.sha256(resp.content).hexdigest() != cur["file_sha256"]:
                raise LocalDataError(f"Release asset for '{key}' failed its SHA-256 check")
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            os.replace(tmp, path)
        return path

    def _load_records(self, key: str) -> list[dict]:
        cur = self._current(key)
        path = self._records_file(key)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != cur["file_sha256"]:
            raise LocalDataError(f"Snapshot file for '{key}' does not match the manifest checksum")
        raw = gzip.decompress(data)
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
        if len(records) != cur["row_count"]:
            raise LocalDataError(f"Snapshot for '{key}' has {len(records)} rows; manifest says {cur['row_count']}")
        return records

    # ------------------------------------------------------------- index
    def _index_signature(self) -> str:
        parts = [f"v{INDEX_FORMAT}"] + [f"{k}:{self._current(k)['records_sha256']}" for k in sorted(self.dataset_keys())]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def index_path(self) -> Path:
        with self._lock:
            if self._index_path and self._index_path.exists():
                return self._index_path
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"govdata-{self._index_signature()}.sqlite"
            if not path.exists():
                self._build_index(path)
            self._index_path = path
            return path

    def _build_index(self, path: Path) -> None:
        tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        if tmp.exists():
            tmp.unlink()
        con = sqlite3.connect(tmp)
        try:
            con.executescript("""
                CREATE TABLE records (id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, rec_id TEXT NOT NULL,
                                      body TEXT NOT NULL, UNIQUE (dataset, rec_id));
                CREATE TABLE codes (rid INTEGER NOT NULL, field TEXT NOT NULL, code TEXT NOT NULL,
                                    hier TEXT NOT NULL);
                CREATE INDEX codes_code ON codes(code);
                CREATE INDEX codes_hier ON codes(hier);
                CREATE VIRTUAL TABLE fts USING fts5(text, content='', detail='column',
                                                    tokenize = 'unicode61 remove_diacritics 2');
            """)
            rowid = 0
            for key in self.dataset_keys():
                meta = self.snapshot_meta(key)
                id_field = meta.get("validation", {}).get("id_field", "_id")
                types = {f["id"]: f.get("type", "") for f in meta.get("fields", [])}
                code_fields = [f for f in types if is_code_field(f)]
                rows, codes, texts = [], [], []
                for rec in self._load_records(key):
                    rowid += 1
                    rows.append((rowid, key, str(rec.get(id_field)), json.dumps(rec, ensure_ascii=False)))
                    texts.append((rowid, index_text(v for k, v in rec.items() if k != id_field)))
                    for f in code_fields:
                        code = normalize_code(rec.get(f), numeric_field=types.get(f) in ("numeric", "int"))
                        if len(code.lstrip("-")) >= 2:
                            codes.append((rowid, f, code, hierarchy_key(code)))
                con.executemany("INSERT INTO records VALUES (?,?,?,?)", rows)
                con.executemany("INSERT INTO codes VALUES (?,?,?,?)", codes)
                con.executemany("INSERT INTO fts(rowid, text) VALUES (?,?)", texts)
            con.execute("INSERT INTO fts(fts) VALUES ('optimize')")
            con.commit()
            con.execute("VACUUM")
        finally:
            con.close()
        os.replace(tmp, path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.index_path()}?mode=ro", uri=True, check_same_thread=False)
        return con

    # ------------------------------------------------------------- tools
    def list_datasets(self) -> dict:
        out = []
        for key in self.dataset_keys():
            meta = self.snapshot_meta(key)
            prov = self.provenance(key)
            out.append({
                "dataset": key,
                "label": self.manifest()["resources"][key].get("label", key),
                "title": prov["dataset_title"],
                "resource_name": prov["resource_name"],
                "publisher": prov["publisher"],
                "row_count": meta["row_count"],
                "record_id_field": meta.get("validation", {}).get("id_field", "_id"),
                "fields": [f["id"] for f in meta.get("fields", [])],
                "code_fields": [f["id"] for f in meta.get("fields", []) if is_code_field(f["id"])],
                "snapshot_version": prov["snapshot_version"],
                "retrieved_at": prov["retrieved_at"],
                "last_verified_at": prov["last_verified_at"],
                "resource_id": prov["resource_id"],
                "source_url": prov["source_url"],
            })
        return {"datasets": out, "note": "Query with search_local_government_records; never request whole datasets."}

    def _validate_filters(self, key: str, filters) -> dict:
        if not filters:
            return {}
        if not isinstance(filters, dict) or len(filters) > MAX_FILTERS:
            raise LocalDataError(f"filters must be an object with at most {MAX_FILTERS} field: value pairs")
        fields = {f["id"] for f in self.snapshot_meta(key).get("fields", [])}
        unknown = [f for f in filters if f not in fields]
        if unknown:
            raise LocalDataError(f"Unknown field(s) {unknown} for '{key}'. Fields: {sorted(fields)}")
        if not all(isinstance(v, (str, int, float, bool)) for v in filters.values()):
            raise LocalDataError("filter values must be scalars (exact match)")
        return filters

    @staticmethod
    def _passes(record: dict, filters: dict) -> bool:
        for k, v in filters.items():
            rv = record.get(k)
            if str(rv if rv is not None else "").strip() != str(v).strip():
                return False
        return True

    def search(self, dataset: str, query: str, filters: dict | None = None, limit: int = DEFAULT_LIMIT,
               offset: int = 0) -> dict:
        """Focused search: customs-code exact / parent / child matches, then full-text (Hebrew + English)."""
        datasets = self.dataset_keys() if dataset in ("", "all", "*", None) else [dataset]
        for key in datasets:
            self._current(key)
        filters_by_ds = {key: self._validate_filters(key, filters) for key in datasets} if filters else {}
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        offset = max(0, int(offset or 0))
        query = (query or "").strip()
        if not query and not filters:
            raise LocalDataError("query or filters is required")

        hits: dict[int, dict] = {}  # records.id -> {"match", "rank", "field"}

        def add(rid, match, rank, field=None):
            cur = hits.get(rid)
            if cur is None or rank < cur["rank"]:
                hits[rid] = {"match": match, "rank": rank, "field": field}

        con = self._connect()
        try:
            ds_ph = ",".join("?" * len(datasets))
            in_ds = f"AND r.dataset IN ({ds_ph})"
            code = code_query(query)
            if code:
                hier = hierarchy_key(code)
                found_code = False
                for rid, f in con.execute(
                        f"SELECT c.rid, c.field FROM codes c JOIN records r ON r.id = c.rid "
                        f"WHERE (c.code = ? OR c.hier = ?) {in_ds}", (code, hier, *datasets)):
                    add(rid, "exact_code", 0, f)
                    found_code = True
                for rid, f in con.execute(
                        f"SELECT c.rid, c.field FROM codes c JOIN records r ON r.id = c.rid "
                        f"WHERE c.field = ? AND c.hier > ? AND c.hier < ? {in_ds} LIMIT {MAX_CANDIDATES}",
                        (OWN_CODE_FIELD, hier, hier + "~", *datasets)):
                    add(rid, "child_code", 2, f)
                    found_code = True
                sign = "-" if hier.startswith("-") else ""
                digits = hier.lstrip("-")
                parents = [sign + digits[:n] for n in range(2, len(digits), 2)]
                # Parent headings/chapters only for a code that exists; otherwise "1347" (a standard number)
                # would return all of chapter 13.
                if parents and found_code:
                    ph = ",".join("?" * len(parents))
                    for rid, f, h in con.execute(
                            f"SELECT c.rid, c.field, c.hier FROM codes c JOIN records r ON r.id = c.rid "
                            f"WHERE c.field = ? AND c.hier IN ({ph}) {in_ds}", (OWN_CODE_FIELD, *parents, *datasets)):
                        add(rid, f"parent_code ({h})", 1 + (len(digits) - len(h.lstrip('-'))) / 100, f)
            # Text search too: a code-like query may be a standard number (ת"י 1347) rather than a tariff item.
            terms = [code.lstrip("-")] if code else sorted(extract_terms(query))
            if not terms and query:
                terms = [t for t in normalize_text(query).split() if t]
            if terms:
                rows = con.execute(
                    f"SELECT f.rowid, bm25(fts) FROM fts f JOIN records r ON r.id = f.rowid "
                    f"WHERE fts MATCH ? {in_ds} ORDER BY bm25(fts) LIMIT {MAX_CANDIDATES}",
                    (fts_query(terms), *datasets)).fetchall()
                bodies = self._bodies(con, [r for r, _ in rows])
                for rid, bm in rows:
                    body = bodies.get(rid)
                    if body is None:
                        continue
                    text = " ".join(str(v) for v in body[2].values() if v is not None)
                    got = {t for t in terms if matched_terms(token_variants(t), text)}
                    if not got:
                        continue
                    coverage = len(got) / len(terms)
                    add(rid, "all_terms" if coverage == 1 else f"terms {sorted(got)}", 3 + (1 - coverage) + bm / 1000)
            if filters and not query:
                for (rid,) in con.execute(f"SELECT r.id FROM records r WHERE 1=1 {in_ds}", datasets):
                    add(rid, "filter", 5)

            bodies = self._bodies(con, list(hits))
        finally:
            con.close()

        ranked = []
        for rid, info in sorted(hits.items(), key=lambda kv: (kv[1]["rank"], kv[0])):
            key, rec_id, body = bodies[rid]
            if filters_by_ds.get(key) and not self._passes(body, filters_by_ds[key]):
                continue
            ranked.append((key, rec_id, info, body))

        page = ranked[offset: offset + limit]
        results = [self._result(key, rid, body, info) for key, rid, info, body in page]
        out = {
            "query": query, "filters": filters or {}, "datasets_searched": datasets,
            "total_matches": len(ranked), "offset": offset, "returned": len(results),
            "next_offset": offset + len(results) if len(ranked) > offset + len(results) else None,
            "records": results,
            "provenance": {key: self.provenance(key) for key in {r["dataset"] for r in results} or datasets},
            "evidence_note": EVIDENCE_NOTE,
        }
        if code:
            out["code_query"] = {"normalized": code, "hierarchy_key": hierarchy_key(code),
                                 "match_types": "exact_code > parent_code (heading/chapter) > child_code"}
        if not results:
            out["note"] = ZERO_RESULT_NOTE
        return out

    @staticmethod
    def _bodies(con, ids: list[int]) -> dict[int, tuple[str, str, dict]]:
        out = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = f"SELECT id, dataset, rec_id, body FROM records WHERE id IN ({','.join('?' * len(chunk))})"
            for rid, key, rec_id, body in con.execute(q, chunk):
                out[rid] = (key, rec_id, json.loads(body))
        return out

    def _result(self, key: str, rid: str, body: dict, info: dict | None = None) -> dict:
        fields = {k: _render_value(v) for k, v in body.items()}
        out = {"dataset": key, "record_id": rid, "fields": fields, "record_line": record_line(body),
               "snapshot_version": self._current(key).get("version", ""),
               "resource_id": self.manifest()["resources"][key]["resource_id"]}
        if info:
            out["match"] = info["match"] + (f" in {info['field']}" if info.get("field") else "")
        return out

    def get_record(self, dataset: str, record_id) -> dict:
        self._current(dataset)
        con = self._connect()
        try:
            row = con.execute("SELECT body FROM records WHERE dataset = ? AND rec_id = ?",
                              (dataset, str(record_id).strip())).fetchone()
        finally:
            con.close()
        if row is None:
            return {"dataset": dataset, "record_id": str(record_id), "found": False,
                    "note": "No record with this id in the local snapshot.",
                    "provenance": self.provenance(dataset)}
        body = json.loads(row[0])
        return {"found": True, **self._result(dataset, str(record_id).strip(), body),
                "provenance": self.provenance(dataset), "evidence_note": EVIDENCE_NOTE}


_default: GovernmentData | None = None
_default_lock = threading.Lock()


def get_default() -> GovernmentData:
    global _default
    with _default_lock:
        if _default is None:
            _default = GovernmentData()
        return _default


_warm_started = False


def warm_up() -> None:
    """Build the default index in a background thread once per process (first query then needn't wait)."""
    global _warm_started
    with _default_lock:
        if _warm_started or (get_setting("GOVDATA_WARMUP", "1") or "1") == "0":
            return
        _warm_started = True

    def build():
        try:
            gd = get_default()
            if gd.dataset_keys():
                gd.index_path()
        except Exception:  # noqa: BLE001 - surfaced later by the first query
            pass

    threading.Thread(target=build, daemon=True, name="govdata-index").start()
