"""Snapshot sync: complete multi-page downloads, validation, atomic replacement, failure handling."""

import gzip
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_government_data.py"
spec = importlib.util.spec_from_file_location("sync_government_data", SCRIPT)
sync = importlib.util.module_from_spec(spec)
sys.modules["sync_government_data"] = sync
spec.loader.exec_module(sync)

RID = "5536eaa1-2e51-406b-aff6-b9ca02801b7c"
PKG = "pkg-customs"
CFG = {"key": "customs_tariff", "resource_id": RID, "label": "Customs tariff"}
FIELDS = [{"id": "_id", "type": "int"}, {"id": "פרט מכס", "type": "text"}, {"id": "תיאור", "type": "text"}]
NOW = datetime(2026, 10, 7, 3, 0, 0, tzinfo=timezone.utc)


def make_records(n, start=1):
    return [{"_id": i, "פרט מכס": f"{8703000000 + i:010d}", "תיאור": f"רכב מנועי {i} / motor car"}
            for i in range(start, start + n)]


class Resp:
    def __init__(self, status=200, doc=None, body=b"", headers=None):
        self.status_code = status
        self._doc = doc
        self.content = body if doc is None else json.dumps(doc, ensure_ascii=False).encode()
        self.headers = headers or {}

    def json(self):
        if self._doc is None:
            raise ValueError("not json")
        return self._doc

    def iter_content(self, n):
        yield self.content

    def close(self):
        pass


class FakeCkan:
    """data.gov.il CKAN fake: resource_show, package_show and paginated datastore_search."""

    def __init__(self, records, page_cap=None, metadata_modified="2026-10-01T00:00:00"):
        self.records = records
        self.page_cap = page_cap  # server-side max rows per page (smaller than the requested limit)
        self.metadata_modified = metadata_modified
        self.calls = []
        self.fail = {}  # action -> list of statuses to return before succeeding
        self.total_override = None
        self.drop_offsets = set()  # offsets whose page silently loses a record
        self.duplicate_offsets = set()

    def get(self, url, headers=None, timeout=None, stream=False):
        parsed = urlparse(url)
        action = parsed.path.rsplit("/", 1)[-1]
        params = dict(parse_qsl(parsed.query))
        self.calls.append((action, params))
        assert headers and "User-Agent" in headers
        queue = self.fail.get(action)
        if queue:
            return Resp(queue.pop(0), {"success": False, "error": {"message": "boom"}})
        if action == "resource_show":
            return Resp(200, {"success": True, "result": {
                "id": RID, "package_id": PKG, "name": "תעריף המכס", "format": "CSV", "datastore_active": True,
                "url": "https://data.gov.il/dataset/x/resource/y/download/tariff.csv",
                "last_modified": "2026-10-01T00:00:00", "metadata_modified": self.metadata_modified}})
        if action == "package_show":
            return Resp(200, {"success": True, "result": {
                "id": PKG, "name": "customs-tariff", "title": "תעריף המכס", "license_title": "ODbL",
                "organization": {"title": "רשות המסים"}, "metadata_modified": self.metadata_modified}})
        if action == "datastore_search":
            assert "q" not in params and "filters" not in params, "full export must not use search"
            limit, offset = int(params.get("limit", 100)), int(params.get("offset", 0))
            total = self.total_override if self.total_override is not None else len(self.records)
            if limit == 0:
                return Resp(200, {"success": True, "result": {"total": total, "fields": FIELDS, "records": []}})
            assert params.get("sort") == "_id asc"
            n = min(limit, self.page_cap or limit)
            page = [dict(r) for r in self.records[offset: offset + n]]
            if offset in self.drop_offsets and len(page) > 10:
                page = page[:5] + page[6:]  # a record silently missing from the middle of a page
            if offset in self.duplicate_offsets and page:
                page[-1] = dict(page[0])
            return Resp(200, {"success": True, "result": {"total": total, "fields": FIELDS, "records": page}})
        return Resp(404, {"success": False, "error": {"message": "unknown"}})


def run(tmp_path, fake, **kw):
    http = sync.Http(session=fake, min_interval=0, sleep=lambda s: None)
    data_dir = tmp_path / "data" / "government"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = sync.load_manifest(data_dir)
    kw.setdefault("now", NOW)
    attempt = sync.sync_resource(http, CFG, manifest, data_dir, tmp_path / "staging", **kw)
    sync.atomic_write_json(data_dir / "manifest.json", manifest)
    return attempt, manifest, data_dir


def read_current(data_dir, manifest):
    cur = manifest["resources"]["customs_tariff"]["current"]
    raw, recs = sync.read_snapshot_records(data_dir / cur["path"])
    return cur, raw, recs


def test_complete_multi_page_download(tmp_path):
    records = make_records(12345)
    fake = FakeCkan(records, page_cap=1000)  # server caps pages below our requested page size
    attempt, manifest, data_dir = run(tmp_path, fake, page_size=5000)
    assert attempt["status"] == "updated", attempt
    cur, raw, recs = read_current(data_dir, manifest)
    assert cur["row_count"] == 12345 == len(recs)
    assert [r["_id"] for r in recs] == list(range(1, 12346))
    assert recs[0]["פרט מכס"] == "8703000001"  # original values (leading zeros, Hebrew) preserved
    assert cur["records_sha256"] == sync.sha256(raw)
    assert cur["file_sha256"] == sync.sha256((data_dir / cur["path"]).read_bytes())
    snap = json.loads((data_dir / cur["snapshot_json"]).read_text("utf-8"))
    assert snap["validation"]["count_matches_total"] and snap["validation"]["duplicate_ids"] == 0
    assert snap["publisher"] == "רשות המסים" and snap["license_title"] == "ODbL"
    assert snap["resource_id"] == RID and snap["source_url"].endswith(RID)
    pages = [p for a, p in fake.calls if a == "datastore_search" and p.get("limit") != "0"]
    assert len(pages) == 13
    # total read before and after the download
    assert sum(1 for a, p in fake.calls if a == "datastore_search" and p.get("limit") == "0") >= 2
    assert sync.verify_manifest(data_dir) == []


def test_gzip_is_deterministic(tmp_path):
    raw = sync.serialize(make_records(10))
    assert sync.gzip_bytes(raw) == sync.gzip_bytes(raw)
    assert gzip.decompress(sync.gzip_bytes(raw)) == raw


def test_missing_rows_rejected_and_previous_snapshot_kept(tmp_path):
    attempt, manifest, data_dir = run(tmp_path, FakeCkan(make_records(3000)), page_size=1000)
    assert attempt["status"] == "updated"
    first = dict(manifest["resources"]["customs_tariff"]["current"])

    bad = FakeCkan(make_records(3100), metadata_modified="2026-10-06T00:00:00")
    bad.drop_offsets = {1000}  # one page silently loses a record
    attempt, manifest, data_dir = run(tmp_path, bad, page_size=1000, now=NOW + timedelta(days=1))
    assert attempt["status"] == "failed" and attempt["error_kind"] == "incomplete"
    entry = manifest["resources"]["customs_tariff"]
    assert entry["current"]["version"] == first["version"]
    assert entry["current"]["row_count"] == 3000
    assert entry["last_attempt"]["status"] == "failed"
    assert sync.verify_manifest(data_dir) == []
    # no partially-written version directory left behind
    versions = [p.name for p in (data_dir / "snapshots" / "customs_tariff").iterdir()]
    assert versions == [first["version"]]


def test_duplicate_ids_rejected(tmp_path):
    fake = FakeCkan(make_records(2500))
    fake.duplicate_offsets = {1000}
    attempt, manifest, _ = run(tmp_path, fake, page_size=1000)
    assert attempt["status"] == "failed" and attempt["error_kind"] == "incomplete"
    assert manifest["resources"]["customs_tariff"].get("current") is None


def test_total_changing_during_download_rejected(tmp_path):
    fake = FakeCkan(make_records(1500))
    original = fake.get
    state = {"n": 0}

    def get(url, **kw):
        if "limit=0" in url:
            state["n"] += 1
            if state["n"] == 2:  # 1 = pre-download total, 2 = post-download re-check
                fake.total_override = 1501
        return original(url, **kw)

    fake.get = get
    attempt, _, _ = run(tmp_path, fake, page_size=1000)
    assert attempt["status"] == "failed" and "changed during download" in attempt["message"]


def test_transient_errors_retried_with_backoff(tmp_path):
    fake = FakeCkan(make_records(50))
    fake.fail["resource_show"] = [503, 429]
    sleeps = []
    http = sync.Http(session=fake, min_interval=0, sleep=sleeps.append, backoff_base=2)
    data_dir = tmp_path / "data" / "government"
    data_dir.mkdir(parents=True)
    manifest = sync.load_manifest(data_dir)
    attempt = sync.sync_resource(http, CFG, manifest, data_dir, tmp_path / "s", now=NOW)
    assert attempt["status"] == "updated"
    assert sleeps[:2] == [2, 4] and http.retries == 2


def test_access_denied_is_not_retried(tmp_path):
    fake = FakeCkan(make_records(50))
    fake.fail["resource_show"] = [403, 403, 403]
    attempt, manifest, _ = run(tmp_path, fake)
    assert attempt["status"] == "failed" and attempt["error_kind"] == "blocked"
    assert sum(1 for a, _ in fake.calls if a == "resource_show") == 1


def test_unchanged_metadata_skips_download_then_weekly_refresh(tmp_path):
    fake = FakeCkan(make_records(100))
    run(tmp_path, fake)
    fake.calls.clear()
    attempt, manifest, _ = run(tmp_path, fake, now=NOW + timedelta(days=1))
    assert attempt["status"] == "unchanged" and "skipped" in attempt["message"]
    assert not [p for a, p in fake.calls if a == "datastore_search" and p.get("limit") != "0"]
    # After FULL_REFRESH_DAYS the data is re-downloaded; identical content creates no new version.
    attempt, manifest, data_dir = run(tmp_path, fake, now=NOW + timedelta(days=9))
    assert attempt["status"] == "unchanged" and "checksum" in attempt["message"]
    assert manifest["resources"]["customs_tariff"]["current"]["verified_at"] == sync.iso(NOW + timedelta(days=9))
    assert len(list((data_dir / "snapshots" / "customs_tariff").iterdir())) == 1


def test_changed_data_creates_new_version_atomically(tmp_path):
    run(tmp_path, FakeCkan(make_records(100)))
    fake = FakeCkan(make_records(120), metadata_modified="2026-10-08T00:00:00")
    attempt, manifest, data_dir = run(tmp_path, fake, now=NOW + timedelta(days=1))
    assert attempt["status"] == "updated"
    cur, _, recs = read_current(data_dir, manifest)
    assert len(recs) == 120
    assert manifest["resources"]["customs_tariff"]["versions"] == ["20261007T030000Z", "20261008T030000Z"]
    assert sync.verify_manifest(data_dir) == []


def test_large_shrink_rejected(tmp_path):
    run(tmp_path, FakeCkan(make_records(1000)))
    attempt, manifest, _ = run(tmp_path, FakeCkan(make_records(100), metadata_modified="x"),
                               now=NOW + timedelta(days=1))
    assert attempt["status"] == "failed" and "shrink" in attempt["message"]
    assert manifest["resources"]["customs_tariff"]["current"]["row_count"] == 1000


def test_oversized_snapshot_goes_to_release_staging(tmp_path):
    attempt, manifest, data_dir = run(tmp_path, FakeCkan(make_records(500)), max_repo_bytes=100)
    cur = manifest["resources"]["customs_tariff"]["current"]
    assert attempt["status"] == "updated" and cur["storage"] == "release" and cur["path"] == ""
    asset = tmp_path / "staging" / cur["release"]["asset_name"]
    assert asset.exists() and sync.sha256(asset.read_bytes()) == cur["file_sha256"]
    assert cur["release"]["url"].startswith("https://github.com/")
    assert not (data_dir / "snapshots" / "customs_tariff" / cur["version"] / "records.jsonl.gz").exists()


def test_empty_dataset_rejected(tmp_path):
    attempt, manifest, _ = run(tmp_path, FakeCkan([]))
    assert attempt["status"] == "failed"
    assert manifest["resources"]["customs_tariff"].get("current") is None


def test_verify_detects_tampered_file(tmp_path):
    _, manifest, data_dir = run(tmp_path, FakeCkan(make_records(10)))
    cur = manifest["resources"]["customs_tariff"]["current"]
    (data_dir / cur["path"]).write_bytes(sync.gzip_bytes(b'{"_id":1}\n'))
    assert any("checksum" in p for p in sync.verify_manifest(data_dir))


def test_parse_csv_file_preserves_values():
    data = "קוד,שם\n0101,סוסים חיים\n,\n0102,בקר\n".encode("utf-8")
    header, records, _ = sync.parse_file(data, "CSV", "https://x/y.csv")
    assert header == ["_row", "קוד", "שם"]
    assert records == [{"_row": 1, "קוד": "0101", "שם": "סוסים חיים"}, {"_row": 3, "קוד": "0102", "שם": "בקר"}]


@pytest.mark.parametrize("bad", ["", "ftp://x"])
def test_file_download_requires_http_url(bad):
    http = sync.Http(session=FakeCkan([]), min_interval=0, sleep=lambda s: None)
    with pytest.raises(sync.SyncError):
        sync.download_file(http, {"download_url": bad})
