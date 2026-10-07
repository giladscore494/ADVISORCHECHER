"""Local indexed queries over validated snapshots (synthetic snapshots + the real committed ones)."""

import gzip
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

import local_data
from govdata_fixtures import FIO_RID, NOW, TARIFF, TARIFF_FIELDS, build_govdata, write_dataset


@pytest.fixture
def gov(tmp_path):
    return local_data.GovernmentData(data_dir=build_govdata(tmp_path), cache_dir=tmp_path / "cache")


def ids(result):
    return [(r["dataset"], r["record_id"], r["match"].split(" ")[0]) for r in result["records"]]


def test_code_normalization_and_hierarchy():
    assert local_data.normalize_code("8703.23.30.00/0") == "8703233000"
    assert local_data.normalize_code("-0010000000/9") == "-0010000000"
    assert local_data.normalize_code(407100000, numeric_field=True) == "0407100000"
    assert local_data.normalize_code("סוסים") == ""
    assert local_data.hierarchy_key("6115000000") == "6115"
    assert local_data.hierarchy_key("1000000000") == "10"  # chapter 10, not "1"
    assert local_data.hierarchy_key("6115100000") == "611510"
    assert local_data.code_query("87.03") == "8703" and local_data.code_query("רכב") == ""


def test_exact_parent_and_child_code_matching(gov):
    r = gov.search("customs_tariff", "8703230000")
    assert ids(r)[0] == ("customs_tariff", "2", "exact_code")
    matches = dict(((d, i), m) for d, i, m in ids(r))
    assert matches[("customs_tariff", "1")] == "parent_code"  # heading 8703
    assert matches[("customs_tariff", "3")] == "child_code"  # 8703233000
    assert ("customs_tariff", "4") not in matches
    # Heading query with dots returns the heading first, then its children.
    r = gov.search("customs_tariff", "87.03")
    assert ids(r)[0] == ("customs_tariff", "1", "exact_code")
    assert {i for _, i, _ in ids(r)} == {"1", "2", "3"}


def test_check_digit_and_numeric_classification_fields(gov):
    r = gov.search("free_import_order", "0407110000/4")
    assert ("free_import_order", "1", "exact_code") in ids(r)
    # The numeric parent column lost its leading zero in the DataStore; it is restored for matching.
    r = gov.search("free_import_order", "0407100000")
    assert ("free_import_order", "1", "exact_code") in ids(r)


def test_hebrew_prefixes_and_english_terms(gov):
    r = gov.search("all", "והרכב החשמלי")  # prefixed forms of רכב / חשמלי
    assert ("customs_tariff", "3") in {(d, i) for d, i, _ in ids(r)}
    assert r["records"][0]["match"].startswith("all_terms")
    r = gov.search("all", "toys")
    found = {(d, i) for d, i, _ in ids(r)}
    assert ("mandatory_standards", "1") in found and ("customs_tariff", "4") in found


def test_standard_number_query_is_not_swallowed_by_chapter_match(gov):
    r = gov.search("all", "1347")
    assert ids(r)[0][:2] == ("mandatory_standards", "2")
    assert not any(m == "parent_code" for _, _, m in ids(r))


def test_filters_exact_match_and_unknown_field(gov):
    r = gov.search("free_import_order", "", filters={"ConfirmationType": "אישור תקן"})
    assert [i for _, i, _ in ids(r)] == ["2"]
    r = gov.search("free_import_order", "9503", filters={"ConfirmationType": "רשיון יבוא"})
    assert r["records"] == []
    with pytest.raises(local_data.LocalDataError, match="Unknown field"):
        gov.search("free_import_order", "x", filters={"nope": "1"})


def test_zero_results_is_never_an_exemption(gov):
    r = gov.search("all", "zzzunknownterm")
    assert r["total_matches"] == 0 and "NOT proof of a legal exemption" in r["note"]


def test_results_bounded_and_carry_provenance(gov):
    r = gov.search("all", "8703", limit=1)
    assert r["returned"] == 1 and r["next_offset"] == 1 and r["total_matches"] >= 3
    rec = r["records"][0]
    assert rec["fields"]["CustomsItemFullClassification"] == "8703000000"  # exact original value
    assert rec["record_line"].startswith("_id: 1 | ")
    prov = r["provenance"]["customs_tariff"]
    assert prov["snapshot_version"] == "20261007T030000Z" and prov["resource_id"]
    assert prov["retrieved_at"] == "2026-10-07T03:00:00Z" and prov["source_url"].startswith("https://data.gov.il/")
    assert "NOT legally binding" in r["evidence_note"]
    r2 = gov.search("all", "8703", limit=1, offset=1)
    assert r2["records"][0]["record_id"] != rec["record_id"]
    with pytest.raises(local_data.LocalDataError):
        gov.search("all", "")
    assert gov.search("all", "8703", limit=500)["returned"] <= local_data.MAX_LIMIT


def test_get_record_and_list_and_status(gov):
    rec = gov.get_record("customs_tariff", "3")
    assert rec["found"] and rec["fields"]["CustomsTariff"] == "פטור"
    assert not gov.get_record("customs_tariff", "999")["found"]
    with pytest.raises(local_data.LocalDataError, match="Unknown dataset"):
        gov.get_record("nope", "1")
    listed = gov.list_datasets()["datasets"]
    assert {d["dataset"] for d in listed} == {"customs_tariff", "free_import_order", "mandatory_standards"}
    tariff = next(d for d in listed if d["dataset"] == "customs_tariff")
    assert tariff["row_count"] == 5 and "CustomsItemFullClassification" in tariff["code_fields"]
    st = gov.status(now=NOW + timedelta(days=1))
    assert {d["freshness"] for d in st["datasets"]} == {"fresh"}
    st = gov.status(now=NOW + timedelta(days=30))
    assert {d["freshness"] for d in st["datasets"]} == {"stale"}


def test_index_rebuilt_when_snapshot_changes(gov, tmp_path):
    first = gov.index_path()
    data_dir = gov.data_dir
    write_dataset(data_dir, "customs_tariff", "5536eaa1-2e51-406b-aff6-b9ca02801b7c", "Customs tariff",
                  TARIFF_FIELDS, TARIFF + [{"_id": 6, "CustomsItem_4_Digits": "8704",
                                            "CustomsItemFullClassification": "8704000000",
                                            "GoodsDescription": "משאיות", "CustomsTariff": ""}])
    fresh = local_data.GovernmentData(data_dir=data_dir, cache_dir=tmp_path / "cache")
    assert fresh.index_path() != first
    assert fresh.get_record("customs_tariff", "6")["found"]


def test_tampered_snapshot_is_refused(tmp_path):
    data_dir = build_govdata(tmp_path)
    manifest = json.loads((data_dir / "manifest.json").read_text("utf-8"))
    path = data_dir / manifest["resources"]["customs_tariff"]["current"]["path"]
    path.write_bytes(gzip.compress(b'{"_id": 1}\n'))
    gd = local_data.GovernmentData(data_dir=data_dir, cache_dir=tmp_path / "cache")
    with pytest.raises(local_data.LocalDataError, match="checksum"):
        gd.index_path()


class FakeReleaseSession:
    def __init__(self, content):
        self.content = content
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        return type("R", (), {"status_code": 200, "content": self.content})()


def test_release_asset_snapshot_downloaded_and_verified(tmp_path):
    data_dir = tmp_path / "data" / "government"
    data_dir.mkdir(parents=True)
    staging = tmp_path / "staging"
    cur = write_dataset(data_dir, "free_import_order", FIO_RID, "Free Import Order",
                        [{"id": "_id", "type": "int"}, {"id": "x", "type": "text"}],
                        [{"_id": i, "x": f"שורה {i}"} for i in range(1, 51)], max_repo_bytes=10, staging_dir=staging)
    assert cur["storage"] == "release" and cur["path"] == ""
    asset = (staging / cur["release"]["asset_name"]).read_bytes()
    session = FakeReleaseSession(asset)
    gd = local_data.GovernmentData(data_dir=data_dir, cache_dir=tmp_path / "cache", session=session)
    assert gd.get_record("free_import_order", "50")["fields"]["x"] == "שורה 50"
    assert session.urls == [cur["release"]["url"]]
    # A corrupted download is rejected.
    bad = local_data.GovernmentData(data_dir=data_dir, cache_dir=tmp_path / "cache2",
                                    session=FakeReleaseSession(b"corrupt"))
    with pytest.raises(local_data.LocalDataError, match="SHA-256"):
        bad.index_path()
    assert hashlib.sha256(asset).hexdigest() == cur["file_sha256"]


REAL_DATA = Path(__file__).resolve().parent.parent / "data" / "government"


@pytest.mark.skipif(not (REAL_DATA / "manifest.json").exists(), reason="no committed snapshots")
def test_real_committed_snapshots_are_complete_and_queryable(tmp_path_factory):
    """The real validated data.gov.il snapshots in the repository: counts, checksums, indexed queries."""
    manifest = json.loads((REAL_DATA / "manifest.json").read_text("utf-8"))
    cache = Path(local_data.tempfile.gettempdir()) / "advisorchecher-govdata-tests"
    gd = local_data.GovernmentData(data_dir=REAL_DATA, cache_dir=cache)
    for key, entry in manifest["resources"].items():
        cur = entry["current"]
        snap = json.loads((REAL_DATA / cur["snapshot_json"]).read_text("utf-8"))
        assert snap["validation"]["count_matches_total"] and snap["validation"]["duplicate_ids"] == 0
        assert snap["row_count"] == cur["row_count"] == snap["validation"]["expected_total"]
        if cur["storage"] == "repo":
            data = (REAL_DATA / cur["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == cur["file_sha256"]
            assert gzip.decompress(data).count(b"\n") == cur["row_count"]
    if "customs_tariff" in manifest["resources"]:
        r = gd.search("customs_tariff", "8703")
        assert r["records"] and r["records"][0]["match"].startswith("exact_code")
        assert r["records"][0]["fields"]["CustomsItemFullClassification"].startswith("8703")
    if "free_import_order" in manifest["resources"]:
        r = gd.search("free_import_order", "0407110000/4")
        assert any(x["match"].startswith("exact_code") for x in r["records"])
