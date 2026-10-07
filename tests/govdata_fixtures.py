"""Builds small but realistic local government snapshots through the real sync code path."""

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "sync_government_data.py"
if "sync_government_data" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sync_government_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_government_data"] = module
    spec.loader.exec_module(module)
sync = sys.modules["sync_government_data"]

TARIFF_RID = "5536eaa1-2e51-406b-aff6-b9ca02801b7c"
FIO_RID = "a36db570-09f2-4521-8e3d-0290eb839c68"
STD_RID = "1a4d94e2-369a-488d-a223-eb1020612fbd"
NOW = datetime(2026, 10, 7, 3, 0, 0, tzinfo=timezone.utc)

TARIFF_FIELDS = [{"id": "_id", "type": "int"}, {"id": "CustomsItem_4_Digits", "type": "text"},
                 {"id": "CustomsItemFullClassification", "type": "text"}, {"id": "GoodsDescription", "type": "text"},
                 {"id": "CustomsTariff", "type": "text"}]
TARIFF = [
    {"_id": 1, "CustomsItem_4_Digits": "8703", "CustomsItemFullClassification": "8703000000",
     "GoodsDescription": "מכוניות נוסעים וכלי רכב מנועיים אחרים / motor cars", "CustomsTariff": ""},
    {"_id": 2, "CustomsItem_4_Digits": "8703", "CustomsItemFullClassification": "8703230000",
     "GoodsDescription": "בעלי נפח צילינדרים העולה על 1,500 סמ\"ק", "CustomsTariff": "7%"},
    {"_id": 3, "CustomsItem_4_Digits": "8703", "CustomsItemFullClassification": "8703233000/0",
     "GoodsDescription": "רכב חשמלי לנוסעים", "CustomsTariff": "פטור"},
    {"_id": 4, "CustomsItem_4_Digits": "9503", "CustomsItemFullClassification": "9503000000",
     "GoodsDescription": "תלת-אופן, קורקינטים, מכוניות דוושה וצעצועים דומים בעלי גלגלים; toys", "CustomsTariff": "12%"},
    {"_id": 5, "CustomsItem_4_Digits": "1000", "CustomsItemFullClassification": "1000000000",
     "GoodsDescription": "דגנים", "CustomsTariff": ""},
]
FIO_FIELDS = [{"id": "_id", "type": "int"}, {"id": "CustomsItemFullClassification", "type": "text"},
              {"id": "RegularityRequirement_CustomsItemFullClassification", "type": "text"},
              {"id": "CustomsItemParent_FullClassification", "type": "numeric"},
              {"id": "ConfirmationType", "type": "text"}, {"id": "RegularityRequirement_Authority", "type": "text"}]
FIO = [
    {"_id": 1, "CustomsItemFullClassification": "0407110000/4",
     "RegularityRequirement_CustomsItemFullClassification": "0407000000",
     "CustomsItemParent_FullClassification": 407100000, "ConfirmationType": "רשיון יבוא",
     "RegularityRequirement_Authority": "משרד החקלאות"},
    {"_id": 2, "CustomsItemFullClassification": "9503000000",
     "RegularityRequirement_CustomsItemFullClassification": "9503000000",
     "CustomsItemParent_FullClassification": 9503000000, "ConfirmationType": "אישור תקן",
     "RegularityRequirement_Authority": "ממונה על התקינה - ת\"י 562"},
]
STD_FIELDS = [{"id": "_id", "type": "int"}, {"id": "standard_number", "type": "text"},
              {"id": "standard_name", "type": "text"}, {"id": "standard_name_en", "type": "text"}]
STD = [
    {"_id": 1, "standard_number": "ת\"י 562 חלק 1", "standard_name": "בטיחות צעצועים: תכונות מכניות",
     "standard_name_en": "Safety of toys: mechanical properties"},
    {"_id": 2, "standard_number": "ת\"י 1347", "standard_name": "ברז ערבוב מכני", "standard_name_en": "Mixing tap"},
]
DATASETS = [
    ("customs_tariff", TARIFF_RID, "Customs tariff", TARIFF_FIELDS, TARIFF),
    ("free_import_order", FIO_RID, "Free Import Order", FIO_FIELDS, FIO),
    ("mandatory_standards", STD_RID, "Official standards", STD_FIELDS, STD),
]


def write_dataset(data_dir: Path, key: str, rid: str, label: str, fields, records, max_repo_bytes=None,
                  staging_dir: Path | None = None) -> dict:
    manifest = sync.load_manifest(data_dir)
    meta = {"resource_id": rid, "resource_name": label, "dataset_id": f"pkg-{key}", "dataset_title": label,
            "publisher": "רשות המסים בישראל", "license_title": "Other (Open)", "license_url": "",
            "source_url": f"https://data.gov.il/dataset/{key}/resource/{rid}", "resource_last_modified": "2026-10-01",
            "dataset_metadata_modified": "2026-10-01", "download_url": "", "datastore_active": True}
    dl = sync.Download(records=[dict(r) for r in records], fields=fields, method="datastore_search",
                       api_url=f"https://data.gov.il/api/3/action/datastore_search?resource_id={rid}",
                       expected_total=len(records), pages=1)
    checks = sync.validate(dl, None)
    raw = sync.serialize(dl.records)
    gz = sync.gzip_bytes(raw)
    version = NOW.strftime("%Y%m%dT%H%M%SZ")
    snap = sync.write_snapshot(data_dir, staging_dir or data_dir / "staging", key, version, meta, dl, checks, raw, gz,
                               sync.iso(NOW), "fp", max_repo_bytes or sync.MAX_REPO_FILE_BYTES)
    current = {k: snap[k] for k in ("version", "retrieved_at", "row_count", "records_sha256", "file_sha256",
                                    "file_bytes", "storage", "path", "dataset_title", "dataset_id", "resource_name",
                                    "publisher", "license_title", "source_url", "data_api_url",
                                    "resource_last_modified")}
    if "release" in snap:
        current["release"] = snap["release"]
    current.update(snapshot_json=f"snapshots/{key}/{version}/snapshot.json", verified_at=sync.iso(NOW))
    manifest["resources"][key] = {"resource_id": rid, "label": label, "current": current,
                                  "last_attempt": {"status": "updated", "at": sync.iso(NOW)}}
    sync.atomic_write_json(data_dir / "manifest.json", manifest)
    return current


def build_govdata(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data" / "government"
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, rid, label, fields, records in DATASETS:
        write_dataset(data_dir, key, rid, label, fields, records)
    return data_dir
