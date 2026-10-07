"""REAL read-only integration tests against data.gov.il (no mocks).

Skipped unless RUN_LIVE_GOV_TESTS=1. The sync workflow runs them right after synchronizing, so the
live official totals and sampled records are compared with the snapshots that were just validated.
"""

import gzip
import json
import os
from pathlib import Path

import pytest

from govdata_fixtures import sync

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("RUN_LIVE_GOV_TESTS") != "1", reason="set RUN_LIVE_GOV_TESTS=1")]

DATA = Path(__file__).resolve().parent.parent / "data" / "government"


@pytest.fixture(scope="module")
def http():
    return sync.Http(min_interval=1.0)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((DATA / "manifest.json").read_text("utf-8"))


def snapshot_records(cur) -> dict:
    with gzip.open(DATA / cur["path"], "rt", encoding="utf-8") as fh:
        return {r["_id"]: r for r in map(json.loads, fh)}


@pytest.mark.parametrize("cfg", sync.RESOURCES, ids=[r["key"] for r in sync.RESOURCES])
def test_live_resource_matches_validated_snapshot(http, manifest, cfg):
    meta = sync.fetch_metadata(http, cfg["resource_id"])
    assert meta["resource_id"] == cfg["resource_id"] and meta["datastore_active"]
    info = sync.datastore_info(http, cfg["resource_id"])
    cur = manifest["resources"][cfg["key"]]["current"]
    assert info["total"] == cur["row_count"], (
        f"{cfg['key']}: live total {info['total']} != snapshot {cur['row_count']} (data changed since the sync?)")
    if cur["storage"] != "repo":
        pytest.skip("snapshot stored as release asset")
    records = snapshot_records(cur)
    assert len(records) == cur["row_count"]
    # Lossless check: live records (first page and one from the middle) equal the stored ones exactly.
    for offset in (0, max(0, info["total"] // 2 - 2)):
        page = http.action("datastore_search", resource_id=cfg["resource_id"], limit=5, offset=offset,
                           sort="_id asc")
        assert page["records"], "empty live page"
        for live in page["records"]:
            live.pop("_full_text", None)
            assert records[live["_id"]] == live, f"record {live['_id']} differs from the snapshot"
