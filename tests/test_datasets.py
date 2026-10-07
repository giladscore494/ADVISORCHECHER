"""JSON, data.gov.il CKAN, CSV and XLSX retrieval through fetch_url (network mocked)."""

import io
import json

import pytest
from openpyxl import Workbook

import fetcher
from fetcher import extract_csv_text, extract_json, extract_xlsx_text, fetch_url
from test_fetcher import FakeResp, FakeSession

CKAN = "https://data.gov.il/api/3/action"
RESOURCE_URL = "https://data.gov.il/dataset/vehicles/resource/053cea08/download/vehicles.csv"


@pytest.fixture(autouse=True)
def public_hosts(monkeypatch):
    monkeypatch.setattr(fetcher, "_is_public_host", lambda host: True)


def fetch(url, body, ctype):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False).encode()
    return fetch_url(url, session=FakeSession([FakeResp(body=body, ctype=ctype)]))


# ------------------------------------------------------------------ JSON
def test_generic_json():
    res = fetch("https://api.example.gov.il/x", {"threshold_kg": 3500, "שם": "תקנה"}, "application/json; charset=utf-8")
    assert res.ok and res.source_type == "json"
    assert '"threshold_kg": 3500' in res.text and "תקנה" in res.text


def test_json_detected_without_content_type():
    res = fetch("https://www.gov.il/data", [{"a": 1}], "application/octet-stream")
    assert res.ok and res.source_type == "json"


def test_invalid_json_reported():
    res = fetch("https://www.gov.il/data", b"{not json", "application/json")
    assert not res.ok and "Invalid JSON" in res.error


# ------------------------------------------------------------------ CKAN
RESOURCE_SHOW = {
    "help": "https://data.gov.il/api/3/action/help_show?name=resource_show",
    "success": True,
    "result": {
        "id": "053cea08-09bc-40ec-8f7a-156f0677aff3",
        "package_id": "8a6e3d1f-0000-4a4b-9c1e-1234567890ab",
        "name": "רשימת כלי רכב פרטיים",
        "description": "Private vehicles registry",
        "format": "CSV",
        "mimetype": "text/csv",
        "url": RESOURCE_URL,
        "last_modified": "2026-09-30T08:12:00",
        "datastore_active": True,
        "size": 1048576,
        "cache_url": None,
        "position": 0,
    },
}


def test_ckan_resource_show_extracts_metadata_and_download_url():
    res = fetch(f"{CKAN}/resource_show?id=053cea08", RESOURCE_SHOW, "application/json")
    assert res.ok and res.source_type == "ckan_resource"
    assert res.resource_url == RESOURCE_URL
    assert res.title == "רשימת כלי רכב פרטיים"
    assert res.metadata["format"] == "CSV" and res.metadata["package_id"].startswith("8a6e3d1f")
    assert res.metadata["last_modified"] == "2026-09-30T08:12:00"
    assert "cache_url" not in res.metadata and "position" not in res.metadata
    assert RESOURCE_URL in res.text and "fetch_url on the resource url" in res.text


def test_ckan_relative_resource_url_is_resolved():
    body = {"success": True, "result": {"url": "/dataset/x/download/file.csv", "format": "CSV", "package_id": "p"}}
    res = fetch(f"{CKAN}/resource_show?id=x", body, "application/json")
    assert res.resource_url == "https://data.gov.il/dataset/x/download/file.csv"


def test_ckan_error_response():
    body = {"success": False, "error": {"__type": "Not Found Error", "message": "Not found: Resource was not found."}}
    res = fetch(f"{CKAN}/resource_show?id=missing", body, "application/json")
    assert not res.ok and "CKAN API error: Not found" in res.error


def test_ckan_package_show_lists_resources():
    body = {"success": True, "result": {
        "title": "רישוי עסקים", "name": "business-licensing", "notes": "Licensing data",
        "organization": {"title": "משרד הפנים"}, "metadata_modified": "2026-08-01",
        "resources": [{"id": "r1", "name": "2026", "format": "XLSX", "url": "https://data.gov.il/r1.xlsx"}],
    }}
    res = fetch(f"{CKAN}/package_show?id=business-licensing", body, "application/json")
    assert res.ok and res.source_type == "ckan_dataset"
    assert "משרד הפנים" in res.text and "https://data.gov.il/r1.xlsx" in res.text and "[XLSX]" in res.text


def test_ckan_datastore_search_renders_bounded_table(monkeypatch):
    monkeypatch.setattr(fetcher, "MAX_TABLE_ROWS", 2)
    body = {"success": True, "result": {
        "resource_id": "053cea08", "total": 5000,
        "fields": [{"id": "_id"}, {"id": "סוג"}, {"id": "משקל"}],
        "records": [{"_id": i, "סוג": "נגרר", "משקל": 750 + i} for i in range(3)],
    }}
    _, _, text, _, meta = extract_json(json.dumps(body).encode())
    assert "Columns: _id | סוג | משקל" in text
    assert "0 | נגרר | 750" in text and "1 | נגרר | 751" in text and "752" not in text
    assert "[showing first 2 of 5000 data rows]" in text
    assert meta["total"] == 5000


# ------------------------------------------------------------------- CSV
def test_csv_utf8_hebrew():
    data = "סוג,משקל מרבי\nנגרר קל,750\nנגרר,3500\n".encode("utf-8-sig")
    res = fetch(RESOURCE_URL, data, "text/csv")
    assert res.ok and res.source_type == "csv"
    assert res.text.startswith("Columns: סוג | משקל מרבי")
    assert "נגרר קל | 750" in res.text


def test_csv_windows_1255_and_semicolons():
    data = "סוג;משקל\nנגרר;750\n".encode("cp1255")
    text = extract_csv_text(data)
    assert "Columns: סוג | משקל" in text and "נגרר | 750" in text


def test_csv_labelled_as_excel_is_still_csv():
    res = fetch("https://data.gov.il/x/file.csv", b"a,b\n1,2\n", "application/vnd.ms-excel")
    assert res.ok and res.source_type == "csv" and "1 | 2" in res.text


def test_csv_rows_bounded(monkeypatch):
    monkeypatch.setattr(fetcher, "MAX_TABLE_ROWS", 3)
    data = ("id,value\n" + "".join(f"{i},v{i}\n" for i in range(10))).encode()
    text = extract_csv_text(data)
    assert "2 | v2" in text and "3 | v3" not in text
    assert "[showing first 3 of 10 data rows]" in text


def test_empty_csv_reported():
    res = fetch("https://data.gov.il/x/empty.csv", b"", "text/csv")
    assert not res.ok


# ------------------------------------------------------------------ XLSX
def make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_reads_sheets():
    data = make_xlsx({
        "תעריפים": [["שירות", "מחיר"], ["בדיקה תקופתית", 250], ["השכרה יומית", 120]],
        "Notes": [["source"], ["Ministry of Transport"]],
    })
    res = fetch("https://data.gov.il/x/rates.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    assert res.ok and res.source_type == "xlsx"
    assert "[sheet: תעריפים]" in res.text and "Columns: שירות | מחיר" in res.text
    assert "בדיקה תקופתית | 250" in res.text and "[sheet: Notes]" in res.text


def test_xlsx_rows_and_sheets_bounded(monkeypatch):
    monkeypatch.setattr(fetcher, "MAX_TABLE_ROWS", 2)
    monkeypatch.setattr(fetcher, "MAX_XLSX_SHEETS", 1)
    data = make_xlsx({"S1": [["n"]] + [[i] for i in range(10)], "S2": [["x"], [1]]})
    text = extract_xlsx_text(data)
    assert "\n0\n1" in text and "\n2" not in text.split("[showing")[0]
    assert "[showing first 2 of 10 data rows]" in text
    assert "[1 more sheets not shown]" in text and "S2" not in text


def test_xlsx_zip_bomb_guard(monkeypatch):
    monkeypatch.setattr(fetcher, "MAX_XLSX_UNCOMPRESSED", 100)
    with pytest.raises(ValueError, match="too large"):
        extract_xlsx_text(make_xlsx({"S": [["a"]]}))


def test_legacy_xls_reported_clearly():
    res = fetch("https://data.gov.il/x/old.xls", b"\xd0\xcf\x11\xe0 legacy ole file", "application/vnd.ms-excel")
    assert not res.ok and res.source_type == "xlsx" and ".xls is not supported" in res.error


# --------------------------------------------------- failed official retrieval
def test_gov_il_403_reported_without_retry():
    session = FakeSession([FakeResp(status=403, body=b"<html>Access denied</html>")])
    res = fetch_url("https://www.gov.il/he/departments/legalInfo/regulation-x", session=session)
    assert not res.ok and res.http_status == 403
    assert "access denied" in res.error and "not retried" in res.error
    assert session.responses == []  # exactly one request, no retry with other headers
