"""Mocked data.gov.il CKAN API: a URL-routing fake session plus realistic fixture payloads.

Fixture contents are invented for tests (they do not claim to reproduce real records), except that the
"מבנים לשימור - JSON-ITM" resource mirrors the id, name and datastore_active=false of the real resource
that a previous run mistook for a standards dataset.
"""

import io
import json
from urllib.parse import parse_qsl, urlparse

from openpyxl import Workbook

API = "https://data.gov.il/api/3/action/"

STD_DATASET_ID = "0f6c1d2e-1111-4a4b-9c1e-000000000001"
STD_RESOURCE_ID = "7a2b3c4d-2222-4b5c-8d9e-000000000002"  # datastore_active = true
STD_XLSX_RESOURCE_ID = "7a2b3c4d-3333-4b5c-8d9e-000000000003"  # datastore_active = false, XLSX download
STD_XLSX_URL = "https://data.gov.il/dataset/official-standards/resource/7a2b3c4d-3333/download/standards.xlsx"

PRESERVATION_DATASET_ID = "c3d4e5f6-4444-4c5d-9e0f-000000000004"
PRESERVATION_RESOURCE_ID = "5d801af3-6e69-4fd0-807a-3c635a5fe7b7"
PRESERVATION_URL = "https://data.gov.il/dataset/preservation-buildings/resource/5d801af3/download/buildings_itm.json"

STD_RECORD_LINE = "1 | 1234 | ברגים ואומים מפלדה | רשמי"


def envelope(result, success=True):
    return json.dumps({"help": "https://data.gov.il/api/3/action/help_show", "success": success,
                       "result": result}, ensure_ascii=False).encode()


def error_envelope(message, kind="Not Found Error"):
    return json.dumps({"success": False, "error": {"__type": kind, "message": message}}).encode()


STD_DATASET = {
    "id": STD_DATASET_ID,
    "name": "official-standards",
    "title": "רשימת תקנים רשמיים",
    "notes": "רשימת התקנים הרשמיים שהוכרזו לפי חוק התקנים, כולל מספר התקן, שם התקן וסטטוס.",
    "organization": {"title": "משרד הכלכלה והתעשייה"},
    "metadata_created": "2019-01-01T00:00:00",
    "metadata_modified": "2026-09-15T10:00:00",
    "license_id": "cc-by",
    "license_title": "Creative Commons Attribution",
    "tags": [{"name": "תקינה"}, {"name": "standards"}],
    "resources": [
        {"id": STD_RESOURCE_ID, "name": "תקנים רשמיים", "format": "CSV", "datastore_active": True,
         "url": "https://data.gov.il/dataset/official-standards/resource/7a2b3c4d-2222/download/standards.csv",
         "last_modified": "2026-09-15T09:00:00"},
        {"id": STD_XLSX_RESOURCE_ID, "name": "תקנים רשמיים (XLSX)", "format": "XLSX", "datastore_active": "False",
         "url": STD_XLSX_URL, "last_modified": "2026-09-01T09:00:00"},
    ],
}

PRESERVATION_DATASET = {
    "id": PRESERVATION_DATASET_ID,
    "name": "preservation-buildings",
    "title": "מבנים לשימור",
    "notes": "שכבת מבנים לשימור בקואורדינטות רשת ישראל החדשה (ITM).",
    "organization": {"title": "רשות מקומית (fixture)"},
    "metadata_modified": "2025-03-01T00:00:00",
    "license_title": "Other (Open)",
    "tags": [{"name": "שימור"}, {"name": "GIS"}],
    "resources": [{"id": PRESERVATION_RESOURCE_ID, "name": "מבנים לשימור - JSON-ITM", "format": "JSON",
                   "datastore_active": False, "url": PRESERVATION_URL}],
}


def resource_of(dataset, resource_id):
    r = next(r for r in dataset["resources"] if r["id"] == resource_id)
    return {**r, "package_id": dataset["id"], "description": "", "mimetype": "", "created": "2020-01-01T00:00:00"}


STD_FIELDS = [{"id": "_id", "type": "int"}, {"id": "מספר תקן", "type": "text"},
              {"id": "שם התקן", "type": "text"}, {"id": "סטטוס", "type": "text"}]
STD_RECORDS = [
    {"_id": 1, "מספר תקן": "1234", "שם התקן": "ברגים ואומים מפלדה", "סטטוס": "רשמי"},
    {"_id": 2, "מספר תקן": "5678", "שם התקן": "צינורות פלסטיק", "סטטוס": "רשמי"},
]


def datastore_result(records, total=None, offset=0, limit=20):
    return {"resource_id": STD_RESOURCE_ID, "fields": STD_FIELDS, "records": records,
            "total": len(records) if total is None else total, "offset": offset, "limit": limit,
            "_links": {"start": "/api/3/action/datastore_search", "next": "/api/3/action/datastore_search?offset=20"}}


def standards_xlsx() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "תקנים"
    ws.append(["מספר תקן", "שם התקן", "סטטוס"])
    ws.append(["1234", "ברגים ואומים מפלדה", "רשמי"])
    ws.append(["9999", "מעליות", "רשמי"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


PRESERVATION_JSON = json.dumps({"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"שם": "בית העם", "כתובת": "רחוב הרצל 1"},
     "geometry": {"type": "Point", "coordinates": [178000, 663000]}}]}, ensure_ascii=False).encode()


class Resp:
    def __init__(self, status=200, body=b"", ctype="application/json", headers=None):
        self.status_code = status
        self.headers = {"Content-Type": ctype, **(headers or {})}
        self._body = body
        self.url = ""
        self.encoding = "utf-8"
        self.is_redirect = False

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i: i + size]

    def close(self):
        pass


class CkanSession:
    """Routes GET requests by (path suffix, required query params). A list of responses is served in
    order (the last one repeats). Every request is recorded."""

    def __init__(self, routes=None):
        self.routes = list(routes or [])
        self.requests: list[dict] = []

    def add(self, path, responses, **params):
        self.routes.insert(0, (path, params, responses if isinstance(responses, list) else [responses]))
        return self

    def get(self, url, headers=None, **kwargs):
        parsed = urlparse(url)
        params = dict(parse_qsl(parsed.query))
        self.requests.append({"url": url, "path": parsed.path, "params": params, "headers": headers or {}})
        for path, required, responses in self.routes:
            if parsed.path.endswith(path) and all(params.get(k) == str(v) for k, v in required.items()):
                resp = responses.pop(0) if len(responses) > 1 else responses[0]
                resp.url = url
                return resp
        resp = Resp(404, error_envelope("Not found"))
        resp.url = url
        return resp

    def calls_to(self, suffix):
        return [r for r in self.requests if r["path"].endswith(suffix)]


def standard_session() -> CkanSession:
    """A data.gov.il mock with the standards dataset and the unrelated preservation dataset."""
    s = CkanSession()
    s.add("/package_search", Resp(body=envelope({"count": 2, "results": [STD_DATASET, PRESERVATION_DATASET]})))
    s.add("/package_show", Resp(body=envelope(STD_DATASET)), id=STD_DATASET_ID)
    s.add("/package_show", Resp(body=envelope(STD_DATASET)), id="official-standards")
    s.add("/package_show", Resp(body=envelope(PRESERVATION_DATASET)), id=PRESERVATION_DATASET_ID)
    s.add("/resource_show", Resp(body=envelope(resource_of(STD_DATASET, STD_RESOURCE_ID))), id=STD_RESOURCE_ID)
    s.add("/resource_show", Resp(body=envelope(resource_of(STD_DATASET, STD_XLSX_RESOURCE_ID))), id=STD_XLSX_RESOURCE_ID)
    s.add("/resource_show", Resp(body=envelope(resource_of(PRESERVATION_DATASET, PRESERVATION_RESOURCE_ID))),
          id=PRESERVATION_RESOURCE_ID)
    s.add("/datastore_search", Resp(body=envelope(datastore_result(STD_RECORDS[:1]))), resource_id=STD_RESOURCE_ID)
    s.add("/download/standards.xlsx", Resp(body=standards_xlsx(),
                                           ctype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    s.add("/download/buildings_itm.json", Resp(body=PRESERVATION_JSON))
    return s
