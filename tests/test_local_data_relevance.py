"""Exact codes and technical standards, scope filters, unrelated-match detection and zero-result reporting."""

import json

import pytest

import agent
import local_data
from govdata_fixtures import STD_RID, write_dataset
from helpers import FakeLLM, fake_fetch, fake_search, text_response, tool_response

TARIFF = [
    {"_id": 1, "CustomsBookTypeID": 1, "CustomsItemFullClassification": "7318000000",
     "GoodsDescription": "ברגים, לולבים, אומים ודיסקיות, מברזל או מפלדה"},
    {"_id": 2, "CustomsBookTypeID": 1, "CustomsItemFullClassification": "7318150000/7",
     "GoodsDescription": "ברגים ולולבים אחרים"},
    {"_id": 3, "CustomsBookTypeID": 1, "CustomsItemFullClassification": "7317000000",
     "GoodsDescription": "מסמרים ונעצים"},
]
REQS = [
    # A requirement attached at heading level applies to the items below it.
    {"_id": 1, "CustomsItemFullClassification": "7318150000/7", "RegularityRequirement_CustomsItemFullClassification":
        "7318000000", "CustomsBookType": "יבוא", "AutonomyRegularityRegionType": "ישראל ואוטונומיה",
     "RegularityRequirementID": "100", "ConfirmationType": "אישור תקן", "FullGoodsDescription": "ברגים"},
    {"_id": 2, "CustomsItemFullClassification": "7318150000/7", "RegularityRequirement_CustomsItemFullClassification":
        "7318150000", "CustomsBookType": "יצוא", "AutonomyRegularityRegionType": "ישראל ואוטונומיה",
     "RegularityRequirementID": "101", "ConfirmationType": "רישיון יצוא", "FullGoodsDescription": "ברגים"},
    {"_id": 3, "CustomsItemFullClassification": "7318150000/7", "RegularityRequirement_CustomsItemFullClassification":
        "7318150000", "CustomsBookType": "יבוא", "AutonomyRegularityRegionType": "אוטונומיה בלבד - עזה",
     "RegularityRequirementID": "102", "ConfirmationType": "צו אלוף", "FullGoodsDescription": "ברגים"},
    # The requirement id 7089 must never match a query for ISO 7089 or the number 7089.
    {"_id": 4, "CustomsItemFullClassification": "8516800000/9", "RegularityRequirement_CustomsItemFullClassification":
        "8516800000", "CustomsBookType": "יבוא", "AutonomyRegularityRegionType": "ישראל ואוטונומיה",
     "RegularityRequirementID": "7089", "ConfirmationType": "אישור", "FullGoodsDescription": "מחממי מים"},
]
STANDARDS = [
    {"_id": 1, "standard_number": "ת\"י 1347", "standard_name": "ברז ערבוב", "note": "מבוסס על ISO 4064"},
    {"_id": 2, "standard_number": "ת\"י 562 חלק 1", "standard_name": "בטיחות צעצועים",
     "standard_number_en": "SI 562 part 1"},
    {"_id": 3, "standard_number": "ת\"י 562 חלק 2", "standard_name": "בטיחות צעצועים: דליקות",
     "standard_number_en": "SI 562 part 2"},
    {"_id": 4, "standard_number": "ת\"י 40321", "standard_name": "מוצר אחר", "note": "ISO 40321 בלבד"},
    {"_id": 5, "standard_number": "ת\"י 900", "standard_name": "אומים", "note": "ISO 4032:2012 adopted"},
]


def fields(records):
    names = []
    for r in records:
        names += [k for k in r if k not in names]
    return [{"id": n, "type": "int" if n == "_id" or n == "CustomsBookTypeID" else "text"} for n in names]


def padded(records):
    names = [f["id"] for f in fields(records)]
    return [{n: r.get(n, "") for n in names} for r in records]


@pytest.fixture
def gov(tmp_path):
    data = tmp_path / "data" / "government"
    data.mkdir(parents=True)
    write_dataset(data, "customs_tariff", "5536eaa1-2e51-406b-aff6-b9ca02801b7c", "Tariff", fields(TARIFF),
                  padded(TARIFF))
    write_dataset(data, "import_regulations", "d9750b40-c0b9-4e05-a08e-ae768a92e9ca", "Orders", fields(REQS),
                  padded(REQS))
    write_dataset(data, "mandatory_standards", STD_RID, "Standards", fields(STANDARDS), padded(STANDARDS))
    return local_data.GovernmentData(data_dir=data, cache_dir=tmp_path / "cache")


def refs(out):
    return [f"{r['dataset']}:{r['record_id']}" for r in out["records"]]


def test_standard_references_are_parsed():
    assert local_data.standard_refs("ISO 4032") == [("ISO", "4032", "")]
    assert local_data.standard_refs("ISO 898-1:2013 and EN 71-1") == [("ISO", "898", "1"), ("EN", "71", "1")]
    assert local_data.standard_refs('ת"י 562 חלק 1') == [("SI", "562", "1")]
    assert local_data.standard_refs("SI 562 part 1") == [("SI", "562", "1")]
    assert local_data.standard_refs("ISO 4032:2012") == [("ISO", "4032", "")]  # year, not a part
    assert local_data.standard_refs("regular text 4032") == []


def test_exact_technical_standard_matching_rejects_noise(gov):
    out = gov.search("all", "ISO 4032")
    # Only the record that cites ISO 4032; not ISO 4064, not ISO 40321, and not "anything mentioning ISO".
    assert refs(out) == ["mandatory_standards:5"] and out["match_strategy"] == "standard_reference"
    assert out["records"][0]["match"].startswith("exact_standard (ISO 4032)")
    zero = gov.search("all", "ISO 7089")
    assert zero["total_matches"] == 0 and "import_regulations:4" not in refs(zero)
    assert zero["zero_result_details"]["match_strategy"] == "standard_reference"
    assert refs(gov.search("mandatory_standards", 'ת"י 1347')) == ["mandatory_standards:1"]
    assert refs(gov.search("mandatory_standards", 'ת"י 562')) == ["mandatory_standards:2", "mandatory_standards:3"]
    assert refs(gov.search("mandatory_standards", "SI 562 part 2")) == ["mandatory_standards:3"]


def test_numbers_in_identifier_fields_are_unrelated(gov):
    out = gov.search("import_regulations", "7089")
    assert out["total_matches"] == 0
    assert out["excluded"]["unrelated"] == 1
    assert "identifier fields" in out["excluded"]["unrelated_examples"][0]["reason"]
    shown = gov.search("import_regulations", "7089", include_unrelated=True)
    assert shown["records"][0]["relevance"] == "unrelated"


def test_partial_term_matches_are_weak_or_unrelated(gov):
    out = gov.search("customs_tariff", "ברגים 7318")
    assert refs(out)[:2] == ["customs_tariff:1", "customs_tariff:2"]
    assert all(r["relevance"] == "direct" for r in out["records"])
    # "ברגים" matches, but the number does not: unrelated, excluded and counted.
    out = gov.search("customs_tariff", "ברגים 9999")
    assert out["total_matches"] == 0 and out["excluded"]["unrelated"] == 2


def test_parent_level_requirement_and_scope_filters(gov):
    out = gov.search("import_regulations", "7318150000")
    matches = {r["record_id"]: r["match"] for r in out["records"]}
    assert matches["1"].startswith("requirement_on_parent_code (7318)") or matches["1"].startswith("exact_code")
    assert set(matches) == {"1", "2", "3"}
    assert out["records"][0]["scope"]["direction"] in ("import", "export")
    imports_israel = gov.search("import_regulations", "7318150000", direction="import", jurisdiction="israel")
    assert refs(imports_israel) == ["import_regulations:1"]
    assert imports_israel["excluded"]["out_of_scope"] == 2
    exports = gov.search("import_regulations", "7318150000", direction="export")
    assert refs(exports) == ["import_regulations:2"]
    with pytest.raises(local_data.LocalDataError):
        gov.search("import_regulations", "7318", direction="sideways")


def test_zero_results_show_query_filters_scope_completeness_and_neighbours(gov):
    out = gov.search("import_regulations", "7317000000", direction="import", jurisdiction="israel",
                     filters={"ConfirmationType": "אישור תקן"})
    assert out["total_matches"] == 0 and "NOT proof" in out["note"]
    z = out["zero_result_details"]
    assert z["query"] == "7317000000" and z["filters"] == {"ConfirmationType": "אישור תקן"}
    assert (z["direction"], z["jurisdiction"], z["match_strategy"]) == ("import", "israel", "customs_code")
    comp = z["dataset_completeness"]["import_regulations"]
    assert comp["row_count"] == comp["expected_total"] == 4 and comp["complete_snapshot"] is True
    assert comp["snapshot_version"] and comp["freshness"]
    assert z["nearest_headings_present"]["import_regulations"][0]["heading"] == "7318"
    assert "NOT proof" in z["conclusion"]


def test_zero_result_details_reach_the_trace(gov):
    fake = FakeLLM([
        tool_response(("search_local_government_records", {"dataset": "import_regulations", "query": "ISO 7089",
                                                            "direction": "import", "jurisdiction": "israel"})),
        text_response("DONE"), text_response("{}"),
    ])
    a = agent.ResearchAgent(fake, search_fn=fake_search, fetch_fn=fake_fetch, gov_data=gov)
    run = a.run("screws")
    q = run.trace["local_queries"][0]
    assert q["total_matches"] == 0 and q["zero_result_details"]["query"] == "ISO 7089"
    assert q["direction"] == "import" and q["jurisdiction"] == "israel"
    out = [json.loads(m["content"]) for m in fake.calls[1]["messages"] if m["role"] == "tool"][0]
    assert out["zero_result_details"]["dataset_completeness"]["import_regulations"]["complete_snapshot"]
