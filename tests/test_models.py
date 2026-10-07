import json

import pytest

from models import extract_json_object, parse_research_result, rank_opportunities
from helpers import valid_report


def test_valid_report_parses_from_fenced_text():
    text = "Here you go:\n```json\n" + json.dumps(valid_report()) + "\n```"
    result = parse_research_result(text)
    assert result.opportunities[0].classification == "A"


def test_empty_result_is_valid():
    result = parse_research_result(json.dumps({"opportunities": [], "no_opportunity_reason": "nothing survived"}))
    assert result.opportunities == []


@pytest.mark.parametrize("mutate", [
    lambda r: r["opportunities"][0].update(classification="D"),
    lambda r: r["opportunities"][0]["scores"].update(legal_risk=11),
    lambda r: r["opportunities"][0].update(primary_sources=[]),
    lambda r: r["opportunities"][0].update(primary_sources=[{"url": "not a url"}]),
    lambda r: r["opportunities"][0].pop("scores"),
    lambda r: r["opportunities"][0].update(business_score=150),
])
def test_invalid_reports_rejected(mutate):
    report = valid_report()
    mutate(report)
    with pytest.raises(ValueError):
        parse_research_result(json.dumps(report))


@pytest.mark.parametrize("text", ["", "no json here", "{bad json,}", "[1, 2]"])
def test_malformed_text_rejected(text):
    with pytest.raises(ValueError):
        parse_research_result(text)


def test_extract_json_object():
    assert extract_json_object('prefix {"a": {"b": 1}} suffix') == '{"a": {"b": 1}}'


def test_ranking_prefers_class_a_and_limits():
    report = valid_report(n=3)
    report["opportunities"][0].update(classification="C", business_score=99)
    report["opportunities"][1].update(classification="A", business_score=50)
    report["opportunities"][2].update(classification="A", business_score=80)
    ranked = rank_opportunities(parse_research_result(json.dumps(report)).opportunities, 2)
    assert [o.business_score for o in ranked] == [80, 50]
