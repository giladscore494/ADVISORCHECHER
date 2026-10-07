"""Pydantic schema for the agent's final output, plus strict parsing."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

SCORE_KEYS = (
    "profit_potential",
    "startup_capital",
    "operational_complexity",
    "regulatory_complexity",
    "legal_risk",
    "regulatory_change_risk",
    "competition",
    "side_business_fit",
    "recurring_revenue",
    "barrier_to_entry",
)


class SourceRef(BaseModel):
    title: str = ""
    url: str
    section: str = ""
    support: str = ""
    # Set by the agent after the run (not by the model): was this URL retrieved and read in this run,
    # and is it an official source? Any value the model supplies is overwritten.
    verified: bool | None = None
    official: bool | None = None
    verification_note: str = ""

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("source url must start with http:// or https://")
        return v


class Scores(BaseModel):
    """Every score is 0-10 where 10 is the most favorable for the founder
    (e.g. legal_risk=10 means very low legal risk, startup_capital=10 means very little capital needed)."""

    profit_potential: int = Field(ge=0, le=10)
    startup_capital: int = Field(ge=0, le=10)
    operational_complexity: int = Field(ge=0, le=10)
    regulatory_complexity: int = Field(ge=0, le=10)
    legal_risk: int = Field(ge=0, le=10)
    regulatory_change_risk: int = Field(ge=0, le=10)
    competition: int = Field(ge=0, le=10)
    side_business_fit: int = Field(ge=0, le=10)
    recurring_revenue: int = Field(ge=0, le=10)
    barrier_to_entry: int = Field(ge=0, le=10)


class Opportunity(BaseModel):
    name: str = Field(min_length=1)
    summary: str = ""
    regulatory_mechanism: str = Field(min_length=1)
    classification: Literal["A", "B", "C"]
    business_thesis: str = Field(min_length=1)
    customer: str = ""
    customer_problem: str = ""
    revenue_model: str = ""
    startup_capital_estimate: str = ""
    existing_competition: list[str] = []
    primary_sources: list[SourceRef] = Field(min_length=1)
    secondary_sources: list[SourceRef] = []
    contradictory_sources_checked: list[SourceRef] = []
    red_team: list[str] = []
    open_legal_questions: list[str] = []
    scores: Scores
    business_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    # Filled in by the agent after validation, not by the model.
    unread_primary_sources: list[str] = []
    verification_status: Literal["verified", "partially_verified", "unverified"] = "unverified"
    verification_notes: list[str] = []
    downgraded_from: str = ""


class ResearchResult(BaseModel):
    opportunities: list[Opportunity] = []
    no_opportunity_reason: str = ""
    research_summary: str = ""


def extract_json_object(text: str) -> str:
    """Strip markdown fences / surrounding prose and return the outermost {...} block."""
    t = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return t[start : end + 1]


AGENT_OPPORTUNITY_FIELDS = ("unread_primary_sources", "verification_status", "verification_notes", "downgraded_from")
AGENT_SOURCE_FIELDS = ("verified", "official", "verification_note")


def _drop_agent_fields(data: dict) -> None:
    """Verification fields are computed by the agent; ignore anything the model put there."""
    for opp in data.get("opportunities") or []:
        if not isinstance(opp, dict):
            continue
        for key in AGENT_OPPORTUNITY_FIELDS:
            opp.pop(key, None)
        for list_key in ("primary_sources", "secondary_sources", "contradictory_sources_checked"):
            for src in opp.get(list_key) or []:
                if isinstance(src, dict):
                    for key in AGENT_SOURCE_FIELDS:
                        src.pop(key, None)


def parse_research_result(text: str) -> ResearchResult:
    """Parse and validate the model's final JSON. Raises ValueError with a readable reason."""
    try:
        data = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    _drop_agent_fields(data)
    try:
        return ResearchResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc


def rank_opportunities(opps: list[Opportunity], limit: int) -> list[Opportunity]:
    """A before B before C, then better-verified evidence, then business_score and confidence."""
    order = {"A": 0, "B": 1, "C": 2}
    verified = {"verified": 0, "partially_verified": 1, "unverified": 2}
    ranked = sorted(
        opps,
        key=lambda o: (order[o.classification], verified[o.verification_status], -o.business_score, -o.confidence),
    )
    return ranked[: max(0, limit)]
