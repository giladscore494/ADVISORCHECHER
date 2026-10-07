"""Pydantic schema for the agent's final output, plus strict parsing."""

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

DimensionStatus = Literal["verified", "partially_verified", "unverified", "contradicted"]

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
    # Supplied by the model: a verbatim quote from the retrieved content supporting the claim, dataset
    # identifiers for data.gov.il evidence, and the legal effective date stated in the legal text (if any).
    excerpt: str = ""
    dataset_id: str = ""
    resource_id: str = ""
    legal_effective_date: str = ""
    # Optional citation details: the document_id from fetch_url and the page the quote is on.
    document_id: str = ""
    page: str = ""
    # Set by the agent after the run (not by the model); any value the model supplies is discarded.
    verified: bool | None = None
    official: bool | None = None
    verification_note: str = ""
    kind: str = ""  # legal_document | dataset | web
    excerpt_verified: bool | None = None
    retrieval_status: str = ""  # ok | failed | rejected | no_matching_records | metadata_only | not_retrieved
    dataset_title: str = ""
    publisher: str = ""
    last_updated: str = ""  # dataset/resource update date, NOT a legal effective date
    matched_pages: list[int] = []  # pages of the stored document on which the excerpt was found
    page_mismatch: bool | None = None  # the cited page differs from where the excerpt actually is

    @field_validator("url")
    @classmethod
    def _http_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("source url must start with http:// or https://")
        return v

    @field_validator("page", mode="before")
    @classmethod
    def _page_text(cls, v) -> str:
        return "" if v is None else str(v)


def _valid_sources(items) -> list:
    """Evidence lists inside checks: drop malformed entries instead of rejecting the whole report."""
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, SourceRef) or (
        isinstance(i, dict) and str(i.get("url", "")).strip().startswith(("http://", "https://")))]


class LegalCheck(BaseModel):
    status: Literal["checked", "not_checked", "contradicted", "not_applicable"] = "not_checked"
    finding: str = ""
    evidence: list[SourceRef] = []
    # Set by the agent.
    verified: bool | None = None
    verification_note: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v) -> str:
        v = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in ("checked", "not_checked", "contradicted", "not_applicable") else "not_checked"

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v) -> list:
        return _valid_sources(v)


class LegalChecks(BaseModel):
    """Every aspect that must be checked before a legal conclusion counts as verified."""

    provision: LegalCheck = Field(default_factory=LegalCheck)
    scope: LegalCheck = Field(default_factory=LegalCheck)
    validity: LegalCheck = Field(default_factory=LegalCheck)
    exceptions: LegalCheck = Field(default_factory=LegalCheck)
    product_classification: LegalCheck = Field(default_factory=LegalCheck)


class BusinessAdvantage(BaseModel):
    claim: str = ""
    # True if the rule is available to every importer on the same terms (then it is not an advantage).
    generally_available: bool | None = None
    differentiator: str = ""
    status: Literal["claimed", "contradicted", "not_claimed"] = "claimed"
    evidence: list[SourceRef] = []
    competitor_evidence: list[SourceRef] = []

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v) -> str:
        v = str(v or "claimed").strip().lower().replace(" ", "_")
        return v if v in ("claimed", "contradicted", "not_claimed") else "claimed"

    @field_validator("evidence", "competitor_evidence", mode="before")
    @classmethod
    def _evidence(cls, v) -> list:
        return _valid_sources(v)


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
    claim_type: Literal["positive", "negative"] | None = None  # negative = exemption / no requirement applies
    legal_checks: LegalChecks = Field(default_factory=LegalChecks)
    business_advantage: BusinessAdvantage = Field(default_factory=BusinessAdvantage)
    # Filled in by the agent after validation, not by the model.
    unread_primary_sources: list[str] = []
    # Source dimension (kept under its original name for compatibility with stored reports).
    verification_status: Literal["verified", "partially_verified", "unverified"] = "unverified"
    verification_notes: list[str] = []
    downgraded_from: str = ""
    legal_verification: DimensionStatus = "unverified"
    business_verification: DimensionStatus = "unverified"
    negative_claim: bool = False
    dimension_notes: dict[str, list[str]] = {}

    @field_validator("claim_type", mode="before")
    @classmethod
    def _claim_type(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("positive", "negative") else None

    @property
    def source_verification(self) -> str:
        return self.verification_status


class ResearchResult(BaseModel):
    opportunities: list[Opportunity] = []
    no_opportunity_reason: str = ""
    research_summary: str = ""
    unresolved_questions: list[str] = []
    # Filled in by the agent: the run's critical-evidence checklist at the end of the research.
    checklist: list[dict] = []


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


AGENT_OPPORTUNITY_FIELDS = ("unread_primary_sources", "verification_status", "verification_notes", "downgraded_from",
                            "legal_verification", "business_verification", "negative_claim", "dimension_notes")
AGENT_SOURCE_FIELDS = (
    "verified", "official", "verification_note", "kind", "excerpt_verified", "retrieval_status",
    "dataset_title", "publisher", "last_updated", "matched_pages", "page_mismatch",
)


def _drop_source_fields(sources) -> None:
    for src in sources or []:
        if isinstance(src, dict):
            for key in AGENT_SOURCE_FIELDS:
                src.pop(key, None)


def _drop_agent_fields(data: dict) -> None:
    """Verification fields are computed by the agent; ignore anything the model put there."""
    for opp in data.get("opportunities") or []:
        if not isinstance(opp, dict):
            continue
        for key in AGENT_OPPORTUNITY_FIELDS:
            opp.pop(key, None)
        for list_key in ("primary_sources", "secondary_sources", "contradictory_sources_checked"):
            _drop_source_fields(opp.get(list_key))
        checks = opp.get("legal_checks")
        if isinstance(checks, dict):
            for check in checks.values():
                if isinstance(check, dict):
                    check.pop("verified", None)
                    check.pop("verification_note", None)
                    _drop_source_fields(check.get("evidence"))
        elif checks is not None:
            opp.pop("legal_checks")
        adv = opp.get("business_advantage")
        if isinstance(adv, dict):
            _drop_source_fields(adv.get("evidence"))
            _drop_source_fields(adv.get("competitor_evidence"))
        elif adv is not None:
            opp.pop("business_advantage")
    data.pop("checklist", None)


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
    """A before B before C, then legal, source and business verification, then business_score and confidence.
    Contradicted conclusions rank last within their class."""
    order = {"A": 0, "B": 1, "C": 2}
    verified = {"verified": 0, "partially_verified": 1, "unverified": 2, "contradicted": 3}
    ranked = sorted(
        opps,
        key=lambda o: (order[o.classification], verified[o.legal_verification], verified[o.verification_status],
                       verified[o.business_verification], -o.business_score, -o.confidence),
    )
    return ranked[: max(0, limit)]
