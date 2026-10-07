"""Three separate, conservative verification dimensions.

SOURCE VERIFIED      The cited text or dataset record was actually retrieved in this run and the quoted
                     excerpt was found in it (computed from retrieved content, never from the model).
LEGAL APPLICABILITY  The provision, its scope, validity, exceptions and the product classification were each
VERIFIED             checked against retrieved, source-verified official evidence sufficient for this specific
                     conclusion. A verified quotation alone never makes a legal conclusion verified.
BUSINESS ADVANTAGE   The finding gives a demonstrable advantage over competing importers (not a generally
VERIFIED             available rule), on top of a verified legal conclusion, with verified evidence for the
                     differentiator and for the competitive situation.

Each dimension is one of VERIFIED, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED. Negative claims
(exemptions, "no requirement applies") need explicit official legal text: zero dataset matches or the
absence of a provision in an incompletely extracted document never establish them.
"""

from __future__ import annotations

import re
from typing import Callable

VERIFIED, PARTIAL, UNVERIFIED, CONTRADICTED = "verified", "partially_verified", "unverified", "contradicted"
STATUSES = (VERIFIED, PARTIAL, UNVERIFIED, CONTRADICTED)
LABELS = {VERIFIED: "VERIFIED", PARTIAL: "PARTIALLY VERIFIED", UNVERIFIED: "UNVERIFIED", CONTRADICTED: "CONTRADICTED"}
RANK = {VERIFIED: 0, PARTIAL: 1, UNVERIFIED: 2, CONTRADICTED: 3}

LEGAL_CHECKS = {
    "provision": "the legal provision itself (official legal text)",
    "scope": "scope and conditions of the provision",
    "validity": "current validity (in force, effective date, amendments, expiry)",
    "exceptions": "exceptions and overriding rules",
    "product_classification": "the product's classification (customs item or legal definition)",
}
# Checks that must rest on official legal text; classification may also rest on official dataset records.
LEGAL_TEXT_REQUIRED = ("provision", "scope", "validity", "exceptions")

NEGATIVE_RE = re.compile(
    r"\bexempt|\bexemption|not required|no requirement|does not apply|doesn't apply|not subject to|"
    r"no licen[cs]e|without (?:a )?licen[cs]e|no permit|no approval|not regulated|no restriction|"
    r"פטור|פטורה|פטורים|אינו טעון|אינה טעונה|אינם טעונים|אין צורך|אין חובה|לא נדרש|אינו נדרש|אינה נדרשת|"
    r"לא חל|אינו חל|אינה חלה|אין דרישה|ללא רישיון|ללא רשיון|ללא אישור|לא חייב|אינו חייב",
    re.IGNORECASE,
)


def is_negative_claim(*texts: str) -> bool:
    """Exemption / absence-of-requirement claims, which need explicit legal text to be established."""
    return any(NEGATIVE_RE.search(t or "") for t in texts)


def source_ok(src) -> bool:
    return bool(src.official and src.verified and src.excerpt_verified)


def source_dimension(sources: list) -> tuple[str, list[str]]:
    """All cited primary sources official, retrieved and excerpt-matched -> VERIFIED; some -> PARTIAL."""
    if not sources:
        return UNVERIFIED, ["No primary source cited."]
    ok = [s for s in sources if source_ok(s)]
    notes = [f"{s.url}: {s.verification_note}" for s in sources if not source_ok(s)]
    if ok and len(ok) == len(sources):
        return VERIFIED, []
    if ok:
        return PARTIAL, notes
    return UNVERIFIED, notes


def _check_evidence_ok(key: str, check) -> tuple[bool, str]:
    ok = [s for s in check.evidence if source_ok(s)]
    if key in LEGAL_TEXT_REQUIRED:
        ok = [s for s in ok if s.kind == "legal_document"]
        if not ok:
            return False, "needs a retrieved official legal text with a matching quote"
    elif not ok:
        return False, "needs retrieved official evidence (legal text or dataset record) with a matching quote"
    return True, ""


def legal_dimension(checks, negative: bool, verify: Callable | None = None) -> tuple[str, list[str]]:
    """Conservative legal-applicability status from the model's per-aspect checks, re-verified here."""
    notes: list[str] = []
    verified, contradicted = {}, []
    for key, label in LEGAL_CHECKS.items():
        check = getattr(checks, key)
        if verify is not None:
            for src in check.evidence:
                verify(src, require_excerpt=True)
        ok, why = _check_evidence_ok(key, check)
        status = check.status
        if status == "contradicted":
            if ok:
                contradicted.append(key)
                check.verified = True
                check.verification_note = "Contradicting evidence verified."
            else:
                check.verified = False
                check.verification_note = f"Claimed contradiction not verified: {why}."
                notes.append(f"{key}: claimed contradiction not verified ({why}).")
            continue
        if status in ("checked", "not_applicable") and ok:
            verified[key] = True
            check.verified = True
            check.verification_note = "Checked against verified official evidence."
        else:
            check.verified = False
            check.verification_note = ("Not checked." if status == "not_checked" else f"Not verified: {why}.")
            notes.append(f"{label}: " + ("not checked" if status == "not_checked" else f"not verified ({why})"))
    if contradicted:
        return CONTRADICTED, [f"Verified official evidence contradicts the conclusion ({', '.join(contradicted)})."] + notes
    if len(verified) == len(LEGAL_CHECKS):
        return VERIFIED, []
    if verified.get("provision"):
        status = PARTIAL
    else:
        status = UNVERIFIED
        notes.insert(0, "The legal provision itself was not verified against retrieved official legal text.")
    if negative:
        notes.insert(0, "Negative claim (exemption / no requirement): NOT established. An exemption needs explicit, "
                        "verified legal text covering scope, validity and exceptions; zero dataset matches or an "
                        "unfound provision are not proof.")
    return status, notes


def business_dimension(adv, legal: str, verify: Callable | None = None) -> tuple[str, list[str]]:
    """Advantage over competing importers, not a rule available to everyone."""
    if verify is not None:
        for src in adv.evidence + adv.competitor_evidence:
            verify(src, require_excerpt=True)
    evidence_ok = [s for s in adv.evidence if s.verified and s.excerpt_verified]
    competitor_ok = [s for s in adv.competitor_evidence if s.verified and s.excerpt_verified]
    notes = []
    if adv.status == "contradicted" and (evidence_ok or competitor_ok):
        return CONTRADICTED, ["Verified evidence shows the advantage is not real (e.g. competitors use the same rule)."]
    if not (adv.claim or adv.differentiator):
        return UNVERIFIED, ["No specific competitive advantage was claimed or demonstrated."]
    if adv.generally_available is True:
        return UNVERIFIED, ["The rule is generally available to all importers: it is not, by itself, an advantage "
                            "over competing importers."]
    if adv.generally_available is None:
        notes.append("Not established whether the rule is available to every competing importer.")
    if not evidence_ok:
        notes.append("No retrieved, quote-matched evidence for the differentiator.")
    if not competitor_ok:
        notes.append("No retrieved, quote-matched evidence about competing importers.")
    if legal != VERIFIED:
        notes.append("The underlying legal conclusion is not fully verified.")
    if not notes:
        return VERIFIED, []
    if evidence_ok or competitor_ok:
        return PARTIAL, notes
    return UNVERIFIED, notes


def counters(findings: list[dict]) -> dict:
    """Separate run-level counters (the UI must never merge them into one 'verified' number)."""
    legal = [f for f in findings if f.get("claim_type") == "legal_conclusion"]
    business = [f for f in findings if f.get("claim_type") == "business_advantage"]
    negative = [f for f in findings if f.get("negative_claim")]
    return {
        "findings": len(findings),
        "source_verified": sum(1 for f in findings if f.get("source_status") == VERIFIED),
        "legal_conclusions": len(legal),
        "legal_verified": sum(1 for f in legal if f.get("legal_status") == VERIFIED),
        "business_claims": len(business),
        "business_verified": sum(1 for f in business if f.get("business_status") == VERIFIED),
        "negative_claims": len(negative),
        "negative_unresolved": sum(1 for f in negative if f.get("legal_status") != VERIFIED),
        "contradicted": sum(1 for f in findings if CONTRADICTED in (f.get("legal_status"), f.get("business_status"))),
    }
