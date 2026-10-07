"""Configurable critical-evidence checklist for legal investigations.

Before a run ends, critical questions that are still open are listed for the model ONCE (a bounded
completion review, not an unconditional loop) together with every strategy already attempted, so it can:
1. identify the missing legal evidence; 2. search already-downloaded documents and the local official data;
3. run targeted Serper searches; 4. try an accessible official alternative (never bypassing 401/403/robots);
5. otherwise mark the question UNRESOLVED with the reason. Questions still open when the run stops are
marked unresolved by the system. Resolving a question requires a reference to retrieved evidence.

Configuration (first match wins): the agent's `checklist` argument, the CRITICAL_EVIDENCE_CHECKLIST setting
(a JSON list, or a path to a JSON file) of {"id", "question"} objects, else DEFAULT_CHECKLIST.
An empty list disables the checklist and the completion review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import get_setting

DEFAULT_CHECKLIST = [
    {"id": "governing_legal_text",
     "question": "Was the official text of the governing law / regulation / order, including the relevant "
                 "schedule or section, retrieved and quoted?"},
    {"id": "product_classification",
     "question": "Is the product's classification (customs tariff item, legal definition) confirmed against an "
                 "official source?"},
    {"id": "scope_and_exceptions",
     "question": "Were the provision's scope, conditions and exceptions (including overriding rules and mandatory "
                 "standards) checked in the legal text?"},
    {"id": "validity",
     "question": "Is the provision currently in force (effective date, amendments, expiry) per an official source?"},
    {"id": "competitive_advantage",
     "question": "Does the conclusion give an advantage over competing importers, rather than describe a rule "
                 "available to everyone?"},
]
STATUSES = ("open", "resolved", "unresolved", "not_applicable")
STRATEGIES = ("document_search", "local_data", "web_search", "official_alternative", "dataset_api", "other")
MAX_ITEMS = 20
EVIDENCE_REF_RE = re.compile(r"^(E\d+|F\d+|doc-[0-9a-f]{8,}(:\d+)?|https?://\S+)$")


def load(spec=None) -> list[dict]:
    """The configured checklist items ({"id", "question"})."""
    if spec is None:
        raw = (get_setting("CRITICAL_EVIDENCE_CHECKLIST") or "").strip()
        if raw:
            if not raw.startswith("["):
                raw = Path(raw).read_text("utf-8")
            spec = json.loads(raw)
        else:
            spec = DEFAULT_CHECKLIST
    items = []
    for i, entry in enumerate(spec[:MAX_ITEMS]):
        if isinstance(entry, str):
            entry = {"id": f"q{i + 1}", "question": entry}
        if not isinstance(entry, dict) or not str(entry.get("question", "")).strip():
            continue
        item_id = re.sub(r"[^a-z0-9_]+", "_", str(entry.get("id") or f"q{i + 1}").lower()).strip("_") or f"q{i + 1}"
        items.append({"id": item_id, "question": str(entry["question"]).strip()[:500]})
    return items


class Checklist:
    def __init__(self, items: list[dict] | None = None):
        self.items: dict[str, dict] = {}
        for it in items or []:
            self.items[it["id"]] = {"id": it["id"], "question": it["question"], "status": "open", "note": "",
                                    "evidence": [], "attempts": [], "updated_by": "", "step": None}

    # -------------------------------------------------------------- state
    def to_state(self) -> list[dict]:
        return list(self.items.values())

    @classmethod
    def from_state(cls, state: list[dict] | None) -> "Checklist":
        c = cls()
        for it in state or []:
            c.items[it["id"]] = dict(it)
        return c

    @property
    def enabled(self) -> bool:
        return bool(self.items)

    def open_items(self) -> list[dict]:
        return [it for it in self.items.values() if it["status"] == "open"]

    # ------------------------------------------------------------ updates
    def update(self, entries: list, evidence_exists, step: int) -> list[dict]:
        """Apply the model's updates. 'resolved' needs at least one existing evidence reference."""
        results = []
        for e in entries[:MAX_ITEMS]:
            if not isinstance(e, dict):
                continue
            item = self.items.get(str(e.get("id", "")).strip())
            if item is None:
                results.append({"id": e.get("id"), "error": f"Unknown checklist id. Known: {sorted(self.items)}"})
                continue
            status = str(e.get("status", item["status"])).strip().lower()
            if status not in STATUSES:
                results.append({"id": item["id"], "error": f"status must be one of {STATUSES}"})
                continue
            refs = [str(r).strip() for r in (e.get("evidence") or []) if str(r).strip()]
            valid = [r for r in refs if EVIDENCE_REF_RE.match(r) and evidence_exists(r)]
            strategy = str(e.get("strategy_attempted", "")).strip()
            if strategy:
                item["attempts"].append({"step": step, "strategy": strategy[:300]})
            note = str(e.get("note", "")).strip()[:1000]
            if status == "resolved" and not valid:
                results.append({"id": item["id"], "status": item["status"],
                                "error": "Not marked resolved: cite at least one retrieved evidence reference "
                                         "(evidence id E#, finding id F#, document id with page, or a retrieved URL)."})
                continue
            if status in ("unresolved", "not_applicable") and not note:
                results.append({"id": item["id"], "status": item["status"],
                                "error": f"A note explaining why the question is {status} is required."})
                continue
            item.update(status=status, note=note or item["note"], updated_by="model", step=step)
            item["evidence"] = sorted(set(item["evidence"]) | set(valid))
            results.append({"id": item["id"], "status": status, "evidence": item["evidence"]})
        return results

    def close_open(self, reason: str) -> int:
        """Mark every still-open question unresolved (the run is ending)."""
        n = 0
        for it in self.open_items():
            it.update(status="unresolved", note=reason, updated_by="system")
            n += 1
        return n

    def summary(self) -> dict:
        counts = {s: 0 for s in STATUSES}
        for it in self.items.values():
            counts[it["status"]] += 1
        return counts


def near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    """True when two queries have (almost) the same words, ignoring order and case."""
    from evidence import normalize_text

    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold
