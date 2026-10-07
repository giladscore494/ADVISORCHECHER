"""Controlled tool-calling research loop.

The model reasons, calls search_web / fetch_url / the live data.gov.il dataset tools / the local
government snapshot tools / update_candidates / record_findings, and we execute those tools under
hard limits (steps, searches, fetches, API calls, local queries, duplicates).
When the model says it is done, or a limit is hit, we ask for the final JSON report, validate it,
and retry once if it is malformed.

Durability: every model response, every completed tool call, every phase change, every caught error
and every finalization attempt is checkpointed through `checkpointer` (see research_store.Checkpointer)
together with the full resumable state (conversation, evidence, provenance, candidates, findings,
counters, token usage). A run interrupted at any point can be resumed with `ResearchAgent.resume()`,
and if no valid final report can be produced a clearly labelled partial report is built from the
saved evidence (partial_report.py) instead of losing the run.
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import unquote, urlparse, urlunparse

import datagov
import local_data
import partial_report
from evidence import excerpt_found, extract_terms
from fetcher import fetch_url
from llm import LLMError, parse_tool_arguments
from models import ResearchResult, SourceRef, parse_research_result, rank_opportunities
from prompts import (
    CUSTOM_INSTRUCTIONS_TEMPLATE,
    FINALIZE_PROMPT,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from search import SearchError, is_primary_source, search_web

PHASES = {
    "mapping": "Mapping regulatory environment",
    "searching": "Searching for laws and regulations",
    "reading": "Reading sources",
    "generating": "Generating candidate opportunities",
    "red_team": "Red-teaming candidates",
    "market": "Validating market and competitors",
    "ranking": "Ranking surviving opportunities",
}

_PHASE_PARAMS = {
    "phase": {"type": "string", "enum": list(PHASES), "description": "Current research phase."},
    "purpose": {"type": "string", "description": "One short public sentence describing this action for the progress display."},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Google web search (Israel region). Returns title, url, snippet, position, primary_source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, Hebrew or English."},
                    "num_results": {"type": "integer", "minimum": 1, "maximum": 20, "default": 10},
                    **_PHASE_PARAMS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Download a web page or PDF and return its readable text (truncated if long).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, **_PHASE_PARAMS},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_government_datasets",
            "description": "Search official Israeli government datasets on data.gov.il (CKAN package_search). "
                           "Returns dataset ids, titles, publishers, formats and a relevance hint. Use Hebrew and English terms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "rows": {"type": "integer", "minimum": 1, "maximum": datagov.MAX_SEARCH_ROWS, "default": 10},
                    "start": {"type": "integer", "minimum": 0, "default": 0, "description": "Pagination offset."},
                    **_PHASE_PARAMS,
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_government_dataset",
            "description": "Get a data.gov.il dataset's metadata (description, publisher, update dates, license, resources "
                           "with datastore_active). Rejects datasets unrelated to the research topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "Dataset id or name from search_government_datasets."},
                    "query": {"type": "string", "description": "What you are looking for in this dataset (Hebrew/English terms)."},
                    **_PHASE_PARAMS,
                },
                "required": ["dataset_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_government_resource",
            "description": "Read records from a data.gov.il resource relevant to `query`. Checks relevance first; uses "
                           "datastore_search when datastore_active, otherwise the official download URL. Returns bounded, "
                           "query-matched records or passages plus provenance (dataset/resource ids, publisher, dates).",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "query": {"type": "string", "description": "Terms that relevant records must contain (Hebrew/English)."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": datagov.MAX_RECORDS, "default": 20},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "filters": {"type": "object", "description": "Optional exact-match column filters, e.g. {\"סוג\": \"נגרר\"}."},
                    **_PHASE_PARAMS,
                },
                "required": ["resource_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_local_government_datasets",
            "description": "List the complete, validated official data.gov.il snapshots stored with this app (customs "
                           "tariff, Free Import Order requirements, additional import orders, official standards, "
                           "standards declarations): fields, row counts, snapshot date. Free; no network.",
            "parameters": {"type": "object", "properties": {**_PHASE_PARAMS}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_government_records",
            "description": "Focused query over a local official snapshot (indexed; never returns whole datasets). "
                           "Supports exact customs classification codes (e.g. 8703.23.00.00/2, 87032300, 8703) with "
                           "parent heading/chapter and child matches, Hebrew/English terms across all fields, and "
                           "exact-match field filters. Returns matching records with exact field values, record ids "
                           "and provenance. Zero matches is NOT proof of an exemption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string",
                                "description": "Dataset key from list_local_government_datasets, or 'all'."},
                    "query": {"type": "string", "description": "Customs code or Hebrew/English terms."},
                    "filters": {"type": "object", "description": "Optional exact-match field filters, e.g. "
                                                                 "{\"ConfirmationType\": \"...\"}."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": local_data.MAX_LIMIT,
                              "default": local_data.DEFAULT_LIMIT},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    **_PHASE_PARAMS,
                },
                "required": ["dataset", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_government_record",
            "description": "Get one record (all exact field values + provenance) from a local official snapshot.",
            "parameters": {
                "type": "object",
                "properties": {"dataset": {"type": "string"}, "record_id": {"type": "string"}, **_PHASE_PARAMS},
                "required": ["dataset", "record_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_government_snapshot_status",
            "description": "Snapshot date, last official verification and freshness of every local dataset. Use the "
                           "live data.gov.il tools when freshness matters or a record is missing locally.",
            "parameters": {"type": "object", "properties": {**_PHASE_PARAMS}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_findings",
            "description": "Save findings as you establish them (they are checkpointed and survive failures) and "
                           "open questions. Each finding needs the source URL (or dataset resource_id) it rests on and "
                           "an exact excerpt from the retrieved content; the system checks the excerpt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string"},
                                "source_url": {"type": "string"},
                                "excerpt": {"type": "string"},
                                "resource_id": {"type": "string"},
                            },
                            "required": ["statement", "source_url", "excerpt"],
                        },
                    },
                    "open_questions": {"type": "array", "items": {"type": "string"}},
                    **_PHASE_PARAMS,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_candidates",
            "description": "Record or update candidate opportunities in the research funnel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "mechanism": {"type": "string"},
                                "status": {"type": "string", "enum": ["candidate", "rejected", "surviving"]},
                                "reason": {"type": "string", "description": "Short reason for the status."},
                            },
                            "required": ["name", "status"],
                        },
                    }
                },
                "required": ["candidates"],
            },
        },
    },
]

MAX_NO_PROGRESS_ROUNDS = 3
MAX_INSTRUCTIONS_CHARS = 5000
NO_RESULT_MESSAGE = "No sufficiently strong opportunity found."
ACCESS_DENIED_STATUSES = (401, 403)
BLOCKED_SOURCE_GUIDANCE = (
    "Access to this official page was denied. Do NOT try to bypass it (no other user agents, proxies, "
    "cached or archived copies of the blocked page). Look for a publicly accessible official alternative instead: "
    "the same document as a PDF on gov.il, the law or regulation text on main.knesset.gov.il, the regulation as "
    "published in Reshumot (רשומות / קובץ תקנות), or a dataset on data.gov.il "
    "(use search_government_datasets). "
    "If no official copy can be retrieved, the finding that depends on this source is UNVERIFIED."
)
DATASET_SOURCE_TYPES = {"csv", "xlsx", "json", "ckan_resource", "ckan_dataset", "ckan_datastore"}
# CKAN actions that return discovery metadata, never dataset contents: not evidence for a claim.
CKAN_METADATA_ACTIONS = ("package_search", "package_show", "resource_show")


def is_ckan_metadata_url(url: str) -> bool:
    path = urlparse(url).path
    return any(path.endswith(f"/action/{action}") for action in CKAN_METADATA_ACTIONS)


@dataclass
class Limits:
    max_steps: int = 25
    max_searches: int = 30
    max_fetches: int = 20
    max_opportunities: int = 5
    max_api_calls: int = 40  # data.gov.il CKAN API calls per run
    max_local_queries: int = 60  # local snapshot queries per run (cheap, but bounded)


@dataclass
class RunResult:
    domain: str
    result: ResearchResult | None = None
    error: str = ""
    stop_reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    status: str = ""  # completed | failed | interrupted
    partial_report: dict | None = None


STATE_VERSION = 1
MAX_STORED_EVENTS = 300
MAX_LOCAL_RESULT_CHARS = 30000
MAX_FINDINGS = 100
MAX_OPEN_QUESTIONS = 50
INTERRUPTED_TOOL_RESULT = ("This tool call was interrupted (server restart or failure) before it completed. "
                           "Call it again if you still need it.")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NullCheckpointer:
    """Used when no durable store is configured (tests, scripts)."""

    failures = 0

    def save(self, label, phase, state, summary, status="running", error=""):
        return None

    def touch(self, label=None, phase=None):
        return None

    def finish(self, status, error="", final_report=None, partial_report=None):
        return None


def normalize_query(q: str) -> str:
    return " ".join(q.lower().split())


def normalize_url(url: str) -> str:
    p = urlparse(url.strip())
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", p.query, ""))


class ResearchAgent:
    def __init__(
        self,
        llm,
        limits: Limits | None = None,
        on_event: Callable[[dict], None] | None = None,
        search_fn: Callable[..., list[dict]] = search_web,
        fetch_fn: Callable[[str], Any] = fetch_url,
        ckan_client: "datagov.CkanClient | None" = None,
        gov_data: "local_data.GovernmentData | None" = None,
        checkpointer=None,
        run_id: str = "",
    ):
        self.llm = llm
        self.limits = limits or Limits()
        self.gov_data = gov_data
        self.checkpointer = checkpointer or NullCheckpointer()
        self.run_id = run_id
        self.ckan = ckan_client or datagov.CkanClient(max_calls=self.limits.max_api_calls)
        self.domain = ""
        # Retrieved content (normalized URL -> text) used to confirm quoted excerpts.
        self.retrieved_text: dict[str, str] = {}
        # resource_id -> provenance, status and the records/passages that were returned.
        self.dataset_evidence: dict[str, dict] = {}
        self.seen_dataset_requests: set[str] = set()
        self.on_event = on_event or (lambda e: None)
        self.search_fn = search_fn
        self.fetch_fn = fetch_fn

        self.seen_queries: set[str] = set()
        self.seen_urls: set[str] = set()
        # normalized URL -> {"ok", "error", "http_status", "official", "source_type"}
        self.fetch_status: dict[str, dict] = {}
        self.url_titles: dict[str, str] = {}
        self.search_count = 0
        self.fetch_count = 0
        self.local_query_count = 0
        self.phase = "mapping"
        self.started = time.monotonic()
        self.elapsed_before = 0.0  # time spent before a resume
        self.instructions = ""
        self.messages: list[dict] = []
        self.step = 0  # last completed step
        self.no_progress_rounds = 0
        self.loop_done = False
        self.finalize_index: int | None = None
        self.final_raw = ""
        self.findings: list[dict] = []
        self.open_questions: list[str] = []
        self.trace: dict[str, Any] = {
            "searches": [], "fetches": [], "model_calls": [], "tool_calls": [],
            "candidates": {}, "events": [], "warnings": [],
            "dataset_searches": [], "dataset_inspections": [], "dataset_reads": [], "api_calls": [],
            "local_queries": [], "api_errors": [], "resumes": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "model_calls": 0},
        }

    # ---------------------------------------------------------------- events
    def _emit(self, message: str, kind: str = "info") -> None:
        event = {
            "kind": kind,
            "message": message,
            "phase": PHASES.get(self.phase, self.phase),
            "searches": self.search_count,
            "fetches": self.fetch_count,
            "candidates": len(self.trace["candidates"]),
            "api_calls": self.ckan.calls,
            "local_queries": self.local_query_count,
            "elapsed_s": self._elapsed(),
            "at": iso_now(),
        }
        self.trace["events"].append(event)
        self.on_event(event)

    def _elapsed(self) -> float:
        return round(self.elapsed_before + time.monotonic() - self.started, 1)

    def _set_phase(self, args: dict) -> None:
        if args.get("phase") in PHASES and args["phase"] != self.phase:
            previous = self.phase
            self.phase = args["phase"]
            self._checkpoint(f"phase: {PHASES[previous]} -> {PHASES[self.phase]}")

    def _api_error(self, source: str, error: str) -> None:
        self.trace["api_errors"].append({"at": iso_now(), "source": source, "error": str(error)[:1000]})

    # ------------------------------------------------------------ durability
    def export_state(self) -> dict:
        """Everything needed to resume this run or to build a partial report (JSON-serializable)."""
        trace = dict(self.trace)
        trace["events"] = trace["events"][-MAX_STORED_EVENTS:]
        trace["api_calls"] = list(self.ckan.log)
        trace["elapsed_s"] = self._elapsed()
        return {
            "state_version": STATE_VERSION,
            "run_id": self.run_id,
            "domain": self.domain,
            "instructions": self.instructions,
            "limits": asdict(self.limits),
            "provider": getattr(self.llm, "provider", ""),
            "model": getattr(self.llm, "model", ""),
            "phase": self.phase,
            "step": self.step,
            "no_progress_rounds": self.no_progress_rounds,
            "loop_done": self.loop_done,
            "finalize_index": self.finalize_index,
            "final_raw": self.final_raw[:50000],
            "messages": self.messages,
            "search_count": self.search_count,
            "fetch_count": self.fetch_count,
            "local_query_count": self.local_query_count,
            "ckan_calls": self.ckan.calls,
            "seen_queries": sorted(self.seen_queries),
            "seen_urls": sorted(self.seen_urls),
            "seen_dataset_requests": sorted(self.seen_dataset_requests),
            "fetch_status": self.fetch_status,
            "url_titles": self.url_titles,
            "retrieved_text": self.retrieved_text,
            "dataset_evidence": self.dataset_evidence,
            "findings": self.findings,
            "open_questions": self.open_questions,
            "trace": trace,
            "elapsed_s": self._elapsed(),
            "checkpoint_at": iso_now(),
        }

    def restore_state(self, state: dict) -> None:
        if state.get("state_version") != STATE_VERSION:
            raise ValueError(f"Unsupported checkpoint state version {state.get('state_version')}")
        self.run_id = state.get("run_id") or self.run_id
        self.domain = state["domain"]
        self.instructions = state.get("instructions", "")
        self.phase = state.get("phase", "mapping")
        self.step = int(state.get("step", 0))
        self.no_progress_rounds = int(state.get("no_progress_rounds", 0))
        self.loop_done = bool(state.get("loop_done"))
        self.finalize_index = state.get("finalize_index")
        self.final_raw = state.get("final_raw", "")
        self.messages = list(state.get("messages", []))
        self.search_count = int(state.get("search_count", 0))
        self.fetch_count = int(state.get("fetch_count", 0))
        self.local_query_count = int(state.get("local_query_count", 0))
        self.ckan.calls = int(state.get("ckan_calls", 0))
        self.seen_queries = set(state.get("seen_queries", []))
        self.seen_urls = set(state.get("seen_urls", []))
        self.seen_dataset_requests = set(state.get("seen_dataset_requests", []))
        self.fetch_status = dict(state.get("fetch_status", {}))
        self.url_titles = dict(state.get("url_titles", {}))
        self.retrieved_text = dict(state.get("retrieved_text", {}))
        self.dataset_evidence = dict(state.get("dataset_evidence", {}))
        self.findings = list(state.get("findings", []))
        self.open_questions = list(state.get("open_questions", []))
        trace = state.get("trace") or {}
        for key, value in trace.items():
            self.trace[key] = value
        for key in ("local_queries", "api_errors", "resumes"):
            self.trace.setdefault(key, [])
        self.trace.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                                              "model_calls": 0})
        self.ckan.log = list(trace.get("api_calls", []))
        self.elapsed_before = float(state.get("elapsed_s", 0.0))
        self.started = time.monotonic()

    def _summary(self, label: str) -> dict:
        statuses = [c.get("status") for c in self.trace["candidates"].values()]
        return {
            "label": label, "step": self.step, "phase": self.phase, "searches": self.search_count,
            "fetches": self.fetch_count, "api_calls": self.ckan.calls, "local_queries": self.local_query_count,
            "candidates": len(statuses), "surviving": statuses.count("surviving"),
            "rejected": statuses.count("rejected"), "findings": len(self.findings),
            "verified_findings": sum(1 for f in self.findings if f.get("verified")),
            "token_usage": dict(self.trace["token_usage"]), "api_errors": len(self.trace["api_errors"]),
            "elapsed_s": self._elapsed(),
        }

    def _checkpoint(self, label: str, status: str = "running", error: str = "") -> None:
        try:
            self.checkpointer.save(label, self.phase, self.export_state(), self._summary(label),
                                   status=status, error=error)
        except Exception as exc:  # noqa: BLE001 - durability problems must not kill the research
            self.trace["warnings"].append(f"Checkpoint failed ({label}): {exc}")

    def _touch(self, label: str) -> None:
        try:
            self.checkpointer.touch(label, self.phase)
        except Exception as exc:  # noqa: BLE001
            self.trace["warnings"].append(f"Heartbeat failed: {exc}")

    # ----------------------------------------------------------------- tools
    def _budget(self) -> dict:
        return {
            "searches_left": self.limits.max_searches - self.search_count,
            "fetches_left": self.limits.max_fetches - self.fetch_count,
        }

    def _tool_search(self, args: dict) -> tuple[dict, bool]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}, False
        key = normalize_query(query)
        if key in self.seen_queries:
            return {"error": "Duplicate search rejected: this exact query was already run. Use the earlier results or a different query."}, False
        if self.search_count >= self.limits.max_searches:
            return {"error": "Search limit reached. Work with the evidence you already have."}, False

        return self._run_search(query, args.get("purpose", ""), args.get("num_results", 10))

    def _run_search(self, query: str, purpose: str, num_results=10) -> tuple[dict, bool]:
        """Execute a search that already passed duplicate and budget checks."""
        self.seen_queries.add(normalize_query(query))
        self.search_count += 1
        self._emit(purpose or f"Searching: {query}", "search")
        entry: dict[str, Any] = {"query": query, "purpose": purpose, "results": [], "error": ""}
        self.trace["searches"].append(entry)
        try:
            results = self.search_fn(query, num_results=num_results)
        except SearchError as exc:
            entry["error"] = str(exc)
            self._api_error("serper", exc)
            self._emit(f"Search failed: {exc}", "error")
            return {"error": str(exc), **self._budget()}, True
        entry["results"] = [{"title": r["title"], "url": r["url"], "primary": r["primary_source"]} for r in results]
        for r in results:
            if r.get("title"):
                self.url_titles.setdefault(normalize_url(r["url"]), r["title"])
        if not results:
            return {"results": [], "note": "No results. Try different wording or language.", **self._budget()}, True
        return {"results": results, **self._budget()}, True

    def _tool_fetch(self, args: dict) -> tuple[dict, bool]:
        url = str(args.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}, False
        key = normalize_url(url)
        if key in self.seen_urls:
            return {"error": "Duplicate fetch rejected: this URL was already read earlier in this run."}, False
        if self.fetch_count >= self.limits.max_fetches:
            return {"error": "Page fetch limit reached. Work with the sources you already read."}, False

        self.seen_urls.add(key)
        self.fetch_count += 1
        self._emit(args.get("purpose") or f"Reading {url}", "fetch")
        res = self.fetch_fn(url)
        official = is_primary_source(res.final_url or url)
        self.trace["fetches"].append({
            "url": url, "ok": res.ok, "source_type": res.source_type, "title": res.title,
            "chars": len(res.text), "truncated": res.truncated, "error": res.error,
            "primary": official, "http_status": res.http_status,
            "resource_url": getattr(res, "resource_url", ""),
        })
        status = {"ok": res.ok, "error": res.error, "http_status": res.http_status,
                  "official": official, "source_type": res.source_type}
        self.fetch_status[key] = status
        if res.final_url:
            self.fetch_status.setdefault(normalize_url(res.final_url), status)

        if not res.ok:
            restricted = getattr(res, "access_restricted", False) or res.http_status in ACCESS_DENIED_STATUSES
            if official:
                self._emit(f"Official source unavailable — evidence not verified: {url} ({res.error})", "error")
            else:
                self._emit(f"Could not read {url}: {res.error}", "error")
            result = {"url": url, "ok": False, "error": res.error, "http_status": res.http_status}
            if official and restricted:
                result["guidance"] = BLOCKED_SOURCE_GUIDANCE
                alternatives = self._search_alternatives(url)
                if alternatives is not None:
                    result["alternative_search"] = alternatives
            return {**result, **self._budget()}, True

        self.retrieved_text[key] = res.text
        if res.final_url:
            self.retrieved_text.setdefault(normalize_url(res.final_url), res.text)
        result = {
            "url": url, "final_url": res.final_url, "ok": True, "source_type": res.source_type,
            "primary_source": official, "title": res.title,
            "truncated": res.truncated, "text": res.text,
        }
        if getattr(res, "resource_url", ""):
            result["resource_url"] = res.resource_url
            result["metadata"] = res.metadata
        return {**result, **self._budget()}, True

    def _alternative_query(self, url: str) -> str:
        """Build a search for an accessible official copy of a blocked page, from its title or URL slug."""
        basis = self.url_titles.get(normalize_url(url), "")
        if not basis:
            segments = [unquote(p) for p in urlparse(url).path.split("/") if p]
            slug = segments[-1] if segments else ""
            slug = re.sub(r"\.(aspx?|html?|php)$", "", slug, flags=re.IGNORECASE)
            basis = re.sub(r"[-_+]+", " ", slug).strip()
        if len(basis) < 4 or basis.isdigit():
            return ""
        return f"{basis} filetype:pdf site:gov.il"

    def _search_alternatives(self, url: str) -> dict | None:
        """On HTTP 401/403 from an official site, run one search (within the normal search budget)
        for a publicly accessible official copy. Never retries or bypasses the blocked URL."""
        query = self._alternative_query(url)
        if not query or normalize_query(query) in self.seen_queries:
            return None
        if self.search_count >= self.limits.max_searches:
            return {"note": "Search budget exhausted; could not look for an alternative official copy."}
        self.trace["warnings"].append(f"Official source blocked ({url}); searched for an accessible official alternative.")
        result, _ = self._run_search(query, "Looking for an accessible official copy of a blocked gov.il page")
        blocked = normalize_url(url)
        if "results" in result:
            result["results"] = [r for r in result["results"] if normalize_url(r["url"]) != blocked]
        return {"query": query, **result}

    # ------------------------------------------------------- government data
    def _dataset_request_key(self, tool: str, args: dict) -> str:
        relevant = {k: v for k, v in args.items() if k not in ("phase", "purpose")}
        return tool + ":" + json.dumps(relevant, sort_keys=True, ensure_ascii=False).lower()

    def _duplicate_dataset_request(self, tool: str, args: dict) -> bool:
        key = self._dataset_request_key(tool, args)
        if key in self.seen_dataset_requests:
            return True
        self.seen_dataset_requests.add(key)
        return False

    def _ckan_failure(self, exc: "datagov.CkanError", what: str) -> dict:
        result = {"ok": False, "error": f"{what}: {exc}", "error_kind": exc.kind, "http_status": exc.http_status}
        if exc.kind == "blocked":
            result["guidance"] = BLOCKED_SOURCE_GUIDANCE
        self._api_error("data.gov.il", f"{what}: {exc}")
        self._emit(f"Official source unavailable — evidence not verified ({what}: {exc})", "error")
        return result

    def _tool_dataset_search(self, args: dict) -> tuple[dict, bool]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}, False
        if self._duplicate_dataset_request("search_government_datasets", args):
            return {"error": "Duplicate dataset search rejected: already run with the same arguments."}, False
        self._emit(args.get("purpose") or f"Searching official Israeli government datasets: {query}", "dataset")
        entry = {"query": query, "results": [], "error": ""}
        self.trace["dataset_searches"].append(entry)
        try:
            found = self.ckan.package_search(query, rows=args.get("rows", 10), start=args.get("start", 0))
        except datagov.CkanError as exc:
            entry["error"] = str(exc)
            return self._ckan_failure(exc, "Dataset search failed"), True
        terms = extract_terms(query, self.domain)
        for d in found["results"]:
            d["likely_relevant"] = datagov.assess_relevance(terms, datagov.metadata_fields(d))["relevant"]
        entry["results"] = [{"id": d["id"], "title": d["title"], "publisher": d["publisher"],
                             "likely_relevant": d["likely_relevant"]} for d in found["results"]]
        found["note"] = ("Discovery only: a search hit is not evidence. Inspect relevant datasets, then read "
                         "records with read_government_resource.")
        return found, True

    def _tool_dataset_inspect(self, args: dict) -> tuple[dict, bool]:
        dataset_id = str(args.get("dataset_id", "")).strip()
        if not dataset_id:
            return {"error": "dataset_id is required"}, False
        if self._duplicate_dataset_request("inspect_government_dataset", args):
            return {"error": "Duplicate dataset inspection rejected: already inspected."}, False
        self._emit(args.get("purpose") or f"Inspecting dataset metadata: {dataset_id}", "dataset")
        try:
            dataset = self.ckan.package_show(dataset_id)
        except datagov.CkanError as exc:
            self.trace["dataset_inspections"].append({"dataset_id": dataset_id, "status": "failed", "error": str(exc)})
            return self._ckan_failure(exc, "Dataset metadata unavailable"), True
        terms = extract_terms(args.get("query", ""), self.domain)
        relevance = datagov.assess_relevance(terms, datagov.metadata_fields(dataset))
        record = {"dataset_id": dataset["id"], "title": dataset["title"], "publisher": dataset["publisher"],
                  "dataset_url": dataset["dataset_url"], "status": "relevant" if relevance["relevant"] else "rejected",
                  "matched_terms": relevance["matched_terms"]}
        self.trace["dataset_inspections"].append(record)
        if not relevance["relevant"]:
            self._emit(f"Dataset rejected: unrelated to research topic — {dataset['title']}", "rejected")
            return {"status": "rejected", "dataset_id": dataset["id"], "title": dataset["title"],
                    "publisher": dataset["publisher"],
                    "reason": "Title, description, publisher and tags do not match the research topic or query. "
                              "Do not use this dataset as evidence; continue searching."}, True
        dataset["status"] = "relevant"
        dataset["metadata_relevance"] = relevance
        dataset["note"] = "Metadata only. Read records with read_government_resource(resource_id, query)."
        return dataset, True

    def _tool_dataset_read(self, args: dict) -> tuple[dict, bool]:
        resource_id = str(args.get("resource_id", "")).strip()
        query = str(args.get("query", "")).strip()
        if not resource_id or not query:
            return {"error": "resource_id and query are required"}, False
        if self._duplicate_dataset_request("read_government_resource", args):
            return {"error": "Duplicate read rejected: these records were already returned."}, False
        self._emit(args.get("purpose") or f"Reading official dataset records: resource {resource_id}", "dataset")
        try:
            out = datagov.read_resource(self.ckan, resource_id, query, self.domain, limit=args.get("limit", 20),
                                        offset=args.get("offset", 0), filters=args.get("filters"))
        except datagov.CkanError as exc:
            self._record_dataset_read(resource_id, {"status": "failed", "error": str(exc), "provenance": {}})
            return self._ckan_failure(exc, "Resource metadata unavailable"), True
        self._record_dataset_read(resource_id, out)
        prov = out["provenance"]
        label = f"{prov.get('dataset_title') or prov.get('dataset_id')} / {prov.get('resource_name') or resource_id}"
        if out["status"] == "rejected":
            self._emit(f"Dataset rejected: unrelated to research topic — {label}", "rejected")
        elif out["status"] == "failed":
            self._emit(f"Official source unavailable — evidence not verified: {label} ({out.get('error', '')})", "error")
            if out.get("access_restricted"):
                out["guidance"] = BLOCKED_SOURCE_GUIDANCE
        elif out["status"] == "no_matching_records":
            self._emit(f"No matching records in {label}", "dataset")
        else:
            self._emit(f"Read official dataset records: {label}", "dataset")
            out["quote_instruction"] = ("To cite this resource, set resource_id and quote an exact record line "
                                        "(or contiguous part of it) as the excerpt.")
        return out, True

    def _record_dataset_read(self, resource_id: str, out: dict) -> None:
        prov = out.get("provenance", {})
        evidence_text = out.get("records") or "\n".join(out.get("passages", []))
        record = {"resource_id": resource_id, "status": out["status"], "error": out.get("error", ""),
                  "reason": out.get("reason", ""), "evidence_text": evidence_text, **prov}
        previous = self.dataset_evidence.get(resource_id)
        if previous and previous["status"] == "ok":
            # Keep earlier successful evidence; append newly returned records (pagination).
            if record["status"] == "ok":
                previous["evidence_text"] += "\n" + evidence_text
        else:
            self.dataset_evidence[resource_id] = record
        self.trace["dataset_reads"].append({k: v for k, v in record.items() if k != "evidence_text"}
                                           | {"records_returned": bool(evidence_text)})
        ok = out["status"] == "ok"
        error = {"rejected": "Dataset rejected: unrelated to the research topic.",
                 "no_matching_records": "No records matching the research query were found.",
                 }.get(out["status"], out.get("error", ""))
        for url in (prov.get("source_url"), prov.get("download_url"), prov.get("metadata_api_url"), prov.get("data_api_url")):
            if not url:
                continue
            key = normalize_url(url)
            if ok:
                self.retrieved_text[key] = self.retrieved_text.get(key, "") + "\n" + evidence_text
            current = self.fetch_status.get(key)
            if current and current.get("ok") and not ok:
                continue
            self.fetch_status[key] = {"ok": ok, "error": "" if ok else error, "http_status": out.get("http_status"),
                                      "official": is_primary_source(url), "source_type": "dataset",
                                      "resource_id": resource_id}

    # ------------------------------------------------------- local snapshots
    def _gov(self) -> "local_data.GovernmentData":
        if self.gov_data is None:
            self.gov_data = local_data.get_default()
        return self.gov_data

    def _local_budget(self) -> dict:
        return {"local_queries_left": self.limits.max_local_queries - self.local_query_count}

    def _local_call(self, tool: str, args: dict, fn) -> tuple[dict, bool]:
        if self._duplicate_dataset_request(tool, args):
            return {"error": "Duplicate local query rejected: already run with the same arguments."}, False
        if self.local_query_count >= self.limits.max_local_queries:
            return {"error": "Local dataset query limit reached. Work with the records you already have."}, False
        self.local_query_count += 1
        entry = {"tool": tool, "args": {k: v for k, v in args.items() if k not in ("phase", "purpose")},
                 "error": "", "total_matches": None, "record_ids": []}
        self.trace["local_queries"].append(entry)
        try:
            out = fn()
        except (local_data.LocalDataError, ValueError, OSError) as exc:
            entry["error"] = str(exc)
            self._api_error("local_snapshot", exc)
            self._emit(f"Local dataset query failed: {exc}", "error")
            return {"error": str(exc), **self._local_budget()}, True
        records = out.get("records") or ([out] if out.get("found") else [])
        entry["total_matches"] = out.get("total_matches", len(records))
        entry["record_ids"] = [f"{r['dataset']}:{r['record_id']}" for r in records][:50]
        prov = out.get("provenance") or {}
        versions = [prov.get("snapshot_version")] if "snapshot_version" in prov else [
            p.get("snapshot_version") for p in prov.values() if isinstance(p, dict)]
        entry["snapshot_versions"] = sorted({v for v in versions if v})
        if records:
            self._record_local_evidence(records, out)
        return {**out, **self._local_budget()}, True

    def _record_local_evidence(self, records: list[dict], out: dict) -> None:
        by_ds: dict[str, list[dict]] = {}
        for r in records:
            by_ds.setdefault(r["dataset"], []).append(r)
        provenance = out.get("provenance") or {}
        for ds, recs in by_ds.items():
            prov = provenance.get(ds) if "dataset" not in provenance else provenance
            prov = prov or self._gov().provenance(ds)
            resource_id = prov.get("resource_id") or recs[0].get("resource_id", "")
            text = "\n".join(r["record_line"] for r in recs)
            previous = self.dataset_evidence.get(resource_id)
            if previous and previous.get("status") == "ok":
                previous["evidence_text"] += "\n" + text
                previous.setdefault("snapshot_version", prov.get("snapshot_version", ""))
            else:
                self.dataset_evidence[resource_id] = {
                    "resource_id": resource_id, "status": "ok", "error": "", "reason": "", "evidence_text": text,
                    "dataset_id": prov.get("dataset_id", ""), "dataset_title": prov.get("dataset_title", ""),
                    "resource_name": prov.get("resource_name", ""), "publisher": prov.get("publisher", ""),
                    "license": prov.get("license", ""), "source_url": prov.get("source_url", ""),
                    "resource_last_modified": prov.get("resource_last_modified", ""),
                    "dataset_last_updated": "", "snapshot_version": prov.get("snapshot_version", ""),
                    "retrieved_at": prov.get("retrieved_at", ""), "source_kind": "local_snapshot",
                }
            url = prov.get("source_url")
            if url:
                key = normalize_url(url)
                self.retrieved_text[key] = self.retrieved_text.get(key, "") + "\n" + text
                self.fetch_status[key] = {"ok": True, "error": "", "http_status": None, "official": is_primary_source(url),
                                          "source_type": "dataset", "resource_id": resource_id}

    @staticmethod
    def _fit(out: dict) -> dict:
        """Keep a local query result within MAX_LOCAL_RESULT_CHARS (drop trailing records, say so)."""
        records = out.get("records")
        if not records:
            return out
        while len(records) > 1 and len(json.dumps(out, ensure_ascii=False)) > MAX_LOCAL_RESULT_CHARS:
            records.pop()
            out["returned"] = len(records)
            out["next_offset"] = out.get("offset", 0) + len(records)
            out["truncated"] = "Result shortened to fit the context budget; use offset to continue."
        return out

    def _tool_local_list(self, args: dict) -> tuple[dict, bool]:
        self._emit(args.get("purpose") or "Listing local official government datasets", "dataset")
        return self._local_call("list_local_government_datasets", args, lambda: self._gov().list_datasets())

    def _tool_local_search(self, args: dict) -> tuple[dict, bool]:
        dataset = str(args.get("dataset", "")).strip() or "all"
        query = str(args.get("query", "")).strip()
        if not query and not args.get("filters"):
            return {"error": "query (or filters) is required"}, False
        self._emit(args.get("purpose") or f"Querying local official dataset {dataset}: {query}", "dataset")

        def run():
            return self._fit(self._gov().search(dataset, query, args.get("filters"),
                                                args.get("limit", local_data.DEFAULT_LIMIT), args.get("offset", 0)))

        return self._local_call("search_local_government_records", args, run)

    def _tool_local_get(self, args: dict) -> tuple[dict, bool]:
        dataset, record_id = str(args.get("dataset", "")).strip(), str(args.get("record_id", "")).strip()
        if not dataset or not record_id:
            return {"error": "dataset and record_id are required"}, False
        self._emit(args.get("purpose") or f"Reading local official record {dataset}:{record_id}", "dataset")
        return self._local_call("get_local_government_record", args,
                                lambda: self._gov().get_record(dataset, record_id))

    def _tool_snapshot_status(self, args: dict) -> tuple[dict, bool]:
        try:
            return self._gov().status(), True
        except (local_data.LocalDataError, ValueError, OSError) as exc:
            return {"error": str(exc)}, True

    # -------------------------------------------------------------- findings
    def _tool_findings(self, args: dict) -> tuple[dict, bool]:
        findings, questions = args.get("findings") or [], args.get("open_questions") or []
        if not isinstance(findings, list) or not isinstance(questions, list):
            return {"error": "findings and open_questions must be lists"}, False
        results = []
        known = {f["statement"] for f in self.findings}
        for f in findings[:20]:
            if not isinstance(f, dict) or not str(f.get("statement", "")).strip():
                continue
            statement = str(f["statement"]).strip()[:2000]
            try:
                src = SourceRef(url=str(f.get("source_url", "")), excerpt=str(f.get("excerpt", ""))[:1000],
                                resource_id=str(f.get("resource_id", "")))
            except ValueError:
                results.append({"statement": statement, "verified": False, "note": "source_url must be http(s)"})
                continue
            self._verify_source(src, require_excerpt=True)
            entry = {"statement": statement, "source_url": src.url, "excerpt": src.excerpt,
                     "resource_id": src.resource_id, "official": bool(src.official),
                     "verified": bool(src.verified and src.official), "kind": src.kind,
                     "verification_note": src.verification_note, "step": self.step + 1, "at": iso_now()}
            if src.kind == "dataset" and src.resource_id in self.dataset_evidence:
                entry["snapshot_version"] = self.dataset_evidence[src.resource_id].get("snapshot_version", "")
            if statement in known:
                self.findings = [entry if x["statement"] == statement else x for x in self.findings]
            elif len(self.findings) < MAX_FINDINGS:
                self.findings.append(entry)
                known.add(statement)
            results.append({"statement": statement[:200], "verified": entry["verified"],
                            "note": entry["verification_note"]})
        for q in questions[:20]:
            q = str(q).strip()[:1000]
            if q and q not in self.open_questions and len(self.open_questions) < MAX_OPEN_QUESTIONS:
                self.open_questions.append(q)
        verified = sum(1 for r in results if r["verified"])
        self._emit(f"Findings recorded: {len(results)} ({verified} verified), open questions: "
                   f"{len(self.open_questions)}", "candidates")
        return {"recorded": len(results), "results": results, "open_questions": len(self.open_questions)}, True

    def _tool_candidates(self, args: dict) -> tuple[dict, bool]:
        items = args.get("candidates")
        if not isinstance(items, list):
            return {"error": "candidates must be a list"}, False
        for c in items:
            if isinstance(c, dict) and c.get("name"):
                self.trace["candidates"][c["name"]] = {
                    "mechanism": c.get("mechanism", self.trace["candidates"].get(c["name"], {}).get("mechanism", "")),
                    "status": c.get("status", "candidate"),
                    "reason": c.get("reason", ""),
                }
        statuses = [c["status"] for c in self.trace["candidates"].values()]
        self._emit(
            f"Candidate funnel: {len(statuses)} total, {statuses.count('rejected')} rejected, "
            f"{statuses.count('surviving')} surviving", "candidates",
        )
        return {"recorded": len(items), "total": len(statuses)}, True

    def _dispatch(self, step: int, name: str, raw_args: str) -> tuple[dict, bool]:
        """Run one tool call. Returns (result_for_model, made_progress)."""
        try:
            args = parse_tool_arguments(raw_args)
        except ValueError as exc:
            result, progress = {"error": str(exc)}, False
        else:
            self._set_phase(args)
            handler = {
                "search_web": self._tool_search,
                "fetch_url": self._tool_fetch,
                "update_candidates": self._tool_candidates,
                "search_government_datasets": self._tool_dataset_search,
                "inspect_government_dataset": self._tool_dataset_inspect,
                "read_government_resource": self._tool_dataset_read,
                "list_local_government_datasets": self._tool_local_list,
                "search_local_government_records": self._tool_local_search,
                "get_local_government_record": self._tool_local_get,
                "get_government_snapshot_status": self._tool_snapshot_status,
                "record_findings": self._tool_findings,
            }.get(name)
            if handler is None:
                result, progress = {"error": f"Unknown tool '{name}'"}, False
            else:
                result, progress = handler(args)
        self.trace["tool_calls"].append({
            "step": step, "tool": name, "args": raw_args[:500],
            "outcome": "ok" if progress and "error" not in result else result.get("error", "")[:200],
        })
        return result, progress

    # ------------------------------------------------------------------ loop
    def _call_llm(self, step, messages: list[dict], **kwargs):
        try:
            resp = self.llm.chat(messages, **kwargs)
        except LLMError as exc:
            self._api_error("model", exc)
            raise
        usage = resp.usage or {}
        totals = self.trace["token_usage"]
        totals["model_calls"] = totals.get("model_calls", 0) + 1
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            totals[k] = totals.get(k, 0) + int(usage.get(k, 0) or 0)
        self.trace["model_calls"].append({
            "step": step, "duration_s": resp.duration_s, "finish_reason": resp.finish_reason,
            "tool_calls": len(resp.tool_calls), "usage": resp.usage,
        })
        return resp

    def build_user_message(self, domain: str, instructions: str = "") -> str:
        lim = self.limits
        content = USER_PROMPT_TEMPLATE.format(
            domain=domain, max_steps=lim.max_steps, max_searches=lim.max_searches,
            max_fetches=lim.max_fetches, max_opportunities=lim.max_opportunities)
        if instructions:
            # Keep the user's text inside its delimiters so it cannot pose as system text.
            safe = re.sub(r"<\s*/?\s*custom_instructions\s*>", "", instructions, flags=re.IGNORECASE)
            content += CUSTOM_INSTRUCTIONS_TEMPLATE.format(instructions=safe)
        return content

    def run(self, domain: str, instructions: str = "") -> RunResult:
        run = RunResult(domain=domain, trace=self.trace, run_id=self.run_id)
        self.domain = domain
        instructions = (instructions or "").strip()
        if len(instructions) > MAX_INSTRUCTIONS_CHARS:
            instructions = instructions[:MAX_INSTRUCTIONS_CHARS]
            self.trace["warnings"].append(f"Custom instructions truncated to {MAX_INSTRUCTIONS_CHARS} characters.")
        self.instructions = instructions
        self.trace["custom_instructions"] = instructions
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_message(domain, instructions)},
        ]
        suffix = " (with custom instructions)" if instructions else ""
        self._emit(f"Starting research on: {domain}{suffix}", "start")
        self._checkpoint("research started")
        return self._drive(run, start_step=1)

    def resume(self, state: dict) -> RunResult:
        """Continue an interrupted run from its last checkpoint (same conversation, budgets and evidence)."""
        self.restore_state(state)
        run = RunResult(domain=self.domain, trace=self.trace, run_id=self.run_id)
        repaired = self._repair_dangling_tool_calls()
        self.trace["resumes"].append({"at": iso_now(), "from_step": self.step, "loop_done": self.loop_done,
                                      "repaired_tool_calls": repaired})
        where = "finalization" if self.loop_done else f"step {self.step + 1}"
        self._emit(f"Resuming research from the last checkpoint ({where}).", "start")
        self._checkpoint(f"resumed at {where}")
        return self._drive(run, start_step=self.step + 1)

    def _repair_dangling_tool_calls(self) -> int:
        """If the process died between a model response and its tool results, answer the missing calls."""
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        repaired = 0
        for i in range(len(self.messages) - 1, -1, -1):
            m = self.messages[i]
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    if tc["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                            {"error": INTERRUPTED_TOOL_RESULT}, ensure_ascii=False)})
                        repaired += 1
                break
        return repaired

    def _drive(self, run: RunResult, start_step: int) -> RunResult:
        try:
            if not self.loop_done:
                fatal = self._research_loop(run, start_step)
                if fatal:
                    return self._finish(run)
                self.loop_done = True
                self._emit(f"Research loop ended ({run.stop_reason}).", "info")
                self.phase = "ranking"
                self._checkpoint(f"research loop ended: {run.stop_reason}")
            else:
                run.stop_reason = self.trace.get("stop_reason") or "resumed for finalization"
            self._finalize(run)
        except Exception as exc:  # noqa: BLE001 - never lose the persisted research to an unexpected error
            run.error = f"Unexpected error: {exc.__class__.__name__}: {exc}"
            self.trace["warnings"].append(run.error)
            self._api_error("agent", run.error)
            self._checkpoint("unexpected error", status="failed", error=run.error)
        except BaseException as exc:  # KeyboardInterrupt / SystemExit: persist, then stop
            run.error = f"Interrupted: {exc.__class__.__name__}"
            self._checkpoint("interrupted", status="interrupted", error=run.error)
            self._finish(run, status="interrupted")
            raise
        return self._finish(run)

    def _research_loop(self, run: RunResult, start_step: int) -> bool:
        """Returns True if the run cannot continue at all (model unavailable on the very first step)."""
        lim = self.limits
        run.stop_reason = "step limit reached"
        for step in range(start_step, lim.max_steps + 1):
            if self.search_count >= lim.max_searches and self.fetch_count >= lim.max_fetches:
                run.stop_reason = "search and fetch budgets exhausted"
                break
            self._touch(f"Waiting for the model (step {step})")
            try:
                resp = self._call_llm(step, self.messages, tools=TOOLS)
            except LLMError as exc:
                self.trace["warnings"].append(f"Model error at step {step}: {exc}")
                self._emit(f"Model error: {exc}", "error")
                run.stop_reason = f"model error: {exc}"
                if step == 1:
                    run.error = str(exc)
                    self._checkpoint("model error at step 1", status="failed", error=run.error)
                    return True
                self._checkpoint(f"model error at step {step}")
                break
            self.messages.append(resp.message)
            self.step = step  # the step is consumed once the model has answered
            self._checkpoint(f"model response (step {step})")
            if not resp.tool_calls:
                run.stop_reason = "model finished research"
                break

            progressed = False
            for tc in resp.tool_calls:
                result, progress = self._dispatch(step, tc.name, tc.arguments)
                progressed = progressed or progress
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": json.dumps(result, ensure_ascii=False)})
                self._checkpoint(f"tool {tc.name} (step {step})")

            self.no_progress_rounds = 0 if progressed else self.no_progress_rounds + 1
            if self.no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
                run.stop_reason = "stopped: repeated duplicate or rejected tool calls"
                self.trace["warnings"].append(run.stop_reason)
                break
        self.trace["stop_reason"] = run.stop_reason
        return False

    def _finalize(self, run: RunResult) -> None:
        self._emit("Writing and validating the final report", "info")
        # On resume after a crash during finalization, restart finalization from a clean history.
        if self.finalize_index is None:
            self.finalize_index = len(self.messages)
        else:
            self.messages = self.messages[: self.finalize_index]
        prompt = FINALIZE_PROMPT.replace("{max_opportunities}", str(self.limits.max_opportunities))
        self.messages.append({"role": "user", "content": prompt})
        last_error = ""
        for attempt in (1, 2):
            self._checkpoint(f"before final report (attempt {attempt})")
            try:
                resp = self._call_llm(f"final-{attempt}", self.messages, json_mode=True)
            except LLMError as exc:
                run.error = f"Final report failed: {exc}"
                self._checkpoint("final report failed", status="failed", error=run.error)
                return
            self.messages.append(resp.message)
            self.final_raw = resp.content or ""
            self._checkpoint(f"final report received (attempt {attempt})")
            try:
                result = parse_research_result(resp.content)
            except ValueError as exc:
                last_error = str(exc)
                self.trace["warnings"].append(f"Final output attempt {attempt} invalid: {last_error[:500]}")
                self.messages.append({"role": "user", "content": REPAIR_PROMPT.format(error=last_error[:2000])})
                continue
            run.result = self._post_process(result)
            return
        run.error = f"The model did not return a valid report after 2 attempts. Last error: {last_error[:1000]}"
        self._checkpoint("final report invalid", status="failed", error=run.error)

    def _verify_source(self, src, require_excerpt: bool) -> None:
        """Fill provenance and verification from what this run actually retrieved (never from the model)."""
        key = normalize_url(src.url)
        status = self.fetch_status.get(key)
        record = self.dataset_evidence.get(src.resource_id) if src.resource_id else None
        if record is None and status and status.get("resource_id"):
            record = self.dataset_evidence.get(status["resource_id"])
        src.official = is_primary_source(src.url)

        if record is not None:
            src.kind = "dataset"
            src.resource_id = record["resource_id"]
            src.dataset_id = record.get("dataset_id") or src.dataset_id
            src.dataset_title = record.get("dataset_title", "")
            src.publisher = record.get("publisher", "")
            src.last_updated = record.get("resource_last_modified") or record.get("dataset_last_updated", "")
            src.official = src.official or is_primary_source(record.get("source_url", ""))
            src.retrieval_status = record["status"]
            retrieved = record["status"] == "ok"
            content = record.get("evidence_text", "")
            failure = {
                "rejected": "Dataset rejected: unrelated to the research topic; not evidence.",
                "no_matching_records": "No records matching the research query were found in this resource.",
            }.get(record["status"], f"Retrieval failed: {record.get('error', '')}")
        else:
            if status:
                src.kind = "dataset" if status.get("source_type") in DATASET_SOURCE_TYPES else (
                    "legal_document" if src.official else "web")
            else:
                src.kind = "dataset" if (src.dataset_id or src.resource_id) else (
                    "legal_document" if src.official else "web")
            retrieved = bool(status and status["ok"])
            src.retrieval_status = "ok" if retrieved else ("failed" if status else "not_retrieved")
            content = self.retrieved_text.get(key, "")
            failure = f"Retrieval failed: {status['error']}" if status else "Not retrieved during this run."
            if retrieved and (is_ckan_metadata_url(src.url) or (status or {}).get("source_type") in ("ckan_resource", "ckan_dataset")):
                src.kind = "dataset"
                src.retrieval_status = "metadata_only"
                retrieved = False
                failure = ("CKAN metadata (search/package/resource description) is not the dataset contents and "
                           "cannot support a claim; read the records with read_government_resource.")

        src.excerpt_verified = bool(retrieved and src.excerpt and excerpt_found(src.excerpt, content))
        src.verified = retrieved and (src.excerpt_verified or not require_excerpt)
        if not retrieved:
            src.verification_note = failure
        elif not src.official:
            src.verification_note = "Retrieved, but not an official source."
        elif not src.excerpt and require_excerpt:
            src.verification_note = "Retrieved, but no supporting excerpt was quoted; the claim is not confirmed."
        elif src.excerpt and not src.excerpt_verified:
            src.verification_note = "Retrieved, but the quoted excerpt was not found in the retrieved content."
        else:
            src.verification_note = "Retrieved during this run" + (
                "; the quoted excerpt was found in the source." if src.excerpt_verified else ".")
        if record is not None and record.get("snapshot_version") and retrieved:
            src.verification_note += (f" Evidence from the local official snapshot {record['snapshot_version']} "
                                      f"(retrieved from data.gov.il {record.get('retrieved_at', '')}).")

    def _verify_opportunity(self, opp) -> None:
        for src in opp.primary_sources:
            self._verify_source(src, require_excerpt=True)
        for src in opp.secondary_sources + opp.contradictory_sources_checked:
            self._verify_source(src, require_excerpt=False)
        official_ok = [s for s in opp.primary_sources if s.official and s.verified]
        legal_ok = [s for s in official_ok if s.kind == "legal_document"]
        unverified = [s for s in opp.primary_sources if not (s.official and s.verified)]
        opp.unread_primary_sources = [s.url for s in opp.primary_sources if s.retrieval_status != "ok"]
        opp.verification_notes = [f"{s.url}: {s.verification_note}" for s in unverified]
        if official_ok and not unverified:
            opp.verification_status = "verified"
        elif official_ok:
            opp.verification_status = "partially_verified"
        else:
            opp.verification_status = "unverified"
            opp.verification_notes.insert(
                0, "Legal finding UNVERIFIED: no official source supporting it was retrieved and confirmed in this run.")
        if any(s.kind == "dataset" for s in opp.primary_sources):
            opp.verification_notes.append(
                "Dataset evidence describes the dataset's contents; it is not by itself proof of a currently "
                "applicable legal obligation. Dataset update dates are not legal effective dates.")
        # Class A: all cited primary evidence verified, including an official legal document.
        if opp.classification == "A":
            reason = ""
            if opp.verification_status != "verified":
                reason = "not all supporting official evidence was retrieved and checked."
            elif not legal_ok:
                reason = "supported only by dataset evidence; no verified official legal text supports the claim."
            if reason:
                opp.classification = "B"
                opp.downgraded_from = "A"
                opp.verification_notes.insert(0, f"Downgraded from A to B: {reason}")
                self.trace["warnings"].append(f"'{opp.name}' downgraded from A to B ({reason})")
        if opp.verification_status != "verified":
            self.trace["warnings"].append(f"'{opp.name}' cites primary sources that were not verified in this run.")

    def _post_process(self, result: ResearchResult) -> ResearchResult:
        for opp in result.opportunities:
            self._verify_opportunity(opp)
        result.opportunities = rank_opportunities(result.opportunities, self.limits.max_opportunities)
        if not result.opportunities and not result.no_opportunity_reason:
            result.no_opportunity_reason = NO_RESULT_MESSAGE
        return result

    def _finish(self, run: RunResult, status: str | None = None) -> RunResult:
        self.trace["stop_reason"] = run.stop_reason
        self.trace["elapsed_s"] = self._elapsed()
        self.trace["api_calls"] = list(self.ckan.log)
        self.trace["findings"] = self.findings
        self.trace["open_questions"] = self.open_questions
        if run.result is not None:
            self.trace["final"] = run.result.model_dump()
        run.status = status or ("completed" if run.result is not None else "failed")
        if run.result is None:
            # Never lose the run: build a clearly labelled report from the persisted evidence only.
            try:
                run.partial_report = partial_report.build(self.export_state(), status=run.status, error=run.error)
            except Exception as exc:  # noqa: BLE001
                self.trace["warnings"].append(f"Partial report generation failed: {exc}")
        try:
            self.checkpointer.save("finished: " + run.status, self.phase, self.export_state(),
                                   self._summary("finished: " + run.status), status=run.status, error=run.error)
            self.checkpointer.finish(run.status, run.error,
                                     final_report=run.result.model_dump() if run.result is not None else None,
                                     partial_report=run.partial_report)
        except Exception as exc:  # noqa: BLE001
            self.trace["warnings"].append(f"Final checkpoint failed: {exc}")
        if status != "interrupted":
            self._emit("Research complete." if not run.error else f"Research ended with an error: {run.error}", "done")
        return run
