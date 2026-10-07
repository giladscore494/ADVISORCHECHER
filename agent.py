"""Controlled tool-calling research loop.

The model reasons, calls search_web / fetch_url / the data.gov.il dataset tools /
update_candidates, and we execute those tools under hard limits (steps, searches,
fetches, API calls, duplicates).
When the model says it is done, or a limit is hit, we ask for the final JSON
report, validate it, and retry once if it is malformed.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import unquote, urlparse, urlunparse

import datagov
from evidence import excerpt_found, extract_terms
from fetcher import fetch_url
from llm import LLMError, parse_tool_arguments
from models import ResearchResult, parse_research_result, rank_opportunities
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


@dataclass
class RunResult:
    domain: str
    result: ResearchResult | None = None
    error: str = ""
    stop_reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict)


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
    ):
        self.llm = llm
        self.limits = limits or Limits()
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
        self.phase = "mapping"
        self.started = time.monotonic()
        self.trace: dict[str, Any] = {
            "searches": [], "fetches": [], "model_calls": [], "tool_calls": [],
            "candidates": {}, "events": [], "warnings": [],
            "dataset_searches": [], "dataset_inspections": [], "dataset_reads": [], "api_calls": [],
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
            "elapsed_s": round(time.monotonic() - self.started, 1),
        }
        self.trace["events"].append(event)
        self.on_event(event)

    def _set_phase(self, args: dict) -> None:
        if args.get("phase") in PHASES:
            self.phase = args["phase"]

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
    def _call_llm(self, step: int, messages: list[dict], **kwargs):
        resp = self.llm.chat(messages, **kwargs)
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
        run = RunResult(domain=domain, trace=self.trace)
        self.domain = domain
        lim = self.limits
        instructions = (instructions or "").strip()
        if len(instructions) > MAX_INSTRUCTIONS_CHARS:
            instructions = instructions[:MAX_INSTRUCTIONS_CHARS]
            self.trace["warnings"].append(f"Custom instructions truncated to {MAX_INSTRUCTIONS_CHARS} characters.")
        self.trace["custom_instructions"] = instructions
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_message(domain, instructions)},
        ]
        suffix = " (with custom instructions)" if instructions else ""
        self._emit(f"Starting research on: {domain}{suffix}", "start")

        no_progress_rounds = 0
        run.stop_reason = "step limit reached"
        for step in range(1, lim.max_steps + 1):
            if self.search_count >= lim.max_searches and self.fetch_count >= lim.max_fetches:
                run.stop_reason = "search and fetch budgets exhausted"
                break
            try:
                resp = self._call_llm(step, messages, tools=TOOLS)
            except LLMError as exc:
                self.trace["warnings"].append(f"Model error at step {step}: {exc}")
                self._emit(f"Model error: {exc}", "error")
                run.stop_reason = f"model error: {exc}"
                if step == 1:
                    run.error = str(exc)
                    return self._finish(run)
                break
            messages.append(resp.message)
            if not resp.tool_calls:
                run.stop_reason = "model finished research"
                break

            progressed = False
            for tc in resp.tool_calls:
                result, progress = self._dispatch(step, tc.name, tc.arguments)
                progressed = progressed or progress
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, ensure_ascii=False)})

            no_progress_rounds = 0 if progressed else no_progress_rounds + 1
            if no_progress_rounds >= MAX_NO_PROGRESS_ROUNDS:
                run.stop_reason = "stopped: repeated duplicate or rejected tool calls"
                self.trace["warnings"].append(run.stop_reason)
                break

        self._emit(f"Research loop ended ({run.stop_reason}).", "info")
        self.phase = "ranking"
        self._finalize(run, messages)
        return self._finish(run)

    def _finalize(self, run: RunResult, messages: list[dict]) -> None:
        self._emit("Writing and validating the final report", "info")
        prompt = FINALIZE_PROMPT.replace("{max_opportunities}", str(self.limits.max_opportunities))
        messages.append({"role": "user", "content": prompt})
        last_error = ""
        for attempt in (1, 2):
            try:
                resp = self._call_llm(f"final-{attempt}", messages, json_mode=True)
            except LLMError as exc:
                run.error = f"Final report failed: {exc}"
                return
            messages.append(resp.message)
            try:
                result = parse_research_result(resp.content)
            except ValueError as exc:
                last_error = str(exc)
                self.trace["warnings"].append(f"Final output attempt {attempt} invalid: {last_error[:500]}")
                messages.append({"role": "user", "content": REPAIR_PROMPT.format(error=last_error[:2000])})
                continue
            run.result = self._post_process(result)
            return
        run.error = f"The model did not return a valid report after 2 attempts. Last error: {last_error[:1000]}"

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

    def _finish(self, run: RunResult) -> RunResult:
        self.trace["stop_reason"] = run.stop_reason
        self.trace["elapsed_s"] = round(time.monotonic() - self.started, 1)
        self.trace["api_calls"] = list(self.ckan.log)
        if run.result is not None:
            self.trace["final"] = run.result.model_dump()
        self._emit("Research complete." if not run.error else f"Research ended with an error: {run.error}", "done")
        return run
