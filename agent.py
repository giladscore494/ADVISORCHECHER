"""Controlled tool-calling research loop.

The model reasons, calls search_web / fetch_url / update_candidates, and we
execute those tools under hard limits (steps, searches, fetches, duplicates).
When the model says it is done, or a limit is hit, we ask for the final JSON
report, validate it, and retry once if it is malformed.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

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


@dataclass
class Limits:
    max_steps: int = 25
    max_searches: int = 30
    max_fetches: int = 20
    max_opportunities: int = 5


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
    ):
        self.llm = llm
        self.limits = limits or Limits()
        self.on_event = on_event or (lambda e: None)
        self.search_fn = search_fn
        self.fetch_fn = fetch_fn

        self.seen_queries: set[str] = set()
        self.seen_urls: set[str] = set()
        self.ok_urls: set[str] = set()
        self.search_count = 0
        self.fetch_count = 0
        self.phase = "mapping"
        self.started = time.monotonic()
        self.trace: dict[str, Any] = {
            "searches": [], "fetches": [], "model_calls": [], "tool_calls": [],
            "candidates": {}, "events": [], "warnings": [],
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

        self.seen_queries.add(key)
        self.search_count += 1
        self._emit(args.get("purpose") or f"Searching: {query}", "search")
        entry: dict[str, Any] = {"query": query, "purpose": args.get("purpose", ""), "results": [], "error": ""}
        self.trace["searches"].append(entry)
        try:
            results = self.search_fn(query, num_results=args.get("num_results", 10))
        except SearchError as exc:
            entry["error"] = str(exc)
            self._emit(f"Search failed: {exc}", "error")
            return {"error": str(exc), **self._budget()}, True
        entry["results"] = [{"title": r["title"], "url": r["url"], "primary": r["primary_source"]} for r in results]
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
        self.trace["fetches"].append({
            "url": url, "ok": res.ok, "source_type": res.source_type, "title": res.title,
            "chars": len(res.text), "truncated": res.truncated, "error": res.error,
            "primary": is_primary_source(res.final_url or url),
        })
        if not res.ok:
            self._emit(f"Could not read {url}: {res.error}", "error")
            return {"url": url, "ok": False, "error": res.error, **self._budget()}, True
        self.ok_urls.add(key)
        if res.final_url:
            self.ok_urls.add(normalize_url(res.final_url))
        return {
            "url": url, "final_url": res.final_url, "ok": True, "source_type": res.source_type,
            "primary_source": is_primary_source(res.final_url or url), "title": res.title,
            "truncated": res.truncated, "text": res.text, **self._budget(),
        }, True

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

    def _post_process(self, result: ResearchResult) -> ResearchResult:
        for opp in result.opportunities:
            opp.unread_primary_sources = [
                s.url for s in opp.primary_sources if normalize_url(s.url) not in self.ok_urls
            ]
            if opp.unread_primary_sources:
                self.trace["warnings"].append(
                    f"'{opp.name}' cites primary sources that were not successfully read in this run."
                )
        result.opportunities = rank_opportunities(result.opportunities, self.limits.max_opportunities)
        if not result.opportunities and not result.no_opportunity_reason:
            result.no_opportunity_reason = NO_RESULT_MESSAGE
        return result

    def _finish(self, run: RunResult) -> RunResult:
        self.trace["stop_reason"] = run.stop_reason
        self.trace["elapsed_s"] = round(time.monotonic() - self.started, 1)
        if run.result is not None:
            self.trace["final"] = run.result.model_dump()
        self._emit("Research complete." if not run.error else f"Research ended with an error: {run.error}", "done")
        return run
