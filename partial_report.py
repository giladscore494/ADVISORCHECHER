"""Fallback report built ONLY from persisted research evidence (no model call, no new conclusions).

Used when a run fails, times out, is interrupted, or the model's final JSON report cannot be produced or
validated. Every item is labelled: VERIFIED (official source retrieved in this run and the quoted excerpt
was found in it), RETRIEVED (content retrieved, no verified claim), UNVERIFIED / FAILED, or WORKING
HYPOTHESIS (the model's candidate funnel, never verified by itself).
"""

from __future__ import annotations

from datetime import datetime, timezone

NOTICE = (
    "PARTIAL REPORT - this is NOT a validated final report. It was generated automatically from the research "
    "evidence that was saved before the run stopped. It contains no new conclusions: candidates are the model's "
    "working hypotheses, and only items marked VERIFIED were confirmed against an official source retrieved "
    "during this run. Dataset records are factual dataset evidence, not legally binding text. Nothing here is "
    "legal advice; verify everything with a qualified professional."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(state: dict, status: str = "failed", error: str = "", checkpoint_at: str = "") -> dict:
    trace = state.get("trace") or {}
    findings = state.get("findings") or []
    candidates = trace.get("candidates") or {}
    fetches = trace.get("fetches") or []
    evidence = state.get("dataset_evidence") or {}
    official_ok = [f for f in fetches if f.get("ok") and f.get("primary")]
    other_ok = [f for f in fetches if f.get("ok") and not f.get("primary")]
    failed = [f for f in fetches if not f.get("ok")]
    report = {
        "type": "partial_report",
        "notice": NOTICE,
        "generated_at": _now(),
        "run_id": state.get("run_id", ""),
        "domain": state.get("domain", ""),
        "custom_instructions": state.get("instructions", ""),
        "status": status,
        "error": error,
        "stop_reason": trace.get("stop_reason", ""),
        "last_checkpoint_at": checkpoint_at or state.get("checkpoint_at", ""),
        "phase": state.get("phase", ""),
        "provider": state.get("provider", ""),
        "model": state.get("model", ""),
        "progress": {
            "model_steps_completed": state.get("step", 0),
            "research_loop_finished": bool(state.get("loop_done")),
            "searches": state.get("search_count", 0),
            "pages_read": state.get("fetch_count", 0),
            "data_gov_il_api_calls": state.get("ckan_calls", 0),
            "local_dataset_queries": state.get("local_query_count", 0),
            "elapsed_s": state.get("elapsed_s", 0),
        },
        "token_usage": trace.get("token_usage", {}),
        "verified_findings": [f for f in findings if f.get("verified")],
        "unverified_findings": [f for f in findings if not f.get("verified")],
        "candidates": [
            {"name": n, "status": c.get("status", ""), "mechanism": c.get("mechanism", ""),
             "reason": c.get("reason", ""), "label": "WORKING HYPOTHESIS - not verified"}
            for n, c in candidates.items()
        ],
        "official_sources_retrieved": [
            {"url": f["url"], "title": f.get("title", ""), "source_type": f.get("source_type", ""),
             "label": "RETRIEVED (official domain); claims need a verified excerpt"} for f in official_ok],
        "other_sources_retrieved": [
            {"url": f["url"], "title": f.get("title", ""), "label": "RETRIEVED (not official; discovery only)"}
            for f in other_ok],
        "failed_retrievals": [
            {"url": f["url"], "error": f.get("error", ""), "http_status": f.get("http_status"),
             "label": "FAILED - anything depending on it is UNVERIFIED"} for f in failed],
        "dataset_evidence": [
            {"resource_id": rid, "dataset_title": e.get("dataset_title", ""), "publisher": e.get("publisher", ""),
             "status": e.get("status", ""), "source": e.get("source_kind", "data.gov.il API"),
             "snapshot_version": e.get("snapshot_version", ""), "source_url": e.get("source_url", ""),
             "records_excerpt": (e.get("evidence_text") or "")[:3000],
             "label": "DATASET EVIDENCE (factual dataset content, not legal text)"}
            for rid, e in evidence.items()],
        "searches": [{"query": s.get("query", ""), "results": len(s.get("results", [])), "error": s.get("error", "")}
                     for s in trace.get("searches", [])],
        "local_queries": trace.get("local_queries", []),
        "open_questions": state.get("open_questions") or [],
        "api_errors": trace.get("api_errors", []),
        "warnings": trace.get("warnings", []),
    }
    if state.get("final_raw"):
        report["unvalidated_model_draft"] = {
            "label": "UNVALIDATED MODEL OUTPUT - failed validation or was not checked; NOT verified, do not rely on it",
            "text": state["final_raw"][:20000],
        }
    return report


def _esc(text) -> str:
    return str(text or "").replace("\n", " ").strip()


def to_markdown(report: dict) -> str:
    lines = [f"# Partial research report: {_esc(report.get('domain'))}", "", f"> **{report['notice']}**", ""]
    lines += [
        f"- Run ID: `{report.get('run_id', '')}`",
        f"- Status: **{report.get('status', '')}**" + (f" - {_esc(report.get('error'))}" if report.get("error") else ""),
        f"- Stop reason: {_esc(report.get('stop_reason')) or 'n/a'}",
        f"- Last checkpoint: {report.get('last_checkpoint_at', '')} (phase: {report.get('phase', '')})",
        f"- Model: {report.get('provider', '')} {report.get('model', '')}",
        f"- Generated: {report.get('generated_at', '')}",
    ]
    p = report.get("progress", {})
    lines.append(f"- Progress: {p.get('model_steps_completed', 0)} model steps, {p.get('searches', 0)} searches, "
                 f"{p.get('pages_read', 0)} pages read, {p.get('data_gov_il_api_calls', 0)} data.gov.il API calls, "
                 f"{p.get('local_dataset_queries', 0)} local dataset queries")
    tu = report.get("token_usage") or {}
    if tu:
        lines.append(f"- Tokens: {tu.get('prompt_tokens', 0)} input, {tu.get('completion_tokens', 0)} output "
                     f"({tu.get('model_calls', 0)} model calls)")
    if report.get("custom_instructions"):
        lines += ["", "## Custom instructions", "", _esc(report["custom_instructions"])]

    def section(title, items, render):
        lines.extend(["", f"## {title}", ""])
        lines.extend(render(i) for i in items) if items else lines.append("_None recorded._")

    section("VERIFIED findings (official source retrieved; excerpt found)", report["verified_findings"],
            lambda f: f"- {_esc(f['statement'])}  \n  Source: {f['source_url']}  \n  > {_esc(f['excerpt'])}")
    section("UNVERIFIED findings", report["unverified_findings"],
            lambda f: f"- {_esc(f['statement'])}  \n  Source: {f['source_url']} - _{_esc(f.get('verification_note'))}_")
    section("Candidate opportunities (WORKING HYPOTHESES - not verified)", report["candidates"],
            lambda c: f"- **{_esc(c['name'])}** [{c['status']}] {_esc(c['mechanism'])}"
                      + (f" - {_esc(c['reason'])}" if c.get("reason") else ""))
    section("Open questions", report["open_questions"], lambda q: f"- {_esc(q)}")
    section("Official sources retrieved", report["official_sources_retrieved"],
            lambda s: f"- [{_esc(s['title']) or s['url']}]({s['url']}) ({s['source_type']})")
    section("Government dataset evidence", report["dataset_evidence"],
            lambda d: f"- {_esc(d['dataset_title'])} / resource `{d['resource_id']}` - {d['status']} via {d['source']}"
                      + (f", snapshot {d['snapshot_version']}" if d.get("snapshot_version") else "")
                      + f" (DATASET EVIDENCE, not legal text)")
    section("Failed or blocked retrievals (dependent findings UNVERIFIED)", report["failed_retrievals"],
            lambda f: f"- {f['url']}: {_esc(f['error'])}")
    section("Other sources retrieved (not official)", report["other_sources_retrieved"],
            lambda s: f"- [{_esc(s['title']) or s['url']}]({s['url']})")
    section("Searches performed", report["searches"],
            lambda s: f"- `{_esc(s['query'])}`: " + (f"error {_esc(s['error'])}" if s["error"] else f"{s['results']} results"))
    section("API errors", report["api_errors"], lambda e: f"- {e.get('at', '')} {e.get('source', '')}: {_esc(e.get('error'))}")
    if report.get("unvalidated_model_draft"):
        d = report["unvalidated_model_draft"]
        lines += ["", f"## {d['label']}", "", "```", d["text"], "```"]
    return "\n".join(lines) + "\n"
