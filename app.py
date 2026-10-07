"""Regulatory Opportunity Hunter: Streamlit UI."""

import html
import json
import re
import time

import streamlit as st

import agent
import llm
import local_data
import partial_report
import research_runner
import research_store
from config import get_int, get_setting
from models import ResearchResult

DISCLAIMER = (
    "This tool performs preliminary research and does not provide legal advice. Any opportunity involving "
    "legal or regulatory interpretation must be independently verified by a qualified professional before "
    "money is invested or operations begin."
)
CLASS_LABELS = {
    "A": "A: Explicit regulatory advantage",
    "B": "B: Plausible but ambiguous (professional review required)",
    "C": "C: Apparent unintended loophole (high regulatory-change risk)",
}
VERIFICATION_LABELS = {
    "verified": "✅ Evidence verified",
    "partially_verified": "⚠️ Partially verified",
    "unverified": "❌ Unverified",
}
EVENT_ICONS = {"search": "🔎", "fetch": "📄", "error": "⚠️", "candidates": "🧪", "done": "✅",
               "dataset": "🗂️", "rejected": "⛔"}
READ_STATUS_ICONS = {"ok": "✅", "rejected": "⛔", "no_matching_records": "∅", "failed": "❌"}
FRESHNESS_ICONS = {"fresh": "🟢", "aging": "🟡", "stale": "🔴", "missing": "⚪"}
STATUS_ICONS = {"running": "⏳", "completed": "✅", "failed": "❌", "interrupted": "⚠️"}
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def md(text) -> str:
    """Escape untrusted external text (titles, publishers, errors, records) before rendering as Markdown."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or ""))

st.set_page_config(page_title="Regulatory Opportunity Hunter", page_icon="🔎", layout="wide")
st.title("Regulatory Opportunity Hunter")
st.caption("Searches Israeli laws, regulations and government guidance for lawful, regulation-created business opportunities.")
st.warning(DISCLAIMER)
# Let each text field and paragraph pick its own direction (Hebrew RTL, English LTR).
st.markdown("<style>textarea, input[type=text] {unicode-bidi: plaintext;}</style>", unsafe_allow_html=True)

local_data.warm_up()


def get_store():
    try:
        return research_store.get_store()
    except research_store.StoreError as exc:
        st.error(f"Research store unavailable: {exc}")
        return None


def run_mode() -> str:
    return (get_setting("RESEARCH_RUN_MODE", "background") or "background").lower()


# ------------------------------------------------------ government data
def show_snapshot_status() -> None:
    try:
        status = local_data.get_default().status()
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Local government datasets unavailable: {md(exc)}")
        return
    datasets = status["datasets"]
    available = [d for d in datasets if d["available"]]
    if not available:
        st.caption("No local government dataset snapshots found; the agent will use the live data.gov.il API only.")
        return
    worst = max(available, key=lambda d: d["age_days"] if d["age_days"] is not None else 1e9)
    icon = FRESHNESS_ICONS.get(worst["freshness"], "•")
    with st.expander(f"{icon} Local official datasets: {len(available)} validated snapshots · oldest verified "
                     f"{worst['last_verified_at'] or 'n/a'} ({worst['freshness']})"):
        st.caption("Complete validated snapshots of data.gov.il resources, synchronized by GitHub Actions. "
                   "They are factual dataset evidence, not legally binding text. " + status["freshness_policy"] + ".")
        st.dataframe([{
            "dataset": d["label"], "rows": d["row_count"], "snapshot": d["retrieved_at"],
            "last verified": d["last_verified_at"], "age (days)": d["age_days"],
            "freshness": f"{FRESHNESS_ICONS.get(d['freshness'], '')} {d['freshness']}",
            "last sync": f"{d['last_attempt_status']} {d['last_attempt_at']}", "resource id": d["resource_id"],
        } for d in datasets], hide_index=True, use_container_width=True)
        run_url = (status.get("last_run") or {}).get("github_run_url")
        if run_url:
            st.caption(f"Last synchronization run: {run_url}")


show_snapshot_status()

# ------------------------------------------------------------------ inputs
providers = list(llm.PROVIDERS)
default_provider = llm.provider_name() if llm.provider_name() in providers else providers[0]
provider = st.selectbox(
    "Model provider", providers, index=providers.index(default_provider),
    format_func=lambda p: f"{llm.PROVIDER_LABELS.get(p, p)} · {llm.provider_model(p)}"
    + ("" if get_setting(llm.api_key_env_name(p)) else " (API key not configured)"),
    help="API keys are read server-side from the environment, .env or Streamlit secrets; they are never shown "
         "or stored with research data. All providers use the same tools (Serper search, fetch_url, data.gov.il, "
         "local datasets).",
)
missing = [k for k in (llm.api_key_env_name(provider), "SERPER_API_KEY") if not get_setting(k)]
if missing:
    st.error(
        f"Missing configuration: {', '.join(missing)}. Set them as environment variables, in a `.env` file, "
        "or in `.streamlit/secrets.toml` (see `.env.example`)."
    )

domain = st.text_input("What domain should we investigate?", placeholder="e.g. equipment rental, mandatory inspections, waste and recycling")
instructions = st.text_area(
    "Custom Research Instructions",
    height=150,
    max_chars=agent.MAX_INSTRUCTIONS_CHARS,
    placeholder=(
        "Optional, Hebrew or English. Describe goals, business constraints, priorities, exclusions or questions. "
        "e.g. Prioritize opportunities needing under ₪30,000 startup capital that can run alongside a full-time job."
    ),
    help="Guides the agent's focus. It cannot override the built-in legal safeguards, source verification, "
    "red-team process or run limits.",
)

with st.expander("Advanced settings"):
    c1, c2, c3, c4, c5 = st.columns(5)
    max_steps = c1.number_input("Max agent steps", 3, 100, get_int("MAX_AGENT_STEPS", 25))
    max_searches = c2.number_input("Max search calls", 1, 100, get_int("MAX_SEARCHES", 30))
    max_fetches = c3.number_input("Max URLs to read", 0, 100, get_int("MAX_FETCHES", 20))
    max_api_calls = c4.number_input("Max data.gov.il API calls", 0, 200, get_int("MAX_CKAN_CALLS", 40))
    max_opps = c5.number_input("Max final opportunities", 1, 10, 5)

start = st.button("Start Research", type="primary", disabled=bool(missing) or not domain.strip())


# ---------------------------------------------------------------- research
def run_inline(store, domain: str, instructions: str, limits: agent.Limits, provider: str) -> str:
    """Synchronous mode (RESEARCH_RUN_MODE=inline): same durable checkpoints, progress shown in place."""
    status = st.status("Researching…", expanded=True)
    with status:
        phase_box = st.empty()
        m1, m2, m3, m4 = st.columns(4)
        searches_m, pages_m, cands_m, elapsed_m = m1.empty(), m2.empty(), m3.empty(), m4.empty()
        current_box = st.empty()
        log_box = st.empty()
    log_lines: list[str] = []

    def on_event(e: dict) -> None:
        phase_box.markdown(f"**Phase:** {e['phase']}")
        searches_m.metric("Searches", e["searches"])
        pages_m.metric("Pages read", e["fetches"])
        cands_m.metric("Candidates", e["candidates"])
        elapsed_m.metric("Elapsed", f"{int(e['elapsed_s'])}s")
        current_box.info(e["message"])
        log_lines.append(f"{EVENT_ICONS.get(e['kind'], '•')} {e['message']}")
        log_box.markdown("\n".join(f"- {line}" for line in log_lines[-12:]))
        status.update(label=f"{e['phase']}: {e['message']}"[:120])

    run_id, run = research_runner.start(store, domain, instructions, limits, provider, background=False,
                                        llm_factory=lambda p: llm.LLMClient(p), on_event=on_event)
    status.update(label="Research complete" if not run.error else "Research ended with an error",
                  state="complete" if not run.error else "error", expanded=False)
    return run_id


if start:
    store = get_store()
    if store is not None:
        limits = agent.Limits(int(max_steps), int(max_searches), int(max_fetches), int(max_opps), int(max_api_calls))
        if run_mode() == "inline":
            run_id = run_inline(store, domain.strip(), instructions.strip(), limits, provider)
        else:
            run_id, _ = research_runner.start(store, domain.strip(), instructions.strip(), limits, provider,
                                              llm_factory=lambda p: llm.LLMClient(p))
        st.session_state["run_id"] = run_id
        st.query_params["run"] = run_id

# ----------------------------------------------------------------- results
def source_mark(s) -> str:
    """✅ official and read, ☑️ read but not official, ❌ not retrieved."""
    if s.verified is None:
        return ""
    if not s.verified:
        return "❌ "
    return "✅ " if s.official else "☑️ "


def source_links(sources) -> None:
    for s in sources:
        label = s.title or s.url
        line = f"- {source_mark(s)}[{label}]({s.url})"
        if s.section:
            line += f" (§ {s.section})"
        if s.support:
            line += f": {s.support}"
        if s.kind == "dataset" and (s.dataset_title or s.resource_id):
            line += (f"  \n  Dataset: {md(s.dataset_title) or 'n/a'} · Publisher: {md(s.publisher) or 'n/a'} · "
                     f"resource `{md(s.resource_id)}` · last updated {md(s.last_updated) or 'n/a'} (not a legal date)")
        if s.legal_effective_date:
            line += f"  \n  Legal effective date (per source): {md(s.legal_effective_date)}"
        if s.excerpt:
            mark = "✓" if s.excerpt_verified else "✗"
            line += f"  \n  > {mark} “{md(s.excerpt)}”"
        if s.verification_note and not (s.verified and s.official):
            line += f"  \n  _{md(s.verification_note)}_"
        st.markdown(line)


def bullet_list(items: list[str]) -> None:
    st.markdown("\n".join(f"- {i}" for i in items) if items else "_None recorded._")


def show_opportunity(opp, expanded: bool) -> None:
    header = (f"{opp.name}  ·  Score {opp.business_score}/100  ·  Class {opp.classification}  ·  "
              f"Confidence {opp.confidence}%  ·  {VERIFICATION_LABELS[opp.verification_status]}")
    with st.expander(header, expanded=expanded):
        st.markdown(f"**{CLASS_LABELS[opp.classification]}**")
        if opp.verification_status == "verified":
            st.success("Official evidence verified: every cited primary source was retrieved and read in this run.")
        else:
            box = st.error if opp.verification_status == "unverified" else st.warning
            title = ("Legal finding UNVERIFIED." if opp.verification_status == "unverified"
                     else "Partially verified: some cited primary sources could not be retrieved or are not official.")
            box("\n".join([f"**{title}** Verify manually before relying on it."] + [f"- {n}" for n in opp.verification_notes]))
        if opp.summary:
            st.markdown(opp.summary)
        st.markdown("#### Business thesis")
        st.markdown(opp.business_thesis)
        st.markdown("#### Regulatory mechanism")
        st.markdown(opp.regulatory_mechanism)
        st.markdown("#### Primary sources")
        source_links(opp.primary_sources)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Startup capital**\n\n{opp.startup_capital_estimate or 'n/a'}")
        c2.markdown(f"**Customer**\n\n{opp.customer or 'n/a'}")
        c3.markdown(f"**Revenue model**\n\n{opp.revenue_model or 'n/a'}")
        if opp.customer_problem:
            st.markdown(f"**Customer problem:** {opp.customer_problem}")

        st.markdown("#### Red team findings")
        bullet_list(opp.red_team)
        st.markdown("#### Open legal questions")
        bullet_list(opp.open_legal_questions)
        st.markdown("#### Existing competition")
        bullet_list(opp.existing_competition)

        st.markdown("#### Scores (0-10, 10 = most favorable)")
        st.dataframe(
            [{"dimension": k.replace("_", " "), "score": v} for k, v in opp.scores.model_dump().items()],
            hide_index=True, use_container_width=False,
        )
        if opp.contradictory_sources_checked:
            st.markdown("#### Contradictory sources checked")
            source_links(opp.contradictory_sources_checked)
        if opp.secondary_sources:
            st.markdown("#### Secondary sources (discovery only, not legal proof)")
            source_links(opp.secondary_sources)


def show_dataset_trace(trace: dict) -> None:
    searches, inspections, reads = (trace.get(k, []) for k in ("dataset_searches", "dataset_inspections", "dataset_reads"))
    calls = trace.get("api_calls", [])
    if not (searches or inspections or reads or calls):
        return
    st.markdown("**Official government datasets (data.gov.il)**")
    failed_calls = [c for c in calls if not c["ok"]]
    st.caption(f"API requests: {sum(1 for c in calls if not c['cached'])} sent, "
               f"{sum(1 for c in calls if c['cached'])} served from cache, {len(failed_calls)} failed")
    for d in searches:
        suffix = f" ❌ {md(d['error'])}" if d["error"] else (
            f" ({len(d['results'])} datasets, {sum(r['likely_relevant'] for r in d['results'])} likely relevant)")
        st.markdown(f"- 🗂️ Dataset search `{md(d['query'])}`{suffix}")
    for d in inspections:
        icon = {"relevant": "✅", "rejected": "⛔"}.get(d["status"], "❌")
        title = md(d.get("title") or d["dataset_id"])
        link = f"[{title}]({d['dataset_url']})" if d.get("dataset_url") else title
        detail = "rejected: unrelated to research topic" if d["status"] == "rejected" else md(d.get("error", "")) or "relevant"
        st.markdown(f"- {icon} Inspected {link} · {md(d.get('publisher', ''))} · {detail}")
    for r in reads:
        icon = READ_STATUS_ICONS.get(r["status"], "•")
        name = md(f"{r.get('dataset_title') or r.get('dataset_id') or ''} / {r.get('resource_name') or r['resource_id']}")
        link = f"[{name}]({r['source_url']})" if r.get("source_url") else name
        detail = {"ok": "records read", "rejected": "rejected: unrelated to research topic",
                  "no_matching_records": "no matching records"}.get(r["status"], md(r.get("error", "")))
        updated = f" · updated {md(r.get('resource_last_modified') or r.get('dataset_last_updated'))}" if (
            r.get("resource_last_modified") or r.get("dataset_last_updated")) else ""
        st.markdown(f"- {icon} {link} · {md(r.get('publisher', ''))}{updated} · {detail}")
    for c in failed_calls:
        st.markdown(f"- ❌ `{md(c['action'])}` {md(c['error'])} ([request]({c['url']}))")


def show_trace(trace: dict) -> None:
    with st.expander("Research Trace"):
        st.caption(
            f"Stop reason: {md(trace.get('stop_reason', 'n/a'))} · Model calls: {len(trace.get('model_calls', []))} · "
            f"Elapsed: {trace.get('elapsed_s', 0)}s"
        )
        st.markdown("**Custom research instructions**")
        if trace.get("custom_instructions"):
            st.markdown(
                f'<div dir="auto" style="white-space: pre-wrap">{html.escape(trace["custom_instructions"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("_None provided._")
        for w in trace.get("warnings", []) + trace.get("llm_warnings", []):
            st.markdown(f"- ⚠️ {w}")
        st.markdown(f"**Searches ({len(trace.get('searches', []))})**")
        for s in trace.get("searches", []):
            suffix = f" ❌ {s['error']}" if s["error"] else f" ({len(s['results'])} results)"
            st.markdown(f"- `{s['query']}`{suffix}")
        st.markdown(f"**URLs inspected ({len(trace.get('fetches', []))})**")
        for f in trace.get("fetches", []):
            mark = "✅" if f["ok"] else "❌"
            tag = "primary" if f["primary"] else "secondary"
            detail = f"{f['source_type']}, {f['chars']} chars" if f["ok"] else f["error"]
            if f.get("resource_url"):
                detail += f" · resource: {f['resource_url']}"
            st.markdown(f"- {mark} [{md(f['title'] or f['url'])}]({f['url']}) · {tag} · {md(detail)}")
        show_dataset_trace(trace)
        show_local_trace(trace)
        show_findings(trace)
        cands = trace.get("candidates", {})
        if cands:
            st.markdown(f"**Candidate funnel ({len(cands)})**")
            st.dataframe(
                [{"candidate": n, **c} for n, c in cands.items()], hide_index=True, use_container_width=True
            )
        tu = trace.get("token_usage") or {}
        if tu.get("model_calls"):
            st.caption(f"Tokens: {tu.get('prompt_tokens', 0)} input · {tu.get('completion_tokens', 0)} output · "
                       f"{tu.get('model_calls', 0)} model calls")
        for e in trace.get("api_errors", []):
            st.markdown(f"- ❌ {md(e.get('source'))}: {md(e.get('error'))}")


def show_local_trace(trace: dict) -> None:
    queries = trace.get("local_queries", [])
    if not queries:
        return
    st.markdown(f"**Local official dataset queries ({len(queries)})**")
    for q in queries:
        args = q.get("args", {})
        what = md(f"{args.get('dataset', '')} {args.get('query', '') or args.get('record_id', '')}".strip())
        detail = f"❌ {md(q['error'])}" if q.get("error") else (
            f"{q.get('total_matches')} matches" + (f" · snapshot {', '.join(q.get('snapshot_versions') or [])}"
                                                   if q.get("snapshot_versions") else ""))
        st.markdown(f"- 🗄️ `{md(q['tool'])}` {what} · {detail}")


def show_findings(trace: dict) -> None:
    findings = trace.get("findings") or []
    questions = trace.get("open_questions") or []
    if findings:
        st.markdown(f"**Recorded findings ({len(findings)})**")
        for f in findings:
            mark = "✅ VERIFIED" if f.get("verified") else "❌ UNVERIFIED"
            st.markdown(f"- {mark}: {md(f['statement'])} ({f['source_url']})  \n  _{md(f.get('verification_note'))}_")
    if questions:
        st.markdown("**Open questions**")
        bullet_list([md(q) for q in questions])


def trace_download(record: dict) -> str:
    state = record.get("state") or {}
    trace = dict(state.get("trace") or {})
    trace.pop("events", None)
    doc = {"run_id": record["run_id"], "status": record["status"], "domain": record.get("domain"),
           "config": record.get("config"), "last_checkpoint_at": record.get("last_checkpoint_at"),
           "checkpoint_seq": record.get("checkpoint_seq"), "summary": record.get("summary"),
           "findings": state.get("findings"), "open_questions": state.get("open_questions"),
           "dataset_evidence_provenance": {k: {kk: vv for kk, vv in v.items() if kk != "evidence_text"}
                                           for k, v in (state.get("dataset_evidence") or {}).items()},
           "trace": trace, "events": (state.get("trace") or {}).get("events", [])}
    return json.dumps(doc, ensure_ascii=False, indent=2, default=str)


def show_run_header(record: dict) -> None:
    summary = record.get("summary") or {}
    icon = STATUS_ICONS.get(record["status"], "•")
    st.markdown(f"**Run ID:** `{record['run_id']}` · **Status:** {icon} {record['status']}")
    st.caption(f"Last successful checkpoint: {record.get('last_checkpoint_at') or 'n/a'} "
               f"(#{record.get('checkpoint_seq', 0)}: {md(record.get('label') or '')}) · "
               f"phase {agent.PHASES.get(record.get('phase') or '', record.get('phase') or '')}")
    if summary:
        c = st.columns(6)
        c[0].metric("Steps", summary.get("step", 0))
        c[1].metric("Searches", summary.get("searches", 0))
        c[2].metric("Pages read", summary.get("fetches", 0))
        c[3].metric("Local queries", summary.get("local_queries", 0))
        c[4].metric("Candidates", summary.get("candidates", 0))
        c[5].metric("Findings ✓", f"{summary.get('verified_findings', 0)}/{summary.get('findings', 0)}")


def show_live(run_id: str) -> None:
    store = get_store()
    record = store.load(run_id, include_state=False) if store else None
    if record is None:
        return
    show_run_header(record)
    events = research_runner.live_events(run_id)
    if events:
        st.info(events[-1]["message"])
        st.markdown("\n".join(f"- {EVENT_ICONS.get(e['kind'], '•')} {md(e['message'])}" for e in events[-12:]))
    if record["status"] != "running":
        st.rerun()


def show_partial(record: dict) -> None:
    report = research_runner.report_for(record)
    if not report:
        return
    st.warning(partial_report.NOTICE)
    c1, c2 = st.columns(2)
    c1.download_button("Download partial report (Markdown)", partial_report.to_markdown(report),
                       file_name=f"partial_report_{record['run_id']}.md", mime="text/markdown")
    c2.download_button("Download partial report (JSON)", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name=f"partial_report_{record['run_id']}.json", mime="application/json")
    with st.expander("Partial report preview", expanded=True):
        st.markdown(partial_report.to_markdown(report))


def show_run(run_id: str) -> None:
    store = get_store()
    if store is None:
        return
    try:
        record = store.load(run_id)
    except research_store.StoreError as exc:
        st.error(f"Could not load run {md(run_id)}: {md(exc)}")
        return
    if record is None:
        st.error(f"Run `{md(run_id)}` was not found in the research store.")
        return
    st.divider()
    st.subheader(f"Results: {record.get('domain') or ''}")
    if record["status"] == "running":
        if research_runner.is_active(run_id) or run_mode() != "inline":
            st.fragment(show_live, run_every=3)(run_id)
        return
    show_run_header(record)
    state = record.get("state") or {}
    trace = dict(state.get("trace") or {})
    trace.setdefault("custom_instructions", state.get("instructions", ""))
    if record.get("error"):
        st.error(record["error"])
    if record["status"] == "completed" and record.get("final_report"):
        result = ResearchResult.model_validate(record["final_report"])
        if result.research_summary:
            st.markdown(result.research_summary)
        if not result.opportunities:
            reason = result.no_opportunity_reason
            st.info(f"**{agent.NO_RESULT_MESSAGE}**" + (f"\n\n{reason}" if reason != agent.NO_RESULT_MESSAGE else ""))
        for i, opp in enumerate(result.opportunities):
            show_opportunity(opp, expanded=(i == 0))
    else:
        if record["status"] == "interrupted":
            st.error("This run was interrupted (server restart, timeout or crash) before it finished. "
                     "Everything up to the last checkpoint was saved.")
        show_partial(record)
        if research_runner.can_resume(record):
            if st.button("Resume research from last checkpoint", key=f"resume-{run_id}"):
                try:
                    if run_mode() == "inline":
                        research_runner.resume(store, run_id, background=False, llm_factory=lambda p: llm.LLMClient(p))
                    else:
                        research_runner.resume(store, run_id, llm_factory=lambda p: llm.LLMClient(p))
                except (research_runner.RunnerError, research_store.StoreError) as exc:
                    st.error(str(exc))
                st.rerun()
    if trace:
        show_trace(trace)
    st.download_button("Download research trace (JSON)", trace_download(record),
                       file_name=f"research_trace_{run_id}.json", mime="application/json", key=f"trace-{run_id}")


def show_recovery() -> None:
    store = get_store()
    if store is None:
        return
    with st.expander("Recover previous run"):
        st.caption(f"Research store: {store.describe()}.")
        if not store.durable_across_redeploys:
            st.caption("⚠️ Checkpoints survive app restarts on this disk, but not container replacement or redeploys "
                       "(e.g. Streamlit Community Cloud). Set RESEARCH_STORE_URL to a PostgreSQL database for "
                       "durable recovery.")
        rid = st.text_input("Run ID", key="recover_run_id", placeholder="e.g. 20261007-120000-1a2b3c4d5e6f7a8b")
        if st.button("Load run", disabled=not rid.strip()):
            st.session_state["run_id"] = rid.strip()
            st.query_params["run"] = rid.strip()
            st.rerun()
        if (get_setting("RESEARCH_LIST_RUNS", "0") or "0") == "1":
            runs = store.list_runs(20)
            if runs:
                st.dataframe([{k: r[k] for k in ("run_id", "status", "domain", "created_at", "last_checkpoint_at")}
                              for r in runs], hide_index=True, use_container_width=True)


current = st.session_state.get("run_id") or st.query_params.get("run")
if current:
    st.session_state["run_id"] = current
    show_run(current)
show_recovery()
st.caption(DISCLAIMER)
