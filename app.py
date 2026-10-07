"""Regulatory Opportunity Hunter: Streamlit UI."""

import html
import json
import time

import streamlit as st

import agent
import llm
from config import get_int, get_setting

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
EVENT_ICONS = {"search": "🔎", "fetch": "📄", "error": "⚠️", "candidates": "🧪", "done": "✅"}

st.set_page_config(page_title="Regulatory Opportunity Hunter", page_icon="🔎", layout="wide")
st.title("Regulatory Opportunity Hunter")
st.caption("Searches Israeli laws, regulations and government guidance for lawful, regulation-created business opportunities.")
st.warning(DISCLAIMER)
# Let each text field and paragraph pick its own direction (Hebrew RTL, English LTR).
st.markdown("<style>textarea, input[type=text] {unicode-bidi: plaintext;}</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ inputs
missing = [k for k in (llm.api_key_env_name(), "SERPER_API_KEY") if not get_setting(k)]
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
    c1, c2, c3, c4 = st.columns(4)
    max_steps = c1.number_input("Max agent steps", 3, 100, get_int("MAX_AGENT_STEPS", 25))
    max_searches = c2.number_input("Max search calls", 1, 100, get_int("MAX_SEARCHES", 30))
    max_fetches = c3.number_input("Max URLs to read", 0, 100, get_int("MAX_FETCHES", 20))
    max_opps = c4.number_input("Max final opportunities", 1, 10, 5)

start = st.button("Start Research", type="primary", disabled=bool(missing) or not domain.strip())


# ---------------------------------------------------------------- research
def run_research(domain: str, instructions: str, limits: agent.Limits) -> agent.RunResult:
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

    try:
        client = llm.LLMClient()
    except llm.LLMError as exc:
        status.update(label="Configuration error", state="error")
        return agent.RunResult(domain=domain, error=str(exc))

    run = agent.ResearchAgent(client, limits=limits, on_event=on_event).run(domain, instructions)
    run.trace["llm_warnings"] = list(client.warnings)
    status.update(label="Research complete" if not run.error else "Research ended with an error",
                  state="complete" if not run.error else "error", expanded=False)
    return run


if start:
    limits = agent.Limits(int(max_steps), int(max_searches), int(max_fetches), int(max_opps))
    st.session_state["run"] = run_research(domain.strip(), instructions.strip(), limits)
    st.session_state["run_at"] = time.strftime("%Y-%m-%d %H:%M")


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
        if s.verification_note and not (s.verified and s.official):
            line += f"  \n  _{s.verification_note}_"
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


def show_trace(trace: dict) -> None:
    with st.expander("Research Trace"):
        st.caption(
            f"Stop reason: {trace.get('stop_reason', 'n/a')} · Model calls: {len(trace.get('model_calls', []))} · "
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
            st.markdown(f"- {mark} [{f['title'] or f['url']}]({f['url']}) · {tag} · {detail}")
        cands = trace.get("candidates", {})
        if cands:
            st.markdown(f"**Candidate funnel ({len(cands)})**")
            st.dataframe(
                [{"candidate": n, **c} for n, c in cands.items()], hide_index=True, use_container_width=True
            )
        st.download_button(
            "Download full trace (JSON)",
            json.dumps(trace, ensure_ascii=False, indent=2),
            file_name="research_trace.json",
            mime="application/json",
        )


run: agent.RunResult | None = st.session_state.get("run")
if run is not None:
    st.divider()
    st.subheader(f"Results: {run.domain}")
    st.caption(f"Run at {st.session_state.get('run_at', '')}")
    if run.error:
        st.error(run.error)
    elif run.result is not None:
        if run.result.research_summary:
            st.markdown(run.result.research_summary)
        if not run.result.opportunities:
            reason = run.result.no_opportunity_reason
            st.info(f"**{agent.NO_RESULT_MESSAGE}**" + (f"\n\n{reason}" if reason != agent.NO_RESULT_MESSAGE else ""))
        for i, opp in enumerate(run.result.opportunities):
            show_opportunity(opp, expanded=(i == 0))
    if run.trace:
        show_trace(run.trace)
    st.caption(DISCLAIMER)
