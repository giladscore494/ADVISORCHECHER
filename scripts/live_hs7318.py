"""Live HS 7318 research run (real model + Serper + official sites) with measurements, for comparison with
run 20261007-025045-54c043110af66f55 (GPT-6.1 Sol: 317 s, ~903K input tokens, 12 model calls, 29 local
queries, 12 searches, 14 fetches). Costs real API tokens: run it deliberately (workflow_dispatch or the
`live-benchmark` PR label in .github/workflows/live-benchmark.yml).

usage: python scripts/live_hs7318.py [--provider openai] [--out metrics.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
import llm  # noqa: E402
from config import get_setting  # noqa: E402

DOMAIN = "ייבוא ברגים, אומים ודיסקיות מפלדה (פרט מכס 7318) לישראל"
INSTRUCTIONS = (
    "Focus on HS 7318 (screws, bolts, nuts, washers of iron or steel) imported into Israel. Determine which Free "
    "Import Order schedule and which mandatory Israeli standards apply (e.g. whether ISO 4032 nuts or ISO 7089 "
    "washers are covered), the exact legal text of the relevant schedules, exceptions, validity, and whether any "
    "rule gives a real advantage over competing importers."
)
PREVIOUS = {"run_id": "20261007-025045-54c043110af66f55", "elapsed_s": 317, "input_tokens": 903_188,
            "model_calls": 12, "local_queries": 29, "searches": 12, "fetches": 14}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--out", default="live_hs7318_metrics.json")
    ns = ap.parse_args()
    missing = [k for k in (llm.api_key_env_name(ns.provider), "SERPER_API_KEY") if not get_setting(k)]
    if missing:
        print(f"Cannot run the live benchmark: missing {', '.join(missing)} (repository secrets).", file=sys.stderr)
        return 2
    client = llm.LLMClient(ns.provider)
    events = []
    a = agent.ResearchAgent(client, on_event=events.append)
    started = time.monotonic()
    run = a.run(DOMAIN, INSTRUCTIONS)
    elapsed = round(time.monotonic() - started, 1)
    tu = run.trace["token_usage"]
    sources = [s for o in (run.result.opportunities if run.result else []) for s in o.primary_sources]
    metrics = {
        "previous_run": PREVIOUS, "provider": ns.provider, "model": client.model, "status": run.status,
        "error": run.error, "stop_reason": run.stop_reason, "elapsed_s": elapsed,
        "input_tokens": tu.get("prompt_tokens"), "cached_input_tokens": tu.get("cached_tokens"),
        "output_tokens": tu.get("completion_tokens"), "reasoning_tokens": tu.get("reasoning_tokens"),
        "model_calls": tu.get("model_calls"), "token_usage_by_phase": run.trace.get("token_usage_by_phase"),
        "per_call": [{k: c.get(k) for k in ("step", "phase", "usage", "estimated_input_tokens")}
                     for c in run.trace["model_calls"]],
        "compactions": len(run.trace.get("compactions", [])),
        "searches": a.search_count, "fetches": a.fetch_count, "local_queries": a.local_query_count,
        "document_queries": a.document_query_count,
        "documents": [{k: d.get(k) for k in ("document_id", "url", "page_count", "chars", "status", "issues")}
                      for d in a.documents.values()],
        "citations": {"primary_sources": len(sources),
                      "excerpt_verified": sum(1 for s in sources if s.excerpt_verified),
                      "with_matched_pages": sum(1 for s in sources if s.matched_pages),
                      "wrong_page": sum(1 for s in sources if s.page_mismatch)},
        "verification_counters": run.trace.get("verification_counters"),
        "opportunities": [{"name": o.name, "class": o.classification, "source": o.verification_status,
                           "legal": o.legal_verification, "business": o.business_verification,
                           "negative_claim": o.negative_claim}
                          for o in (run.result.opportunities if run.result else [])],
        "checklist": {i["id"]: i["status"] for i in run.trace.get("checklist", [])},
        "unresolved_questions": run.result.unresolved_questions if run.result else [],
        "warnings": run.trace.get("warnings", [])[-20:],
    }
    Path(ns.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), "utf-8")
    lines = [f"## Live HS 7318 benchmark ({ns.provider} {client.model})", "",
             "| measure | previous run | this run |", "|---|---|---|",
             f"| status | completed | {run.status} |",
             f"| elapsed (s) | {PREVIOUS['elapsed_s']} | {elapsed} |",
             f"| input tokens | {PREVIOUS['input_tokens']:,} | {tu.get('prompt_tokens', 0):,} "
             f"({tu.get('cached_tokens', 0):,} cached) |",
             f"| model calls | {PREVIOUS['model_calls']} | {tu.get('model_calls')} |",
             f"| searches / fetches / local queries | 12 / 14 / 29 | {a.search_count} / {a.fetch_count} / "
             f"{a.local_query_count} (+{a.document_query_count} document) |",
             f"| documents complete | n/a | {sum(d['status'] == 'complete' for d in a.documents.values())}/"
             f"{len(a.documents)} |",
             f"| verification | 10/11 'verified' | {json.dumps(run.trace.get('verification_counters'))} |"]
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if run.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
