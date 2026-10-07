"""Deterministic HS 7318 integration benchmark: replays the profile of run 20261007-025045-54c043110af66f55
(GPT-6.1 Sol: 12 model calls, 12 Serper searches, 14 page fetches, 29 local dataset queries) against a given
code tree, so the code before and after a change can be compared on identical inputs.

What is real: the code under test (agent loop, fetcher/extraction, verification, compaction), the committed
government dataset snapshots (all 29 local queries run against them), the PDF/HTML parsing.
What is simulated: the model's decisions (a fixed script), Serper results and the fetched documents
(synthetic Hebrew legal PDFs/HTML of realistic size, incl. a 140-page order whose schedule is on page 120),
and the model's reasoning payload (1,200 reasoning tokens per call, replayed as Responses API reasoning items).
Tokens are ESTIMATED from the exact Responses API payload (llm.to_responses_input + tools) with
context.estimate_text_tokens (~4 ASCII / ~2 Hebrew characters per token); they are comparable between code
versions, not a provider bill.

usage:
  python scripts/benchmark_hs7318.py --compare <base_tree> [--json out.json]   # base vs this tree
  python scripts/benchmark_hs7318.py --repo <tree>                              # one tree (prints JSON)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
THIS_REPO = HERE.parent
REASONING_TOKENS_PER_CALL = 1200
GOV = "https://www.gov.il/BlobFolder/legalinfo"
FIO_PDF = f"{GOV}/free-import-order/he/free_import_order_2014.pdf"
STD_PDF = f"{GOV}/standards-order/he/standards_import_groups.pdf"
CERT_PDF = f"{GOV}/conformity/he/conformity_procedure.pdf"
HTML = [f"https://www.gov.il/he/departments/legalInfo/import-guide-{i:02d}" for i in range(11)]
SCHEDULE_QUOTE = 'פרט 7318.15 - ברגים ולולבים אחרים: אישור תקן לפי ת"י 1347 חלק 2'
EARLY_QUOTE = "יבואן יגיש לממונה את המסמכים הנדרשים לפי התוספת הרלוונטית לפני שחרור הטובין"
DEEP_HTML_QUOTE = "סעיף 47 - יבוא אישי של ברגים ואומים בכמות שאינה מסחרית פטור מאישור תקן"
FAKE_QUOTE = "כל הברגים פטורים מכל דרישה לפי צו יבוא חופשי ללא יוצא מן הכלל"


# ------------------------------------------------------------------ inputs
def hebrew_paragraph(i: int) -> str:
    return (f"({i}) הוראות לעניין יבוא טובין: היבואן ימסור לממונה על התקינה את תעודת ההתאמה, את דוח הבדיקה של "
            f"מעבדה מוסמכת ואת פרטי היצרן, והכול בהתאם לתנאים שנקבעו בתוספת ובנוהל שפורסם ברשומות; הממונה רשאי "
            f"לדרוש מסמכים נוספים או לסרב לשחרור הטובין אם לא הוכח כי הם מתאימים לדרישות התקן הרשמי")


def pdf_pages(n: int, special: dict[int, str]) -> list[str]:
    pages = []
    for p in range(1, n + 1):
        if p in special:
            pages.append(special[p])
            continue
        lines = [f'צו יבוא חופשי, התשע"ד-2014 - עמוד {p}', f"סעיף {p} - הגדרות ותחולה",
                 EARLY_QUOTE if p == 2 else f"({p}א) {EARLY_QUOTE[:60]} בעניין פרט {p}"]
        lines += [hebrew_paragraph(p * 10 + k)[:180] for k in range(5)]
        pages.append("\n".join(lines))
    return pages


def html_page(i: int) -> bytes:
    paras = [f"<p>{hebrew_paragraph(i * 100 + k)}</p>" for k in range(70 if i == 3 else 12 + i * 4)]
    if i == 3:  # the decisive sentence is far beyond 15,000 characters
        paras.append(f"<p>{DEEP_HTML_QUOTE}</p>")
    body = "".join(paras)
    return (f"<html><head><title>מדריך יבוא {i}</title></head><body><main><h1>מדריך יבוא {i}</h1>{body}"
            "</main></body></html>").encode("utf-8")


def build_documents(make_text_pdf) -> dict[str, tuple[bytes, str]]:
    schedule = "\n".join(["תוספת שנייה", "(סעיף 2(א))", SCHEDULE_QUOTE,
                          "ISO 4032 ו-ISO 7089 אינם תקנים רשמיים לעניין צו זה"]
                         + [hebrew_paragraph(1200 + k)[:180] for k in range(4)])
    docs = {
        FIO_PDF: (make_text_pdf(pdf_pages(140, {120: schedule})), "application/pdf"),
        STD_PDF: (make_text_pdf(pdf_pages(80, {}), ), "application/pdf"),
        CERT_PDF: (make_text_pdf(pdf_pages(30, {}), ), "application/pdf"),
    }
    for i, url in enumerate(HTML):
        docs[url] = (html_page(i), "text/html; charset=utf-8")
    return docs


def serper(query: str, num_results: int = 10) -> list[dict]:
    """Deterministic results; neighbouring queries overlap like real searches do."""
    seed = int(hashlib.sha256(query.encode()).hexdigest(), 16)
    pool = [FIO_PDF, STD_PDF, CERT_PDF] + HTML + [f"https://www.example-importers.co.il/article-{k}" for k in range(8)]
    out = []
    for k in range(10):
        url = pool[(seed + k * 3) % len(pool)]
        if any(r["url"] == url for r in out):
            continue
        out.append({"title": f"תוצאה {k} - צו יבוא חופשי ברגים", "url": url,
                    "snippet": "צו יבוא חופשי, התשע\"ד-2014 - תוספת שנייה: דרישות לאישור תקן ליבוא ברגים, אומים "
                               "ודיסקיות מפלדה; ראו גם ת\"י 1347 ונוהל הממונה על התקינה " * 2,
                    "position": k + 1, "primary_source": "gov.il" in url})
    return out[:num_results]


# 28 searches + list_local_government_datasets = the 29 local queries of the original run.
LOCAL_QUERIES = [
    ("customs_tariff", "7318"), ("free_import_order", "7318"), ("import_regulations", "7318"), ("all", "ברגים"),
    ("mandatory_standards", "ISO 4032"), ("mandatory_standards", "ISO 7089"), ("standards_declarations", "ISO 4032"),
    ("all", "אומים"), ("customs_tariff", "7318150000"), ("free_import_order", "7318150000"),
    ("import_regulations", "7318150000"), ("mandatory_standards", "ברגים"), ("standards_declarations", "ברגים"),
    ("all", "דיסקיות"), ("all", "7318.15"), ("mandatory_standards", "ISO 898"), ("customs_tariff", "אומים"),
    ("free_import_order", "7318160000"), ("customs_tariff", "7318160000"), ("all", "ISO 7089"), ("all", "4032"),
    ("mandatory_standards", "ת\"י 1347"), ("free_import_order", "ברגים"), ("import_regulations", "ברגים"),
    ("free_import_order", "73"), ("all", "לולבים"),
    ("standards_declarations", "אומים"), ("all", "ISO 4032 nuts"),
]


# ------------------------------------------------------------------ model
def scripted_turns(after: bool, doc_ids: dict, tariff_line: str) -> list[list[tuple[str, dict]]]:
    """10 research turns, then DONE, then the final report: 12 model calls, as in the original run."""
    local = [("search_local_government_records", {"dataset": d, "query": q, "phase": "reading"}) for d, q in LOCAL_QUERIES]
    searches = [("search_web", {"query": q, "phase": "searching"}) for q in (
        "צו יבוא חופשי ברגים 7318", "free import order screws Israel standard", "אישור תקן ברגים אומים יבוא",
        "ת\"י 1347 ברגים", "ISO 4032 Israel mandatory standard", "ISO 7089 washers Israel import",
        "תוספת שנייה צו יבוא חופשי", "ממונה על התקינה ברגים פטור", "יבוא אישי ברגים פטור", "קבוצות יבוא תקנים ברגים",
        "רשומות צו התקנים ברגים", "מכון התקנים ברגים בדיקה")]
    fetches = [("fetch_url", {"url": u, "phase": "reading"}) for u in [FIO_PDF, HTML[0], STD_PDF, HTML[1], HTML[2],
                                                                       HTML[3], CERT_PDF] + HTML[4:11]]
    findings = [
        {"statement": "Item 7318.15 requires a standard approval under the second schedule", "source_url": FIO_PDF,
         "excerpt": SCHEDULE_QUOTE, "page": "120", "claim_type": "legal_conclusion"},
        {"statement": "The importer must file documents before release", "source_url": FIO_PDF,
         "excerpt": EARLY_QUOTE, "page": "2"},
        {"statement": "Personal non-commercial imports of screws are exempt", "source_url": HTML[3],
         "excerpt": DEEP_HTML_QUOTE, "claim_type": "legal_conclusion", "negative": True},
        {"statement": "No Free Import Order requirement exists for 7318 (zero local records)", "source_url": FIO_PDF,
         "excerpt": FAKE_QUOTE, "negative": True},
        {"statement": "Tariff line 7318.15 exists", "source_url": "https://data.gov.il/dataset/customs_tariff",
         "excerpt": tariff_line, "resource_id": "5536eaa1-2e51-406b-aff6-b9ca02801b7c"},
        {"statement": "The schedule item is cited on the wrong page", "source_url": FIO_PDF, "excerpt": SCHEDULE_QUOTE,
         "page": "12", "claim_type": "legal_conclusion"},
    ]
    turns = [
        [("list_local_government_datasets", {}), ("get_government_snapshot_status", {})] + searches[0:2],
        local[0:4],
        searches[2:4] + fetches[0:2],
        local[4:8],
        searches[4:6] + fetches[2:4],
        local[8:12],
        searches[6:8] + fetches[4:7],
        local[12:17] + searches[8:10],
        fetches[7:11] + local[17:23] + [("record_findings", {"findings": findings[:3]})],
        searches[10:12] + fetches[11:14] + local[23:28] + [
            ("record_findings", {"findings": findings[3:]}),
            ("update_candidates", {"candidates": [{"name": "Certified screw import service", "status": "surviving"},
                                                  {"name": "Personal-import exemption", "status": "candidate"}]})],
    ]
    if after:  # the same turns, plus the new document / checklist tools (no extra model calls)
        fio = doc_ids[FIO_PDF]
        turns[3].append(("search_document", {"document_id": fio, "query": "תוספת שנייה 7318"}))
        turns[5].append(("read_document_range", {"document_id": fio, "start_page": 120, "end_page": 120}))
        turns[7].append(("search_document", {"document_id": doc_ids[STD_PDF], "query": "ISO 4032"}))
        turns[9].append(("update_checklist", {"items": [
            {"id": "governing_legal_text", "status": "resolved", "evidence": ["F1"]},
            {"id": "product_classification", "status": "resolved", "evidence": ["F5"]},
            {"id": "scope_and_exceptions", "status": "unresolved", "note": "exceptions to the schedule not found",
             "strategy_attempted": "search_document on the order; web search"},
            {"id": "validity", "status": "unresolved", "note": "no official amendment history retrieved"},
            {"id": "competitive_advantage", "status": "unresolved", "note": "rule applies to every importer"}]}))
    return turns


def final_report(tariff_line: str) -> dict:
    def src(url, excerpt, page="", resource_id=""):
        return {"title": "src", "url": url, "excerpt": excerpt, "page": page, "resource_id": resource_id,
                "section": "", "support": ""}

    scores = {k: 6 for k in ("profit_potential", "startup_capital", "operational_complexity", "regulatory_complexity",
                             "legal_risk", "regulatory_change_risk", "competition", "side_business_fit",
                             "recurring_revenue", "barrier_to_entry")}
    base = {"summary": "", "customer": "", "customer_problem": "", "revenue_model": "", "startup_capital_estimate": "",
            "existing_competition": [], "secondary_sources": [], "contradictory_sources_checked": [], "red_team": [],
            "open_legal_questions": [], "scores": scores, "business_score": 60, "confidence": 50}
    tariff_src = src("https://data.gov.il/dataset/customs_tariff", tariff_line,
                     resource_id="5536eaa1-2e51-406b-aff6-b9ca02801b7c")
    return {"research_summary": "HS 7318 import requirements.", "no_opportunity_reason": "", "opportunities": [
        {**base, "name": "Certified screw import service", "classification": "A",
         "regulatory_mechanism": "Second schedule item 7318.15 requires a standard approval.",
         "business_thesis": "Importers need certification support.",
         "primary_sources": [src(FIO_PDF, SCHEDULE_QUOTE, "120"), tariff_src],
         "legal_checks": {"provision": {"status": "checked", "evidence": [src(FIO_PDF, SCHEDULE_QUOTE, "120")]},
                          "product_classification": {"status": "checked", "evidence": [tariff_src]}},
         "business_advantage": {"claim": "Faster approvals", "generally_available": True}},
        {**base, "name": "Personal-import exemption reseller", "classification": "B",
         "regulatory_mechanism": "Personal imports of screws are exempt (פטור) from a standard approval.",
         "business_thesis": "Exploit the exemption.",
         "primary_sources": [src(HTML[3], DEEP_HTML_QUOTE), src(FIO_PDF, FAKE_QUOTE)]},
    ]}


class ScriptedLLM:
    """Replays the turns; attaches a Responses-style reasoning item to every answer (as GPT-6.1 Sol does)."""

    provider, model = "openai", "gpt-6.1-sol (scripted)"

    def __init__(self, llm_module, turns, report):
        self.llm = llm_module
        self.turns = list(turns)
        self.report = report
        self.calls = []
        self.warnings = []

    def chat(self, messages, tools=None, json_mode=False):
        payload = self.llm.to_responses_input(messages)
        tool_defs = self.llm.to_responses_tools(tools) if tools else []
        self.calls.append({"input": payload, "tools": tool_defs, "json_mode": json_mode})
        n = len(self.calls)
        reasoning = {"type": "reasoning", "id": f"rs_{n}", "summary": [],
                     "encrypted_content": "R" * (REASONING_TOKENS_PER_CALL * 4)}
        if json_mode:
            text = json.dumps(self.report, ensure_ascii=False)
            return self.llm.ChatResponse(content=text, tool_calls=[], finish_reason="stop",
                                         message={"role": "assistant", "content": text,
                                                  "_responses_output": [reasoning, {"role": "assistant", "content": text}]},
                                         usage={})
        if not self.turns:
            return self.llm.ChatResponse(content="DONE", tool_calls=[], finish_reason="stop", usage={},
                                         message={"role": "assistant", "content": "DONE",
                                                  "_responses_output": [reasoning, {"role": "assistant", "content": "DONE"}]})
        calls = [self.llm.ToolCall(id=f"call_{n}_{i}", name=name, arguments=json.dumps(args, ensure_ascii=False))
                 for i, (name, args) in enumerate(self.turns.pop(0))]
        replay = [reasoning] + [{"type": "function_call", "call_id": c.id, "name": c.name, "arguments": c.arguments}
                                for c in calls]
        return self.llm.ChatResponse(content="", tool_calls=calls, finish_reason="tool_calls", usage={}, message={
            "role": "assistant", "content": "", "_responses_output": replay,
            "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                           for c in calls]})


def payload_tokens(estimate, call) -> int:
    return estimate(json.dumps(call["input"], ensure_ascii=False)) + estimate(json.dumps(call["tools"], ensure_ascii=False))


# ------------------------------------------------------------------- run
def run_tree(tree: Path, budget: int | None) -> dict:
    sys.path[:0] = [str(tree), str(THIS_REPO / "tests"), str(THIS_REPO)]
    tmp = Path(tempfile.mkdtemp(prefix="bench-"))
    os.environ.update(DOCUMENT_CACHE_DIR=str(tmp / "docs"), GOVDATA_CACHE_DIR=str(tmp / "gov"), GOVDATA_WARMUP="0")
    for name in ("CRITICAL_EVIDENCE_CHECKLIST", "CONTEXT_BUDGET_TOKENS", "DOCUMENT_CACHE_MAX_AGE_HOURS"):
        os.environ.pop(name, None)  # always the default configuration
    if budget is not None:
        os.environ["CONTEXT_BUDGET_TOKENS"] = str(budget)
    import importlib

    agent = importlib.import_module("agent")
    fetcher = importlib.import_module("fetcher")
    llm = importlib.import_module("llm")
    local_data = importlib.import_module("local_data")
    sys.path.insert(0, str(THIS_REPO))
    from context import estimate_text_tokens  # one estimator for both trees
    from helpers import make_text_pdf

    after = hasattr(agent, "context")
    docs = build_documents(make_text_pdf)
    doc_ids = {u: "doc-" + hashlib.sha256(b).hexdigest()[:16] for u, (b, _) in docs.items()}
    gov = local_data.GovernmentData(data_dir=Path(tree) / "data" / "government", cache_dir=tmp / "gov")
    tariff = gov.search("customs_tariff", "7318150000", limit=5)
    tariff_line = next(r["record_line"] for r in tariff["records"] if r["record_id"] == "8297")

    class Resp:
        def __init__(self, url, body, ctype):
            self.status_code, self.url, self.encoding, self.is_redirect = 200, url, "utf-8", False
            self.headers, self._body = {"Content-Type": ctype}, body

        def iter_content(self, size):
            for i in range(0, len(self._body), size):
                yield self._body[i:i + size]

        def close(self):
            pass

    class Session:
        def get(self, url, **kw):
            body, ctype = docs[url]
            return Resp(url, body, ctype)

    fetcher._is_public_host = lambda host: True
    fetch = lambda url: fetcher.fetch_url(url, session=Session(), respect_robots=False)
    scripted = ScriptedLLM(llm, scripted_turns(after, doc_ids, tariff_line), final_report(tariff_line))
    a = agent.ResearchAgent(scripted, search_fn=serper, fetch_fn=fetch, gov_data=gov,
                            limits=agent.Limits(max_steps=25, max_searches=30, max_fetches=20))
    gov.index_path()  # build the index before timing
    started = time.monotonic()
    run = a.run("יבוא ברגים, אומים ודיסקיות (פרט מכס 7318)")
    elapsed = time.monotonic() - started

    per_call = [payload_tokens(estimate_text_tokens, c) for c in scripted.calls]
    trace = run.trace
    # Legal document extraction completeness (the 140-page order; schedule on page 120).
    fio_fetch = next(f for f in trace["fetches"] if f["url"] == FIO_PDF)
    full_doc = fetcher.fetch_url(FIO_PDF, session=Session(), respect_robots=False, max_chars=10**9) if after else None
    if after:
        meta = a.docs.meta(doc_ids[FIO_PDF])
        extraction = {"pages_extracted": meta["page_count"], "pages_total": 140, "chars_available": meta["chars"],
                      "status": meta["status"], "schedule_page_120_retrievable": bool(
                          a.docs.locate_excerpt(doc_ids[FIO_PDF], SCHEDULE_QUOTE))}
    else:
        text = a.retrieved_text.get(agent.normalize_url(FIO_PDF), "")
        pages = sorted({int(p) for p in __import__("re").findall(r"\[page (\d+)\]", text)})
        extraction = {"pages_extracted": len(pages), "pages_total": 140, "chars_available": len(text),
                      "status": "truncated" if fio_fetch["truncated"] else "complete",
                      "schedule_page_120_retrievable": SCHEDULE_QUOTE in text,
                      "hebrew_characters_in_text": sum("֐" <= c <= "׿" for c in text)}
    # Citation accuracy over the recorded findings (2 real quotes beyond the old limits, 1 early, 1 record,
    # 1 fabricated, 1 wrong page).
    findings = a.findings
    real = [f for f in findings if f["excerpt"] != FAKE_QUOTE]
    citations = {"real_quotes": len(real),
                 "real_quotes_source_verified": sum(1 for f in real if f.get("verified")),
                 "fabricated_quotes_accepted": sum(1 for f in findings if f["excerpt"] == FAKE_QUOTE and f.get("verified"))}
    if after:
        citations["wrong_page_citations_flagged"] = sum(1 for f in findings if "is wrong" in f["verification_note"])
    result = run.result
    opps = []
    for o in (result.opportunities if result else []):
        entry = {"name": o.name, "class": o.classification, "source": o.verification_status}
        if after:
            entry.update(legal=o.legal_verification, business=o.business_verification, negative_claim=o.negative_claim)
        opps.append(entry)
    local_noise = {}
    for q in trace["local_queries"]:
        key = f"{q['args'].get('dataset')}:{q['args'].get('query')}"
        if key in ("mandatory_standards:ISO 4032", "mandatory_standards:ISO 7089", "all:4032", "all:ISO 7089"):
            local_noise[key] = q.get("total_matches")
    out = {
        "tree": str(tree), "version": "after" if after else "before",
        "model_calls": len(scripted.calls), "searches": a.search_count, "fetches": a.fetch_count,
        "local_queries": a.local_query_count, "document_queries": getattr(a, "document_query_count", 0),
        "estimated_input_tokens_total": sum(per_call), "estimated_input_tokens_per_call": per_call,
        "reasoning_tokens_replayed": sum(sum(1 for it in c["input"] if it.get("type") == "reasoning")
                                         for c in scripted.calls) * REASONING_TOKENS_PER_CALL,
        "compactions": len(trace.get("compactions", [])), "elapsed_s_local": round(elapsed, 2),
        "stop_reason": run.stop_reason, "extraction_fio_order": extraction, "citations": citations,
        "findings": [{k: f.get(k) for k in ("statement", "verified", "source_status", "legal_status",
                                            "matched_pages")} for f in findings],
        "opportunities": opps, "local_noise_total_matches": local_noise,
        "verified_findings_counter": f"{sum(1 for f in findings if f.get('verified'))}/{len(findings)}",
    }
    if after:
        out["verification_counters"] = trace.get("verification_counters")
        out["checklist"] = {i["id"]: i["status"] for i in trace.get("checklist", [])}
        out["token_usage_by_phase_estimated"] = {k: v.get("estimated_input_tokens") for k, v in
                                                 trace.get("token_usage_by_phase", {}).items()}
        out["unresolved_questions"] = result.unresolved_questions if result else []
    return out


def compare(base: Path, json_out: str | None) -> None:
    def child(tree, budget=None):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--repo", str(tree)]
        if budget is not None:
            cmd += ["--budget", str(budget)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode:
            raise SystemExit(proc.stderr[-4000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    before, after = child(base), child(THIS_REPO)
    no_compaction = child(THIS_REPO, budget=10**9)
    report = {"before": before, "after": after, "after_without_compaction": no_compaction}
    b, a = before["estimated_input_tokens_total"], after["estimated_input_tokens_total"]
    rows = [
        ("Model calls", before["model_calls"], after["model_calls"]),
        ("Searches / fetches / local queries", f"{before['searches']} / {before['fetches']} / {before['local_queries']}",
         f"{after['searches']} / {after['fetches']} / {after['local_queries']} (+{after['document_queries']} document)"),
        ("Estimated input tokens (all calls)", f"{b:,}", f"{a:,} ({(1 - a / b) * 100:.0f}% less)"),
        ("  of which replayed reasoning", f"{before['reasoning_tokens_replayed']:,}", f"{after['reasoning_tokens_replayed']:,}"),
        ("Estimated input tokens, last call", f"{before['estimated_input_tokens_per_call'][-1]:,}",
         f"{after['estimated_input_tokens_per_call'][-1]:,}"),
        ("Context compactions", 0, after["compactions"]),
        ("140-page order: pages / chars available", f"{before['extraction_fio_order']['pages_extracted']} / "
         f"{before['extraction_fio_order']['chars_available']:,}",
         f"{after['extraction_fio_order']['pages_extracted']} / {after['extraction_fio_order']['chars_available']:,}"),
        ("Schedule (page 120) retrievable", before["extraction_fio_order"]["schedule_page_120_retrievable"],
         after["extraction_fio_order"]["schedule_page_120_retrievable"]),
        ("Real quotes source-verified", f"{before['citations']['real_quotes_source_verified']}/"
         f"{before['citations']['real_quotes']}", f"{after['citations']['real_quotes_source_verified']}/"
         f"{after['citations']['real_quotes']}"),
        ("Fabricated quotes accepted", before["citations"]["fabricated_quotes_accepted"],
         after["citations"]["fabricated_quotes_accepted"]),
        ("Wrong page citations flagged", "n/a", after["citations"].get("wrong_page_citations_flagged")),
        ("Local runtime (s, no network/model latency)", before["elapsed_s_local"], after["elapsed_s_local"]),
    ]
    print(f"{'Measure':45} | {'before':>28} | after")
    for name, x, y in rows:
        print(f"{name:45} | {str(x):>28} | {y}")
    print(f"\nAfter without compaction (budget disabled): {no_compaction['estimated_input_tokens_total']:,} "
          f"estimated input tokens")
    print("Local noise (total matches) before:", before["local_noise_total_matches"])
    print("Local noise (total matches) after: ", after["local_noise_total_matches"])
    print("Opportunities before:", before["opportunities"])
    print("Opportunities after: ", after["opportunities"])
    print("Verification counters after:", after["verification_counters"], "checklist:", after["checklist"])
    if json_out:
        Path(json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--compare")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--json")
    ns = ap.parse_args()
    if ns.compare:
        compare(Path(ns.compare).resolve(), ns.json)
    else:
        print(json.dumps(run_tree(Path(ns.repo or THIS_REPO).resolve(), ns.budget), ensure_ascii=False))
