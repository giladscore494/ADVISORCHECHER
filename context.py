"""Context management for the research loop: token estimates, history compaction and the research digest.

Why: every model call resends the conversation. Before compaction a run kept every fetched page, every
search snippet and every dataset record in the history forever (and, for the Responses API, every earlier
reasoning item), so 12 calls cost ~900K input tokens. Now:

* Every tool result gets an evidence id (E1, E2, ...) and is stored in full in the run's evidence ledger
  (checkpointed). The model can always retrieve it again with get_evidence(evidence_id).
* When the history exceeds the context budget, tool results older than the most recent rounds are
  replaced by compact, deterministic stubs (ids, URLs, document ids, page numbers, record ids, short
  quotes) and old reasoning payloads are dropped. Compaction is sticky (an already compacted message never
  changes again), so the provider's prompt-prefix cache stays valid between compactions.
* A research digest is appended to each call after compaction: recorded findings with their verbatim
  quotes and citations (never compacted), documents, completed-phase summaries, queries already tried,
  the critical-evidence checklist and the evidence index.
"""

from __future__ import annotations

import json
from typing import Any

# Rough token estimate: ~4 ASCII characters per token, ~2 per non-ASCII (Hebrew) character. Used for the
# context budget and for comparable measurements; providers report the real usage separately.
ASCII_CHARS_PER_TOKEN = 4.0
OTHER_CHARS_PER_TOKEN = 2.0
SNIPPET = 240
MAX_DIGEST_EVIDENCE = 80
MAX_DIGEST_QUERIES = 40
REASONING_KEEP_CHARS = 400


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = len(text.encode("ascii", "ignore"))
    return int(ascii_chars / ASCII_CHARS_PER_TOKEN + (len(text) - ascii_chars) / OTHER_CHARS_PER_TOKEN) + 1


def message_tokens(m: dict) -> int:
    total = 4 + estimate_text_tokens(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        total += estimate_text_tokens(tc.get("function", {}).get("arguments") or "") + 8
    total += estimate_text_tokens(m.get("reasoning_content") or "")
    for item in m.get("_responses_output") or []:
        if item.get("type") == "reasoning":
            total += estimate_text_tokens(item.get("encrypted_content") or "")
    return total


def estimate_tokens(messages: list[dict]) -> int:
    return sum(message_tokens(m) for m in messages)


def _short(text: Any, n: int = SNIPPET) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def summarize_result(tool: str, args: dict, result: dict) -> dict:
    """A compact stub that keeps identifiers, citations and short quotes of a tool result."""
    if "error" in result and len(result) <= 4:
        return {k: result[k] for k in ("error", "document_id") if k in result}
    if tool == "search_web":
        return {"query": args.get("query", ""), "results": [
            {"title": _short(r.get("title"), 90), "url": r.get("url"), "primary_source": r.get("primary_source")}
            for r in (result.get("results") or [])[:10]]}
    if tool == "fetch_url":
        out = {k: result.get(k) for k in ("url", "ok", "document_id", "title", "source_type", "primary_source",
                                          "error", "http_status") if result.get(k) not in (None, "")}
        if result.get("document"):
            out["document"] = {k: result["document"].get(k) for k in ("unit", "page_count", "status")}
        out["text"] = "[compacted: use search_document / read_document_range on document_id]"
        return out
    if tool == "search_document":
        return {"document_id": result.get("document_id"), "query": result.get("query"),
                "pages_with_matches": result.get("pages_with_matches", [])[:30],
                "results": [{"page": h.get("page"), "lines": f"{h.get('first_line')}-{h.get('last_line')}",
                             "text": _short(h.get("text"))} for h in (result.get("results") or [])[:5]]}
    if tool == "read_document_range":
        return {"document_id": result.get("document_id"), "source_url": result.get("source_url"),
                "pages": [{"page": p.get("page"), "start": _short(p.get("text"), 160)}
                          for p in result.get("pages") or []],
                "text": "[compacted: read the pages again or use get_evidence]"}
    if tool == "get_document_status":
        return {k: result.get(k) for k in ("document_id", "status", "page_count", "pages_without_text", "issues")}
    if tool in ("search_local_government_records", "get_local_government_record"):
        recs = result.get("records") or ([result] if result.get("found") else [])
        out = {"query": args.get("query") or args.get("record_id"), "dataset": args.get("dataset"),
               "total_matches": result.get("total_matches", len(recs)),
               "records": [{"ref": f"{r.get('dataset')}:{r.get('record_id')}", "match": r.get("match", ""),
                            "line": _short(r.get("record_line"), 200)} for r in recs[:8]]}
        if not recs:
            out["note"] = "Zero matches (not proof of an exemption)."
        return out
    if tool == "search_government_datasets":
        return {"query": args.get("query"), "results": [
            {"id": d.get("id"), "title": _short(d.get("title"), 90), "likely_relevant": d.get("likely_relevant")}
            for d in (result.get("results") or [])[:10]]}
    if tool == "inspect_government_dataset":
        return {k: result.get(k) for k in ("status", "dataset_id", "id", "title", "publisher") if result.get(k)} | {
            "resources": [{"id": r.get("id"), "name": _short(r.get("name"), 80)} for r in result.get("resources") or []][:10]}
    if tool == "read_government_resource":
        return {"status": result.get("status"), "resource_id": args.get("resource_id"), "query": args.get("query"),
                "records": _short(result.get("records"), 600) if isinstance(result.get("records"), str) else None}
    if tool == "get_evidence":
        return {"retrieved_evidence_id": result.get("evidence_id"), "note": "re-retrieved evidence (compacted again)"}
    # record_findings, update_candidates, update_checklist, list/status tools: already small.
    text = json.dumps(result, ensure_ascii=False)
    return result if len(text) <= 1500 else {"summary": _short(text, 1200)}


def stub_content(evidence_id: str, tool: str, args: dict, result: dict) -> str:
    out: dict = {"evidence_id": evidence_id, "compacted": True}
    out.update({k: v for k, v in summarize_result(tool, args, result).items() if k not in out})
    out["full_result"] = f"get_evidence('{evidence_id}')"
    return json.dumps(out, ensure_ascii=False)


def strip_old_reasoning(m: dict) -> bool:
    """Drop replayed reasoning payloads from an older assistant message. Returns True if it changed."""
    changed = False
    if m.get("_responses_output"):
        kept = [it for it in m["_responses_output"] if it.get("type") != "reasoning"]
        if len(kept) != len(m["_responses_output"]):
            m["_responses_output"] = kept
            changed = True
    rc = m.get("reasoning_content")
    if rc and len(rc) > REASONING_KEEP_CHARS:
        # Kept (shortened) rather than removed: some chat endpoints expect the field on tool-call turns.
        m["reasoning_content"] = rc[:REASONING_KEEP_CHARS] + " […earlier reasoning compacted]"
        changed = True
    return changed


def recent_start(messages: list[dict], keep_rounds: int) -> int:
    """Index of the first message of the last `keep_rounds` assistant rounds (never before the 2 prompts)."""
    seen = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            seen += 1
            if seen >= keep_rounds:
                return max(i, 2)
    return 2


def compact(messages: list[dict], ledger: dict, keep_rounds: int) -> int:
    """Compact everything before the recent window in place. Returns the number of messages changed."""
    changed = 0
    for m in messages[: recent_start(messages, keep_rounds)]:
        if m.get("role") == "tool" and m.get("_evidence_id") and not m.get("_compacted"):
            entry = ledger.get(m["_evidence_id"])
            if entry is None:
                continue
            m["content"] = stub_content(entry["id"], entry["tool"], entry.get("args") or {}, entry["result"])
            m["_compacted"] = True
            changed += 1
        elif m.get("role") == "assistant" and strip_old_reasoning(m):
            changed += 1
    return changed
