import json
import re

from fetcher import FetchResult
from llm import ChatResponse, ToolCall


def make_pdf(text: str) -> bytes:
    """Build a minimal valid one-page PDF containing `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


# A sentence present in every mocked document, so reports can quote it as their supporting excerpt.
EXCERPT = "תקנה 4: בעל נגרר יבצע בדיקה תקופתית אחת לשנה"
SOURCE_TEXT = f"פרק ב.\n{EXCERPT}, בתחנת בדיקה מורשית.\n§5 ..."


def legal_checks(url, excerpt=None, status="checked") -> dict:
    """Every legal-applicability aspect checked against the same quoted official text."""
    ev = [{"url": url, "excerpt": excerpt or EXCERPT}]
    return {k: {"status": status, "finding": f"{k} checked", "evidence": ev}
            for k in ("provision", "scope", "validity", "exceptions", "product_classification")}


def valid_report(url="https://www.gov.il/he/departments/legalInfo/regulation-x", n=1, cls="A", legal=True) -> dict:
    opp = {
        "name": "Inspection equipment rental",
        "summary": "Rent certified equipment needed for mandatory periodic checks.",
        "regulatory_mechanism": "Regulation X §4 requires a periodic check; self-check allowed under §5.",
        "classification": cls,
        "business_thesis": "Owners must check yearly; renting the device is cheaper than a service visit.",
        "customer": "Small fleet owners",
        "customer_problem": "Annual compliance cost",
        "revenue_model": "Daily rental",
        "startup_capital_estimate": "ILS 15,000",
        "existing_competition": ["Two service firms"],
        "primary_sources": [{"title": "Regulation X", "url": url, "section": "4", "support": "Duty to inspect",
                             "excerpt": EXCERPT}],
        "secondary_sources": [],
        "contradictory_sources_checked": [],
        "red_team": ["VERIFIED FACT: no licence needed for rental"],
        "open_legal_questions": ["Does §5 cover commercial users?"],
        "scores": {k: 7 for k in (
            "profit_potential", "startup_capital", "operational_complexity", "regulatory_complexity",
            "legal_risk", "regulatory_change_risk", "competition", "side_business_fit",
            "recurring_revenue", "barrier_to_entry")},
        "business_score": 70,
        "confidence": 60,
    }
    if legal:
        opp["legal_checks"] = legal_checks(url)
    return {"research_summary": "Checked.", "no_opportunity_reason": "", "opportunities": [dict(opp, name=f"{opp['name']} {i}") for i in range(n)]}


def tool_response(*calls) -> ChatResponse:
    tcs = [ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args)) for i, (name, args) in enumerate(calls)]
    return ChatResponse(
        content="", tool_calls=tcs, finish_reason="tool_calls",
        message={"role": "assistant", "content": "", "tool_calls": [
            {"id": t.id, "type": "function", "function": {"name": t.name, "arguments": t.arguments}} for t in tcs]},
    )


def text_response(text: str) -> ChatResponse:
    return ChatResponse(content=text, tool_calls=[], finish_reason="stop", message={"role": "assistant", "content": text})


class FakeLLM:
    """Replays scripted responses; once the script is exhausted, repeats the last one."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.warnings = []

    def chat(self, messages, tools=None, json_mode=False):
        self.calls.append({"messages": list(messages), "tools": tools, "json_mode": json_mode})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def fake_search(query, num_results=10):
    return [{"title": f"Result for {query}", "url": "https://www.gov.il/he/departments/legalInfo/regulation-x",
             "snippet": "...", "position": 1, "primary_source": True}]


def fake_fetch(url):
    return FetchResult(url=url, final_url=url, ok=True, source_type="html", title="Regulation X", text=SOURCE_TEXT)


_HEB = re.compile(r"[֐-׿]")


def _bidi_runs(line: str) -> list[tuple[str, bool]]:
    """Split a logical right-to-left line into (text, is_rtl) runs, resolving neutrals like the Unicode bidi
    algorithm in an RTL paragraph: a neutral between two left-to-right characters stays left-to-right."""
    kinds = []
    for ch in line:
        kinds.append("R" if _HEB.match(ch) else ("L" if ch.isascii() and ch.isalnum() else "N"))
    resolved = []
    for i, k in enumerate(kinds):
        if k != "N":
            resolved.append(k == "R")
            continue
        prev = next((kinds[j] for j in range(i - 1, -1, -1) if kinds[j] != "N"), "R")
        nxt = next((kinds[j] for j in range(i + 1, len(kinds)) if kinds[j] != "N"), "R")
        resolved.append(not (prev == "L" and nxt == "L"))
    runs: list[tuple[str, bool]] = []
    for ch, rtl in zip(line, resolved):
        if runs and runs[-1][1] == rtl:
            runs[-1] = (runs[-1][0] + ch, rtl)
        else:
            runs.append((ch, rtl))
    return runs


def make_text_pdf(pages: list[str], blank_pages: tuple[int, ...] = (), font_size: int = 10) -> bytes:
    """Multi-page PDF with arbitrary Unicode text (a ToUnicode CMap maps one byte per character).
    Hebrew lines are laid out like real government PDFs: glyphs stored in visual order and positioned right to
    left. `blank_pages` (1-based) get no text at all, like scanned image pages."""
    chars = sorted({c for p in pages for c in p if c != "\n"} | {" "})
    if len(chars) > 220:
        raise ValueError("too many distinct characters for a one-byte encoding")
    code = {c: 0x21 + i for i, c in enumerate(chars)}

    def enc(s):
        return "".join("\\%03o" % code[c] for c in s)

    items = [f"<{code[c]:02X}> <{ord(c):04X}>" for c in chars]
    cmap = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap", "/CMapName /Custom def",
            "/CMapType 2 def", "1 begincodespacerange <00> <FF> endcodespacerange"]
    for i in range(0, len(items), 100):
        chunk = items[i:i + 100]
        cmap += [f"{len(chunk)} beginbfchar", *chunk, "endbfchar"]
    cmap += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    cmap_b = "\n".join(cmap).encode()
    width = font_size * 0.5
    objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>",
            # A non-standard font name, so parsers use the /Widths array (as with fonts embedded in real PDFs).
            3: (b"<< /Type /Font /Subtype /TrueType /BaseFont /FixtureHebrew /FirstChar 33 /LastChar 255 /Widths ["
                + b" ".join([b"500"] * 223) + b"] /FontDescriptor 4 0 R /ToUnicode 5 0 R >>"),
            4: (b"<< /Type /FontDescriptor /FontName /FixtureHebrew /Flags 32 /FontBBox [0 -200 1000 900] "
                b"/ItalicAngle 0 /Ascent 900 /Descent -200 /CapHeight 700 /StemV 80 /MissingWidth 500 >>"),
            5: b"<< /Length %d >>\nstream\n" % len(cmap_b) + cmap_b + b"\nendstream"}
    n = len(pages)
    objs[2] = ("<< /Type /Pages /Kids [" + " ".join(f"{6 + 2 * i} 0 R" for i in range(n)) + f"] /Count {n} >>").encode()
    for i, text in enumerate(pages):
        ops, y = ["BT", f"/F1 {font_size} Tf"], 800
        if i + 1 not in blank_pages:
            for line in text.split("\n"):
                if _HEB.search(line):
                    x = 570.0
                    for run, rtl in _bidi_runs(line):
                        x -= len(run) * width
                        ops.append(f"1 0 0 1 {x:.1f} {y} Tm ({enc(run[::-1] if rtl else run)}) Tj")
                else:
                    ops.append(f"1 0 0 1 40 {y} Tm ({enc(line)}) Tj")
                y -= 14
        ops.append("ET")
        stream = "\n".join(ops).encode()
        objs[6 + 2 * i] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Contents {7 + 2 * i} 0 R "
                           f"/Resources << /Font << /F1 3 0 R >> >> >>").encode()
        objs[7 + 2 * i] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    out, offsets = bytearray(b"%PDF-1.4\n"), {}
    for k in sorted(objs):
        offsets[k] = len(out)
        out += b"%d 0 obj\n" % k + objs[k] + b"\nendobj\n"
    xref, total = len(out), max(objs) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % total
    for k in range(1, total):
        out += b"%010d 00000 n \n" % offsets[k]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (total, xref)
    return bytes(out)
