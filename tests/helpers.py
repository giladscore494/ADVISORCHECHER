import json

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


def valid_report(url="https://www.gov.il/he/departments/legalInfo/regulation-x", n=1, cls="A") -> dict:
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
        "primary_sources": [{"title": "Regulation X", "url": url, "section": "4", "support": "Duty to inspect"}],
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
    return FetchResult(url=url, final_url=url, ok=True, source_type="html", title="Regulation X", text="§4 ...")
