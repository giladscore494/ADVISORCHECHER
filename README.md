# Regulatory Opportunity Hunter

An AI research agent that searches Israeli laws, regulations, government guidance, licensing rules,
exemptions, thresholds, mandatory requirements and incentives to find **lawful business opportunities
created by regulation**. It is a discovery tool: it produces evidence-backed hypotheses that are worth
further business and professional legal validation.

> **Disclaimer.** This tool performs preliminary research and does not provide legal advice. Any
> opportunity involving legal or regulatory interpretation must be independently verified by a qualified
> professional before money is invested or operations begin.

## What it does

1. You enter a domain (e.g. *equipment rental*, *mandatory inspections*, *waste and recycling*) and click **Start Research**.
2. The agent works on its own, in Hebrew and English: it maps the regulatory environment, searches for
   laws and regulations, reads the actual sources (HTML and PDF), generates candidate mechanisms,
   **red-teams** each one (looking for overriding laws, licensing, standards, zoning, tax and so on),
   checks the market (competitors, pricing, demand) and ranks what survives.
3. The UI shows live progress (phase, searches, pages read, candidates, current action) and then only the
   strongest 3-5 opportunities. If nothing survives, it says **"No sufficiently strong opportunity found."**
4. Each opportunity shows its classification (A: explicit advantage, B: plausible but ambiguous,
   C: apparent loophole), scores, business thesis, regulatory mechanism, clickable primary sources,
   red-team findings and open legal questions.
5. A **Research Trace** lists every search, every URL inspected (succeeded or failed) and the candidate
   funnel, and can be downloaded as JSON for audit.

## Architecture

```
app.py        Streamlit UI: input, live progress, result cards, research trace
agent.py      Controlled tool-calling loop, limits, duplicate detection, trace, finalization
llm.py        Provider-agnostic chat client (Kimi by default, GLM ready) over the OpenAI-compatible API
search.py     search_web(query, num_results) via Serper, plus primary-source domain detection
fetcher.py    fetch_url(url): HTML text extraction (BeautifulSoup) and PDF text (pypdf), with safety limits
prompts.py    System prompt (research strategy, red team, legal safety), final-report prompt
models.py     Pydantic schema for the final report, strict JSON parsing and validation, ranking
config.py     Settings from environment, .env, or Streamlit secrets
tests/        Unit tests plus a Streamlit AppTest UI test (all network calls mocked)
```

### Agent loop

The model gets three tools: `search_web`, `fetch_url` and `update_candidates` (which records the
candidate funnel). Each step:

1. The model is called with the conversation plus the tool definitions.
2. If it requests tools, `agent.py` executes them under hard limits and returns the results as tool messages.
3. If it stops calling tools (or a limit is hit), the agent asks for the final report in JSON mode.
4. The report is validated with Pydantic. If it is malformed, the validation error is sent back once for a
   repair attempt. If it is still invalid, the run reports an error. Malformed output is never shown as a result.

Limits and safeguards:
- `MAX_AGENT_STEPS` (default 25), maximum searches (30), maximum page fetches (20), all adjustable in the UI.
- Identical searches (case and whitespace normalized) and identical URLs (fragment and trailing slash
  normalized) are rejected.
- Three consecutive rounds with no successful tool call end the loop gracefully.
- A failed search or fetch (403, 404, timeout, unparsable PDF) is returned to the model as an error. It
  never kills the run.
- Primary sources cited in the final report that were not actually read during the run are flagged in the UI.

The model's private reasoning is never displayed. The UI shows only the short public `purpose` the model
attaches to each tool call (e.g. "Checking contradictory licensing requirements").

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in the keys
streamlit run app.py
```

Instead of `.env`, you can put the same keys in `.streamlit/secrets.toml` (handy for Streamlit Cloud).
Both files are git-ignored. Never commit credentials.

### Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `KIMI_API_KEY` | yes | | Moonshot / Kimi API key |
| `SERPER_API_KEY` | yes | | https://serper.dev API key |
| `KIMI_BASE_URL` | no | `https://api.moonshot.ai/v1` | OpenAI-compatible endpoint |
| `KIMI_MODEL` | no | `kimi-k3` | |
| `KIMI_REASONING_EFFORT` | no | `max` | Sent as `reasoning_effort`; dropped automatically if the endpoint rejects it |
| `LLM_TIMEOUT` | no | `600` | Seconds per model call |
| `MAX_AGENT_STEPS` | no | `25` | Default for the UI setting |
| `MAX_SEARCHES` / `MAX_FETCHES` | no | `30` / `20` | Defaults for the UI settings |
| `LLM_PROVIDER` | no | `kimi` | `kimi` or `glm` |
| `GLM_API_KEY`, `GLM_BASE_URL`, `GLM_MODEL` | if `glm` | `https://api.z.ai/api/paas/v4`, `glm-5.3` | |

## Model

The default model is **Kimi K3** (`kimi-k3`) with `reasoning_effort=max`, called through Moonshot's
OpenAI-compatible API using the `openai` Python SDK. K3 does not accept sampling parameters, so none are
sent. On tool-call turns, the assistant's `reasoning_content` is echoed back in the history as Kimi's
thinking models require, but it is never displayed.

To switch providers, set `LLM_PROVIDER=glm` (or add an entry to `PROVIDERS` in `llm.py`). Nothing else
depends on provider details.

## Search (Serper)

`search.search_web(query, num_results=10)` calls `https://google.serper.dev/search` with `gl=il` and
returns `title`, `url`, `snippet`, `position` and `primary_source`. Official Israeli domains (`gov.il`
and its subdomains, `knesset.gov.il`, `court.gov.il`, `boi.org.il`, `sii.org.il`) are marked as primary.
The prompt tells the model that secondary sources (law firms, news, blogs) support discovery and market
validation only, never a legal conclusion.

## Source fetching

`fetcher.fetch_url(url)`:
- HTML: removes scripts, styles, nav, header, footer and forms, prefers `<main>` or `<article>`, and collapses whitespace.
- PDF: detected by content type, `%PDF` magic bytes or a `.pdf` path. Extracts text with `pypdf` (first 60
  pages) and labels each page. OCR is not supported: a scanned PDF is reported as having no extractable text.
- Limits: 10s connect / 30s read timeout, 15 MB download cap, 15,000 extracted characters (truncation is flagged).
- Only http(s). Private, loopback and link-local hosts are refused, and redirects are followed manually so
  each hop is checked.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

These cover HTML and PDF parsing, fetch failure handling, JSON validation, duplicate-tool detection,
limits, final-report repair, the LLM client (with mocked SDK responses) and the Streamlit UI flow
(`streamlit.testing.v1.AppTest`). No API keys are needed.

## Known MVP limitations

- No OCR. Scanned PDFs and JavaScript-rendered pages produce no text.
- Some gov.il pages render content client-side or block automated clients, so the agent may need to fall
  back to PDFs or other mirrors of the official text.
- Progress updates arrive per tool call. A long model call (max reasoning) shows no intermediate updates.
- Everything is kept in memory for one session. There is no persistence, authentication or caching.
- Output quality depends entirely on the model's research. Always check the cited sources yourself.
