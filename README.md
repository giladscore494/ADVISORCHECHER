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
   Optionally add **Custom Research Instructions** (Hebrew or English): goals, business constraints, priorities,
   exclusions or questions. They steer the research but cannot override the legal safeguards, source
   verification, red-team process or run limits, and they are shown in the Research Trace.
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
fetcher.py    fetch_url(url): HTML, PDF, JSON, CSV, XLSX; retries, robots.txt, SSRF and size limits
datagov.py    data.gov.il CKAN client (package_search/package_show/resource_show/datastore_search) and
              the dataset read pipeline with relevance checks and provenance
evidence.py   Hebrew/English term matching (dataset relevance) and quoted-excerpt verification
prompts.py    System prompt (research strategy, red team, legal safety), final-report prompt
models.py     Pydantic schema for the final report, strict JSON parsing and validation, ranking
config.py     Settings from environment, .env, or Streamlit secrets
scripts/      smoke_datagov.py: optional read-only live check against data.gov.il
tests/        Unit tests plus a Streamlit AppTest UI test (all network calls mocked)
```

### Agent loop

The model gets these tools:
- `search_web`: discover websites and documents (Serper).
- `fetch_url`: read an individual law, regulation, guidance page or PDF.
- `search_government_datasets`, `inspect_government_dataset`, `read_government_resource`: structured
  official data from data.gov.il (see below).
- `update_candidates`: records the candidate funnel.

Each step:

1. The model is called with the conversation plus the tool definitions.
2. If it requests tools, `agent.py` executes them under hard limits and returns the results as tool messages.
3. If it stops calling tools (or a limit is hit), the agent asks for the final report in JSON mode.
4. The report is validated with Pydantic. If it is malformed, the validation error is sent back once for a
   repair attempt. If it is still invalid, the run reports an error. Malformed output is never shown as a result.

Limits and safeguards:
- `MAX_AGENT_STEPS` (default 25), maximum searches (30), page fetches (20) and data.gov.il API calls (40),
  all adjustable in the UI.
- Identical searches (case and whitespace normalized), identical URLs (fragment and trailing slash
  normalized) and identical dataset requests are rejected.
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
| `MAX_SEARCHES` / `MAX_FETCHES` / `MAX_CKAN_CALLS` | no | `30` / `20` / `40` | Defaults for the UI settings |
| `CKAN_BASE_URL` | no | `https://data.gov.il/api/3/action/` | CKAN API base |
| `CKAN_USER_AGENT` | no | `RegulatoryOpportunityHunter/0.1 (...)` | Honest client identifier sent to the API; set it if data.gov.il documents a required client header |
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

## Official government datasets (data.gov.il)

`datagov.py` talks to the public CKAN API at `https://data.gov.il/api/3/action/` (no API key; read-only).

| Tool | CKAN calls | Returns |
|---|---|---|
| `search_government_datasets(query, rows, start)` | `package_search` (`q`, `rows` ≤ 20, `start`) | Dataset ids, titles, publishers, formats, update dates and a relevance hint. Discovery only, never evidence. |
| `inspect_government_dataset(dataset_id, query)` | `package_show` | Description, publisher, dates, license and resources with `datastore_active`. Unrelated datasets are rejected. |
| `read_government_resource(resource_id, query, limit, offset, filters)` | `resource_show` → `package_show` → `datastore_search` **or** official download | Bounded, query-matched records or passages plus provenance. |

**Read pipeline.** `resource_show` returns metadata, not data.
1. The resource's dataset (title, description, publisher, tags) and the resource name and description are
   checked against the research topic and query.
2. **Unrelated datasets are rejected before their contents are read.** A run once mistook
   `5d801af3-…` "מבנים לשימור - JSON-ITM" for a standards dataset; it is now a regression test.
3. If the metadata is relevant:
   - With `datastore_active == true`, the agent calls `datastore_search` (`resource_id`, `q`, optional
     `filters`, `limit` ≤ 100, `offset` pagination).
   - Otherwise it downloads the resource's official URL (JSON, CSV, XLSX, HTML or PDF). The download is
     capped at 10 MB, scans at most 20,000 rows, and goes through the same SSRF, redirect and robots.txt
     checks as `fetch_url`.
4. Rows are ranked by how many query terms they match. At most 50 records or passages are returned, with
   cells capped at 200 characters and control characters stripped.
5. If nothing matches, the result is "no matching records". A schema sample is shown, marked as not evidence.

**Relevance** is deterministic and works in Hebrew and English. Matching handles Hebrew prefixes (ו/ה/ב/ל/מ/ש/כ),
common inflections and English plurals. A distinctive match is required: a term of 4+ characters, or two terms.

**Envelope validation.** Each call checks the HTTP status, `success == true` and the expected `result`
shape. Malformed or failed responses become tool errors; the run continues.

**Access and politeness.**
- 401/403 is never retried or bypassed.
- 429/5xx and connection errors are retried at most twice, with exponential backoff (1s, 2s; `Retry-After`
  honoured, capped at 10s).
- Calls are spaced at least 0.5s apart, identical calls are served from an in-memory per-run cache, and a
  per-run API call limit applies.
- robots.txt is applied to web pages and resource downloads, not to the documented CKAN API endpoints.

**Provenance.** Every dataset read keeps the dataset id and title, publisher, license, resource id and name,
the resource page URL, the download and API URLs, the update dates and the retrieval status. The update
dates are publication dates, **not legal effective dates**. All of this appears in the Research Trace,
along with every failed API request.

**Live smoke test** (read-only, a few calls): `python scripts/smoke_datagov.py "רכב"`.

## Source fetching

`fetcher.fetch_url(url)`:
- HTML: removes scripts, styles, nav, header, footer and forms, prefers `<main>` or `<article>`, and collapses whitespace.
- PDF: detected by content type, `%PDF` magic bytes or a `.pdf` path. Extracts text with `pypdf` (first 60
  pages) and labels each page. OCR is not supported: a scanned PDF is reported as having no extractable text.
- JSON: pretty-printed. data.gov.il **CKAN** API responses get a readable rendering:
  - `resource_show` returns the resource metadata and the downloadable `resource_url`, which the agent fetches next.
  - `package_show` lists the dataset's resources.
  - `datastore_search` renders records as a table.
  - CKAN errors (`success: false`) are reported as failures.
- CSV and XLSX (official datasets):
  - CSV is decoded as UTF-8, then Windows-1255 (common in Israeli government files).
  - XLSX is read with `openpyxl`. Legacy `.xls` files are reported as unsupported.
  - Output is bounded to 200 rows per file or sheet and 5 sheets, with a zip-bomb guard on XLSX.
- Limits: 10s connect / 30s read timeout, 15 MB download cap, 15,000 extracted characters (truncation is flagged).
- Transient 429/5xx and connection errors: at most 2 retries with exponential backoff.
- robots.txt is respected: an explicit `Disallow` blocks the fetch, while a missing robots.txt allows it.
- Only http(s). Private, loopback and link-local hosts are refused, and redirects are followed manually so
  each hop is checked.
- HTTP 401/403 is reported as access denied and **never retried or bypassed**. When an official site
  blocks a page, the agent runs one search for a publicly accessible official copy (a gov.il PDF,
  Knesset text, Reshumot or a data.gov.il dataset). That search counts against the normal search budget.

## Evidence verification

After the final report is validated, the agent checks every cited source against what was actually retrieved
in the run. The model cannot set these fields itself.
- Every primary source must quote an `excerpt`. A source is **verified** only if it was retrieved in this
  run, is official, and the excerpt appears in the retrieved content (the exact text or a dataset record line).
  A government domain alone verifies nothing.
- Dataset sources are matched by `resource_id`. A rejected dataset, one with no matching records, or CKAN
  *metadata* (`package_search`, `package_show`, `resource_show`, even when fetched with `fetch_url`) never
  verifies a claim. Only records actually returned from the resource count.
- Each source is marked ✅ (official and read), ☑️ (read but not official) or ❌ (not retrieved, with the
  reason, e.g. HTTP 403).
- Each opportunity is **verified** (all primary sources verified), **partially verified** or **unverified**.
  An unverified legal finding is flagged prominently in the UI.
- **Class A requires fully verified official evidence that includes official legal text.** A government
  dataset is evidence about its contents, not proof of a currently applicable legal obligation. Otherwise
  the opportunity is downgraded to B and the downgrade is shown.
- Within a class, better-verified opportunities rank first.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

These cover:
- HTML, PDF, JSON, CSV and XLSX parsing; retries, 403 handling and robots.txt
- the data.gov.il CKAN client and pipeline, using a mocked API with realistic fixtures
- relevance rejection, provenance and excerpt verification
- duplicate-tool detection, limits and final-report repair
- the LLM client (mocked SDK responses)
- the Streamlit UI flow (`streamlit.testing.v1.AppTest`)

No API keys or network access are needed.

## Known MVP limitations

- No OCR. Scanned PDFs and JavaScript-rendered pages produce no text.
- Some gov.il pages render content client-side or block automated clients, so the agent may need to fall
  back to PDFs or other mirrors of the official text.
- Progress updates arrive per tool call. A long model call (max reasoning) shows no intermediate updates.
- Everything is kept in memory for one session. There is no persistence or authentication; caching is per run.
- Dataset relevance and excerpt checks are lexical. They reject clearly unrelated datasets and unquoted claims,
  but they cannot confirm that a quoted passage legally means what the model says it means.
- Output quality depends entirely on the model's research. Always check the cited sources yourself.
