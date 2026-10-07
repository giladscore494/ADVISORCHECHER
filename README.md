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
app.py        Streamlit UI: input, provider choice, snapshot freshness, run status/recovery, results, trace
agent.py      Controlled tool-calling loop, limits, duplicate detection, trace, checkpoints, resume, finalization
llm.py        Provider-agnostic client: Kimi / GLM (Chat Completions) and OpenAI gpt-6.1-sol (Responses API)
local_data.py Indexed (SQLite FTS5 + customs-code) queries over the validated government snapshots
research_store.py   Durable run store (SQLite or PostgreSQL): checkpoints, status, reports
research_runner.py  Background worker thread + heartbeat; start / resume / recover runs
partial_report.py   Fallback report built only from persisted evidence (no model call)
search.py     search_web(query, num_results) via Serper, plus primary-source domain detection
fetcher.py    fetch_url(url): HTML, PDF, JSON, CSV, XLSX; retries, robots.txt, SSRF and size limits
datagov.py    data.gov.il CKAN client (package_search/package_show/resource_show/datastore_search) and
              the dataset read pipeline with relevance checks and provenance
evidence.py   Hebrew/English term matching (dataset relevance) and quoted-excerpt verification
prompts.py    System prompt (research strategy, red team, legal safety), final-report prompt
models.py     Pydantic schema for the final report, strict JSON parsing and validation, ranking
config.py     Settings from environment, .env, or Streamlit secrets
scripts/      sync_government_data.py: complete, validated data.gov.il snapshots (run by GitHub Actions)
              smoke_datagov.py: optional read-only live check against data.gov.il
data/government/  manifest.json + versioned snapshots (snapshots/<dataset>/<version>/records.jsonl.gz)
.github/workflows/  sync-government-data.yml (daily + manual), its branch test, tests.yml (pytest on PRs)
tests/        Unit, recovery (real process kill), PostgreSQL, AppTest UI and live data.gov.il tests
```

### Agent loop

The model gets these tools:
- `search_web`: discover websites and documents (Serper).
- `fetch_url`: read an individual law, regulation, guidance page or PDF.
- `search_government_datasets`, `inspect_government_dataset`, `read_government_resource`: structured
  official data from data.gov.il (see below).
- `list_local_government_datasets`, `search_local_government_records`, `get_local_government_record`,
  `get_government_snapshot_status`: the complete local snapshots (see below), queried first for customs,
  import-requirement and standards questions.
- `update_candidates`: records the candidate funnel.
- `record_findings`: saves findings (statement + source + exact excerpt, checked immediately) and open
  questions as they are established, so they survive a later failure.

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
| `LLM_PROVIDER` | no | `kimi` | Default provider in the UI: `kimi`, `glm` or `openai` |
| `GLM_API_KEY`, `GLM_BASE_URL`, `GLM_MODEL` | if `glm` | `https://api.z.ai/api/paas/v4`, `glm-5.3` | |
| `OPENAI_API_KEY` | if `openai` | | Server-side only |
| `OPENAI_MODEL` / `OPENAI_REASONING_EFFORT` / `OPENAI_BASE_URL` | no | `gpt-6.1-sol` / `high` / `https://api.openai.com/v1` | Responses API |
| `RESEARCH_STORE_URL` | recommended | SQLite `.research_runs/runs.sqlite` | `postgresql://user:pass@host:5432/db` for durable recovery on hosts with ephemeral disks (Streamlit Community Cloud), or `sqlite:////abs/path.sqlite` |
| `RESEARCH_RUN_MODE` | no | `background` | `background` (worker thread, page polls the store) or `inline` (runs inside the page script; tests) |
| `RESEARCH_LIST_RUNS` | no | `0` | `1` lists recent runs in "Recover previous run" (only for private, single-user deployments) |
| `GOVDATA_DIR` / `GOVDATA_CACHE_DIR` | no | `data/government` / system temp | Snapshot location / where the SQLite index and release assets are cached |
| `GOVDATA_WARMUP` | no | `1` | Build the local index in the background when the app starts |

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

## OpenAI provider (gpt-6.1-sol)

Choose **OpenAI** in the "Model provider" selector (or `LLM_PROVIDER=openai`). The adapter in `llm.py` uses
the **Responses API with function calling**: `model=gpt-6.1-sol`, `reasoning={"effort": "high"}`,
`store=false`, `include=["reasoning.encrypted_content"]`. It sends exactly the app's own function tools
(Serper `search_web`, `fetch_url`, the live CKAN tools, the local dataset tools, `update_candidates`,
`record_findings`); OpenAI built-in tools such as web search are never enabled, so legal discovery stays on
Serper and every tool runs inside this app. The internal chat-style history is converted on every call
(`function_call` / `function_call_output` items), and encrypted reasoning items are kept in the history so
a checkpointed run can be resumed. If an endpoint rejects replayed reasoning items, the adapter continues
without them. Kimi and GLM keep using Chat Completions unchanged. API keys are read server-side only
(environment, `.env`, Streamlit secrets); they are never shown in the UI and are redacted from stored runs.

## Government dataset snapshots

`scripts/sync_government_data.py` downloads **complete** official datasets from data.gov.il and the
`Sync government data` workflow (`.github/workflows/sync-government-data.yml`) runs it daily
(02:23 UTC) and on demand (`workflow_dispatch`, inputs `force` and `only`).

| Key | Resource id | Official resource |
|---|---|---|
| `customs_tariff` | `5536eaa1-2e51-406b-aff6-b9ca02801b7c` | ספר סיווג טובין ביבוא - תעריף המכס ומס קניה (Israel Tax Authority) |
| `free_import_order` | `a36db570-09f2-4521-8e3d-0290eb839c68` | דרישות חוקיות - צו יבוא חופשי (Israel Tax Authority) |
| `mandatory_standards` | `1a4d94e2-369a-488d-a223-eb1020612fbd` | מאגר תקנים רשמיים (Ministry of Economy) |
| `import_regulations` | `d9750b40-c0b9-4e05-a08e-ae768a92e9ca` | דרישות חוקיות - צוים נוספים (Israel Tax Authority) |
| `standards_declarations` | `d8611d0e-f5c8-4552-8615-da37e920f07b` | אכרזת תקנים ברשומות (Ministry of Economy) |

For every resource the script:
1. reads official metadata with `resource_show` and `package_show` (dataset, publisher, license, dates);
2. discovers the retrieval method: the DataStore when `datastore_active`, otherwise the resource's official
   download URL (CSV / XLSX / JSON);
3. reads the full-table total with `datastore_search?limit=0` (no `q`, no filters: **not** a search-result
   count), pages through every row ordered by `_id`, and re-reads the total afterwards;
4. validates: rows == official total, `_id` strictly increasing and unique, the same schema on every page and
   record, non-empty, and no unexplained shrink (> 50%) versus the previous snapshot;
5. writes UTF-8 JSONL exactly as served (deterministic gzip) plus `snapshot.json` (dataset name, publisher,
   retrieval date, resource id, license, row count, source URL, API URL, schema, validation results, SHA-256
   of the content and of the file) into a temporary directory, re-reads and re-checks it from disk, then
   atomically renames it into place and atomically rewrites `data/government/manifest.json`.

A failed, blocked or incomplete download never replaces the last complete snapshot: the manifest keeps
pointing at it and records the failed attempt. Unchanged datasets are skipped (fingerprint of official
metadata + DataStore total); a full re-download is forced every 7 days and creates a new version only if the
content checksum changed. Requests are read-only GETs, spaced >= 1 s apart, with timeouts and exponential
backoff for 429/5xx/connection errors only; HTTP 401/403 is never retried or bypassed.

Storage: snapshots up to 45 MiB (gzip) are committed to the repository; larger ones are published as a
GitHub Release asset (`govdata-<key>-<version>`) with the manifest still committed, and the app downloads
them from the release URL and verifies their SHA-256. Nothing is ever truncated. The workflow uses minimal
permissions (`contents: write` on the job only), a concurrency group that never cancels a running sync,
job/step timeouts, `--verify` of every snapshot before committing, real read-only live tests
(`tests/test_live_datagov.py`: live totals and sampled live records must equal the snapshot exactly), and
uploads `sync.log` / `live-tests.log` / the manifest as a run artifact.
`sync-government-data-branch-test.yml` runs the same job when the sync code changes on a `claude/**` branch
(`workflow_dispatch` only works once a workflow is on the default branch).

The Streamlit app reads the snapshots from its own checkout (Streamlit Cloud redeploys on each commit), so
no Serper or model call is needed to obtain them.

## Local government data tools

`local_data.py` builds a SQLite index once per snapshot version (about 15 s for all five datasets, in a
background thread at app start; cached in `GOVDATA_CACHE_DIR`): an FTS5 full-text index with Hebrew
prefix-stripped variants (ו/ה/ב/ל/מ/ש/כ) and English terms across all fields, and a customs-code table over
classification fields. Customs queries such as `8703.23.30.00/0`, `87032330`, `87.03` return **exact**
matches, **parent** heading/chapter rows and **child** items, in that order (check digits are ignored;
numeric columns whose leading zero was lost in the DataStore are restored). A code-like query that is not a
tariff item (e.g. a standard number `1347`) falls back to text search.

The model never receives a dataset: every query returns at most 25 records (trimmed further to fit the
context budget), each with its exact field values, record id, a quotable `record_line`, the snapshot
version and provenance (resource id, publisher, license, source URL, retrieval and verification dates).
Zero matches comes with an explicit note that it is not proof of an exemption. Local records count as
**dataset evidence**, never as legally binding text: a claim supported only by dataset evidence is never
class A. The live CKAN tools remain for freshness checks, records missing locally and other datasets, and
Serper / `fetch_url` remain the independent legal-source verification path. The UI shows each snapshot's
date, last official verification and freshness (fresh <= 2 days, aging <= 8, stale beyond).

## Durable research, recovery and partial reports

Every run gets a unique run ID (`YYYYMMDD-HHMMSS-<64 random bits>`, also put in the page URL as `?run=`)
and is executed by `research_runner.py` in a worker thread owned by the server process, so Streamlit
reruns, widget clicks and browser disconnects do not stop it. The agent checkpoints its **complete,
resumable state** (instructions and configuration, the conversation, searches, retrieved content and
dataset evidence with provenance, candidates and status, recorded findings with verification, open
questions, counters, token usage, API errors, current phase, timestamp) to the research store:

- after every model response and every completed tool call,
- at every research phase change,
- before every final-report model call and after receiving it,
- immediately when an error is caught (model, Serper, data.gov.il, local data, unexpected exceptions).

A heartbeat thread refreshes the run every 20 s while the worker is alive; a "running" run without a
heartbeat for 120 s is reported as **interrupted** (server restart, crash, container recycling). Status is
one of running / completed / failed / interrupted.

Recovery: the results area shows the run ID, status, last successful checkpoint and progress, and offers
**Download partial report** (Markdown / JSON), **Download research trace** (JSON), **Resume research from
last checkpoint**, and **Recover previous run** by run ID. Resume restores the exact conversation and
evidence and continues with the remaining budget; if the process died between a model response and its tool
results, the missing tool results are answered as "interrupted, call again". A run that died during
finalization resumes at finalization without repeating research.

If the final JSON report cannot be produced or validated (model error, timeout, invalid JSON twice, crash),
`partial_report.py` builds a report **only from saved evidence**, without a model call: VERIFIED findings
(official source retrieved in this run and the excerpt found in it), UNVERIFIED findings, candidates labelled
as WORKING HYPOTHESES, official / non-official / failed sources, dataset evidence (labelled as not legal
text), searches, open questions, API errors, and any rejected model draft labelled UNVALIDATED.

Storage (`RESEARCH_STORE_URL`):
- **PostgreSQL** (recommended for Streamlit Community Cloud or any host with an ephemeral disk), e.g. a free
  Supabase / Neon database: `postgresql://user:password@host:5432/dbname`. Tables are created automatically.
- **SQLite** (default `.research_runs/runs.sqlite`, WAL + `synchronous=FULL`): survives process restarts on
  the same disk, **not** container replacement or redeploys; the UI says so.

Research content is private: it is stored only in this store (git-ignored locally), never in the repository;
known API keys are redacted before anything is written. Run IDs are unguessable; listing recent runs in the
UI is off unless `RESEARCH_LIST_RUNS=1` (the app has no authentication; enable it only on private deploys).

Crash recovery is tested with a real process kill: `tests/test_recovery.py` starts a worker process, lets
it research, sends SIGKILL (mid-research and during final JSON generation), then verifies that a new process
sees the run as interrupted, can build the partial report, and resumes it to a verified final report.

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
- complete multi-page snapshot downloads, exact counts, duplicate / overlap detection, schema checks,
  atomic replacement, failed downloads keeping the previous snapshot, incremental skips, release storage
- local indexed queries (exact / parent / child customs codes, Hebrew prefixes, English terms, filters,
  provenance, zero-result notes) on synthetic snapshots and on the real committed snapshots
- the research store on SQLite and on a real throwaway PostgreSQL server (skipped if not installed)
- checkpoints after every tool call, crash during final JSON generation, invalid final JSON, resume,
  KeyboardInterrupt, and a real SIGKILL of a worker process followed by recovery and resume
- the OpenAI Responses adapter, including an end-to-end run where the model drives Serper, fetch_url and
  local data tools; Serper request construction and errors
- the recovery UI (partial report, downloads, resume, recover by ID, interrupted detection, background runs)
- HTML, PDF, JSON, CSV and XLSX parsing; retries, 403 handling and robots.txt
- the data.gov.il CKAN client and pipeline, using a mocked API with realistic fixtures
- relevance rejection, provenance and excerpt verification
- duplicate-tool detection, limits and final-report repair
- the LLM client (mocked SDK responses)
- the Streamlit UI flow (`streamlit.testing.v1.AppTest`)

No API keys or network access are needed. Real read-only data.gov.il integration tests run with
`RUN_LIVE_GOV_TESTS=1 python -m pytest -m live` (the sync workflow runs them after every sync).

## Known MVP limitations

- No OCR. Scanned PDFs and JavaScript-rendered pages produce no text.
- Some gov.il pages render content client-side or block automated clients, so the agent may need to fall
  back to PDFs or other mirrors of the official text.
- Progress updates arrive per tool call. A long model call (max reasoning) shows no intermediate updates.
- There is no authentication. Runs are recoverable by their unguessable run ID; do not share it.
- Resuming continues the saved conversation; a model call that was in flight when the process died is
  repeated (and billed) again. With the default SQLite store, recovery does not survive container
  replacement: configure PostgreSQL for that.
- The background worker lives in the Streamlit server process. A crash is recoverable (resume), but a run
  does not continue by itself while no server process is running.
- Dataset relevance and excerpt checks are lexical. They reject clearly unrelated datasets and unquoted claims,
  but they cannot confirm that a quoted passage legally means what the model says it means.
- Output quality depends entirely on the model's research. Always check the cited sources yourself.
