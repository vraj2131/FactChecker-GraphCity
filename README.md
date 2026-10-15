# FactGraph City

A fact-checking tool that takes a claim written in plain English and renders a
**3D evidence graph**: the claim as a node at the center, retrieved evidence
as surrounding nodes, and edges colored by whether each piece of evidence
supports, refutes, or is merely related to the claim. The verdict
(`verified` / `rejected` / `not_enough_info`) comes with a calibrated
confidence score, not a bare label.

Built on top of the [FEVER](https://fever.ai/) claim-verification dataset,
with live retrieval layered on top so it can also answer claims FEVER never
saw.

---

## 1. What it does, in one pass

```
claim text
   │
   ▼
retrieval   ──  11+ retrievers run in parallel (Wikipedia/FAISS, live
                Wikipedia, Guardian, NewsAPI, GDELT, fact-checkers,
                DuckDuckGo, Reddit/Bluesky, arXiv/PubMed/OpenAlex,
                FRED/SEC/CoinGecko/World Bank — only the ones relevant
                to the claim's domain actually fire)
   │
   ▼
dedupe + rank  ──  URL/near-duplicate removal, trust-weighted ranking
   │
   ▼
NLI stance   ──  DeBERTa entailment model labels each source as
                 supports / refutes / neutral, with a second stronger
                 model re-checking borderline cases
   │
   ▼
LLM classification  ──  Groq-hosted LLM reads claim + sources together,
                        buckets each source as direct_support /
                        direct_refute / correlated_context /
                        insufficient, and gives an overall verdict +
                        its own confidence
   │
   ▼
confidence scoring  ──  5 weighted signals combined and calibrated
                        (see §6)
   │
   ▼
graph JSON  ──  nodes + edges + verdict + confidence → rendered as a
                3D force-directed graph in the browser
```

## 2. How a final decision gets made

This is the part worth understanding before you read any code: **no single
model decides the verdict.** Three independent signals have to line up:

1. **NLI stance** — a DeBERTa model trained on FEVER/ANLI decides, per
   source, whether the evidence text entails, contradicts, or is neutral to
   the claim. This is a classic textual-entailment judgment, not a language
   model's opinion.
2. **LLM classification** — a Groq-hosted LLM (currently rotating across
   `openai/gpt-oss-120b` / `gpt-oss-20b` / `gpt-oss-safeguard-20b` to spread
   across daily token budgets) reads the claim alongside every retrieved
   source and labels each one's relationship to the claim, then gives an
   overall supported/refuted/insufficient call plus its own confidence.
   Fact-check sources get special handling: if the LLM reaches a
   supported/refuted verdict but doesn't explicitly tag a fact-check source
   as direct evidence, it's anchored to the verdict direction anyway, since
   fact-check sources are retrieved specifically because they already
   address the claim.
3. **Confidence scoring** — the two signals above, plus source trust/
   relevance and how many *independent kinds* of source agree (not just how
   many sources), are combined into one weighted score:

   | Signal | Weight | What it captures |
   |---|---|---|
   | Directional (NLI) | 0.35 | Trust-weighted support/refute agreement among direct sources |
   | LLM confidence | 0.25 | The LLM's own stated confidence |
   | Evidence quality | 0.20 | Avg(trust × relevance) of direct sources |
   | Corroboration | 0.15 | Count of *distinct source types* agreeing, not just source count |
   | Coverage | 0.05 | Breadth of retriever types touched at all |

   The weighted sum is piecewise-calibrated (floor 0.05, ceiling 0.95 — the
   system never claims absolute certainty or absolute zero), then
   thresholded: **≥ 0.65 commits** to `verified`/`rejected`; **0.50–0.65**
   returns `not_enough_info` but attaches a provisional `leaning_verdict` so
   the signal isn't thrown away; below that, plain `not_enough_info`.

   The 0.65 commit bar is deliberate: a 200-claim internal benchmark showed
   the system had **zero confidently-wrong answers** at that bar, at the
   cost of answering fewer claims overall. For a fact-checker, "I don't
   know" is a recoverable answer — a confident wrong one is not — so the
   system is tuned to abstain rather than guess.

Every claim also gets a **quantifier check**: words like *always, never,
only, exactly, entire, globally* make a claim false on one counterexample
even if most sources support the general trend, and the LLM prompt
explicitly treats a documented exception as outright refutation rather than
"mostly true."

## 3. Project structure

```
backend/
  app/
    retrieval/        one file per retriever (wikipedia, guardian, newsapi,
                       gdelt, factcheck, duckduckgo, reddit, bluesky,
                       arxiv, pubmed, openalex, fred, sec_edgar,
                       coingecko, worldbank, fever, livewiki)
    models/            embedding, NLI, LLM, calibration wrappers
    services/          retrieval orchestration, stance, confidence,
                       graph building, context expansion, verify-claim
    schemas/           pydantic request/response contracts
    api/                FastAPI routes (verify-claim, sources, health)
    preprocessing/     text normalization, dedup, snippet extraction
  scripts/             one-off data-prep + evaluation scripts
  scripts/batch_harness/   200-claim scored benchmark harness (see §7)
frontend/              React + Vite + react-force-graph-3d
data/
  raw/fever/           FEVER dataset goes here (you provide it, §4.2)
  processed/           parquet files produced by prepare_fever.py
  artifacts/faiss/     FAISS indexes produced by build_faiss_indexes.py
  cache/               on-disk retrieval/NLI/LLM response cache
```

## 4. Reproducing it from scratch

### 4.1 Prerequisites

- Python 3.11+ and Node 18+
- A free-tier API key for each external service you want enabled (all
  optional individually — the pipeline degrades gracefully if a key is
  missing, it just skips that retriever)

### 4.2 Clone and set up the backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4.3 API keys — what you need and where to get it

| Variable | Required? | Used for | Where to get it |
|---|---|---|---|
| `GROQ_API_KEY` | **Required** — no fallback | LLM classification stage (every verdict needs this) | [console.groq.com](https://console.groq.com) → API Keys, free tier |
| `NEWSAPI_KEY` | Optional | Live News source group | [newsapi.org](https://newsapi.org) → free tier, 100 req/day |
| `GUARDIAN_API_KEY` | Optional | Live News source group | [open-platform.theguardian.com](https://open-platform.theguardian.com) → free, 500 req/day |
| `FACTCHECK_API_KEY` | Optional | Fact-Checkers source group | [Google Fact Check Tools API](https://console.cloud.google.com) → enable "Fact Check Tools API", create an API key |
| `GDELT_API_KEY` | Not needed | — | GDELT's API is keyless; leave the variable blank, it's kept only for config parity |
| `HF_TOKEN` | Optional | Raises Hugging Face's anonymous download rate limit for the embedding/NLI models | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `BLUESKY_IDENTIFIER` + `BLUESKY_APP_PASSWORD` | Optional (pair) | Social Media source group | Your handle (`you.bsky.social`) + an **app password** (not your login password) from [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords) |

No key at all is needed for: Wikipedia/FAISS, live Wikipedia, DuckDuckGo,
Reddit, arXiv, PubMed, OpenAlex, FRED, SEC EDGAR, CoinGecko, World Bank —
these are keyless public APIs or local data. Missing an *optional* key just
disables that one retriever at startup; the pipeline still runs on whatever
keys you do provide. `GROQ_API_KEY` is the only one with no fallback —
without it the pipeline cannot produce a verdict.

Copy the field names below into a `.env` file at the project root.

```bash
# App
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# Models — these defaults are what the project was built and evaluated
# against; changing them is reasonable but untested
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# FAISS — leave as-is unless you move the data directory
FAISS_INDEX_PATH=data/artifacts/faiss/
EMBEDDING_CACHE_PATH=data/cache/embeddings/

# External API keys — each one only disables ITS OWN retriever if missing
NEWSAPI_KEY=              # newsapi.org — free tier, 100 req/day
GUARDIAN_API_KEY=         # open-platform.theguardian.com — free, 500 req/day
FACTCHECK_API_KEY=        # Google Fact Check Tools API (console.cloud.google.com)
GDELT_API_KEY=            # GDELT is keyless; leave blank, kept for parity

# Confidence thresholds — see §6; 0.65/0.65 is the validated default
SUPPORT_THRESHOLD=0.65
REFUTE_THRESHOLD=0.65
CORRELATION_THRESHOLD=0.50

# Required for the LLM classification stage — no fallback, the pipeline
# cannot produce a verdict without this
GROQ_API_KEY=              # console.groq.com — free tier

# Optional, for embedding model downloads if you hit HF rate limits
HF_TOKEN=

# Optional — only needed to enable the "Social Media" source group
BLUESKY_IDENTIFIER=        # your-handle.bsky.social
BLUESKY_APP_PASSWORD=      # an app password, not your main password —
                            # generate one at bsky.app/settings/app-passwords
```

No key is needed for: Wikipedia/FAISS, live Wikipedia, DuckDuckGo, Reddit,
arXiv, PubMed, OpenAlex, FRED, SEC EDGAR, CoinGecko, World Bank — these are
either keyless public APIs or local data.

### 4.4 Get the FEVER dataset

Download `train.jsonl`, `dev.jsonl`, and the `wiki-pages/` archive from the
[official FEVER dataset page](https://fever.ai/dataset/fever.html) and place
them at:

```
data/raw/fever/train.jsonl
data/raw/fever/dev.jsonl
data/raw/fever/wiki-pages/
```

(The full Wikipedia dump is large — `prepare_wiki_evidence.py` supports a
`--sample-page-count` flag if you just want to stand the system up quickly
rather than reproduce the full index.)

### 4.5 Run the data-prep pipeline, in order

```bash
# 1. Parse FEVER claims + evidence into parquet
python -m backend.scripts.prepare_fever

# 2. Extract matching Wikipedia sentences for each evidence pointer
python -m backend.scripts.prepare_wiki_evidence

# 3. Embed claims + evidence and build FAISS indexes
python -m backend.scripts.build_faiss_indexes
```

Each script writes into `data/processed/` and `data/artifacts/faiss/`. Pass
`--help` to any of them for sampling/batch-size flags — defaults are sane
for a first run.

### 4.6 Start the backend

```bash
source venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

`GET http://localhost:8000/health` should return OK once models finish
loading (the embedding/NLI models load lazily on the first request, so the
first call will be slow).

### 4.7 Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173` and talks to `http://localhost:8000` by
default (override with a `VITE_API_URL` env var in `frontend/.env` if your
backend runs elsewhere).

### 4.8 Use it

Type a claim into the search bar and hit verify. Two fields are yours to
adjust before submitting:

- **Deep NLI toggle** — re-checks borderline stance labels with a second,
  stronger NLI model. More accurate, a few seconds slower. On by default.
- **Source groups** — which retriever categories to use for this claim:
  Wikipedia (always on), Live News, Fact-Checkers, Scientific, Financial/
  Crypto, Web Search, Social Media. Defaults to everything except Web
  Search and Social Media (those are opt-in: DuckDuckGo can be slow, and
  social sources are low-trust by design).

The response renders as a rotatable 3D graph: click any node to see its
source text, trust score, and how the LLM classified it.

## 5. Running it without the UI

```bash
curl -X POST http://localhost:8000/api/v1/verify-claim \
  -H "Content-Type: application/json" \
  -d '{"claim_text": "The Great Wall of China is visible from space with the naked eye.", "deep_nli": true}'
```

Returns the full graph JSON (verdict, confidence, nodes, edges) — the same
contract the frontend consumes.

## 6. Confidence, calibration, and thresholds — the knobs that matter

If you change the LLM model or retune thresholds, re-validate against the
200-claim scored benchmark before trusting the new numbers — see §7. In
particular: **`GROQ_MAX_TOKENS` (in `backend/app/utils/constants.py`) must
be re-measured per model.** This project previously shipped a silent ~70%
response-truncation bug after a Groq model swap, because the token budget
was sized for the old model's (much terser) completions — the LLM cache key
now includes `max_tokens` specifically so a budget fix can't be masked by
stale cached truncated responses.

## 7. Evaluating changes: the 200-claim benchmark harness

`backend/scripts/batch_harness/` runs the full pipeline end-to-end over a
fixed 200-claim corpus with known ground-truth labels (TRUE/FALSE/
CONTESTED), screenshots every graph, and scores the run into a PDF:

```bash
./run_harness.sh
```

Env vars you can override: `CLAIMS_MODULE` / `GT_MODULE` to point at a
different claim corpus (two independent 200-claim sets already exist —
`claims.py` and `claims_v2.py`), `COMPARE_TO` to diff against a previous
run's results in the output PDF, `GROQ_MODEL_CHAIN` to change which models
rotate through. Designed to run unattended (works well inside `tmux`): it
resumes from a crash-safe manifest rather than restarting, and rotates
across Groq models automatically when one's daily token budget is spent.

Treat this harness as a regression gate, not a one-off report: `analyze_run.py`
accepts `--fail-under-accuracy` / `--fail-under-coverage` and exits non-zero
on either regression, so it can be wired into CI for any retrieval,
prompting, or confidence-formula change.

## 8. Known limitations (carried into the benchmark's own roadmap)

- **No arithmetic reasoning.** Superlative/comparative claims ("largest",
  "more than X") can't be resolved against a real reference dataset yet —
  retrieval finds the right documents, but nothing compares numbers.
- **Coverage is the main lever, not accuracy.** The system is tuned to
  abstain rather than guess; raising coverage without reintroducing
  confident-wrong answers is the active area of work (see the `analysis.pdf`
  produced by §7 for the current most-promising approach).
- **Groq's model catalog is unstable.** Models have been silently removed
  from the catalog mid-project twice. Verify `GROQ_MODEL_CHAIN` against
  `GET https://api.groq.com/openai/v1/models` before a long run.
