# scj_rag — Supreme Court Judgments RAG (POC)

Retrieval-Augmented Generation over Indian Supreme Court judgments (2025).  
Answers questions with per-claim citations to source cases.

## Pipeline

```
Parquet (S3)
    │
    ▼
scripts/ingest.py
    │  pdf_to_text()   → plain text
    │  chunk_text()    → ~800-word chunks
    │  embed()         → 384-dim vectors (bge-small-en-v1.5, local)
    ▼
Postgres + pgvector
    │  judgments table  (metadata)
    │  chunks table     (text + embedding)
    ▼
scripts/ask.py  →  src/rag.py
    │  embed(question)  → query vector
    │  ANN search       → top-k chunks
    │  Claude API       → grounded answer with [case_id, date] citations
    ▼
logs/usage.csv  (token cost log, gitignored)
```

## Setup

```bash
# 1. venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. env vars
cp .env.example .env   # fill in ANTHROPIC_API_KEY and DATABASE_URL

# 3. database
createdb scj_rag
psql -d scj_rag -c "CREATE EXTENSION vector;"
psql -d scj_rag -f sql/01_schema.sql

# 4. download data (public S3, no AWS account needed)
aws s3 sync --no-sign-request \
    s3://indian-supreme-court-judgments/metadata/parquet/year=2025/ \
    ./data/raw/year=2025/
```

## Run

```bash
# Smoke test (first 100 judgments)
python -m scripts.ingest --year 2025 --limit 100

# Full ingest
python -m scripts.ingest --year 2025

# Ask a question
python -m scripts.ask "What did the Supreme Court rule about Article 21 in 2025?"
```

## Env vars

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `DATABASE_URL` | Yes | — | e.g. `postgresql://localhost/scj_rag` |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Override the Claude model |
| `MAX_TOKENS` | No | `1024` | Max tokens in Claude response |

## Tests

```bash
pytest scj_rag/tests/ -v
```

10 unit tests covering `chunk_text` and `find_pdf` (no DB or API needed).

## Status

| Component | Status |
|---|---|
| Schema + ingestion pipeline | Done |
| Embedding (bge-small, local) | Done |
| Vector search (pgvector ANN) | Done |
| RAG answer + citations | Done |
| Cost logging (CSV) | Done |
| Unit tests | Done (10 passing) |
| Hybrid retrieval (BM25 + rerank) | Sirius Phase 2 (ARY-22) |
| Multi-year corpus | Sirius Phase 2 |
| Production API endpoint | Sirius Phase 2 (ARY-31) |

## Out of scope (Sirius Phase 2)

| Feature | Linear ticket |
|---|---|
| BM25 + pgvector + Cohere Rerank hybrid retrieval | ARY-22 |
| Voyage embedding pipeline + HNSW indexes | ARY-19 |
| LLM critique pass / source-grounding validator | ARY-23, ARY-18 |
| Internal F3 search API endpoint | ARY-31 |
| Evaluation harness + golden test set | ARY-26 |
