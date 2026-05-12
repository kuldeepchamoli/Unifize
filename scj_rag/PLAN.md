# Supreme Court Judgments RAG — Beginner-Friendly Build Plan

> **Goal:** Build a Retrieval-Augmented Generation (RAG) system that can answer questions about Indian Supreme Court judgments from 2025, with citations to the source cases.
>
> **You have done this before:** your `legal_rag/` folder already implements the same pattern (Postgres + pgvector + bge-small embeddings + Claude). We are reusing that stack so nothing is new — only the data source changes.
>
> **Scope choices (defaults — speak up if you disagree):**
> - Year: `2025` only.
> - Approach: smoke-test on ~100 judgments first, scale to full year after it works.
> - Folder: `scj_rag/` (this folder), parallel to `legal_rag/`.

---

## 0. The big picture before you touch a keyboard

A RAG pipeline has **five moving parts**. Keep this mental model — every step below maps to one of these:

1. **Raw data** — Parquet files from the public S3 bucket containing judgment text + metadata.
2. **Database** — Postgres with the `pgvector` extension. Two tables: one for the judgment metadata, one for the text chunks + their embedding vectors.
3. **Embedder** — A local model (`bge-small-en-v1.5`) that turns text into 384-dimensional vectors. No API key, no cost.
4. **Retriever** — A Python function that takes a user's question, embeds it, and asks Postgres "give me the 5 chunks whose vectors are closest to this one".
5. **Generator** — Claude (via the Anthropic API). It receives the question + retrieved chunks and writes the final answer with citations.

If any step feels confusing later, ask yourself **which of these five it belongs to** — it always belongs to exactly one.

---

## 1. Prerequisites — what you should already have on the Mac

Run these checks first. If any command says "not found", we install it in the corresponding section below.

```bash
# Check Homebrew (Mac package manager)
brew --version

# Check Python (we need 3.10+)
python3 --version

# Check git
git --version
```

**You already have:**
- An Anthropic API key (used in `legal_rag/`). We will reuse it.

**You will install fresh:**
- Postgres 16, pgvector extension, AWS CLI, a fresh Python virtual environment.

---

## 2. Set up the project folder structure

Your folder will end up looking like this:

```
scj_rag/
├── PLAN.md                # this file
├── README.md              # short usage notes (we write later)
├── requirements.txt       # Python deps
├── .env                   # API keys (NEVER commit)
├── .gitignore             # tells git to ignore .env, data/, .venv/
├── data/
│   └── raw/year=2025/     # parquet files downloaded from S3
├── sql/
│   └── 01_schema.sql      # CREATE TABLE statements
├── src/
│   ├── __init__.py
│   ├── config.py          # load env vars in one place
│   ├── db.py              # Postgres connection helper
│   ├── embed.py           # load bge-small, embed text
│   ├── preprocess.py      # HTML → plain text → chunks
│   ├── retrieve.py        # vector search query
│   └── rag.py             # the full ask-question → answer pipeline
└── scripts/
    ├── __init__.py
    ├── inspect_data.py    # one-off: open a parquet, print columns
    ├── ingest.py          # parquet → preprocess → embed → Postgres
    └── ask.py             # CLI: take a question, print an answer
```

**Why this layout?** It mirrors `legal_rag/`. Reusing a layout you've seen before means less cognitive load.

**Commands to create it** (run from `/Users/Ram/Desktop/Mtech/unifize/scj_rag/`):

```bash
cd /Users/Ram/Desktop/Mtech/unifize/scj_rag
mkdir -p data/raw sql src scripts
touch src/__init__.py scripts/__init__.py
touch .env .gitignore requirements.txt README.md
```

---

## 3. Python virtual environment

**Why a venv?** It is a *project-local* Python install. Packages you install here do not pollute your system Python or other projects. Beginner mistake to avoid: installing everything globally with `pip install`.

```bash
cd /Users/Ram/Desktop/Mtech/unifize/scj_rag
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your shell prompt should show `(.venv)` at the start. Every Python command for this project must be run with the venv activated.

**To leave the venv later:** `deactivate`.
**To re-enter next session:** `source .venv/bin/activate` from the project folder.

---

## 4. `requirements.txt` — Python dependencies

Create the file with this exact content (we will install in section 5). These versions match what you used in `legal_rag/` so the imports look identical.

```
# Postgres + pgvector
psycopg[binary]==3.2.3
pgvector==0.3.6

# Claude API
anthropic==0.39.0

# Env vars
python-dotenv==1.0.1

# Data
pandas==2.2.3
pyarrow==17.0.0           # reads parquet files
beautifulsoup4==4.12.3    # strips HTML out of raw_html field
tqdm==4.67.0              # progress bars

# Embeddings — local, no API key
sentence-transformers==3.3.1
```

**What is new vs legal_rag?**
- `pyarrow` — to read parquet files (your complaints data was CSV, judgments are parquet).
- `beautifulsoup4` — to strip HTML tags out of judgment text (judgments come as `raw_html`).

---

## 5. Install Python deps

```bash
# venv must be active — check your prompt shows (.venv)
pip install --upgrade pip
pip install -r requirements.txt
```

This will download ~2 GB (the sentence-transformers and torch wheels are large). Be patient.

**Verify:**
```bash
python -c "import psycopg, pgvector, anthropic, sentence_transformers, pyarrow, bs4; print('all imports ok')"
```

If you see `all imports ok`, move on.

---

## 6. Install Postgres 16

**Why Postgres?** It is a relational database. We use it because the `pgvector` extension lets us store and search vectors *inside the same database* as our regular tables — no second system needed.

```bash
brew install postgresql@16
brew services start postgresql@16
```

The second command runs Postgres as a background service that auto-starts on login.

**Add Postgres to your PATH** so `psql` and `createdb` work in any terminal. Add this line to `~/.zshrc`:

```bash
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
```

Then reload: `source ~/.zshrc`.

**Verify:**
```bash
psql --version       # should print "psql (PostgreSQL) 16.x"
psql postgres -c "SELECT 1;"   # should print "1"
```

---

## 7. Install the pgvector extension

`pgvector` is what makes Postgres able to do vector similarity search.

```bash
brew install pgvector
```

This installs the extension files. We still need to *enable* it inside our specific database — that happens in step 8.

---

## 8. Create the database

```bash
createdb scj_rag
psql -d scj_rag -c "CREATE EXTENSION vector;"
psql -d scj_rag -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

The last command should print a version like `0.8.0`. That confirms pgvector is active in your `scj_rag` database.

---

## 9. Create the tables — `sql/01_schema.sql`

**Design rationale (read this — it is the most important design decision):**
- Each Supreme Court judgment is LONG (often tens of pages). You cannot embed an entire judgment as one vector — you would lose all detail.
- So we split each judgment into **chunks** (~800 tokens each), embed each chunk, and store one row per chunk.
- We keep judgment-level metadata (case_id, title, judges, date, citation) in a *separate* table to avoid duplicating it on every chunk.
- A foreign key links chunks back to their parent judgment.

Write this into `sql/01_schema.sql`:

```sql
-- One row per judgment
CREATE TABLE IF NOT EXISTS judgments (
    case_id          TEXT PRIMARY KEY,
    title            TEXT,
    petitioner       TEXT,
    respondent       TEXT,
    judge            TEXT,
    author_judge     TEXT,
    citation         TEXT,
    decision_date    DATE,
    disposal_nature  TEXT,
    court            TEXT,
    full_text        TEXT       -- cleaned plain text, kept for debugging / future BM25
);

-- One row per chunk of a judgment
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    case_id    TEXT NOT NULL REFERENCES judgments(case_id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,            -- 0,1,2,... within a judgment
    text       TEXT NOT NULL,
    embedding  vector(384) NOT NULL,        -- bge-small-en-v1.5 output dim
    UNIQUE (case_id, chunk_idx)
);

-- Approximate-nearest-neighbour index for fast cosine search
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

**Apply it:**
```bash
psql -d scj_rag -f sql/01_schema.sql
psql -d scj_rag -c "\dt"     # should list 'judgments' and 'chunks'
```

---

## 10. Install AWS CLI and download the data

**Why AWS CLI and not Athena?** Athena (from the GitHub tutorial) is a query engine that costs money per scan. The S3 bucket itself is *public* — we can download the parquet files directly for free.

```bash
brew install awscli
aws --version
```

**Look at what's available** (no AWS account needed because the bucket is public — we pass `--no-sign-request`):

```bash
aws s3 ls --no-sign-request s3://indian-supreme-court-judgments/metadata/parquet/
```

You should see a list of `year=YYYY/` folders.

**Download year 2025** into `data/raw/`:

```bash
aws s3 sync --no-sign-request \
    s3://indian-supreme-court-judgments/metadata/parquet/year=2025/ \
    ./data/raw/year=2025/
```

**Verify:**
```bash
ls -lh data/raw/year=2025/
```

You should see one or more `.parquet` files.

---

## 11. Inspect the data BEFORE you write any pipeline code

This is the single most important habit for a beginner. **Always look at the data first.** Many bugs come from assumptions about column names or value formats.

Create `scripts/inspect_data.py`:

```python
"""One-off: peek at the downloaded parquet to understand its shape."""
import pandas as pd
from pathlib import Path

PARQUET_DIR = Path(__file__).parent.parent / "data" / "raw" / "year=2025"
files = list(PARQUET_DIR.glob("*.parquet"))
print(f"Found {len(files)} parquet file(s)")

df = pd.read_parquet(files[0])
print(f"\nShape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst row (metadata only):")
print(df.drop(columns=["raw_html"], errors="ignore").iloc[0])
print(f"\nFirst 500 chars of raw_html:")
print(df["raw_html"].iloc[0][:500])
```

Run: `python scripts/inspect_data.py`

**What to verify:**
- Column names match what the GitHub tutorial promised (`case_id`, `title`, `decision_date`, `raw_html`, etc.).
- `raw_html` actually contains HTML.
- `decision_date` is parseable as a date.

If anything is off, **fix the schema in step 9 before continuing** — easier now than after you have ingested 1000 docs.

---

## 12. Config + DB helpers — `src/config.py` and `src/db.py`

Same pattern as `legal_rag/`. Centralise env vars and DB connections so the rest of the code stays clean.

**`.env`** (copy from `legal_rag/.env` — same Anthropic key):
```
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://localhost/scj_rag
```

**`.gitignore`** — protect secrets and big files:
```
.venv/
.env
data/
__pycache__/
*.pyc
```

**`src/config.py`** — load env once:
```python
import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
```

**`src/db.py`** — a helper that returns a psycopg connection with pgvector registered:
```python
import psycopg
from pgvector.psycopg import register_vector
from .config import DATABASE_URL

def connect():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn
```

---

## 13. Preprocessing — `src/preprocess.py`

Two jobs: (a) strip HTML, (b) split into chunks.

```python
from bs4 import BeautifulSoup

def html_to_text(raw_html: str) -> str:
    """Strip HTML tags, collapse whitespace."""
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into ~chunk_size-word chunks with overlap.

    WHY words and not tokens? bge-small's tokenizer ≈ 1 word ≈ 1.3 tokens.
    800 words ≈ 1000 tokens, well under bge-small's 512-token limit per call
    — sentence-transformers truncates silently, which we want.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
```

---

## 14. Embeddings — `src/embed.py`

```python
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model = None

def get_model() -> SentenceTransformer:
    """Lazy-load the model so importing this file is cheap."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns a list of 384-dim float lists."""
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # cosine == dot product after normalisation
        show_progress_bar=False,
    )
    return vectors.tolist()
```

---

## 15. Ingestion — `scripts/ingest.py`

This is the biggest script. It glues preprocessing + embedding + DB writes together.

```python
"""Read parquet → preprocess → embed → insert into Postgres."""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.db import connect
from src.embed import embed
from src.preprocess import html_to_text, chunk_text

def main(year: int, limit: int | None):
    parquet_dir = Path(__file__).parent.parent / "data" / "raw" / f"year={year}"
    df = pd.concat(pd.read_parquet(p) for p in parquet_dir.glob("*.parquet"))
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} judgments from {year}")

    conn = connect()
    with conn.cursor() as cur:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Judgments"):
            full_text = html_to_text(row.get("raw_html", ""))
            if not full_text:
                continue

            # Insert judgment metadata (skip if already there)
            cur.execute("""
                INSERT INTO judgments
                    (case_id, title, petitioner, respondent, judge, author_judge,
                     citation, decision_date, disposal_nature, court, full_text)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (case_id) DO NOTHING
            """, (
                row["case_id"], row.get("title"), row.get("petitioner"),
                row.get("respondent"), row.get("judge"), row.get("author_judge"),
                row.get("citation"), row.get("decision_date"),
                row.get("disposal_nature"), row.get("court"), full_text,
            ))

            # Chunk + embed
            chunks = chunk_text(full_text)
            if not chunks:
                continue
            vectors = embed(chunks)

            # Bulk insert chunks
            cur.executemany("""
                INSERT INTO chunks (case_id, chunk_idx, text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (case_id, chunk_idx) DO NOTHING
            """, [
                (row["case_id"], i, chunk, vec)
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            ])
            conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test cap: process only first N judgments")
    args = ap.parse_args()
    main(args.year, args.limit)
```

**Run the smoke test first:**
```bash
python -m scripts.ingest --year 2025 --limit 100
```

**Sanity check after it finishes:**
```bash
psql -d scj_rag -c "SELECT COUNT(*) FROM judgments;"
psql -d scj_rag -c "SELECT COUNT(*) FROM chunks;"
```
You should see ~100 judgments and several thousand chunks.

Then run without `--limit` for the full year.

---

## 16. Retrieval — `src/retrieve.py`

```python
from .db import connect
from .embed import embed

def search(question: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks most relevant to the question."""
    q_vec = embed([question])[0]
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.text, c.case_id, j.title, j.decision_date, j.citation,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN judgments j USING (case_id)
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """, (q_vec, q_vec, k))
        rows = cur.fetchall()
    conn.close()
    return [
        {"text": r[0], "case_id": r[1], "title": r[2],
         "date": r[3], "citation": r[4], "similarity": r[5]}
        for r in rows
    ]
```

**WHY `<=>`?** That is the pgvector cosine-distance operator. `1 - distance` converts it to a similarity score where 1.0 = identical.

---

## 17. RAG pipeline — `src/rag.py`

```python
import anthropic
from .config import ANTHROPIC_API_KEY
from .retrieve import search

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM = """You are an assistant answering questions about Indian Supreme Court
judgments from 2025. Use ONLY the provided excerpts. After every factual claim,
cite the source as [case_id, decision_date]. If the excerpts do not contain the
answer, say so plainly — do not guess."""

def answer(question: str, k: int = 5) -> str:
    hits = search(question, k=k)
    if not hits:
        return "No relevant judgments found."

    context = "\n\n---\n\n".join(
        f"[{h['case_id']}, {h['date']}] {h['title']}\n{h['text']}"
        for h in hits
    )
    prompt = f"Question: {question}\n\nExcerpts:\n{context}\n\nAnswer:"

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text
```

---

## 18. CLI wrapper — `scripts/ask.py`

```python
import argparse
from src.rag import answer

ap = argparse.ArgumentParser()
ap.add_argument("question", help="Your question, in quotes")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

print(answer(args.question, k=args.k))
```

**Try it:**
```bash
python -m scripts.ask "What did the Supreme Court rule about Article 21 in 2025?"
```

---

## 19. What "done" looks like

By the end you should be able to:
1. Run `python -m scripts.ingest --year 2025 --limit 100` and have data in Postgres.
2. Run `python -m scripts.ask "<any question>"` and get a Claude-generated answer with `[case_id, date]` citations after each claim.
3. Verify a citation by querying Postgres directly: `psql -d scj_rag -c "SELECT title, decision_date FROM judgments WHERE case_id = '...';"`

---

## 20. Common beginner pitfalls (read before running anything)

1. **Venv not activated.** Symptom: `ModuleNotFoundError`. Fix: `source .venv/bin/activate`.
2. **`psql: command not found`.** Fix: add Postgres to PATH (step 6).
3. **`psycopg.OperationalError: connection refused`.** Postgres service is not running. Fix: `brew services start postgresql@16`.
4. **`ERROR: extension "vector" is not available`.** You forgot `brew install pgvector`, or you installed pgvector against a different Postgres version.
5. **`embedding` insert fails with "expected vector".** You forgot `register_vector(conn)` in `db.py`.
6. **Embedding step is slow.** Expected. First time the model downloads (~150 MB), then each batch of 64 chunks takes 1–2 s on M-series Mac. For 100 judgments expect ~5–10 minutes total.
7. **`.env` accidentally committed.** Run `git rm --cached .env` immediately, then ensure it is in `.gitignore`.

---

## 21. Build order — the exact sequence I recommend

Do these in order. Do not jump ahead. Each box should pass its verify step before you start the next.

| # | Section | Verify it worked |
|---|---------|------------------|
| 1 | Section 2 — folders | `ls scj_rag/` shows all subdirs |
| 2 | Section 3 — venv | prompt shows `(.venv)` |
| 3 | Sections 4–5 — Python deps | `python -c "import psycopg, anthropic"` runs |
| 4 | Section 6 — Postgres | `psql postgres -c "SELECT 1;"` prints `1` |
| 5 | Sections 7–8 — pgvector + DB | `\dx` in psql lists `vector` |
| 6 | Section 9 — schema | `\dt` lists `judgments` and `chunks` |
| 7 | Section 10 — download data | `ls data/raw/year=2025/*.parquet` shows files |
| 8 | Section 11 — inspect | inspect script prints columns and HTML preview |
| 9 | Sections 12–14 — helper modules | `python -c "from src.embed import embed; print(len(embed(['hi'])[0]))"` prints `384` |
| 10 | Section 15 — ingest (smoke, `--limit 100`) | row counts in both tables look right |
| 11 | Sections 16–18 — ask | Claude answers a sample question with citations |
| 12 | (Optional) full-year ingest | rerun ingest without `--limit` |

---

When you're ready, tell me **"start section 2"** (or whichever section you want to begin with) and I will walk you through it one small block at a time — the same way we did `legal_rag/`.
