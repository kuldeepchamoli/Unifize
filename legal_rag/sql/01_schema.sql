-- Run this AGAINST the legal_rag database (not the default `postgres` db).
-- It is idempotent: safe to re-run; CREATE EXTENSION / CREATE TABLE use IF NOT EXISTS.

-- pgvector ships as a Postgres extension. CREATE EXTENSION registers the
-- `vector` data type and operators (<=>, <->) inside this database.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per legal document (e.g. a single judgment).
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,                 -- which dataset / file this came from
    title        TEXT,                          -- case name, citation, etc.
    raw_text     TEXT,                          -- full original text (nullable: chunks can be the full source)
    metadata     JSONB NOT NULL DEFAULT '{}',   -- court, date, judges, etc.
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk we will embed and search against.
-- 384 = output dimension of BAAI/bge-small-en-v1.5 (sentence-transformers).
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,                  -- 0,1,2,... order within the document
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR(384),                   -- nullable: we may insert text first, embed later
    token_count  INT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- HNSW index for fast approximate nearest-neighbour search on cosine distance.
-- vector_cosine_ops means: when you write `embedding <=> query_vec`, Postgres
-- uses cosine distance (1 - cosine similarity). Smaller = more similar.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- A plain B-tree on document_id helps when joining chunks back to documents.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
