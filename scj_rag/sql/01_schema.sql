

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

