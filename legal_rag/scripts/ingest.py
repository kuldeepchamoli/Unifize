"""Ingest the Indian Acts CSV into Postgres + pgvector.

Pipeline:
    CSV  ->  filter (optional sample size)
         ->  insert one row per Act into `documents`
         ->  build "{Act name}, Section {N}\\n{text}" strings
         ->  embed in batches of 64 with sentence-transformers
         ->  insert one row per chunk into `chunks` with the embedding

Run:
    python -m scripts.ingest               # default: 30 acts (fast smoke test)
    python -m scripts.ingest --limit 100   # 100 acts
    python -m scripts.ingest --limit 0     # full dataset (35,892 chunks)
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import RAW_DATA_DIR
from src.db import get_conn
from src.embed import embed

CSV_PATH = RAW_DATA_DIR / "acts_csv.csv"
SOURCE_TAG = "kaggle:aayaniqbal2005/indian-supreme-court-judgments-section-wise"
EMBED_BATCH = 64


def load_dataframe(limit_acts: int) -> pd.DataFrame:
    """Read CSV, optionally restrict to the first N unique acts."""
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["text", "name"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    if limit_acts > 0:
        wanted = df["name"].drop_duplicates().head(limit_acts).tolist()
        df = df[df["name"].isin(wanted)].copy()

    # Sort so chunk_index is stable across runs.
    df = df.sort_values(["name", "section"], kind="stable").reset_index(drop=True)
    df["chunk_index"] = df.groupby("name").cumcount()
    return df


def insert_documents(conn, names: list[str]) -> dict[str, int]:
    """Insert one row per Act; return {name: document_id}."""
    name_to_id: dict[str, int] = {}
    with conn.cursor() as cur:
        for name in names:
            cur.execute(
                "INSERT INTO documents (source, title, metadata) "
                "VALUES (%s, %s, %s::jsonb) RETURNING id",
                (SOURCE_TAG, name, '{}'),
            )
            name_to_id[name] = cur.fetchone()[0]
    return name_to_id


def insert_chunks(conn, df: pd.DataFrame, name_to_id: dict[str, int]) -> None:
    """Embed and insert chunks in batches."""
    texts_for_embedding = [
        f"{row.name}, Section {row.section}\n{row.text}"
        for row in df.itertuples(index=False)
    ]

    rows_to_insert = []
    with conn.cursor() as cur:
        for start in tqdm(range(0, len(df), EMBED_BATCH), desc="embed+insert"):
            end = start + EMBED_BATCH
            batch_df = df.iloc[start:end]
            batch_texts = texts_for_embedding[start:end]
            batch_vecs = embed(batch_texts)

            for (_, row), vec in zip(batch_df.iterrows(), batch_vecs):
                rows_to_insert.append((
                    name_to_id[row["name"]],
                    int(row["chunk_index"]),
                    str(row["text"]),
                    vec,
                    int(row["text_length"]) if pd.notna(row["text_length"]) else None,
                ))

            # Flush in chunks of 500 to keep memory bounded.
            if len(rows_to_insert) >= 500:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, chunk_text, embedding, token_count) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    rows_to_insert,
                )
                rows_to_insert = []

        if rows_to_insert:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, chunk_text, embedding, token_count) "
                "VALUES (%s, %s, %s, %s, %s)",
                rows_to_insert,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30,
                        help="Number of unique Acts to ingest. 0 = all.")
    parser.add_argument("--wipe", action="store_true", default=True,
                        help="Truncate documents and chunks before inserting.")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found at {CSV_PATH}. Did you run the kaggle download?")

    print(f"Loading CSV from {CSV_PATH} ...")
    df = load_dataframe(args.limit)
    n_acts = df["name"].nunique()
    n_chunks = len(df)
    print(f"Will ingest {n_acts} acts -> {n_chunks} chunks")

    with get_conn() as conn:
        if args.wipe:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")

        name_to_id = insert_documents(conn, df["name"].drop_duplicates().tolist())
        insert_chunks(conn, df, name_to_id)
        conn.commit()

    print(f"Done. Inserted {n_acts} documents and {n_chunks} chunks.")


if __name__ == "__main__":
    main()
