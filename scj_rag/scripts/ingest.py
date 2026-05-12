"""Read parquet → preprocess → embed → insert into Postgres."""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.db import connect
from src.embed import embed
from src.preprocess import html_to_text, chunk_text

def _v(val):
    """Return None for any pandas NA/NaN, else the value as-is.

    pd.where(pd.notnull(...), None) catches float NaN but not pd.NA
    (the NAType used in nullable-integer/string columns from parquet).
    pd.isna() handles both, so we use it here at value access time.
    """
    try:
        return None if pd.isna(val) else val
    except (TypeError, ValueError):
        return val


def _date(val):
    """Parse a date string (any common format) to datetime.date, else None.

    Postgres rejects non-ISO strings like "15-01-2025" (DD-MM-YYYY) via
    psycopg. pd.to_datetime with dayfirst=True handles that format, and
    errors='coerce' turns unparseable values into NaT -> None.
    """
    if _v(val) is None:
        return None
    try:
        return pd.to_datetime(val, dayfirst=True, errors="coerce").date()
    except Exception:
        return None


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
                _v(row["case_id"]), _v(row.get("title")), _v(row.get("petitioner")),
                _v(row.get("respondent")), _v(row.get("judge")), _v(row.get("author_judge")),
                _v(row.get("citation")), _date(row.get("decision_date")),
                _v(row.get("disposal_nature")), _v(row.get("court")), full_text,
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
