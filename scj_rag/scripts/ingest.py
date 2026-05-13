"""Read parquet → extract PDF text → embed → insert into Postgres."""
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.db import connect
from src.embed import embed
from src.preprocess import pdf_to_text, find_pdf, chunk_text


def _v(val):
    """Return None for any pandas NA/NaN, else the value as-is."""
    try:
        return None if pd.isna(val) else val
    except (TypeError, ValueError):
        return val


def _date(val):
    """Parse a date string (any common format) to datetime.date, else None."""
    if _v(val) is None:
        return None
    try:
        return pd.to_datetime(val, dayfirst=True, errors="coerce").date()
    except Exception:
        return None


def main(year: int, pdf_dir: Path, limit: int | None):
    parquet_dir = Path(__file__).parent.parent / "data" / "raw" / f"year={year}"
    df = pd.concat(
        pd.read_parquet(p) for p in parquet_dir.glob("*.parquet")
    ).drop_duplicates(subset=["case_id"])

    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} judgments from {year}")

    skipped = 0
    with connect() as conn:
        with conn.cursor() as cur:
            # Fetch already-ingested case_ids to short-circuit expensive PDF work
            cur.execute("SELECT case_id FROM judgments")
            already_done = {r[0] for r in cur.fetchall()}

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Judgments"):
                case_id = _v(row["case_id"])
                if case_id in already_done:
                    continue

                path_value = _v(row.get("path"))
                pdf_path = find_pdf(pdf_dir, path_value) if path_value else None

                if pdf_path is None:
                    skipped += 1
                    continue

                full_text = pdf_to_text(pdf_path)
                if not full_text:
                    skipped += 1
                    continue

                cur.execute("""
                    INSERT INTO judgments
                        (case_id, title, petitioner, respondent, judge, author_judge,
                         citation, decision_date, disposal_nature, court, full_text)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (case_id) DO NOTHING
                """, (
                    case_id, _v(row.get("title")), _v(row.get("petitioner")),
                    _v(row.get("respondent")), _v(row.get("judge")), _v(row.get("author_judge")),
                    _v(row.get("citation")), _date(row.get("decision_date")),
                    _v(row.get("disposal_nature")), _v(row.get("court")), full_text,
                ))

                chunks = chunk_text(full_text)
                if not chunks:
                    continue
                vectors = embed(chunks)

                cur.executemany("""
                    INSERT INTO chunks (case_id, chunk_idx, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (case_id, chunk_idx) DO NOTHING
                """, [
                    (case_id, i, chunk, vec)
                    for i, (chunk, vec) in enumerate(zip(chunks, vectors))
                ])
                conn.commit()

    print(f"Done. Skipped {skipped} judgments (no PDF found or empty text).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/raw/year=2025/pdfs"),
        help="Directory containing extracted PDF files",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test cap: process only first N judgments")
    args = ap.parse_args()
    main(args.year, args.pdf_dir, args.limit)
