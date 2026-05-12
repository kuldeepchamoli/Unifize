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
