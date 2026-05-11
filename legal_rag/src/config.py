"""Central config: load .env once, expose typed settings.

Importing this module triggers `load_dotenv()`, so any other module that
reads os.environ AFTER importing config will see the .env values.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root = parent of `src/`. This lets us locate .env
# regardless of which directory the script was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


# Anthropic — chat model only; the SDK reads ANTHROPIC_API_KEY from env automatically
ANTHROPIC_API_KEY = _required("ANTHROPIC_API_KEY")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "claude-haiku-4-5-20251001")

# Embeddings — local sentence-transformers model.
# 384 = output dim of BAAI/bge-small-en-v1.5. Must match VECTOR(N) in sql/01_schema.sql.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

# Postgres — psycopg accepts a libpq-style connection string
PG_CONN_STR = (
    f"host={os.environ.get('PGHOST', 'localhost')} "
    f"port={os.environ.get('PGPORT', '5432')} "
    f"user={os.environ.get('PGUSER', os.environ.get('USER', 'postgres'))} "
    f"password={os.environ.get('PGPASSWORD', '')} "
    f"dbname={os.environ.get('PGDATABASE', 'legal_rag')}"
)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
