import os
from dotenv import load_dotenv
load_dotenv()

def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val

ANTHROPIC_API_KEY = _require("ANTHROPIC_API_KEY")
DATABASE_URL = _require("DATABASE_URL")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
