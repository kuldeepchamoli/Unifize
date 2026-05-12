import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384