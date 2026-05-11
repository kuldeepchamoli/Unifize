"""Local embedding model wrapper.

We use sentence-transformers (HuggingFace) instead of an API.
- First call: downloads the model from HF (~130MB for bge-small-en-v1.5)
  into ~/.cache/huggingface/, then loads it into memory.
- Subsequent calls: uses the cached model. No network, no API key.
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL, EMBED_DIM

_model: SentenceTransformer | None = None  # module-level singleton, loaded lazily


def _get_model() -> SentenceTransformer:
    """Load the embedding model the first time it's needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of strings.

    Returns a (len(texts), EMBED_DIM) float32 numpy array.

    `normalize_embeddings=True` makes every vector unit-length. With unit
    vectors, cosine distance and inner-product distance are equivalent and
    pgvector's HNSW index search is fastest.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)
