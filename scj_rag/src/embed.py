from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model = None

def get_model() -> SentenceTransformer:
    """Lazy-load the model so importing this file is cheap."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns a list of 384-dim float lists."""
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # cosine == dot product after normalisation
        show_progress_bar=False,
    )
    return vectors.tolist()
