from bs4 import BeautifulSoup

def html_to_text(raw_html: str) -> str:
    """Strip HTML tags, collapse whitespace."""
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    return " ".join(text.split())

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into ~chunk_size-word chunks with overlap.

    WHY words and not tokens? bge-small's tokenizer ≈ 1 word ≈ 1.3 tokens.
    800 words ≈ 1000 tokens, well under bge-small's 512-token limit per call
    — sentence-transformers truncates silently, which we want.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks
