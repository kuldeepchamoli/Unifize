from pathlib import Path
import pdfplumber


def pdf_to_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF, collapse whitespace."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = " ".join(pages)
        return " ".join(text.split())
    except Exception:
        return ""


def find_pdf(pdf_dir: Path, path_value: str) -> Path | None:
    """Given the parquet 'path' column value (e.g. '2025_1_81_92'),
    find the matching PDF file (e.g. '2025_1_81_92_EN.pdf') in pdf_dir.

    The tar uses the naming convention: <path>_EN.pdf
    We try that first, then fall back to a glob in case the suffix differs.
    """
    if not path_value:
        return None
    candidate = pdf_dir / f"{path_value}_EN.pdf"
    if candidate.exists():
        return candidate
    matches = list(pdf_dir.glob(f"{path_value}*.pdf"))
    return matches[0] if matches else None


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
