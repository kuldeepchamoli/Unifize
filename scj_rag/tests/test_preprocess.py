from pathlib import Path
import pytest
from scj_rag.src.preprocess import chunk_text, find_pdf


# --- chunk_text ---

def test_chunk_text_empty():
    assert chunk_text("") == []

def test_chunk_text_short_text_is_single_chunk():
    result = chunk_text("hello world", chunk_size=800, overlap=100)
    assert result == ["hello world"]

def test_chunk_text_exact_chunk_size():
    words = ["w"] * 800
    result = chunk_text(" ".join(words), chunk_size=800, overlap=0)
    assert len(result) == 1
    assert len(result[0].split()) == 800

def test_chunk_text_overlap_produces_multiple_chunks():
    words = ["w"] * 1000
    result = chunk_text(" ".join(words), chunk_size=800, overlap=100)
    assert len(result) == 2

def test_chunk_text_overlap_content():
    words = [str(i) for i in range(1000)]
    chunks = chunk_text(" ".join(words), chunk_size=800, overlap=100)
    # Last 100 words of chunk 0 should appear at start of chunk 1
    tail = chunks[0].split()[-100:]
    head = chunks[1].split()[:100]
    assert tail == head

def test_chunk_text_no_empty_chunks():
    words = ["x"] * 2000
    chunks = chunk_text(" ".join(words), chunk_size=800, overlap=100)
    assert all(len(c) > 0 for c in chunks)


# --- find_pdf ---

def test_find_pdf_returns_none_for_empty_path(tmp_path):
    assert find_pdf(tmp_path, "") is None

def test_find_pdf_returns_none_when_no_match(tmp_path):
    assert find_pdf(tmp_path, "2025_1_81_92") is None

def test_find_pdf_finds_exact_en_suffix(tmp_path):
    pdf = tmp_path / "2025_1_81_92_EN.pdf"
    pdf.touch()
    result = find_pdf(tmp_path, "2025_1_81_92")
    assert result == pdf

def test_find_pdf_falls_back_to_glob(tmp_path):
    pdf = tmp_path / "2025_1_81_92_HI.pdf"
    pdf.touch()
    result = find_pdf(tmp_path, "2025_1_81_92")
    assert result == pdf
