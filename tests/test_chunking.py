import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunking import TextChunker
from ingestion.txt_loader import TXTLoader
from ingestion.text_cleaner import TextCleaner

SAMPLE_HR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents", "hr", "employee_handbook.txt")


@pytest.fixture
def sample_text():
    loader = TXTLoader()
    cleaner = TextCleaner()
    raw = loader.extract(SAMPLE_HR)
    return cleaner.clean(raw)


def test_chunker_returns_chunks(sample_text):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text, document_id="test-doc-001")
    assert len(chunks) > 1


def test_chunk_ids_are_unique(sample_text):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text, document_id="test-doc-001")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_has_required_fields(sample_text):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text, document_id="test-doc-001")
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id == "test-doc-001"
        assert chunk.text
        assert chunk.token_count > 0
        assert isinstance(chunk.chunk_index, int)


def test_chunk_size_respected(sample_text):
    chunk_size = 100
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=10)
    chunks = chunker.chunk(sample_text, document_id="test-doc-002")
    for chunk in chunks[:-1]:
        assert chunk.token_count <= chunk_size


def test_chunk_overlap_creates_coverage(sample_text):
    chunker_no_overlap = TextChunker(chunk_size=150, chunk_overlap=0)
    chunker_with_overlap = TextChunker(chunk_size=150, chunk_overlap=50)
    chunks_no = chunker_no_overlap.chunk(sample_text, document_id="doc-a")
    chunks_with = chunker_with_overlap.chunk(sample_text, document_id="doc-b")
    assert len(chunks_with) >= len(chunks_no)


def test_chunk_index_sequential(sample_text):
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk(sample_text, document_id="test-doc-003")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_empty_text_produces_no_chunks():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk("", document_id="empty-doc")
    assert len(chunks) == 0


def test_short_text_produces_single_chunk():
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    text = "This is a very short document."
    chunks = chunker.chunk(text, document_id="short-doc")
    assert len(chunks) == 1
    assert "short document" in chunks[0].text
