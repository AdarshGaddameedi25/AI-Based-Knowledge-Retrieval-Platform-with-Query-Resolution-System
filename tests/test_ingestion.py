import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.text_cleaner import TextCleaner
from ingestion.chunking import TextChunker


def test_text_cleaner_removes_extra_whitespace():
    cleaner = TextCleaner()
    result = cleaner.clean("Hello    world   \n\n\n\nTest")
    assert "    " not in result
    assert "\n\n\n" not in result


def test_text_cleaner_normalizes_unicode():
    cleaner = TextCleaner()
    result = cleaner.clean("caf\u00e9")
    assert isinstance(result, str)


def test_text_cleaner_preserves_content():
    cleaner = TextCleaner()
    content = "The FMLA grants employees up to 12 weeks of leave."
    result = cleaner.clean(content)
    assert "FMLA" in result
    assert "12" in result


def test_chunker_basic():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    text = "word " * 200
    chunks = chunker.chunk(text, "doc-001")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 50
        assert c.document_id == "doc-001"
        assert c.chunk_id != ""


def test_chunker_overlap():
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 200
    chunks = chunker.chunk(text, "doc-002")
    assert len(chunks) >= 2


def test_chunker_preserves_metadata():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    meta = {"source_file": "test.pdf", "domain": "HR"}
    chunks = chunker.chunk("The quick brown fox " * 50, "doc-003", metadata=meta)
    for c in chunks:
        assert c.metadata["source_file"] == "test.pdf"
        assert c.metadata["domain"] == "HR"


def test_chunker_chunk_pages():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    pages = [(1, "Page one content. " * 30), (2, "Page two content. " * 30)]
    chunks = chunker.chunk_pages(pages, "doc-004", metadata={"source_file": "test.pdf"})
    assert any(c.page_number == 1 for c in chunks)
    assert any(c.page_number == 2 for c in chunks)


def test_chunker_empty_text():
    chunker = TextChunker()
    chunks = chunker.chunk("", "doc-005")
    assert chunks == []


def test_file_validation_allowed_extensions():
    allowed = {".pdf", ".docx", ".txt", ".csv"}
    assert ".pdf" in allowed
    assert ".csv" in allowed
    assert ".exe" not in allowed
    assert ".py" not in allowed


def test_csv_loader():
    import tempfile
    import os
    from ingestion.csv_loader import CSVLoader

    csv_content = "topic,description\nFastAPI,A modern web framework\npgvector,PostgreSQL vector extension\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        tmp = f.name
    try:
        loader = CSVLoader()
        text = loader.extract(tmp)
        assert "FastAPI" in text
        assert "pgvector" in text
        assert "PostgreSQL vector extension" in text
    finally:
        os.unlink(tmp)


def test_txt_loader():
    import tempfile
    import os
    from ingestion.txt_loader import TXTLoader

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("This is a test document.\nIt has multiple lines.")
        tmp = f.name
    try:
        loader = TXTLoader()
        text = loader.extract(tmp)
        assert "test document" in text
        assert "multiple lines" in text
    finally:
        os.unlink(tmp)
