import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import DocumentLoader
from ingestion.txt_loader import TXTLoader
from ingestion.csv_loader import CSVLoader
from ingestion.text_cleaner import TextCleaner

SAMPLE_HR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents", "hr", "employee_handbook.txt")
SAMPLE_TECH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents", "technology", "cloud_security.txt")


def test_txt_loader_extracts_text():
    loader = TXTLoader()
    text = loader.extract(SAMPLE_HR)
    assert len(text) > 100
    assert "ACME" in text


def test_csv_loader(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("Name,Age,Department\nAlice,30,Engineering\nBob,25,HR\n")
    loader = CSVLoader()
    text = loader.extract(str(csv_file))
    assert "Alice" in text
    assert "Engineering" in text
    assert "Bob" in text


def test_document_loader_txt():
    loader = DocumentLoader()
    doc = loader.load(SAMPLE_HR)
    assert doc.file_type == "TXT"
    assert doc.document_id is not None
    assert len(doc.text) > 50


def test_document_loader_unsupported_type(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("data")
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load(str(bad_file))


def test_document_loader_missing_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/path/file.txt")


def test_text_cleaner_removes_extra_whitespace():
    cleaner = TextCleaner()
    raw = "Hello    world.\n\n\n\nThis  is   a  test."
    cleaned = cleaner.clean(raw)
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_text_cleaner_on_sample_document():
    loader = TXTLoader()
    cleaner = TextCleaner()
    raw = loader.extract(SAMPLE_TECH)
    cleaned = cleaner.clean(raw)
    assert len(cleaned) > 0
    assert cleaned == cleaned.strip()


def test_load_directory():
    loader = DocumentLoader()
    hr_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_documents", "hr")
    docs = loader.load_directory(hr_dir)
    assert len(docs) >= 2
    for doc in docs:
        assert doc.file_type in {"TXT", "PDF", "DOCX", "CSV"}
        assert len(doc.text) > 0
