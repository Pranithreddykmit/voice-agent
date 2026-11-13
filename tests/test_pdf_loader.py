"""
Unit tests for PDFLoader.
"""

import pytest
from app.ingestion.pdf_loader import PDFLoader, PDFExtractionError


def test_pdf_loader_multi_page(sample_pdf_bytes):
    pages = PDFLoader.load_pdf(sample_pdf_bytes, source_name="regulations.pdf")
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Attendance Rule" in pages[0]["text"]
    assert pages[0]["source"] == "regulations.pdf"
    assert pages[0]["is_empty"] is False

    assert pages[1]["page_number"] == 2
    assert "Grading Rule" in pages[1]["text"]
    assert pages[1]["source"] == "regulations.pdf"
    assert pages[1]["is_empty"] is False


def test_pdf_loader_empty_page(sample_empty_pdf_bytes):
    pages = PDFLoader.load_pdf(sample_empty_pdf_bytes, source_name="empty_page.pdf")
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert pages[0]["is_empty"] is True
    assert pages[0]["char_count"] == 0


def test_pdf_loader_zero_bytes():
    with pytest.raises(PDFExtractionError) as exc_info:
        PDFLoader.load_pdf(b"", source_name="zero.pdf")
    assert "empty" in str(exc_info.value).lower()


def test_pdf_loader_corrupted_bytes():
    with pytest.raises(PDFExtractionError):
        PDFLoader.load_pdf(b"not a valid pdf content", source_name="corrupt.pdf")
