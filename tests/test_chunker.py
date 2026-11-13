"""
Unit tests for StructureAwareChunker.
"""

import pytest
from app.ingestion.chunker import StructureAwareChunker


def test_chunker_basic():
    chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "text": "Paragraph 1 about attendance.\n\nParagraph 2 with details.", "source": "test.pdf"}
    ]
    chunks = chunker.chunk_pages(pages, document_id="doc_123", source_name="test.pdf")
    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc_123"
    assert chunks[0].source == "test.pdf"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id is not None


def test_chunker_deterministic_ids():
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=50)
    pages = [
        {"page_number": 1, "text": "Consistent content for deterministic ID testing.", "source": "sample.pdf"}
    ]
    chunks_run1 = chunker.chunk_pages(pages, document_id="doc_abc", source_name="sample.pdf")
    chunks_run2 = chunker.chunk_pages(pages, document_id="doc_abc", source_name="sample.pdf")

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text


def test_chunker_telugu_text():
    chunker = StructureAwareChunker(chunk_size=150, chunk_overlap=30)
    telugu_text = (
        "విశ్వవిద్యాలయ నిబంధనల ప్రకారం సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి ప్రతి విద్యార్థికి కనీసం 75% హాజరు తప్పనిసరి.\n\n"
        "అనారోగ్య కారణాల వల్ల 65% నుండి 75% వరకు హాజరు ఉన్నవారికి మినహాయింపు లభిస్తుంది."
    )
    pages = [{"page_number": 1, "text": telugu_text, "source": "telugu_doc.pdf"}]
    chunks = chunker.chunk_pages(pages, document_id="doc_tel", source_name="telugu_doc.pdf")

    assert len(chunks) >= 1
    assert "75% హాజరు" in chunks[0].text
    assert chunks[0].page == 1


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        StructureAwareChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        StructureAwareChunker(chunk_size=100, chunk_overlap=150)
