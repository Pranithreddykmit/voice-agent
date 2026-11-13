"""
Unit tests for BM25Index and Tokenizer.
"""

import pytest
from app.retrieval.bm25 import BM25Index
from app.retrieval.tokenizer import tokenize_text
from app.schemas import DocumentChunk


def test_tokenizer_english_and_telugu():
    text = "Attendance is 75%. కనీస హాజరు 75% తప్పనిసరి!"
    tokens = tokenize_text(text)
    assert "attendance" in tokens
    assert "75" in tokens
    assert "కనీస" in tokens
    assert "హాజరు" in tokens
    assert "తప్పనిసరి" in tokens
    assert "!" not in tokens
    assert "%" not in tokens


def test_bm25_search_english():
    index = BM25Index()
    chunks = [
        DocumentChunk(
            chunk_id="c1", document_id="doc1", source="rules.pdf", page=1, chunk_index=0,
            text="Every registered student must maintain 75% minimum attendance.", char_count=65
        ),
        DocumentChunk(
            chunk_id="c2", document_id="doc1", source="rules.pdf", page=2, chunk_index=1,
            text="Grade O corresponds to 90% to 100% marks.", char_count=42
        )
    ]
    index.add_chunks(chunks)

    results = index.search("minimum attendance", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].retrieval_method == "bm25"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_bm25_search_telugu():
    index = BM25Index()
    chunks = [
        DocumentChunk(
            chunk_id="c_te1", document_id="doc_te", source="rules.pdf", page=1, chunk_index=0,
            text="సెమిస్టర్ పరీక్షలకు కనీసం 75% హాజరు తప్పనిసరి.", char_count=45
        ),
        DocumentChunk(
            chunk_id="c_te2", document_id="doc_te", source="rules.pdf", page=3, chunk_index=1,
            text="రీవాల్యుయేషన్ ఫీజు రూ. 1000 చెల్లించాలి.", char_count=40
        )
    ]
    index.add_chunks(chunks)

    results = index.search("రీవాల్యుయేషన్ ఫీజు", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c_te2"
    assert results[0].page == 3


def test_bm25_empty_index_and_query():
    index = BM25Index()
    assert index.search("anything") == []

    index.add_chunks([
        DocumentChunk(
            chunk_id="c1", document_id="d1", source="s.pdf", page=1, chunk_index=0,
            text="Sample text.", char_count=12
        )
    ])
    assert index.search("") == []
