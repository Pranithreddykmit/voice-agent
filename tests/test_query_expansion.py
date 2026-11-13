"""
Focused Unit Tests for Non-Destructive Retrieval Query Expansion.
Verifies:
1. Original transcript unchanged
2. Expansion adds synonymous variants
3. Telugu-only queries still work
4. English-only queries still work
5. Code-mixed queries still work
6. Wi-Fi failure case is recovered ("విటై" -> expands with "wifi", "wi-fi", "వైఫై")
7. No unrelated retrieval regressions
"""

import pytest
from app.retrieval.query_expansion import expand_retrieval_query
from app.schemas import DocumentChunk
from app.retrieval.bm25 import BM25Index


def test_original_transcript_unchanged():
    """Verify that the original query string is not mutated and remains the prefix."""
    raw_query = "కాలేజీ విటై password ఎలా మార్చాలి?"
    expanded = expand_retrieval_query(raw_query)
    
    # Original query must remain identical as prefix
    assert expanded.startswith(raw_query)
    assert raw_query == "కాలేజీ విటై password ఎలా మార్చాలి?"


def test_expansion_adds_variants():
    """Verify that expansion appends academic and acoustic equivalents."""
    query = "What is the fee payment deadline?"
    expanded = expand_retrieval_query(query)
    
    assert "fee" in expanded
    assert "deadline" in expanded
    assert "ఫీజు" in expanded
    assert "గడువు" in expanded or "డెడ్‌లైన్" in expanded


def test_telugu_only_query_expansion():
    """Verify pure Telugu queries expand with English academic equivalents."""
    query = "హాస్టల్ దరఖాస్తు గడువు ఎప్పుడు?"
    expanded = expand_retrieval_query(query)
    
    assert query in expanded
    assert "hostel" in expanded
    assert "application" in expanded
    assert "deadline" in expanded


def test_english_only_query_expansion():
    """Verify pure English queries expand with Telugu equivalents."""
    query = "library book return policy"
    expanded = expand_retrieval_query(query)
    
    assert query in expanded
    assert "లైబ్రరీ" in expanded or "గ్రంథాలయం" in expanded


def test_code_mixed_query_expansion():
    """Verify code-mixed Telugu-English queries expand accurately."""
    query = "Semester exams ki minimum attendance entha undali?"
    expanded = expand_retrieval_query(query)
    
    assert query in expanded
    assert "హాజరు" in expanded or "అటెండెన్స్" in expanded
    assert "పరీక్షలు" in expanded or "సెమిస్టర్" in expanded


def test_wifi_failure_case_recovered():
    """
    Specifically verifies the isolated ASR failure recovery:
    Spoken 'వైఫై' -> ASR 'విటై'.
    Query expansion must add 'wifi', 'wi-fi', and 'వైఫై' to enable BM25 & dense matches.
    """
    asr_transcript = "కాలేజీ క్యాంపస్ లో విటై సెట్టింగ్స్ మార్చడం ఎలా?"
    expanded = expand_retrieval_query(asr_transcript)
    
    # Ensure all variants are present in the retrieval query
    assert "wifi" in expanded.lower()
    assert "వైఫై" in expanded
    
    # Test with BM25 Index containing Wi-Fi IT knowledge chunk
    bm25 = BM25Index()
    wifi_chunk = DocumentChunk(
        chunk_id="chunk_wifi_it_01",
        document_id="doc_campus_it",
        source="campus_it_guide.pdf",
        page=1,
        chunk_index=0,
        text="Campus Wi-Fi credentials and connection setup can be completed at the IT Helpdesk.",
        char_count=85
    )
    unrelated_chunk = DocumentChunk(
        chunk_id="chunk_hostel_01",
        document_id="doc_hostel",
        source="hostel_manual.pdf",
        page=1,
        chunk_index=0,
        text="Hostel accommodation rules and dining hall timings.",
        char_count=52
    )
    bm25.add_chunks([wifi_chunk, unrelated_chunk])
    
    # Without expansion, 'విటై' misses the Wi-Fi chunk in BM25
    raw_results = bm25.search(query=asr_transcript, top_k=1)
    assert len(raw_results) == 0
    
    # With expansion, 'wifi' matches the Wi-Fi chunk as Rank #1
    expanded_results = bm25.search(query=expanded, top_k=1)
    assert len(expanded_results) > 0
    assert expanded_results[0].chunk_id == "chunk_wifi_it_01"


def test_unrelated_query_no_hallucinated_expansion():
    """Verify queries with no matching academic concepts remain clean without false expansions."""
    query = "మొక్కల పెరుగుదల మరియు కిరణజన్య సంయోగ క్రియ"
    expanded = expand_retrieval_query(query)
    
    # No academic synonyms should be falsely attached
    assert expanded == query
