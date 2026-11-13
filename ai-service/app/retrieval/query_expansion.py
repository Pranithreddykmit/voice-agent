"""
Non-Destructive Retrieval Query Expansion Layer.
Expands bilingual Telugu-English academic queries and known ASR phonetic variants
strictly for internal dense and sparse retrieval WITHOUT mutating user transcripts,
conversation history, or LLM generation prompts.
"""

import re
from typing import Dict, List, Set

# Bilingual Domain Synonyms and ASR Phonetic Variant Clusters
# Key: canonical concept identifier
# Value: set of equivalent terms across English, Telugu Unicode, Romanized Telugu, and known Whisper phonetic slips
ACADEMIC_EXPANSION_CLUSTERS: List[Set[str]] = [
    {"attendance", "అటెండెన్స్", "హాజరు", "కనీస హాజరు"},
    {"semester", "సెమిస్టర్"},
    {"exams", "exam", "examination", "ఎగ్జామ్స్", "పరీక్షలు", "పరీక్ష"},
    {"hostel", "హాస్టల్", "వసతి"},
    {"library", "లైబ్రరీ", "గ్రంథాలయం"},
    {"fee", "fees", "ఫీజు"},
    {"medical leave", "medical", "మెడికల్ లీవ్", "వైద్య సెలవు"},
    {"wifi", "wi-fi", "వైఫై", "విటై"},
    {"application", "అప్లికేషన్", "దరఖాస్తు"},
    {"deadline", "డెడ్‌లైన్", "డెడ్లైన్", "గడువు"},
    {"revaluation", "రీవాల్యుయేషన్", "recounting"},
    {"hall ticket", "హాల్ టికెట్"},
    {"condonation", "కండోనేషన్"},
    {"dean", "డీన్"},
    {"qdrant", "క్యూడ్రాంట్"},
    {"postgresql", "పోస్ట్‌గ్రేస్"},
    {"credits", "క్రెడిట్లు", "sgpa", "cgpa"},
]


def expand_retrieval_query(query: str) -> str:
    """
    Expands a search query with bilingual academic equivalents and ASR phonetic variants.
    
    Rules:
    1. Preserves original query verbatim as the prefix.
    2. Identifies matching synonym clusters via substring and token matching.
    3. Appends missing variants to enhance BM25 sparse keyword and dense embedding recall.
    4. Never alters the original query string in-place.
    
    Example:
        Input:  "కాలేజీ విటై password ఎలా మార్చాలి?"
        Output: "కాలేజీ విటై password ఎలా మార్చాలి? wifi wi-fi వైఫై"
    """
    if not query or not query.strip():
        return query

    q_lower = query.lower()
    expanded_terms: List[str] = []
    seen_in_query: Set[str] = set(re.findall(r"[\u0C00-\u0C7F]+|[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", q_lower))

    for cluster in ACADEMIC_EXPANSION_CLUSTERS:
        # Check if any member of the cluster is present in the query
        cluster_matched = False
        for term in cluster:
            term_lower = term.lower()
            if " " in term_lower:
                # Multi-word phrase check
                if term_lower in q_lower:
                    cluster_matched = True
                    break
            else:
                # Single token check
                if term_lower in seen_in_query or term_lower in q_lower:
                    cluster_matched = True
                    break

        # If matched, collect all other terms in the cluster not yet in the query
        if cluster_matched:
            for term in cluster:
                term_lower = term.lower()
                if term_lower not in seen_in_query and term_lower not in q_lower:
                    if term not in expanded_terms:
                        expanded_terms.append(term)

    if not expanded_terms:
        return query

    # Append expanded variants cleanly
    return f"{query.strip()} {' '.join(expanded_terms)}"
