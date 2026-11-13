"""
Multilingual and Indic Tokenizer for Sparse BM25 Retrieval.
Splits text into cohesive word tokens preserving Telugu Unicode sequences and alphanumeric terms.
"""

import re
from typing import List

# Matches English words, numbers, and Telugu Unicode script words (\u0C00-\u0C7F)
TOKEN_PATTERN = re.compile(r"[\u0C00-\u0C7F]+|[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes input text into a list of normalized tokens.
    - English words are lowercased.
    - Telugu Unicode words are preserved intact.
    - Punctuation, symbols, and formatting noise are filtered out.
    """
    if not text:
        return []
    
    matches = TOKEN_PATTERN.findall(text)
    tokens: List[str] = []
    for token in matches:
        # Lowercase if ASCII
        t_clean = token.strip()
        if t_clean:
            tokens.append(t_clean.lower())
    return tokens
