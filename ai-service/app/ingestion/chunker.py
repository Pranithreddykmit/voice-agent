"""
Structure-Aware Document Chunker.
Splits document pages on natural structural boundaries (paragraphs, lines, sentences)
while maintaining deterministic IDs, page metadata, and overlap.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from app.schemas import DocumentChunk


class StructureAwareChunker:
    """
    Chunks extracted document pages into semantically cohesive passages
    respecting structural boundaries and generating deterministic identifiers.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def generate_chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
        """
        Generate a deterministic SHA-256 hex string ID based on document, page, index, and content.
        """
        raw_key = f"{document_id}:{page_number}:{chunk_index}:{text[:64]}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def _split_into_structural_units(self, text: str) -> List[str]:
        """
        Splits text into cohesive structural units (paragraphs, headings, bullet points, or sentences).
        Supports Telugu sentence boundaries (periods, dandas '।', question marks, exclamation marks).
        """
        # First split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for p in paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                # Split large paragraphs into sentences or clauses
                # Regex matches English and Indic sentence terminators
                sentence_pattern = r"(?<=[.!?।\n])\s+"
                sentences = [s.strip() for s in re.split(sentence_pattern, p) if s.strip()]
                for s in sentences:
                    if len(s) <= self.chunk_size:
                        units.append(s)
                    else:
                        # Fallback for unusually long unbroken text: split on whitespace
                        words = s.split()
                        current_word_chunk = []
                        current_len = 0
                        for w in words:
                            if current_len + len(w) + 1 > self.chunk_size and current_word_chunk:
                                units.append(" ".join(current_word_chunk))
                                current_word_chunk = [w]
                                current_len = len(w)
                            else:
                                current_word_chunk.append(w)
                                current_len += len(w) + 1
                        if current_word_chunk:
                            units.append(" ".join(current_word_chunk))

        return units

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        source_name: str
    ) -> List[DocumentChunk]:
        """
        Convert extracted pages into structured chunks with metadata.

        Args:
            pages: List of page dictionaries from PDFLoader.
            document_id: Unique document identifier.
            source_name: Original file name.

        Returns:
            List of DocumentChunk instances.
        """
        chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        for page in pages:
            page_num = page["page_number"]
            page_text = page["text"].strip()

            if not page_text:
                continue

            structural_units = self._split_into_structural_units(page_text)
            if not structural_units:
                continue

            current_chunk_units: List[str] = []
            current_chunk_len = 0

            for unit in structural_units:
                unit_len = len(unit)
                # Check if adding this unit exceeds chunk_size
                if current_chunk_len + unit_len + 1 > self.chunk_size and current_chunk_units:
                    chunk_text = "\n".join(current_chunk_units).strip()
                    if chunk_text:
                        chunk_id = self.generate_chunk_id(document_id, page_num, global_chunk_index, chunk_text)
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                document_id=document_id,
                                source=source_name,
                                page=page_num,
                                chunk_index=global_chunk_index,
                                text=chunk_text,
                                char_count=len(chunk_text)
                            )
                        )
                        global_chunk_index += 1

                    # Retain overlap from previous units
                    overlap_units: List[str] = []
                    overlap_len = 0
                    for prev_unit in reversed(current_chunk_units):
                        if overlap_len + len(prev_unit) + 1 <= self.chunk_overlap:
                            overlap_units.insert(0, prev_unit)
                            overlap_len += len(prev_unit) + 1
                        else:
                            break

                    current_chunk_units = overlap_units + [unit]
                    current_chunk_len = sum(len(u) + 1 for u in current_chunk_units)
                else:
                    current_chunk_units.append(unit)
                    current_chunk_len += unit_len + 1

            # Append the remaining chunk for this page
            if current_chunk_units:
                chunk_text = "\n".join(current_chunk_units).strip()
                if chunk_text:
                    chunk_id = self.generate_chunk_id(document_id, page_num, global_chunk_index, chunk_text)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            source=source_name,
                            page=page_num,
                            chunk_index=global_chunk_index,
                            text=chunk_text,
                            char_count=len(chunk_text)
                        )
                    )
                    global_chunk_index += 1

        return chunks
