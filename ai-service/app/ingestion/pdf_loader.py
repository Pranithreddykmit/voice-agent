"""
PDF Document Loader using PyMuPDF (fitz).
Extracts text page-by-page preserving structural metadata.
"""

import logging
from typing import List, Dict, Any, Union
import pymupdf  # fitz

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails due to corruption, empty files, or invalid formatting."""
    pass


class PDFLoader:
    """
    Extracts text page-by-page from PDF documents.
    Preserves page numbers, source filename, and detects empty/scanned pages.
    """

    @staticmethod
    def load_pdf(
        file_input: Union[str, bytes],
        source_name: str = "document.pdf"
    ) -> List[Dict[str, Any]]:
        """
        Load and extract text from a PDF file path or raw bytes.

        Args:
            file_input: File path (str) or raw binary content (bytes).
            source_name: Name of the source file for metadata citation.

        Returns:
            List of dictionaries containing:
                - page_number: int (1-indexed)
                - text: str (extracted text)
                - source: str (source filename)
                - char_count: int
                - is_empty: bool

        Raises:
            PDFExtractionError: If the document is invalid, password-protected, or unreadable.
        """
        doc = None
        try:
            if isinstance(file_input, bytes):
                if len(file_input) == 0:
                    raise PDFExtractionError(f"Supplied PDF file '{source_name}' is empty (0 bytes).")
                doc = pymupdf.open(stream=file_input, filetype="pdf")
            elif isinstance(file_input, str):
                doc = pymupdf.open(file_input)
            else:
                raise PDFExtractionError("Invalid input type: expected str (path) or bytes.")

            if doc.is_encrypted:
                raise PDFExtractionError(f"PDF '{source_name}' is password-protected and cannot be parsed.")

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFExtractionError(f"PDF '{source_name}' contains zero pages.")

            extracted_pages: List[Dict[str, Any]] = []
            non_empty_count = 0

            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()

                is_empty = len(text) == 0
                if is_empty:
                    logger.warning(
                        f"Page {page_idx + 1} in '{source_name}' has no extractable text. "
                        "Note: OCR for scanned pages is planned for future phases."
                    )
                else:
                    non_empty_count += 1

                extracted_pages.append({
                    "page_number": page_idx + 1,
                    "text": text,
                    "source": source_name,
                    "char_count": len(text),
                    "is_empty": is_empty
                })

            if non_empty_count == 0:
                logger.warning(
                    f"PDF '{source_name}' has {total_pages} pages but no extractable text was found "
                    "(may contain only scanned images)."
                )

            logger.info(
                f"Successfully parsed '{source_name}': {total_pages} pages ({non_empty_count} with text)."
            )
            return extracted_pages

        except PDFExtractionError:
            raise
        except Exception as e:
            logger.error(f"Failed to extract PDF '{source_name}': {str(e)}", exc_info=True)
            raise PDFExtractionError(f"Failed to process PDF '{source_name}': {str(e)}") from e
        finally:
            if doc is not None:
                doc.close()
