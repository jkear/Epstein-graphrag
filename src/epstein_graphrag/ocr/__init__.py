"""OCR pipeline — Marker + Ollama/DeepSeek with forensic context.

This package provides OCR capabilities for processing PDF documents with
domain-specific context for forensic document analysis.

Key modules:
- marker_pipeline: Marker + DeepSeek-OCR-MLX pipeline
- marker_pipeline_ollama: Marker + Ollama Vision pipeline
- ollama_ocr: Ollama vision model OCR interface
- deepseek_ocr: DeepSeek-OCR-MLX interface
- forensic_prompts: Forensic-aware OCR prompts
- forensic_marker_processor: Custom Marker LLM processor with forensic context
- duplicate_detector: Detect duplicate PDFs with different redactions
- redaction_merger: Merge content across duplicate documents
"""

from epstein_graphrag.ocr.duplicate_detector import (
    DocumentFingerprint,
    DuplicateDetector,
    DuplicateGroup,
    RedactionRegion,
    detect_duplicates_in_directory,
)
from epstein_graphrag.ocr.forensic_prompts import (
    FORENSIC_OCR_PROMPT,
    FORENSIC_PHOTOGRAPH_PROMPT,
    IDENTITY_DOCUMENT_OCR_PROMPT,
    LEGAL_DOCUMENT_OCR_PROMPT,
    REDACTION_AWARE_OCR_PROMPT,
    get_forensic_ocr_prompt,
)
from epstein_graphrag.ocr.redaction_merger import (
    MergedDocument,
    RedactionFillin,
    RedactionMerger,
    merge_duplicate_group,
)

__all__ = [
    # Duplicate detection
    "DuplicateDetector",
    "DocumentFingerprint",
    "DuplicateGroup",
    "RedactionRegion",
    "detect_duplicates_in_directory",
    # Redaction merging
    "RedactionMerger",
    "RedactionFillin",
    "MergedDocument",
    "merge_duplicate_group",
    # Forensic prompts
    "FORENSIC_OCR_PROMPT",
    "FORENSIC_PHOTOGRAPH_PROMPT",
    "IDENTITY_DOCUMENT_OCR_PROMPT",
    "LEGAL_DOCUMENT_OCR_PROMPT",
    "REDACTION_AWARE_OCR_PROMPT",
    "get_forensic_ocr_prompt",
]
