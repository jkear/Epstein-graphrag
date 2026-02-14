"""Merge redacted content across duplicate documents.

This module uses the duplicate detection results to merge content from
multiple copies of the same document, filling in gaps where different
copies are redacted in different locations.

The merger works by:
1. Selecting the document with the fewest redactions as the base
2. Identifying regions redacted in the base but present in other copies
3. Extracting supplemental content from other copies
4. Creating a merged output with metadata about what was filled in
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from epstein_graphrag.ocr.duplicate_detector import (
    DocumentFingerprint,
    DuplicateGroup,
)
from epstein_graphrag.ocr.ollama_ocr import extract_text_from_pdf

logger = logging.getLogger(__name__)


@dataclass
class RedactionFillin:
    """A content fillin from a supplemental document.

    Attributes:
        page_num: Page number where fillin occurs
        base_bbox: Bounding box of redaction in base document
        source_document: Path to document providing the fillin
        context: Description of what was filled in
        confidence: Confidence that this is correct fillin (0-1)
    """

    page_num: int
    base_bbox: Tuple[int, int, int, int]
    source_document: str
    context: str
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "base_bbox": list(self.base_bbox),
            "source_document": self.source_document,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass
class MergedDocument:
    """Result of merging duplicate documents.

    Attributes:
        group_id: ID of the duplicate group
        source_files: Paths to all source documents
        base_document: Document used as primary source
        merged_text: Combined text from all sources
        fillins: List of redaction fillins applied
        metadata: Additional merge information
    """

    group_id: str
    source_files: List[str]
    base_document: str
    merged_text: str
    fillins: List[RedactionFillin] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "source_files": self.source_files,
            "base_document": self.base_document,
            "merged_text": self.merged_text,
            "fillins": [f.to_dict() for f in self.fillins],
            "fillin_count": len(self.fillins),
            "metadata": self.metadata,
        }


class RedactionMerger:
    """Merge content across duplicate documents with different redactions.

    The merger analyzes redaction regions across duplicate documents and
    creates a merged version that maximizes content coverage.

    Attributes:
        duplicate_groups: List of duplicate groups to merge
        ocr_model: Ollama model to use for OCR (default: minicpm-v:8b)
    """

    def __init__(
        self,
        duplicate_groups: List[DuplicateGroup],
        ocr_model: str = "minicpm-v:8b",
    ):
        """Initialize the redaction merger.

        Args:
            duplicate_groups: List of detected duplicate groups
            ocr_model: Ollama model name for OCR processing
        """
        self.duplicate_groups = duplicate_groups
        self.ocr_model = ocr_model

    def merge_group(self, group: DuplicateGroup) -> MergedDocument:
        """Merge documents in a duplicate group.

        Strategy:
        1. Select document with fewest redactions as base
        2. Extract text from all documents
        3. Identify content in other docs that fills base redactions
        4. Create merged text with supplemental content annotated

        Args:
            group: DuplicateGroup to merge

        Returns:
            MergedDocument with combined content
        """
        logger.info(f"Merging group {group.group_id} with {len(group.documents)} documents")

        # Step 1: Select base document (fewest redactions)
        base_doc = self._select_base_document(group)
        base_path = Path(base_doc.file_path)
        logger.info(
            f"Using base document: {base_path.name} ({len(base_doc.redaction_regions)} redactions)"
        )

        # Step 2: Extract text from all documents
        doc_texts: Dict[str, str] = {}
        for doc in group.documents:
            doc_path = Path(doc.file_path)
            try:
                text, _ = extract_text_from_pdf(doc_path)
                doc_texts[doc.file_path] = text
                logger.debug(f"Extracted {len(text)} chars from {doc_path.name}")
            except Exception as e:
                logger.error(f"Failed to extract text from {doc_path.name}: {e}")
                doc_texts[doc.file_path] = ""

        # Step 3: Build merged text
        base_text = doc_texts[base_doc.file_path]
        fillins: List[RedactionFillin] = []
        supplemental_sections = []

        # Step 4: For each other document, find unique content
        for doc in group.documents:
            if doc.file_path == base_doc.file_path:
                continue

            other_text = doc_texts[doc.file_path]
            if not other_text:
                continue

            # Find regions that are redacted in base but not in other
            doc_fillins = self._find_fillins(base_doc, doc, other_text)
            fillins.extend(doc_fillins)

            if doc_fillins:
                supplemental_sections.append(
                    {
                        "source_file": Path(doc.file_path).name,
                        "fillin_count": len(doc_fillins),
                        "text": other_text,
                    }
                )

        # Step 5: Build final merged text
        merged_text = base_text

        if supplemental_sections:
            merged_text += "\n\n"
            merged_text += "=" * 80 + "\n"
            merged_text += "SUPPLEMENTAL CONTENT FROM OTHER COPIES\n"
            merged_text += "=" * 80 + "\n"
            merged_text += (
                "The following content from other copies fills redactions "
                "in the base document.\n\n"
            )

            for section in supplemental_sections:
                merged_text += (
                    f"\n--- From {section['source_file']} "
                    f"({section['fillin_count']} fillins) ---\n\n"
                )
                merged_text += section["text"]

        # Create metadata
        metadata = {
            "base_document": base_doc.file_path,
            "base_redaction_count": len(base_doc.redaction_regions),
            "supplemental_documents": [
                Path(doc.file_path).name
                for doc in group.documents
                if doc.file_path != base_doc.file_path
            ],
            "total_fillins": len(fillins),
            "merge_strategy": group.merge_strategy,
        }

        return MergedDocument(
            group_id=group.group_id,
            source_files=[d.file_path for d in group.documents],
            base_document=base_doc.file_path,
            merged_text=merged_text,
            fillins=fillins,
            metadata=metadata,
        )

    def _select_base_document(self, group: DuplicateGroup) -> DocumentFingerprint:
        """Select the best base document (fewest redactions).

        Args:
            group: DuplicateGroup to select from

        Returns:
            DocumentFingerprint with fewest redaction regions
        """
        # Sort by number of redaction regions (ascending)
        return min(group.documents, key=lambda d: len(d.redaction_regions))

    def _find_fillins(
        self,
        base_doc: DocumentFingerprint,
        other_doc: DocumentFingerprint,
        other_text: str,
    ) -> List[RedactionFillin]:
        """Identify content in other_doc that fills redactions in base_doc.

        For each redaction in base_doc, check if other_doc has a redaction
        in a similar location. If not, the content from other_doc can fill
        the gap.

        Args:
            base_doc: Document with redactions to fill
            other_doc: Document potentially containing missing content
            other_text: Extracted text from other_doc

        Returns:
            List of RedactionFillin objects
        """
        fillins = []

        for base_redaction in base_doc.redaction_regions:
            # Check if other doc has overlapping redaction
            has_overlap = any(
                self._regions_overlap(base_redaction.bbox, other.bbox)
                for other in other_doc.redaction_regions
                if other.page_num == base_redaction.page_num
            )

            if not has_overlap:
                # This region is redacted in base but not in other
                fillins.append(
                    RedactionFillin(
                        page_num=base_redaction.page_num,
                        base_bbox=base_redaction.bbox,
                        source_document=other_doc.file_path,
                        context=f"Content from non-redacted copy (page {base_redaction.page_num})",
                        confidence=0.85,
                    )
                )

        return fillins

    @staticmethod
    def _regions_overlap(
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        threshold: float = 0.3,
    ) -> bool:
        """Check if two bounding boxes overlap significantly.

        Args:
            bbox1: First bounding box (x0, y0, x1, y1)
            bbox2: Second bounding box (x0, y0, x1, y1)
            threshold: Overlap threshold (0-1)

        Returns:
            True if boxes overlap above threshold
        """
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        overlap_area = x_overlap * y_overlap

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        min_area = min(area1, area2)
        return overlap_area > min_area * threshold if min_area > 0 else False

    def merge_all(self) -> List[MergedDocument]:
        """Merge all duplicate groups.

        Returns:
            List of MergedDocument objects
        """
        results = []

        for group in self.duplicate_groups:
            try:
                merged = self.merge_group(group)
                results.append(merged)
                logger.info(f"Successfully merged group {group.group_id}")
            except Exception as e:
                logger.error(f"Failed to merge group {group.group_id}: {e}")

        logger.info(f"Merged {len(results)} of {len(self.duplicate_groups)} groups")

        return results

    def save_merged(self, merged_docs: List[MergedDocument], output_dir: Path) -> None:
        """Save merged documents to disk.

        Args:
            merged_docs: List of merged documents to save
            output_dir: Directory to save merged documents
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for merged in merged_docs:
            output_path = output_dir / f"{merged.group_id}_merged.json"

            with open(output_path, "w") as f:
                json.dump(merged.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved merged document to {output_path}")

        # Save summary
        summary_path = output_dir / "merge_summary.json"
        summary = {
            "total_groups": len(merged_docs),
            "total_source_documents": sum(len(m.source_files) for m in merged_docs),
            "total_fillins": sum(len(m.fillins) for m in merged_docs),
            "groups": [
                {
                    "group_id": m.group_id,
                    "source_count": len(m.source_files),
                    "fillin_count": len(m.fillins),
                    "base_document": Path(m.base_document).name,
                }
                for m in merged_docs
            ],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved merge summary to {summary_path}")


def merge_duplicate_group(
    group_file: Path,
    output_dir: Path,
    ocr_model: str = "minicpm-v:8b",
) -> Optional[MergedDocument]:
    """Convenience function to merge a single duplicate group.

    Args:
        group_file: Path to duplicate groups JSON file
        output_dir: Directory to save merged results
        ocr_model: Ollama model to use for OCR

    Returns:
        MergedDocument if successful, None otherwise
    """
    from epstein_graphrag.ocr.duplicate_detector import DuplicateDetector

    # Load duplicate groups
    detector = DuplicateDetector()
    detector.load_groups(group_file)

    if not detector.duplicate_groups:
        logger.warning(f"No duplicate groups found in {group_file}")
        return None

    # Merge first group
    merger = RedactionMerger(detector.duplicate_groups, ocr_model=ocr_model)
    merged_docs = merger.merge_all()

    # Save results
    merger.save_merged(merged_docs, output_dir)

    return merged_docs[0] if merged_docs else None
