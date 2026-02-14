"""Duplicate PDF detection and redaction tracking for forensic documents.

This module provides functionality to:
1. Detect duplicate PDFs that are redacted in different locations
2. Track redaction locations across duplicates
3. Generate fingerprints for document similarity comparison

The detection uses multiple methods:
- File hash for exact duplicates
- Text hash for documents with similar content (different redactions)
- Perceptual hash for visual similarity
- Redaction detection via dark region analysis
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# Optional: PyPDF2 for text extraction
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader

        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False
        PdfReader = None  # type: ignore

# Optional: scipy for connected components in redaction detection
try:
    from scipy import ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RedactionRegion:
    """A redacted region in a PDF page.

    Attributes:
        page_num: Page number (1-indexed)
        bbox: Bounding box (x0, y0, x1, y1) in image coordinates
        confidence: Detection confidence score (0-1)
        surrounding_text: Text detected around the redaction
        area_pixels: Size of redaction in pixels
    """

    page_num: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    surrounding_text: str = ""
    area_pixels: int = 0

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "surrounding_text": self.surrounding_text,
            "area_pixels": self.area_pixels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedactionRegion":
        return cls(
            page_num=data["page_num"],
            bbox=tuple(data["bbox"]),
            confidence=data["confidence"],
            surrounding_text=data.get("surrounding_text", ""),
            area_pixels=data.get("area_pixels", 0),
        )


@dataclass
class DocumentFingerprint:
    """Fingerprint for duplicate detection.

    Attributes:
        file_path: Path to the PDF file
        file_hash: SHA256 hash of the entire file
        page_count: Number of pages in the document
        text_hash: MD5 hash of extracted text (excluding common patterns)
        perceptual_hash: Average hash of first page layout
        redaction_regions: List of detected redaction regions
        file_size: Size of file in bytes
    """

    file_path: str
    file_hash: str
    page_count: int
    text_hash: str
    perceptual_hash: str
    redaction_regions: List[RedactionRegion] = field(default_factory=list)
    file_size: int = 0

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "text_hash": self.text_hash,
            "perceptual_hash": self.perceptual_hash,
            "redaction_regions": [r.to_dict() for r in self.redaction_regions],
            "file_size": self.file_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentFingerprint":
        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            page_count=data["page_count"],
            text_hash=data["text_hash"],
            perceptual_hash=data["perceptual_hash"],
            redaction_regions=[
                RedactionRegion.from_dict(r) for r in data.get("redaction_regions", [])
            ],
            file_size=data.get("file_size", 0),
        )


@dataclass
class DuplicateGroup:
    """A group of duplicate documents.

    Documents in a group are believed to be the same source document
    with different redactions applied.

    Attributes:
        group_id: Unique identifier for this group
        documents: List of fingerprints in this group
        merge_strategy: Strategy for merging ('union', 'longest', 'first')
    """

    group_id: str
    documents: List[DocumentFingerprint]
    merge_strategy: str = "union"

    def get_best_source_for_page(self, page_num: int) -> Optional[DocumentFingerprint]:
        """Find the document with the fewest redactions on a given page.

        Args:
            page_num: Page number to check (1-indexed)

        Returns:
            DocumentFingerprint with fewest redactions on this page, or None
        """
        best_doc = None
        min_redactions = float("inf")

        for doc in self.documents:
            redaction_count = sum(1 for r in doc.redaction_regions if r.page_num == page_num)
            if redaction_count < min_redactions:
                min_redactions = redaction_count
                best_doc = doc

        return best_doc

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "merge_strategy": self.merge_strategy,
            "documents": [d.to_dict() for d in self.documents],
            "document_count": len(self.documents),
        }


class DuplicateDetector:
    """Detects duplicate PDFs and tracks redactions across copies.

    This detector uses multiple methods to identify duplicates:
    1. Exact file hash (for identical files)
    2. Text hash similarity (for files with different redactions)
    3. Perceptual hash similarity (for visual layout matching)
    4. Redaction region comparison (to identify different redaction patterns)

    Attributes:
        similarity_threshold: Threshold for considering documents similar (0-1)
        page_match_threshold: Threshold for perceptual hash matching (0-1)
        min_redaction_size: Minimum pixel area for redaction detection
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        page_match_threshold: float = 0.9,
        min_redaction_size: int = 100,
        detect_redactions: bool = True,
    ):
        """Initialize the duplicate detector.

        Args:
            similarity_threshold: Threshold for text/perceptual similarity (0-1)
            page_match_threshold: Threshold for page perceptual hash match (0-1)
            min_redaction_size: Minimum pixels for redaction region
            detect_redactions: Whether to detect redaction regions
        """
        self.similarity_threshold = similarity_threshold
        self.page_match_threshold = page_match_threshold
        self.min_redaction_size = min_redaction_size
        self.detect_redactions = detect_redactions
        self.duplicate_groups: List[DuplicateGroup] = []

    def compute_file_hash(self, pdf_path: Path) -> str:
        """Compute SHA256 hash of the entire PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            SHA256 hash as hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_text_hash(self, pdf_path: Path) -> str:
        """Compute hash of extracted text.

        Excludes common headers/footers that might differ between copies.
        Falls back to empty string if PyPDF2 is not available.

        Args:
            pdf_path: Path to PDF file

        Returns:
            MD5 hash of extracted text as hexadecimal string
        """
        if not PYPDF_AVAILABLE:
            logger.warning("PyPDF2 not available, skipping text hash computation")
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page in reader.pages:
                text = page.extract_text() or ""
                # Remove common headers/footers (short lines)
                lines = text.split("\n")
                filtered = [line for line in lines if len(line.strip()) > 10]
                text_parts.append(" ".join(filtered))

            combined = " ".join(text_parts)
            return hashlib.md5(combined.encode()).hexdigest()

        except Exception as e:
            logger.warning(f"Failed to compute text hash for {pdf_path.name}: {e}")
            return ""

    def compute_perceptual_hash(self, pdf_path: Path, dpi: int = 72) -> str:
        """Compute perceptual hash of first page layout.

        Uses average hash algorithm on a resized grayscale version.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion

        Returns:
            Binary string representation of perceptual hash
        """
        try:
            images = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)

            if not images:
                return ""

            # Resize to 32x32 and convert to grayscale
            img = images[0].resize((32, 32), Image.Resampling.LANCZOS).convert("L")

            # Compute average hash
            pixels = np.array(img).flatten()
            avg = pixels.mean()
            binary_hash = (pixels > avg).astype(int)

            return "".join(map(str, binary_hash))

        except Exception as e:
            logger.warning(f"Failed to compute perceptual hash for {pdf_path.name}: {e}")
            return ""

    def detect_redactions(self, pdf_path: Path, dpi: int = 150) -> List[RedactionRegion]:
        """Detect blacked-out/redacted regions in PDF.

        Uses dark pixel detection and connected component analysis.
        Falls back to simple thresholding if scipy is not available.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion

        Returns:
            List of detected RedactionRegion objects
        """
        if not self.detect_redactions:
            return []

        redactions = []

        try:
            images = convert_from_path(str(pdf_path), dpi=dpi)

            for page_num, img in enumerate(images, start=1):
                # Convert to numpy array
                pixels = np.array(img)

                # Detect dark regions (potential redactions)
                # Using threshold on darkness
                if len(pixels.shape) == 3:
                    grayscale = np.mean(pixels[:, :, :3], axis=2)
                else:
                    grayscale = pixels

                dark_mask = grayscale < 50  # Very dark pixels

                if not SCIPY_AVAILABLE:
                    # Fallback: simple contiguous region detection
                    redactions.extend(self._detect_redactions_simple(dark_mask, page_num))
                else:
                    redactions.extend(self._detect_redactions_scipy(dark_mask, page_num))

            logger.info(
                f"Detected {len(redactions)} potential redaction regions in {pdf_path.name}"
            )
            return redactions

        except Exception as e:
            logger.error(f"Failed to detect redactions in {pdf_path}: {e}")
            return []

    def _detect_redactions_simple(
        self, dark_mask: np.ndarray, page_num: int
    ) -> List[RedactionRegion]:
        """Simple redaction detection without scipy.

        Uses a sliding window approach to find dark regions.
        """
        redactions = []
        height, width = dark_mask.shape

        # Scan for dark horizontal stripes (common redaction pattern)
        for y in range(0, height - 10, 5):
            # Check if this row is mostly dark
            row_dark_ratio = np.sum(dark_mask[y : y + 10, :]) / (10 * width)

            if row_dark_ratio > 0.95:  # 95% dark in this band
                # Find the extent of the dark region
                y_start = y
                while y < height - 10 and np.sum(dark_mask[y : y + 10, :]) / (10 * width) > 0.9:
                    y += 5

                # Find horizontal extent
                x_start = 0
                x_end = width
                row_slice = dark_mask[y_start:y, :]
                col_darks = np.sum(row_slice, axis=0) > (y - y_start) * 0.9

                # Find contiguous dark columns
                dark_cols = np.where(col_darks)[0]
                if len(dark_cols) > 0:
                    x_start = int(dark_cols[0])
                    x_end = int(dark_cols[-1])

                area = (x_end - x_start) * (y - y_start)

                if area >= self.min_redaction_size:
                    redactions.append(
                        RedactionRegion(
                            page_num=page_num,
                            bbox=(x_start, y_start, x_end, y),
                            confidence=0.7,
                            area_pixels=area,
                        )
                    )

        return redactions

    def _detect_redactions_scipy(
        self, dark_mask: np.ndarray, page_num: int
    ) -> List[RedactionRegion]:
        """Redaction detection using scipy for connected components.

        More accurate detection of irregular redaction shapes.
        """
        redactions = []

        # Find connected components
        labeled, num_features = ndimage.label(dark_mask)

        for i in range(1, num_features + 1):
            component = labeled == i
            area = np.sum(component)

            # Filter by size
            if area < self.min_redaction_size:
                continue

            # Get bounding box
            rows, cols = np.where(component)
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            x0, x1 = int(cols.min()), int(cols.max()) + 1

            # Check aspect ratio (redactions are usually somewhat rectangular)
            width = x1 - x0
            height = y1 - y0
            aspect = width / height if height > 0 else 0

            if aspect < 0.1 or aspect > 10:  # Filter out extreme aspect ratios
                continue

            redactions.append(
                RedactionRegion(
                    page_num=page_num,
                    bbox=(x0, y0, x1, y1),
                    confidence=0.85,
                    area_pixels=int(area),
                )
            )

        return redactions

    def fingerprint_document(self, pdf_path: Path, dpi: int = 150) -> DocumentFingerprint:
        """Generate fingerprint for a PDF document.

        Computes all hashes and detects redactions for comprehensive
        document identification.

        Args:
            pdf_path: Path to PDF file
            dpi: DPI for redaction detection images

        Returns:
            DocumentFingerprint with all computed features
        """
        logger.info(f"Fingerprinting {pdf_path.name}...")

        file_size = pdf_path.stat().st_size
        file_hash = self.compute_file_hash(pdf_path)

        # Get page count
        if PYPDF_AVAILABLE:
            try:
                page_count = len(PdfReader(str(pdf_path)).pages)
            except Exception:
                page_count = 1
        else:
            page_count = 1

        text_hash = self.compute_text_hash(pdf_path)
        perceptual_hash = self.compute_perceptual_hash(pdf_path)

        redaction_regions = []
        if self.detect_redactions:
            redaction_regions = self.detect_redactions(pdf_path, dpi)

        return DocumentFingerprint(
            file_path=str(pdf_path),
            file_hash=file_hash,
            page_count=page_count,
            text_hash=text_hash,
            perceptual_hash=perceptual_hash,
            redaction_regions=redaction_regions,
            file_size=file_size,
        )

    def find_duplicates(self, fingerprints: List[DocumentFingerprint]) -> List[DuplicateGroup]:
        """Group documents by similarity (potential duplicates with different redactions).

        Args:
            fingerprints: List of document fingerprints to analyze

        Returns:
            List of DuplicateGroup objects, each containing similar documents
        """
        groups = []
        processed = set()

        for i, fp1 in enumerate(fingerprints):
            if i in processed:
                continue

            # Start a new group
            current_group = [fp1]
            processed.add(i)

            # Find similar documents
            for j, fp2 in enumerate(fingerprints):
                if j <= i or j in processed:
                    continue

                if self._are_similar(fp1, fp2):
                    current_group.append(fp2)
                    processed.add(j)

            # Only create groups with multiple documents
            if len(current_group) > 1:
                # Generate group ID from hashes
                hash_input = json.dumps(sorted([fp.file_hash[:16] for fp in current_group]))
                group_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]

                groups.append(
                    DuplicateGroup(
                        group_id=group_id,
                        documents=current_group,
                        merge_strategy="union",
                    )
                )

        self.duplicate_groups = groups
        logger.info(f"Found {len(groups)} duplicate groups from {len(fingerprints)} documents")

        return groups

    def _are_similar(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint) -> bool:
        """Check if two fingerprints represent similar documents.

        Args:
            fp1: First document fingerprint
            fp2: Second document fingerprint

        Returns:
            True if documents are likely duplicates with different redactions
        """
        # Must have same page count
        if fp1.page_count != fp2.page_count:
            return False

        # Exact file hash match (identical files)
        if fp1.file_hash == fp2.file_hash:
            return True

        # Text hash match (same content, potentially different formatting)
        if fp1.text_hash and fp2.text_hash and fp1.text_hash == fp2.text_hash:
            return True

        # Perceptual hash similarity (similar layout)
        if fp1.perceptual_hash and fp2.perceptual_hash:
            similarity = self._hash_similarity(fp1.perceptual_hash, fp2.perceptual_hash)
            if similarity >= self.page_match_threshold:
                return True

        # File size similarity (within 10%)
        if fp1.file_size and fp2.file_size:
            size_ratio = min(fp1.file_size, fp2.file_size) / max(fp1.file_size, fp2.file_size)
            if size_ratio > 0.9:
                # Close file size + same page count = likely duplicates
                return True

        return False

    @staticmethod
    def _hash_similarity(hash1: str, hash2: str) -> float:
        """Compute similarity between two binary strings (hamming similarity).

        Args:
            hash1: First binary string
            hash2: Second binary string

        Returns:
            Similarity ratio (0-1)
        """
        if len(hash1) != len(hash2):
            return 0.0

        matching = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matching / len(hash1)

    def save_groups(self, output_path: Path) -> None:
        """Save duplicate groups to JSON file.

        Args:
            output_path: Path to save the groups JSON
        """
        data = {
            "groups": [g.to_dict() for g in self.duplicate_groups],
            "total_groups": len(self.duplicate_groups),
            "total_documents": sum(len(g.documents) for g in self.duplicate_groups),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self.duplicate_groups)} duplicate groups to {output_path}")

    def load_groups(self, input_path: Path) -> None:
        """Load duplicate groups from JSON file.

        Args:
            input_path: Path to load groups from
        """
        with open(input_path) as f:
            data = json.load(f)

        self.duplicate_groups = []
        for group_data in data.get("groups", []):
            docs = [DocumentFingerprint.from_dict(d) for d in group_data["documents"]]
            self.duplicate_groups.append(
                DuplicateGroup(
                    group_id=group_data["group_id"],
                    documents=docs,
                    merge_strategy=group_data.get("merge_strategy", "union"),
                )
            )

        logger.info(f"Loaded {len(self.duplicate_groups)} duplicate groups from {input_path}")


def detect_duplicates_in_directory(
    pdf_dir: Path,
    output_path: Path | None = None,
    detect_redactions: bool = True,
) -> List[DuplicateGroup]:
    """Convenience function to detect duplicates in a directory.

    Args:
        pdf_dir: Directory containing PDF files
        output_path: Optional path to save duplicate groups JSON
        detect_redactions: Whether to detect redaction regions

    Returns:
        List of DuplicateGroup objects
    """
    detector = DuplicateDetector(detect_redactions=detect_redactions)

    # Find all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    pdf_files.extend(pdf_dir.glob("*.PDF"))

    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    # Generate fingerprints
    fingerprints = []
    for pdf_path in pdf_files:
        try:
            fp = detector.fingerprint_document(pdf_path)
            fingerprints.append(fp)
        except Exception as e:
            logger.error(f"Failed to fingerprint {pdf_path.name}: {e}")

    # Find duplicates
    groups = detector.find_duplicates(fingerprints)

    # Save if requested
    if output_path:
        detector.save_groups(output_path)

    return groups
