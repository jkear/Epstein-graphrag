"""OCR quality validation.

Detects low-quality OCR outputs:
- Repeated phrases/words (hallucination)
- Very short text (OCR failure)
- Low entropy/repetitive content
- Gibberish or non-sensical output
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """OCR quality assessment."""

    doc_id: str
    is_valid: bool
    confidence_score: float
    issues: list[str]
    text_length: int
    unique_word_ratio: float
    repetition_ratio: float


def clean_repetition_loops(text: str, max_repeats: int = 3) -> tuple[str, int]:
    """Remove repeated lines and sentences from LLM output.

    Vision models sometimes get stuck in a generation loop, repeating the
    same content dozens of times.  Two patterns are detected:

    1. **Line-level**: Same line repeated on consecutive lines.
    2. **Sentence-level**: Same sentence repeated within a single long line
       (e.g. a 16K-char line that is two sentences repeated 100 times).

    Both are collapsed to at most ``max_repeats`` occurrences.

    Args:
        text: Raw OCR/LLM output text.
        max_repeats: Keep at most this many copies of any repeated unit.

    Returns:
        Tuple of (cleaned_text, number_of_repetitions_removed).
    """
    removed = 0

    # --- Pass 1: Global line-level deduplication ---
    # Catches both consecutive ("A A A A") and alternating ("A B A B A B") patterns.
    # Any line appearing > max_repeats times total is capped.
    lines = text.split("\n")
    line_counts: Counter[str] = Counter()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip very short lines from duplicate detection (bullets, blank lines)
        if len(stripped) < 20:
            cleaned_lines.append(line)
            continue

        line_counts[stripped] += 1
        if line_counts[stripped] <= max_repeats:
            cleaned_lines.append(line)
        else:
            removed += 1

    text = "\n".join(cleaned_lines)

    # --- Pass 2: Sentence-level deduplication within long lines ---
    # Catches: "Sentence A. Sentence B. Sentence A. Sentence B. ..." on one line
    import re

    final_lines: list[str] = []
    for line in text.split("\n"):
        if len(line) < 500:
            final_lines.append(line)
            continue

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", line)
        if len(sentences) < 6:
            final_lines.append(line)
            continue

        # Count sentence occurrences
        counts = Counter(s.strip() for s in sentences if len(s.strip()) > 20)
        if not counts:
            final_lines.append(line)
            continue

        most_common_count = counts.most_common(1)[0][1]
        if most_common_count <= max_repeats:
            final_lines.append(line)
            continue

        # Deduplicate: keep max_repeats of each sentence
        seen: Counter[str] = Counter()
        kept: list[str] = []
        for s in sentences:
            key = s.strip()
            if len(key) <= 20:
                kept.append(s)
                continue
            seen[key] += 1
            if seen[key] <= max_repeats:
                kept.append(s)
            else:
                removed += 1

        final_lines.append(" ".join(kept))

    return "\n".join(final_lines), removed


def detect_repetition(text: str, window_size: int = 10) -> float:
    """Detect repeating patterns in text.

    Returns ratio of repeated sequences to total sequences (0.0 - 1.0).
    High values (>0.5) indicate repetitive/hallucinated text.
    """
    if len(text) < window_size * 2:
        return 0.0

    # Sample sequences throughout text
    sequences = []
    step = max(1, len(text) // 100)  # Sample 100 positions
    for i in range(0, len(text) - window_size, step):
        sequences.append(text[i : i + window_size])

    if not sequences:
        return 0.0

    # Count duplicates
    counter = Counter(sequences)
    duplicates = sum(count - 1 for count in counter.values() if count > 1)
    total = len(sequences)

    return duplicates / total if total > 0 else 0.0


def calculate_unique_word_ratio(text: str) -> float:
    """Calculate ratio of unique words to total words.

    Low values (<0.3) indicate repetitive text.
    """
    words = text.lower().split()
    if len(words) < 10:
        return 0.0

    unique = len(set(words))
    total = len(words)

    return unique / total if total > 0 else 0.0


def detect_number_sequence(text: str) -> bool:
    """Detect if text is just counting numbers (e.g., '1. 2. 3. 4...')."""
    # Check if text is mostly digits and punctuation
    cleaned = text.replace(" ", "").replace(".", "").replace("\n", "")
    if not cleaned:
        return False

    digit_ratio = sum(c.isdigit() for c in cleaned) / len(cleaned)
    return digit_ratio > 0.8


def validate_ocr_quality(
    doc_id: str,
    text: str,
    metadata: dict | None = None,
    min_text_length: int = 20,
    max_repetition_ratio: float = 0.5,
    min_unique_word_ratio: float = 0.3,
) -> QualityReport:
    """Validate OCR output quality.

    Args:
        doc_id: Document identifier
        text: OCR extracted text
        metadata: Optional metadata dict
        min_text_length: Minimum acceptable text length
        max_repetition_ratio: Maximum acceptable repetition ratio
        min_unique_word_ratio: Minimum unique word ratio

    Returns:
        QualityReport with validation results
    """
    issues = []

    # Check 1: Text length
    text_length = len(text.strip())
    if text_length < min_text_length:
        issues.append(f"Text too short ({text_length} chars)")

    # Check 2: Repetition detection
    repetition_ratio = detect_repetition(text)
    if repetition_ratio > max_repetition_ratio:
        issues.append(f"High repetition detected ({repetition_ratio:.1%} repeated sequences)")

    # Check 3: Unique word ratio
    unique_word_ratio = calculate_unique_word_ratio(text)
    if unique_word_ratio > 0 and unique_word_ratio < min_unique_word_ratio:
        issues.append(f"Low vocabulary diversity ({unique_word_ratio:.1%} unique words)")

    # Check 4: Number sequence hallucination
    if detect_number_sequence(text):
        issues.append("Detected number sequence hallucination (1. 2. 3...)")

    # Check 5: Common hallucination patterns
    hallucination_phrases = [
        "object\nobject\nobject",
        "Do not change the text",
        "a blue bottle, a white bottle",
        "\\n\\n\\n\\n\\n",
    ]
    for phrase in hallucination_phrases:
        if phrase in text and text.count(phrase) > 3:
            issues.append(f"Detected hallucination pattern: '{phrase[:30]}...'")

    # Calculate confidence score (0.0 - 1.0)
    confidence_score = 1.0
    if issues:
        # Reduce confidence based on severity
        confidence_score -= len(issues) * 0.2
        confidence_score = max(0.0, confidence_score)

    is_valid = len(issues) == 0

    return QualityReport(
        doc_id=doc_id,
        is_valid=is_valid,
        confidence_score=confidence_score,
        issues=issues,
        text_length=text_length,
        unique_word_ratio=unique_word_ratio,
        repetition_ratio=repetition_ratio,
    )


def validate_ocr_file(ocr_file: Path, **kwargs) -> QualityReport:
    """Validate OCR quality from JSON file.

    Args:
        ocr_file: Path to OCR result JSON
        **kwargs: Additional arguments passed to validate_ocr_quality

    Returns:
        QualityReport
    """
    with open(ocr_file) as f:
        data = json.load(f)

    doc_id = data.get("doc_id", ocr_file.stem)
    text = data.get("text", "")
    metadata = data.get("metadata", {})

    return validate_ocr_quality(doc_id, text, metadata, **kwargs)


def batch_validate(
    ocr_dir: Path,
    output_report: Path | None = None,
) -> dict:
    """Validate all OCR files in a directory.

    Args:
        ocr_dir: Directory containing OCR JSON files
        output_report: Optional path to save validation report

    Returns:
        Dict with statistics and list of failed documents
    """
    ocr_files = sorted(ocr_dir.glob("*.json"))
    ocr_files = [f for f in ocr_files if not f.name.endswith(".error.json")]

    results = []
    valid_count = 0
    invalid_count = 0

    logger.info(f"Validating {len(ocr_files)} OCR files...")

    for ocr_file in ocr_files:
        try:
            report = validate_ocr_file(ocr_file)
            results.append(report)

            if report.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                logger.warning(f"{report.doc_id}: INVALID - {', '.join(report.issues)}")
        except Exception as e:
            logger.error(f"Failed to validate {ocr_file.name}: {e}")
            invalid_count += 1

    # Generate statistics
    stats = {
        "total": len(ocr_files),
        "valid": valid_count,
        "invalid": invalid_count,
        "success_rate": valid_count / len(ocr_files) if ocr_files else 0.0,
        "failed_docs": [r.doc_id for r in results if not r.is_valid],
        "reports": [
            {
                "doc_id": r.doc_id,
                "is_valid": r.is_valid,
                "confidence_score": r.confidence_score,
                "issues": r.issues,
                "text_length": r.text_length,
                "unique_word_ratio": r.unique_word_ratio,
                "repetition_ratio": r.repetition_ratio,
            }
            for r in results
        ],
    }

    # Save report
    if output_report:
        with open(output_report, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Validation report saved to {output_report}")

    return stats
