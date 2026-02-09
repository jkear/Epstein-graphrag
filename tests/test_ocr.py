"""Tests for Marker + Gemini OCR pipeline."""

from pathlib import Path

from epstein_graphrag.ocr.marker_pipeline import (
    OCRResult,
    ProcessingTrack,
)


def test_ocr_result_structure():
    """OCR results have required fields."""
    result = OCRResult(
        doc_id="EFTA00001234",
        track=ProcessingTrack.TEXT,
        text="sample text",
        confidence=0.95,
        metadata={"page_count": 1},
    )
    assert result.doc_id == "EFTA00001234"
    assert result.track == ProcessingTrack.TEXT
    assert result.confidence > 0
    assert result.vision_analysis is None


def test_photograph_result_structure():
    """Photograph analysis results have vision-specific fields."""
    result = OCRResult(
        doc_id="EFTA00005678",
        track=ProcessingTrack.PHOTOGRAPH,
        text="",
        confidence=0.0,
        vision_analysis={
            "scene_description": "Empty room with concrete walls",
            "objects_detected": ["bed frame", "camera mount"],
            "anomalies_noted": ["lock on exterior of door"],
            "faces_detected": [],
        },
        metadata={"page_count": 1},
    )
    assert result.vision_analysis is not None
    assert "scene_description" in result.vision_analysis
    assert "objects_detected" in result.vision_analysis
    assert "anomalies_noted" in result.vision_analysis
    assert "faces_detected" in result.vision_analysis
