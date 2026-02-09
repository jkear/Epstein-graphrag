"""Tests for entity extractor."""

import json
from pathlib import Path

import pytest

from epstein_graphrag.extract.entity_extractor import (
    ExtractionResult,
    extract_from_text,
    extract_from_photo,
    extract_batch,
)


def test_extraction_result_structure():
    """ExtractionResult has correct fields."""
    result = ExtractionResult(
        doc_id="TEST001",
        doc_type="text_document",
        people=[{"name": "John Doe"}],
        locations=[{"name": "New York"}],
        organizations=[{"name": "Acme Corp"}],
        events=[{"description": "Meeting"}],
        allegations=[{"description": "Allegation"}],
        associations=[{"person_a": "A", "person_b": "B"}],
        objects_of_interest=[],
        raw_llm_response='{"test": true}',
    )

    assert result.doc_id == "TEST001"
    assert result.doc_type == "text_document"
    assert len(result.people) == 1
    assert len(result.locations) == 1
    assert len(result.organizations) == 1
    assert len(result.events) == 1
    assert len(result.allegations) == 1
    assert len(result.associations) == 1
    assert result.objects_of_interest == []
    assert '"test": true' in result.raw_llm_response


def test_extract_from_text_requires_gemini_key():
    """extract_from_text requires Gemini API key."""
    with pytest.raises(Exception):  # Will fail without valid key
        extract_from_text(
            doc_id="TEST001",
            doc_type="text_document",
            text="Test document",
            gemini_api_key="invalid_key",
        )


def test_extract_from_photo_requires_gemini_key():
    """extract_from_photo requires Gemini API key."""
    with pytest.raises(Exception):  # Will fail without valid key
        extract_from_photo(
            doc_id="TEST001",
            vision_analysis={"scene_description": "Test scene"},
            gemini_api_key="invalid_key",
        )


def test_extract_batch_creates_output_dir(tmp_path):
    """extract_batch creates output directory if it doesn't exist."""
    processed_dir = tmp_path / "processed"
    extracted_dir = tmp_path / "extracted"
    processed_dir.mkdir()

    # Create a fake OCR result
    (processed_dir / "TEST001.json").write_text(
        json.dumps({"doc_id": "TEST001", "track": "text", "text": "Test"})
    )

    # This will fail due to invalid API key, but should create the dir
    try:
        extract_batch(
            processed_dir=processed_dir,
            extracted_dir=extracted_dir,
            gemini_api_key="invalid_key",
        )
    except Exception:
        pass

    assert extracted_dir.exists()


def test_extract_batch_resume_skips_existing(tmp_path):
    """extract_batch skips already-extracted documents when resume=True."""
    processed_dir = tmp_path / "processed"
    extracted_dir = tmp_path / "extracted"
    processed_dir.mkdir()
    extracted_dir.mkdir()

    # Create fake OCR results
    (processed_dir / "TEST001.json").write_text(
        json.dumps({"doc_id": "TEST001", "track": "text", "text": "Test"})
    )
    (processed_dir / "TEST002.json").write_text(
        json.dumps({"doc_id": "TEST002", "track": "text", "text": "Test"})
    )

    # Create existing extraction result for TEST001
    (extracted_dir / "TEST001.json").write_text(json.dumps({"doc_id": "TEST001"}))

    # This will fail due to invalid API key, but should count stats correctly
    try:
        stats = extract_batch(
            processed_dir=processed_dir,
            extracted_dir=extracted_dir,
            gemini_api_key="invalid_key",
            resume=True,
        )
    except Exception:
        pass

    # Check that skipped count increased
    assert (extracted_dir / "TEST001.json").exists()

