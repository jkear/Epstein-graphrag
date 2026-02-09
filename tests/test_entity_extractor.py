"""Tests for entity extractor."""

import pytest


def test_entity_extraction_not_implemented():
    """Entity extraction raises NotImplementedError until Task 5."""
    from epstein_graphrag.extract.entity_extractor import extract_entities

    with pytest.raises(NotImplementedError):
        extract_entities("some text", "DOC001")
