"""Tests for pattern detector."""

import pytest

from epstein_graphrag.config import Config
from epstein_graphrag.query.pattern_detector import PatternDetector


def test_corroboration_not_implemented():
    """Corroboration detection raises NotImplementedError until Task 8."""
    config = Config()
    detector = PatternDetector(config)
    with pytest.raises(NotImplementedError):
        detector.detect_corroboration([])


def test_temporal_clusters_not_implemented():
    """Temporal clustering raises NotImplementedError until Task 8."""
    config = Config()
    detector = PatternDetector(config)
    with pytest.raises(NotImplementedError):
        detector.detect_temporal_clusters([])


def test_evidence_gaps_not_implemented():
    """Evidence gap detection raises NotImplementedError until Task 8."""
    config = Config()
    detector = PatternDetector(config)
    with pytest.raises(NotImplementedError):
        detector.detect_evidence_gaps({})
