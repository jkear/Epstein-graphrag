"""Proactive intelligence query engine.

Fires graph traversal AND vector similarity simultaneously,
merges results, expands context via graph hops, and synthesizes
a sourced response using a local LLM.
"""

import logging

from epstein_graphrag.config import Config
from epstein_graphrag.query.hybrid_search import HybridSearcher
from epstein_graphrag.query.pattern_detector import PatternDetector

logger = logging.getLogger(__name__)


class QueryEngine:
    """Proactive intelligence query engine.

    Every query triggers both graph traversal and vector search,
    then contextual expansion, pattern detection, and LLM synthesis.
    """

    def __init__(self, config: Config):
        self.config = config
        self.searcher = HybridSearcher(config)
        self.pattern_detector = PatternDetector(config)

    def query(self, question: str) -> dict:
        """Execute a proactive intelligence query.

        Args:
            question: Natural language question.

        Returns:
            Dictionary with 'answer', 'sources', 'patterns', and 'graph_context'.
        """
        raise NotImplementedError("Query engine not yet implemented — see Task 8")

    def close(self):
        """Release resources."""
        self.searcher.close()
