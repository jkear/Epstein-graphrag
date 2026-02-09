"""Evidence chain report generator.

Generates structured reports showing chains of evidence
linking entities, with source citations.
"""

import logging

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


def generate_evidence_chain(query_result: dict, config: Config) -> str:
    """Generate a formatted evidence chain report.

    Args:
        query_result: Result from the query engine.
        config: Application configuration.

    Returns:
        Formatted report string.
    """
    raise NotImplementedError("Evidence chain reports not yet implemented — see Task 9")
