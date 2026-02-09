"""Natural language to Cypher query translation.

Converts user questions into Cypher queries against the
Epstein evidence graph schema.
"""

import logging

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


def generate_cypher(question: str, config: Config) -> str:
    """Translate a natural language question to a Cypher query.

    Args:
        question: Natural language question.
        config: Application configuration.

    Returns:
        Cypher query string.
    """
    raise NotImplementedError("Cypher generation not yet implemented — see Task 8")
