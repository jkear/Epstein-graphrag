"""Entity deduplication and alias resolution.

Resolves name variants to canonical forms before graph ingestion.
"J. Epstein", "Jeffrey Epstein", "Epstein, Jeffrey" all resolve
to the same canonical name.
"""

import json
import logging

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


class AliasResolver:
    """Resolves entity name variants to canonical names.

    Maintains an alias table that maps alternate names to
    canonical forms. The table is seeded with known aliases
    and grows during processing.
    """

    def __init__(self, config: Config):
        self.config = config
        self.alias_table: dict[str, str] = {}
        self._load_alias_table()

    def _load_alias_table(self) -> None:
        """Load the alias table from disk."""
        path = self.config.alias_table_path
        if path.exists():
            self.alias_table = json.loads(path.read_text())
            logger.info(f"Loaded {len(self.alias_table)} aliases")

    def save(self) -> None:
        """Persist the alias table to disk."""
        path = self.config.alias_table_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.alias_table, indent=2))
        logger.info(f"Saved {len(self.alias_table)} aliases")

    def resolve(self, name: str) -> str:
        """Resolve a name to its canonical form.

        Args:
            name: The name to resolve.

        Returns:
            The canonical name, or the input name if no alias exists.
        """
        normalized = name.strip()
        return self.alias_table.get(normalized, normalized)

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register a new alias mapping.

        Args:
            alias: The alternate name.
            canonical: The canonical name it maps to.
        """
        self.alias_table[alias.strip()] = canonical.strip()
