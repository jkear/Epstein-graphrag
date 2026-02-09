"""Tests for entity deduplication / alias resolution."""

from epstein_graphrag.config import Config
from epstein_graphrag.graph.dedup import AliasResolver


def test_alias_resolver_passthrough():
    """Unknown names pass through unchanged."""
    config = Config()
    resolver = AliasResolver(config)
    assert resolver.resolve("Unknown Person") == "Unknown Person"


def test_alias_resolver_add_and_resolve():
    """Added aliases resolve to canonical names."""
    config = Config()
    resolver = AliasResolver(config)
    resolver.add_alias("J. Epstein", "Jeffrey Epstein")
    resolver.add_alias("Epstein, Jeffrey", "Jeffrey Epstein")

    assert resolver.resolve("J. Epstein") == "Jeffrey Epstein"
    assert resolver.resolve("Epstein, Jeffrey") == "Jeffrey Epstein"
    assert resolver.resolve("Jeffrey Epstein") == "Jeffrey Epstein"


def test_alias_resolver_strips_whitespace():
    """Alias resolution strips leading/trailing whitespace."""
    config = Config()
    resolver = AliasResolver(config)
    resolver.add_alias("  J. Epstein  ", "Jeffrey Epstein")
    assert resolver.resolve("J. Epstein") == "Jeffrey Epstein"
