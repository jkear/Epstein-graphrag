"""Central configuration for the Epstein GraphRAG system."""

from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    """All configuration for the Epstein GraphRAG system.

    Paths are relative to the project root unless absolute.
    Neo4j credentials should be overridden via environment or .env file.
    """

    # Paths
    data_root: Path = Path("/Users/jordankearfott/locDocuments/files")
    processed_dir: Path = Path("data/processed")
    extracted_dir: Path = Path("data/extracted")
    manifest_path: Path = Path("data/manifest.json")
    alias_table_path: Path = Path("data/alias_table.json")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Gemini (free tier)
    gemini_model: str = "gemini-2.0-flash"

    # Local LLM (Ollama)
    local_llm_model: str = "mlx-community/Mistral-7B-Instruct-v0.3"
    local_llm_host: str = "http://localhost:11434"

    # Embeddings
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = 768

    # Processing
    batch_size: int = 100
    ocr_confidence_threshold: float = 0.7

    # Query engine
    graph_expansion_hops: int = 3
    vector_top_k: int = 20
    min_corroboration_count: int = 2
