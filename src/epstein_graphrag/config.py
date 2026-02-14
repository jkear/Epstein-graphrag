"""Central configuration for the Epstein GraphRAG system."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from project root
load_dotenv()


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
    neo4j_uri: str = Field(default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default=os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default=os.getenv("NEO4J_PASSWORD", "password"))

    # API Keys for multi-provider extraction
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    deepseek_api_key: str = Field(default=os.getenv("DEEPSEEK_API_KEY", ""))

    # Model configurations
    gemini_model: str = "gemini-2.5-flash"  # More stable than gemini-3-flash-preview

    # LM Studio (MLX backend on Apple Silicon)
    lmstudio_base_url: str = Field(
        default=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    )
    lmstudio_model: str = Field(
        default=os.getenv("LMSTUDIO_MODEL", "qwen/qwen3-vl-8b")
    )
    lmstudio_temperature: float = 0.3
    lmstudio_max_tokens: int = 2000

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
