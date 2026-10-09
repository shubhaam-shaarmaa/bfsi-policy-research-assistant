"""Application Configuration using Pydantic Settings.

Supports environment variables loaded from .env, with defaults suitable
for local development and easy override in containerized / production deployments.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core App
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_NAME: str = "BFSI Policy Research Assistant"
    LOG_LEVEL: str = "INFO"

    # Paths (relative to PROJECT_ROOT by default)
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    DATA_SYNTHETIC_DIR: Path = PROJECT_ROOT / "data" / "synthetic"
    DATA_VECTORSTORE_DIR: Path = PROJECT_ROOT / "data" / "vectorstore"

    # LLM Settings (Stage 5)
    ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER: str = "anthropic"
    LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    LLM_TEMPERATURE: float = 0.0

    # Embedding Settings (Stage 3)
    EMBEDDING_PROVIDER: Literal["local", "voyage", "openai"] = "local"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector DB Settings (Stage 3)
    VECTOR_DB_PROVIDER: Literal["qdrant", "chromadb"] = "qdrant"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: Path = PROJECT_ROOT / "data" / "vectorstore" / "qdrant_embedded"
    COLLECTION_NAME: str = "bfsi_policy_circulars"

    # Retrieval Settings (Stage 4)
    RETRIEVAL_HYBRID_ENABLED: bool = True
    RERANKER_ENABLED: bool = False
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    def ensure_directories(self) -> None:
        """Create necessary project data directories if they don't exist."""
        for path in [
            self.DATA_RAW_DIR,
            self.DATA_PROCESSED_DIR,
            self.DATA_SYNTHETIC_DIR,
            self.DATA_VECTORSTORE_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
