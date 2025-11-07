"""
Configuration management for FastAPI application.
Uses Pydantic Settings for type-safe environment configuration.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Application
    app_name: str = "AI Job Agent"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Security
    secret_key: str = "change-me-in-production"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:70b"
    temperature: float = 0.1
    max_tokens: int = 2048
    context_window: int = 128000

    # API Keys
    adzuna_app_id: Optional[str] = None
    adzuna_app_key: Optional[str] = None
    scrapingdog_api_key: Optional[str] = None

    # RAG Settings
    chroma_db_path: str = "./data/chroma_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    templates_dir: Path = base_dir / "templates"
    static_dir: Path = base_dir / "static"
    data_dir: Path = base_dir / "data"

    @property
    def ollama_api_url(self) -> str:
        """Full Ollama API URL for generate endpoint."""
        return f"{self.ollama_base_url}/api/generate"

    @property
    def ollama_embeddings_url(self) -> str:
        """Full Ollama API URL for embeddings endpoint."""
        return f"{self.ollama_base_url}/api/embeddings"


# Global settings instance
settings = Settings()
