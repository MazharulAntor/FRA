"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from .env (see .env.example)."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["anthropic", "openai", "openrouter", "lmstudio", "ollama"] = (
        "anthropic"
    )
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    lmstudio_base_url: str = "http://127.0.0.1:1234/v1"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "claude-sonnet-4-5"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "./.chroma"
    upload_dir: str = "./uploads"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 8
    max_upload_mb: int = 50
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings."""
    return Settings()
