"""Application configuration via Pydantic Settings.

All values are read from environment variables (or .env file).
No secrets are ever hardcoded here.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Provider API Keys ─────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance (no trailing slash)",
    )

    # ── Infrastructure ────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://redis:6379",
        description="Redis connection URL",
    )

    # ── Budget & Routing ──────────────────────────────────────────
    monthly_budget_usd: float = Field(
        default=50.0,
        description="Monthly spend limit in USD; AUTO strategy switches to CHEAPEST at 80%",
    )
    default_strategy: str = Field(
        default="balanced",
        description="Default routing strategy: cheapest | fastest | quality | balanced | auto",
    )

    # ── Semantic Cache ────────────────────────────────────────────
    cache_similarity_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for cache hits (0.0–1.0)",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for cached responses in seconds",
    )

    # ── Application ───────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG | INFO | WARNING | ERROR",
    )
    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )


def get_settings() -> Settings:
    """Return a Settings instance (reads .env on first call)."""
    return Settings()
