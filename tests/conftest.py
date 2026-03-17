"""Shared pytest fixtures for the LLM Gateway test suite.

Fixtures are added here as each phase is implemented.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_messages() -> list[dict]:
    """Return a minimal valid messages list for testing."""
    return [{"role": "user", "content": "Hello, world!"}]


@pytest.fixture
def settings():
    """Return a Settings instance with safe test defaults."""
    from app.config import Settings

    return Settings(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        ollama_base_url="http://localhost:11434",
        redis_url="redis://localhost:6379",
        monthly_budget_usd=50.0,
        default_strategy="balanced",
        cache_similarity_threshold=0.92,
        cache_ttl_seconds=3600,
        log_level="DEBUG",
        environment="test",
    )
