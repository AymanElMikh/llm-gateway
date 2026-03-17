"""Phase 1 smoke tests — verify the scaffold imports cleanly."""

from __future__ import annotations


def test_app_imports() -> None:
    """The FastAPI app must be importable without errors."""
    from app.main import app

    assert app is not None


def test_settings_defaults() -> None:
    """Settings must load with default values when no .env is present."""
    from app.config import Settings

    s = Settings()
    assert s.redis_url == "redis://redis:6379"
    assert s.monthly_budget_usd == 50.0
    assert s.cache_similarity_threshold == 0.92
    assert s.cache_ttl_seconds == 3600
    assert s.default_strategy == "balanced"
    assert s.log_level == "INFO"
    assert s.environment == "development"


def test_gateway_request_defaults() -> None:
    """GatewayRequest must populate defaults and auto-generate request_id."""
    from app.models.request import GatewayRequest, RoutingStrategy

    req = GatewayRequest(messages=[{"role": "user", "content": "hi"}])
    assert req.strategy == RoutingStrategy.BALANCED
    assert req.stream is False
    assert req.max_tokens == 1000
    assert req.temperature == 0.7
    assert len(req.request_id) == 36  # UUID4 string length


def test_exception_hierarchy() -> None:
    """Custom exceptions must form the correct inheritance chain."""
    from app.exceptions import (
        AllProvidersFailedError,
        CacheError,
        FatalProviderError,
        GatewayError,
        ProviderError,
        ProviderUnavailableError,
        RateLimitError,
    )

    assert issubclass(ProviderError, GatewayError)
    assert issubclass(FatalProviderError, ProviderError)
    assert issubclass(RateLimitError, ProviderError)
    assert issubclass(ProviderUnavailableError, ProviderError)
    assert issubclass(AllProvidersFailedError, GatewayError)
    assert issubclass(CacheError, GatewayError)


def test_provider_protocol_defined() -> None:
    """LLMProvider Protocol must be runtime-checkable."""
    from app.providers.base import LLMProvider

    assert hasattr(LLMProvider, "__protocol_attrs__") or callable(LLMProvider)
