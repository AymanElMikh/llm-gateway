"""Tests for app/core/dispatcher.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.dispatcher import Dispatcher
from app.core.router import CostRouter, ModelConfig
from app.exceptions import (
    AllProvidersFailedError,
    FatalProviderError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.models.request import GatewayRequest, RoutingStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_request(**kwargs) -> GatewayRequest:
    """Return a minimal GatewayRequest, overriding defaults with kwargs."""
    defaults: dict = dict(
        messages=[{"role": "user", "content": "hello"}],
        strategy=RoutingStrategy.CHEAPEST,
    )
    return GatewayRequest(**(defaults | kwargs))


def _make_model(provider: str = "openai", model_id: str = "gpt-4o-mini") -> ModelConfig:
    """Return a ModelConfig suitable for unit testing."""
    return ModelConfig(
        provider=provider,
        model_id=model_id,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        avg_latency_ms=500,
        max_context_tokens=8_000,
        quality_score=0.8,
    )


async def _async_chunks(*items: str):
    """Async generator that yields the given items."""
    for item in items:
        yield item


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def primary_model() -> ModelConfig:
    """Primary model returned by the mock router."""
    return _make_model("openai", "gpt-4o-mini")


@pytest.fixture
def fallback_model() -> ModelConfig:
    """Fallback model in the chain returned by the mock router."""
    return _make_model("anthropic", "claude-haiku-4-5")


@pytest.fixture
def mock_openai_provider() -> MagicMock:
    """Mock OpenAI provider with a successful complete() and stream()."""
    provider = MagicMock()
    provider.name = "openai"
    provider.supported_models = ["gpt-4o-mini"]
    provider.complete = AsyncMock(return_value="Hello from OpenAI!")
    provider.stream = MagicMock(side_effect=lambda *_: _async_chunks("Hello", " world"))
    return provider


@pytest.fixture
def mock_anthropic_provider() -> MagicMock:
    """Mock Anthropic provider with a successful complete() and stream()."""
    provider = MagicMock()
    provider.name = "anthropic"
    provider.supported_models = ["claude-haiku-4-5"]
    provider.complete = AsyncMock(return_value="Hello from Anthropic!")
    provider.stream = MagicMock(side_effect=lambda *_: _async_chunks("Hi", " there"))
    return provider


@pytest.fixture
def mock_router(primary_model: ModelConfig, fallback_model: ModelConfig) -> MagicMock:
    """Mock CostRouter that always returns primary_model and [fallback_model]."""
    router = MagicMock(spec=CostRouter)
    router.select.return_value = primary_model
    router.build_fallback_chain.return_value = [fallback_model]
    return router


@pytest.fixture
def dispatcher(
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
    mock_router: MagicMock,
) -> Dispatcher:
    """Dispatcher wired with mock providers and router."""
    return Dispatcher(
        providers={"openai": mock_openai_provider, "anthropic": mock_anthropic_provider},
        router=mock_router,
    )


# ── dispatch() — happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_success_returns_text(
    dispatcher: Dispatcher, mock_openai_provider: MagicMock
) -> None:
    """Successful dispatch returns the provider's completion text."""
    result = await dispatcher.dispatch(_make_request())

    assert result == "Hello from OpenAI!"
    mock_openai_provider.complete.assert_called_once()


# ── dispatch() — retry on RateLimitError ─────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_retries_on_rate_limit_then_succeeds(
    dispatcher: Dispatcher, mock_openai_provider: MagicMock
) -> None:
    """RateLimitError triggers retries; success on the third attempt."""
    mock_openai_provider.complete.side_effect = [
        RateLimitError("rate limited", provider="openai"),
        RateLimitError("rate limited", provider="openai"),
        "Success after retries!",
    ]

    with patch("app.core.dispatcher.asyncio.sleep") as mock_sleep:
        result = await dispatcher.dispatch(_make_request())

    assert result == "Success after retries!"
    assert mock_openai_provider.complete.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_dispatch_rate_limit_exhausted_falls_back_to_next_provider(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """Exhausting rate-limit retries on the primary falls back to the next provider."""
    mock_openai_provider.complete.side_effect = RateLimitError(
        "always rate limited", provider="openai"
    )

    with patch("app.core.dispatcher.asyncio.sleep"):
        result = await dispatcher.dispatch(_make_request())

    assert result == "Hello from Anthropic!"
    mock_anthropic_provider.complete.assert_called_once()


# ── dispatch() — fallback on FatalProviderError ───────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_fallback_after_fatal_error(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """FatalProviderError on primary causes fallback to the next provider."""
    mock_openai_provider.complete.side_effect = FatalProviderError(
        "fatal", provider="openai"
    )

    result = await dispatcher.dispatch(_make_request())

    assert result == "Hello from Anthropic!"
    mock_anthropic_provider.complete.assert_called_once()


# ── dispatch() — AllProvidersFailedError ─────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_all_providers_failed_raises(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """AllProvidersFailedError raised when every provider in the chain fails."""
    mock_openai_provider.complete.side_effect = FatalProviderError(
        "fatal", provider="openai"
    )
    mock_anthropic_provider.complete.side_effect = FatalProviderError(
        "fatal", provider="anthropic"
    )

    with pytest.raises(AllProvidersFailedError):
        await dispatcher.dispatch(_make_request())


# ── dispatch() — timeout ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_timeout_raises_fatal_then_falls_back(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """asyncio.TimeoutError from wait_for raises FatalProviderError, triggering fallback."""
    with patch(
        "app.core.dispatcher.asyncio.wait_for",
        side_effect=[asyncio.TimeoutError, "Hello from Anthropic!"],
    ):
        result = await dispatcher.dispatch(_make_request())

    assert result == "Hello from Anthropic!"


@pytest.mark.asyncio
async def test_dispatch_timeout_all_fail_raises(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """AllProvidersFailedError when every provider times out."""
    with patch(
        "app.core.dispatcher.asyncio.wait_for",
        side_effect=asyncio.TimeoutError,
    ):
        with pytest.raises(AllProvidersFailedError):
            await dispatcher.dispatch(_make_request())


# ── dispatch() — ProviderUnavailableError retries ────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_retries_on_provider_unavailable(
    dispatcher: Dispatcher, mock_openai_provider: MagicMock
) -> None:
    """ProviderUnavailableError triggers retries with backoff."""
    mock_openai_provider.complete.side_effect = [
        ProviderUnavailableError("unavailable", provider="openai"),
        ProviderUnavailableError("unavailable", provider="openai"),
        "Recovered!",
    ]

    with patch("app.core.dispatcher.asyncio.sleep") as mock_sleep:
        result = await dispatcher.dispatch(_make_request())

    assert result == "Recovered!"
    assert mock_openai_provider.complete.call_count == 3
    assert mock_sleep.call_count == 2


# ── dispatch() — unregistered provider skipped ────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_skips_unregistered_provider(
    mock_router: MagicMock, mock_anthropic_provider: MagicMock, primary_model: ModelConfig
) -> None:
    """A model whose provider is not in the registry is silently skipped."""
    # Only anthropic is registered; primary model is openai (not registered)
    dispatcher = Dispatcher(
        providers={"anthropic": mock_anthropic_provider},
        router=mock_router,
    )

    result = await dispatcher.dispatch(_make_request())

    assert result == "Hello from Anthropic!"


# ── dispatch_stream() — happy path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_stream_yields_all_chunks(
    dispatcher: Dispatcher, mock_openai_provider: MagicMock
) -> None:
    """dispatch_stream yields every chunk from the provider stream."""
    mock_openai_provider.stream = MagicMock(
        side_effect=lambda *_: _async_chunks("Hello", " world", "!")
    )

    chunks: list[str] = []
    async for chunk in dispatcher.dispatch_stream(_make_request(stream=True)):
        chunks.append(chunk)

    assert chunks == ["Hello", " world", "!"]


# ── dispatch_stream() — fallback ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_stream_fallback_on_fatal_error(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """dispatch_stream falls back to the next provider when the primary raises."""

    async def _failing_stream(*_):
        raise FatalProviderError("fatal stream", provider="openai")
        yield  # make this an async generator  # noqa: unreachable

    mock_openai_provider.stream = MagicMock(side_effect=lambda *_: _failing_stream())
    mock_anthropic_provider.stream = MagicMock(
        side_effect=lambda *_: _async_chunks("Hi", " there")
    )

    chunks: list[str] = []
    with patch("app.core.dispatcher.asyncio.sleep"):
        async for chunk in dispatcher.dispatch_stream(_make_request(stream=True)):
            chunks.append(chunk)

    assert chunks == ["Hi", " there"]


@pytest.mark.asyncio
async def test_dispatch_stream_all_fail_raises(
    dispatcher: Dispatcher,
    mock_openai_provider: MagicMock,
    mock_anthropic_provider: MagicMock,
) -> None:
    """AllProvidersFailedError raised when every streaming provider fails."""

    async def _failing_stream(*_):
        raise FatalProviderError("fatal", provider="x")
        yield  # noqa: unreachable

    mock_openai_provider.stream = MagicMock(side_effect=lambda *_: _failing_stream())
    mock_anthropic_provider.stream = MagicMock(side_effect=lambda *_: _failing_stream())

    with pytest.raises(AllProvidersFailedError):
        async for _ in dispatcher.dispatch_stream(_make_request(stream=True)):
            pass
