"""Tests for the three LLM provider implementations.

All external HTTP calls are intercepted with respx so no real network
traffic is made during the test suite.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.exceptions import ProviderUnavailableError, RateLimitError
from app.models.request import GatewayRequest
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def gateway_request() -> GatewayRequest:
    """Minimal valid GatewayRequest for provider tests."""
    return GatewayRequest(messages=[{"role": "user", "content": "Hello"}])


@pytest.fixture
def openai_provider(settings) -> OpenAIProvider:
    """OpenAIProvider wired with test settings."""
    return OpenAIProvider(settings)


@pytest.fixture
def anthropic_provider(settings) -> AnthropicProvider:
    """AnthropicProvider wired with test settings."""
    return AnthropicProvider(settings)


@pytest.fixture
def ollama_provider(settings) -> OllamaProvider:
    """OllamaProvider wired with test settings."""
    return OllamaProvider(settings)


# ── OpenAI ────────────────────────────────────────────────────────────────────


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    @respx.mock
    async def test_complete_returns_content(self, openai_provider, gateway_request):
        """Happy path: complete() returns the assistant message content."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello back!"}}],
                },
            )
        )
        result = await openai_provider.complete("gpt-4o-mini", gateway_request)
        assert result == "Hello back!"

    @respx.mock
    async def test_complete_sends_correct_payload(self, openai_provider, gateway_request):
        """complete() sends model, messages, max_tokens, temperature, stream=False."""
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )
        )
        await openai_provider.complete("gpt-4o-mini", gateway_request)
        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "gpt-4o-mini"
        assert sent["stream"] is False
        assert sent["messages"] == gateway_request.messages

    @respx.mock
    async def test_complete_raises_rate_limit_on_429(self, openai_provider, gateway_request):
        """complete() raises RateLimitError on HTTP 429."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(RateLimitError) as exc_info:
            await openai_provider.complete("gpt-4o-mini", gateway_request)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 429

    @respx.mock
    async def test_complete_raises_unavailable_on_503(self, openai_provider, gateway_request):
        """complete() raises ProviderUnavailableError on HTTP 5xx."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(503)
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await openai_provider.complete("gpt-4o-mini", gateway_request)
        assert exc_info.value.provider == "openai"
        assert exc_info.value.status_code == 503

    @respx.mock
    async def test_stream_yields_chunks(self, openai_provider, gateway_request):
        """stream() yields individual content strings from SSE lines."""
        sse_body = (
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, content=sse_body.encode())
        )
        chunks = [c async for c in openai_provider.stream("gpt-4o-mini", gateway_request)]
        assert chunks == ["Hello", " world"]

    @respx.mock
    async def test_stream_raises_rate_limit_on_429(self, openai_provider, gateway_request):
        """stream() raises RateLimitError on HTTP 429."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(RateLimitError):
            async for _ in openai_provider.stream("gpt-4o-mini", gateway_request):
                pass

    @respx.mock
    async def test_health_check_returns_true_on_200(self, openai_provider):
        """health_check() returns True when /v1/models responds 200."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200)
        )
        assert await openai_provider.health_check() is True

    @respx.mock
    async def test_health_check_returns_false_on_request_error(self, openai_provider):
        """health_check() returns False on network failure."""
        respx.get("https://api.openai.com/v1/models").mock(
            side_effect=httpx.RequestError("connection refused")
        )
        assert await openai_provider.health_check() is False


# ── Anthropic ─────────────────────────────────────────────────────────────────


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    @respx.mock
    async def test_complete_returns_content(self, anthropic_provider, gateway_request):
        """Happy path: complete() returns the first content block text."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"content": [{"text": "Hello back!"}]},
            )
        )
        result = await anthropic_provider.complete("claude-haiku-4-5", gateway_request)
        assert result == "Hello back!"

    @respx.mock
    async def test_complete_extracts_system_message(self, anthropic_provider):
        """System messages are lifted out of messages[] into a top-level system key."""
        request = GatewayRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={"content": [{"text": "ok"}]},
            )
        )
        await anthropic_provider.complete("claude-haiku-4-5", request)
        sent = json.loads(route.calls[0].request.content)
        assert sent["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in sent["messages"])

    @respx.mock
    async def test_complete_raises_rate_limit_on_429(self, anthropic_provider, gateway_request):
        """complete() raises RateLimitError on HTTP 429."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(RateLimitError) as exc_info:
            await anthropic_provider.complete("claude-haiku-4-5", gateway_request)
        assert exc_info.value.provider == "anthropic"

    @respx.mock
    async def test_complete_raises_unavailable_on_500(self, anthropic_provider, gateway_request):
        """complete() raises ProviderUnavailableError on HTTP 500."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await anthropic_provider.complete("claude-haiku-4-5", gateway_request)
        assert exc_info.value.provider == "anthropic"

    @respx.mock
    async def test_stream_yields_chunks(self, anthropic_provider, gateway_request):
        """stream() yields text from content_block_delta SSE events."""
        sse_body = (
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}\n\n'
            'data: {"type": "message_stop"}\n\n'
        )
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, content=sse_body.encode())
        )
        chunks = [c async for c in anthropic_provider.stream("claude-haiku-4-5", gateway_request)]
        assert chunks == ["Hello", " world"]

    @respx.mock
    async def test_stream_raises_rate_limit_on_429(self, anthropic_provider, gateway_request):
        """stream() raises RateLimitError on HTTP 429."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(RateLimitError):
            async for _ in anthropic_provider.stream("claude-haiku-4-5", gateway_request):
                pass


# ── Ollama ────────────────────────────────────────────────────────────────────


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    @respx.mock
    async def test_complete_returns_content(self, ollama_provider, gateway_request):
        """Happy path: complete() returns message.content from Ollama response."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "Bonjour!"}},
            )
        )
        result = await ollama_provider.complete("mistral:7b", gateway_request)
        assert result == "Bonjour!"

    @respx.mock
    async def test_complete_sends_correct_payload(self, ollama_provider, gateway_request):
        """complete() sends model, messages, stream=False, and options."""
        route = respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200,
                json={"message": {"content": "ok"}},
            )
        )
        await ollama_provider.complete("mistral:7b", gateway_request)
        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "mistral:7b"
        assert sent["stream"] is False
        assert "options" in sent

    @respx.mock
    async def test_complete_raises_unavailable_on_503(self, ollama_provider, gateway_request):
        """complete() raises ProviderUnavailableError on HTTP 503."""
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(503)
        )
        with pytest.raises(ProviderUnavailableError) as exc_info:
            await ollama_provider.complete("mistral:7b", gateway_request)
        assert exc_info.value.provider == "ollama"

    @respx.mock
    async def test_stream_yields_chunks(self, ollama_provider, gateway_request):
        """stream() yields content from newline-delimited JSON stream."""
        ndjson_body = (
            '{"message": {"content": "Hello"}, "done": false}\n'
            '{"message": {"content": " world"}, "done": true}\n'
        )
        respx.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, content=ndjson_body.encode())
        )
        chunks = [c async for c in ollama_provider.stream("mistral:7b", gateway_request)]
        assert chunks == ["Hello", " world"]

    @respx.mock
    async def test_health_check_returns_true_on_200(self, ollama_provider):
        """health_check() returns True when /api/tags responds 200."""
        respx.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200)
        )
        assert await ollama_provider.health_check() is True

    @respx.mock
    async def test_health_check_returns_false_when_not_running(self, ollama_provider):
        """health_check() returns False when Ollama is not running."""
        respx.get("http://localhost:11434/api/tags").mock(
            side_effect=httpx.RequestError("connection refused")
        )
        assert await ollama_provider.health_check() is False
