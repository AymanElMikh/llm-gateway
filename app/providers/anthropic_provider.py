"""Anthropic provider implementation.

Uses httpx.AsyncClient to call the Anthropic Messages API directly.
The request format differs from OpenAI: maps GatewayRequest fields to
Anthropic's ``system`` / ``messages`` schema and normalises the SSE
chunk format back to plain text tokens.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.exceptions import ProviderUnavailableError, RateLimitError
from app.models.request import GatewayRequest

_BASE_URL = "https://api.anthropic.com/v1"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    """Calls the Anthropic API (https://api.anthropic.com/v1/messages)."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads ANTHROPIC_API_KEY)."""
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported Anthropic model IDs."""
        return ["claude-haiku-4-5", "claude-sonnet-4-6"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return the full completion text from the Anthropic API."""
        payload = self._build_payload(model_id, request, stream=False)
        response = await self._client.post("/messages", json=payload)
        self._raise_for_status(response)
        return response.json()["content"][0]["text"]

    async def stream(self, model_id: str, request: GatewayRequest) -> AsyncIterator[str]:
        """Yield text chunks from the Anthropic streaming API."""
        payload = self._build_payload(model_id, request, stream=True)
        async with self._client.stream("POST", "/messages", json=payload) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text

    async def health_check(self) -> bool:
        """Return True if the Anthropic API is reachable (uses a minimal request)."""
        try:
            # Anthropic has no free list endpoint; a 401 still proves reachability.
            response = await self._client.post(
                "/messages",
                json={
                    "model": "claude-haiku-4-5",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
            return response.status_code not in (500, 502, 503, 504)
        except httpx.RequestError:
            return False

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_payload(
        self, model_id: str, request: GatewayRequest, *, stream: bool
    ) -> dict:
        """Build the JSON body for the Anthropic messages endpoint.

        Anthropic keeps system messages separate from the conversation list.
        """
        system_content = next(
            (m["content"] for m in request.messages if m["role"] == "system"),
            None,
        )
        messages = [m for m in request.messages if m["role"] != "system"]

        payload: dict = {
            "model": model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }
        if system_content:
            payload["system"] = system_content
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP error codes to typed gateway exceptions."""
        if response.status_code == 429:
            raise RateLimitError(
                "Anthropic rate limit exceeded",
                provider=self.name,
                status_code=429,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Anthropic returned {response.status_code}",
                provider=self.name,
                status_code=response.status_code,
            )
        response.raise_for_status()
