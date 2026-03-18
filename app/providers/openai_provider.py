"""OpenAI provider implementation.

Uses httpx.AsyncClient to call the OpenAI Chat Completions API directly,
without the openai SDK, for full control over request/response handling.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.exceptions import ProviderUnavailableError, RateLimitError
from app.models.request import GatewayRequest

_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider:
    """Calls the OpenAI API (https://api.openai.com/v1/chat/completions)."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads OPENAI_API_KEY)."""
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported OpenAI model IDs."""
        return ["gpt-4o-mini", "gpt-4o"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return the full completion text from the OpenAI API."""
        payload = self._build_payload(model_id, request, stream=False)
        response = await self._client.post("/chat/completions", json=payload)
        self._raise_for_status(response)
        return response.json()["choices"][0]["message"]["content"]

    async def stream(self, model_id: str, request: GatewayRequest) -> AsyncIterator[str]:
        """Yield text chunks from the OpenAI streaming API."""
        payload = self._build_payload(model_id, request, stream=True)
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                content = data["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content

    async def health_check(self) -> bool:
        """Return True if the OpenAI API is reachable."""
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_payload(
        self, model_id: str, request: GatewayRequest, *, stream: bool
    ) -> dict:
        """Build the JSON body for the OpenAI chat completions endpoint."""
        return {
            "model": model_id,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP error codes to typed gateway exceptions."""
        if response.status_code == 429:
            raise RateLimitError(
                "OpenAI rate limit exceeded",
                provider=self.name,
                status_code=429,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"OpenAI returned {response.status_code}",
                provider=self.name,
                status_code=response.status_code,
            )
        response.raise_for_status()
