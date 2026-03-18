"""Ollama provider implementation.

Calls a local Ollama instance via its REST API. Falls back gracefully
when Ollama is not running — health_check returns False rather than raising.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.exceptions import ProviderUnavailableError, RateLimitError
from app.models.request import GatewayRequest


class OllamaProvider:
    """Calls a local Ollama instance (http://{OLLAMA_BASE_URL}/api/chat)."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads OLLAMA_BASE_URL)."""
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=60.0,  # local models can be slow to respond
        )

    @property
    def name(self) -> str:
        """Return provider name."""
        return "ollama"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported Ollama model IDs."""
        return ["mistral:7b"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return the full completion text from the local Ollama instance."""
        payload = self._build_payload(model_id, request, stream=False)
        response = await self._client.post("/api/chat", json=payload)
        self._raise_for_status(response)
        return response.json()["message"]["content"]

    async def stream(self, model_id: str, request: GatewayRequest) -> AsyncIterator[str]:
        """Yield text chunks from the Ollama streaming API."""
        payload = self._build_payload(model_id, request, stream=True)
        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            self._raise_for_status(response)
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break

    async def health_check(self) -> bool:
        """Return True if the local Ollama instance is reachable."""
        try:
            response = await self._client.get("/api/tags")
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
        """Build the JSON body for the Ollama chat endpoint."""
        return {
            "model": model_id,
            "messages": request.messages,
            "stream": stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP error codes to typed gateway exceptions."""
        if response.status_code == 429:
            raise RateLimitError(
                "Ollama rate limit exceeded",
                provider=self.name,
                status_code=429,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Ollama returned {response.status_code}",
                provider=self.name,
                status_code=response.status_code,
            )
        response.raise_for_status()
