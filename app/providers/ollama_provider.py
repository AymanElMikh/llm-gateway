"""Ollama provider implementation.

Calls a local Ollama instance via its REST API. Falls back gracefully
when Ollama is not running — health_check returns False rather than raising.

Implemented in Phase 2.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.config import Settings
from app.models.request import GatewayRequest


class OllamaProvider:
    """Calls a local Ollama instance (http://{OLLAMA_BASE_URL}/api/chat).

    Implemented in Phase 2.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads OLLAMA_BASE_URL)."""
        self.settings = settings

    @property
    def name(self) -> str:
        """Return provider name."""
        return "ollama"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported Ollama model IDs."""
        return ["mistral:7b"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return a full completion from the Ollama API."""
        raise NotImplementedError("Implemented in Phase 2")

    async def stream(
        self, model_id: str, request: GatewayRequest
    ) -> AsyncIterator[str]:
        """Yield text chunks from the Ollama streaming API."""
        raise NotImplementedError("Implemented in Phase 2")
        yield  # pragma: no cover

    async def health_check(self) -> bool:
        """Return True if the local Ollama instance is reachable."""
        raise NotImplementedError("Implemented in Phase 2")
