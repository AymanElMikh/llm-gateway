"""OpenAI provider implementation.

Uses httpx.AsyncClient to call the OpenAI Chat Completions API directly,
without the openai SDK, for full control over request/response handling.

Implemented in Phase 2.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.config import Settings
from app.models.request import GatewayRequest


class OpenAIProvider:
    """Calls the OpenAI API (https://api.openai.com/v1/chat/completions).

    Implemented in Phase 2.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads OPENAI_API_KEY)."""
        self.settings = settings

    @property
    def name(self) -> str:
        """Return provider name."""
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported OpenAI model IDs."""
        return ["gpt-4o-mini", "gpt-4o"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return a full completion from the OpenAI API."""
        raise NotImplementedError("Implemented in Phase 2")

    async def stream(
        self, model_id: str, request: GatewayRequest
    ) -> AsyncIterator[str]:
        """Yield text chunks from the OpenAI streaming API."""
        raise NotImplementedError("Implemented in Phase 2")
        yield  # pragma: no cover

    async def health_check(self) -> bool:
        """Return True if the OpenAI API is reachable."""
        raise NotImplementedError("Implemented in Phase 2")
