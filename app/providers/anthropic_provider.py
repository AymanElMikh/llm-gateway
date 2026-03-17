"""Anthropic provider implementation.

Uses httpx.AsyncClient to call the Anthropic Messages API directly.
The request format differs from OpenAI: maps GatewayRequest fields to
Anthropic's ``system`` / ``messages`` schema and normalises the SSE
chunk format back to plain text tokens.

Implemented in Phase 2.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.config import Settings
from app.models.request import GatewayRequest


class AnthropicProvider:
    """Calls the Anthropic API (https://api.anthropic.com/v1/messages).

    Implemented in Phase 2.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads ANTHROPIC_API_KEY)."""
        self.settings = settings

    @property
    def name(self) -> str:
        """Return provider name."""
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        """Return list of supported Anthropic model IDs."""
        return ["claude-haiku-4-5", "claude-sonnet-4-6"]

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return a full completion from the Anthropic API."""
        raise NotImplementedError("Implemented in Phase 2")

    async def stream(
        self, model_id: str, request: GatewayRequest
    ) -> AsyncIterator[str]:
        """Yield text chunks from the Anthropic streaming API."""
        raise NotImplementedError("Implemented in Phase 2")
        yield  # pragma: no cover

    async def health_check(self) -> bool:
        """Return True if the Anthropic API is reachable."""
        raise NotImplementedError("Implemented in Phase 2")
