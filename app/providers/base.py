"""Provider Protocol definition.

All LLM provider classes must satisfy this structural interface.
We use Protocol (typing) rather than ABC so that providers are
validated via duck typing — no explicit inheritance required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.models.request import GatewayRequest


@runtime_checkable
class LLMProvider(Protocol):
    """Structural interface that every LLM provider must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable provider name (e.g. 'openai')."""
        ...

    @property
    def supported_models(self) -> list[str]:
        """List of model IDs supported by this provider."""
        ...

    async def complete(self, model_id: str, request: GatewayRequest) -> str:
        """Return the full completion text for a non-streaming request."""
        ...

    async def stream(
        self, model_id: str, request: GatewayRequest
    ) -> AsyncIterator[str]:
        """Yield text chunks for a streaming request."""
        ...

    async def health_check(self) -> bool:
        """Return True if the provider is reachable and operational."""
        ...
