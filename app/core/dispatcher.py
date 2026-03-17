"""Dispatcher: orchestrates provider selection, retry logic, and fallback chains.

The Dispatcher receives a GatewayRequest, asks the CostRouter for a ranked
list of models, and attempts each model in order until one succeeds.
Retries with exponential back-off are applied per-model before moving on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.router import CostRouter
    from app.models.request import GatewayRequest
    from app.providers.base import LLMProvider


class Dispatcher:
    """Dispatches LLM requests with retry logic and provider fallback.

    Implemented in Phase 4.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        router: CostRouter,
    ) -> None:
        """Initialise with a provider registry and a cost router."""
        self.providers = providers
        self.router = router

    async def dispatch(self, request: GatewayRequest) -> str:
        """Dispatch a non-streaming request and return the generated text."""
        raise NotImplementedError("Implemented in Phase 4")

    async def dispatch_stream(
        self, request: GatewayRequest
    ) -> AsyncIterator[str]:
        """Dispatch a streaming request and yield text chunks."""
        raise NotImplementedError("Implemented in Phase 4")
        # Satisfy the async generator type
        yield  # pragma: no cover
