"""Dispatcher: orchestrates provider selection, retry logic, and fallback chains.

The Dispatcher receives a GatewayRequest, asks the CostRouter for a ranked
list of models, and attempts each model in order until one succeeds.
Retries with exponential back-off are applied per-model before moving on.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import structlog

from app.exceptions import (
    AllProvidersFailedError,
    FatalProviderError,
    ProviderUnavailableError,
    RateLimitError,
)

if TYPE_CHECKING:
    from app.core.router import CostRouter, ModelConfig
    from app.models.request import GatewayRequest
    from app.providers.base import LLMProvider

log = structlog.get_logger(__name__)

_CALL_TIMEOUT_SECONDS: float = 30.0
_MAX_RETRIES: int = 3


class Dispatcher:
    """Dispatches LLM requests with retry logic and provider fallback.

    Selects a primary model via CostRouter, then works through the fallback
    chain until a provider responds successfully or all have been exhausted.
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
        """Dispatch a non-streaming request and return the generated text.

        Selects the primary model via the router, then works through the
        fallback chain.  Each model is tried with exponential-backoff retries
        before the next one is attempted.

        Args:
            request: The gateway request to fulfil.

        Returns:
            The full completion text from the first successful provider.

        Raises:
            AllProvidersFailedError: If every model in the chain fails.
        """
        primary = self.router.select(
            strategy=request.strategy,
            prompt_tokens=_count_prompt_tokens(request),
            current_spend=0.0,  # live spend injected from TokenTracker in Phase 7
        )
        chain = [primary, *self.router.build_fallback_chain(primary)]

        last_error: Exception | None = None
        for model_config in chain:
            provider = self.providers.get(model_config.provider)
            if provider is None:
                log.warning(
                    "provider_not_registered",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    request_id=request.request_id,
                )
                continue
            try:
                return await self._call_with_retry(provider, model_config, request)
            except FatalProviderError as exc:
                log.warning(
                    "provider_failed_trying_next",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    error=str(exc),
                    request_id=request.request_id,
                )
                last_error = exc

        raise AllProvidersFailedError(
            f"All providers failed for request {request.request_id}"
        ) from last_error

    async def dispatch_stream(
        self, request: GatewayRequest
    ) -> AsyncGenerator[str, None]:
        """Dispatch a streaming request and yield text chunks.

        Same fallback and retry semantics as dispatch(), but delegates to
        provider.stream() and yields each chunk as it arrives.

        Args:
            request: The gateway request to fulfil.

        Yields:
            Text chunks from the first successful provider stream.

        Raises:
            AllProvidersFailedError: If every model in the chain fails.
        """
        primary = self.router.select(
            strategy=request.strategy,
            prompt_tokens=_count_prompt_tokens(request),
            current_spend=0.0,
        )
        chain = [primary, *self.router.build_fallback_chain(primary)]

        last_error: Exception | None = None
        for model_config in chain:
            provider = self.providers.get(model_config.provider)
            if provider is None:
                continue
            try:
                async for chunk in self._stream_with_retry(
                    provider, model_config, request
                ):
                    yield chunk
                return
            except FatalProviderError as exc:
                log.warning(
                    "stream_provider_failed_trying_next",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    error=str(exc),
                    request_id=request.request_id,
                )
                last_error = exc

        raise AllProvidersFailedError(
            f"All providers failed for streaming request {request.request_id}"
        ) from last_error

    async def _call_with_retry(
        self,
        provider: LLMProvider,
        model_config: ModelConfig,
        request: GatewayRequest,
        max_retries: int = _MAX_RETRIES,
    ) -> str:
        """Call provider.complete() with exponential-backoff retries.

        Args:
            provider: The LLM provider to call.
            model_config: Target model configuration.
            request: The gateway request.
            max_retries: Maximum number of attempts.

        Returns:
            The completion text on success.

        Raises:
            FatalProviderError: After all retries are exhausted or on timeout.
        """
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    provider.complete(model_config.model_id, request),
                    timeout=_CALL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "provider_timeout",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    attempt=attempt,
                    request_id=request.request_id,
                )
                raise FatalProviderError(
                    f"Request to {model_config.provider}/{model_config.model_id} timed out",
                    provider=model_config.provider,
                )
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise FatalProviderError(
                        f"Rate limit on {model_config.provider}/{model_config.model_id} "
                        f"after {max_retries} attempts",
                        provider=model_config.provider,
                    )
                backoff = 2**attempt + random.uniform(0.0, 0.5)  # noqa: S311
                log.info(
                    "rate_limit_backoff",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    attempt=attempt,
                    backoff_seconds=round(backoff, 3),
                    request_id=request.request_id,
                )
                await asyncio.sleep(backoff)
            except ProviderUnavailableError:
                if attempt == max_retries - 1:
                    raise FatalProviderError(
                        f"Provider {model_config.provider}/{model_config.model_id} "
                        f"unavailable after {max_retries} attempts",
                        provider=model_config.provider,
                    )
                backoff = float(2**attempt)
                log.info(
                    "provider_unavailable_backoff",
                    provider=model_config.provider,
                    model=model_config.model_id,
                    attempt=attempt,
                    backoff_seconds=backoff,
                    request_id=request.request_id,
                )
                await asyncio.sleep(backoff)

        # Unreachable — all paths above either return or raise — but keeps
        # the type-checker satisfied.
        raise FatalProviderError(  # pragma: no cover
            f"Exhausted retries for {model_config.provider}/{model_config.model_id}",
            provider=model_config.provider,
        )

    async def _stream_with_retry(
        self,
        provider: LLMProvider,
        model_config: ModelConfig,
        request: GatewayRequest,
        max_retries: int = _MAX_RETRIES,
    ) -> AsyncGenerator[str, None]:
        """Stream from provider.stream() with retry logic.

        Retries the entire stream on transient errors.  Note: if chunks have
        already been yielded before an error occurs, the caller may see partial
        duplicate output on retry — this is an acceptable trade-off for the
        current implementation.

        Args:
            provider: The LLM provider to stream from.
            model_config: Target model configuration.
            request: The gateway request.
            max_retries: Maximum number of attempts.

        Yields:
            Text chunks from the provider stream.

        Raises:
            FatalProviderError: After all retries are exhausted.
        """
        for attempt in range(max_retries):
            try:
                async for chunk in provider.stream(model_config.model_id, request):
                    yield chunk
                return  # stream completed successfully
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise FatalProviderError(
                        f"Rate limit on {model_config.provider}/{model_config.model_id} "
                        f"after {max_retries} attempts",
                        provider=model_config.provider,
                    )
                backoff = 2**attempt + random.uniform(0.0, 0.5)  # noqa: S311
                await asyncio.sleep(backoff)
            except ProviderUnavailableError:
                if attempt == max_retries - 1:
                    raise FatalProviderError(
                        f"Provider {model_config.provider}/{model_config.model_id} "
                        f"unavailable after {max_retries} attempts",
                        provider=model_config.provider,
                    )
                await asyncio.sleep(float(2**attempt))

        raise FatalProviderError(  # pragma: no cover
            f"Exhausted stream retries for {model_config.provider}/{model_config.model_id}",
            provider=model_config.provider,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _count_prompt_tokens(request: GatewayRequest) -> int:
    """Rough token estimate: total characters in all message contents divided by 4."""
    return sum(len(str(m.get("content", ""))) for m in request.messages) // 4
