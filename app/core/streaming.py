"""SSE streaming utilities.

Wraps a provider's raw async token generator with the gateway's
Server-Sent Events envelope format and integrates token tracking.

Implemented in Phase 6.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tracking.token_tracker import TokenTracker


async def stream_response(
    provider_stream: AsyncIterator[str],
    tracker: TokenTracker,
    request_id: str,
) -> AsyncIterator[str]:
    """Wrap a provider stream in the SSE envelope format.

    Yields ``data: <json>\\n\\n`` lines followed by ``data: [DONE]\\n\\n``.
    Implemented in Phase 6.
    """
    raise NotImplementedError("Implemented in Phase 6")
    yield  # pragma: no cover
