"""Token usage tracker backed by Redis.

Records per-request token counts and costs in both Redis (for the
/usage admin endpoint) and Prometheus metrics (for scraping).

Implemented in Phase 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis


class TokenTracker:
    """Records token usage and cost per request in Redis and Prometheus.

    Implemented in Phase 7.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        """Initialise with a Redis client."""
        self.redis = redis_client

    async def record(
        self,
        request_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Persist token counts and increment Prometheus cost counters."""
        raise NotImplementedError("Implemented in Phase 7")
