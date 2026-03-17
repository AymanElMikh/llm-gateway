"""Semantic cache using sentence-transformer embeddings stored in Redis.

Cache keys are derived from a stable MD5 hash of the prompt text.
Embeddings are stored as raw bytes and similarity is computed via cosine
distance at query time (no Redis-Stack / RediSearch required).

Implemented in Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from app.config import Settings


class SemanticCache:
    """Cache LLM responses keyed by prompt embedding similarity.

    Implemented in Phase 5.
    """

    def __init__(
        self,
        redis_client: "aioredis.Redis",
        settings: "Settings",
    ) -> None:
        """Initialise with a Redis client and application settings."""
        self.redis = redis_client
        self.settings = settings

    async def get(self, prompt: str) -> str | None:
        """Return a cached response if a similar prompt exists, else None."""
        raise NotImplementedError("Implemented in Phase 5")

    async def set(self, prompt: str, response: str) -> None:
        """Store a prompt-response pair in the cache."""
        raise NotImplementedError("Implemented in Phase 5")
