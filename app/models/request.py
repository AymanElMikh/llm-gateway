"""Request models for the LLM Gateway API."""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class RoutingStrategy(str, Enum):
    """Strategy used by CostRouter to select the target model."""

    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    QUALITY = "quality"
    BALANCED = "balanced"
    AUTO = "auto"


class GatewayRequest(BaseModel):
    """Unified request shape accepted by the gateway chat endpoint."""

    messages: list[dict] = Field(
        description="Conversation history in OpenAI message format: [{role, content}, ...]"
    )
    model: str | None = Field(
        default=None,
        description="Target model ID. If None, CostRouter selects based on strategy.",
    )
    strategy: RoutingStrategy = Field(
        default=RoutingStrategy.BALANCED,
        description="Routing strategy used when model is not explicitly specified.",
    )
    stream: bool = Field(
        default=False,
        description="If True, the response is returned as a Server-Sent Events stream.",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = most random).",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request. Auto-generated if not provided.",
    )
