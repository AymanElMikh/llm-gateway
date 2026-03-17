"""Response models for the LLM Gateway API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GatewayResponse(BaseModel):
    """Non-streaming response returned by the gateway chat endpoint."""

    request_id: str = Field(description="Echoed request identifier.")
    content: str = Field(description="Generated text content.")
    provider: str = Field(description="Name of the provider that served this request.")
    model: str = Field(description="Model ID that generated the response.")
    input_tokens: int = Field(description="Number of tokens in the prompt.")
    output_tokens: int = Field(description="Number of tokens in the completion.")
    cached: bool = Field(
        default=False,
        description="True if the response was served from the semantic cache.",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Estimated cost in USD for this request.",
    )


class StreamChunk(BaseModel):
    """Single chunk emitted in a Server-Sent Events streaming response."""

    content: str = Field(description="Token or partial text in this chunk.")
    id: str = Field(description="Request ID this chunk belongs to.")
    done: bool = Field(
        default=False,
        description="True on the final chunk; content will be empty.",
    )
