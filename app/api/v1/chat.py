"""Chat completions endpoint — POST /v1/chat/completions.

Wires the semantic cache, dispatcher, and token tracker together.
Full implementation in Phase 6; this is the Phase 1 stub.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_cache, get_dispatcher, get_tracker
from app.models.request import GatewayRequest

router = APIRouter(tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(
    request: GatewayRequest,
    cache: object = Depends(get_cache),
    dispatcher: object = Depends(get_dispatcher),
    tracker: object = Depends(get_tracker),
) -> JSONResponse:
    """Handle a chat completion request (stub — implemented in Phase 6)."""
    return JSONResponse(
        status_code=501,
        content={"error": "Not implemented yet — Phase 6"},
    )
