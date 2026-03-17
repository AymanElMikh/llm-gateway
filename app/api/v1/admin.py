"""Admin endpoints — health, usage, and Prometheus metrics.

GET /v1/health    — liveness + provider status
GET /v1/admin/usage — aggregated token and cost statistics
GET /v1/metrics   — Prometheus text-format scrape endpoint

Full implementation in Phase 6/7; this is the Phase 1 stub.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["admin"])


@router.get("/admin/usage")
async def usage() -> JSONResponse:
    """Return aggregated usage statistics (stub — implemented in Phase 6)."""
    return JSONResponse(
        status_code=501,
        content={"error": "Not implemented yet — Phase 6"},
    )


@router.get("/metrics")
async def metrics() -> JSONResponse:
    """Return Prometheus metrics in text format (stub — implemented in Phase 7)."""
    return JSONResponse(
        status_code=501,
        content={"error": "Not implemented yet — Phase 7"},
    )
