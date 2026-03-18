"""FastAPI application entry point.

Defines the app instance, lifespan handler, and mounts all API routers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse

from app.api.v1 import admin, chat
from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI, settings : Settings = Depends(get_settings())) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup initialises all shared resources and stores them on app.state.
    Shutdown releases connections cleanly.
    """

    # Configure structured logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            __import__("logging").getLevelName(settings.log_level)
        ),
    )

    logger.info(
        "llm_gateway_starting",
        environment=settings.environment,
        redis_url=settings.redis_url,
        default_strategy=settings.default_strategy,
        monthly_budget_usd=settings.monthly_budget_usd,
    )

    # Phase 1 stub — real initialisation added in Phase 6
    app.state.settings = settings

    yield

    logger.info("llm_gateway_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title="LLM Gateway",
        description="Async LLM Gateway with Semantic Cache & Cost Routing",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )

    # Mount API routers
    application.include_router(chat.router, prefix="/v1")
    application.include_router(admin.router, prefix="/v1")

    @application.get("/v1/health", tags=["health"])
    async def health() -> JSONResponse:
        """Return a simple liveness check."""
        return JSONResponse({"status": "ok"})

    return application


app = create_app()
