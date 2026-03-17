"""FastAPI dependency injection stubs.

Each function retrieves a shared resource from app.state, which is
populated during the lifespan startup in main.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from app.cache.semantic_cache import SemanticCache
    from app.core.dispatcher import Dispatcher
    from app.tracking.token_tracker import TokenTracker


def get_cache(request: Request) -> SemanticCache:
    """Return the SemanticCache instance from application state."""
    return request.app.state.cache


def get_dispatcher(request: Request) -> Dispatcher:
    """Return the Dispatcher instance from application state."""
    return request.app.state.dispatcher


def get_tracker(request: Request) -> TokenTracker:
    """Return the TokenTracker instance from application state."""
    return request.app.state.tracker
