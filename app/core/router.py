"""Cost-aware model router.

CostRouter selects the optimal LLM model based on the requested routing
strategy (cheapest / fastest / quality / balanced / auto) and the current
monthly spend. It also builds ordered fallback chains and estimates costs.

Implemented in Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.request import RoutingStrategy


class CostRouter:
    """Selects LLM models based on cost, latency, quality, and budget.

    Implemented in Phase 3.
    """

    def select(
        self,
        strategy: RoutingStrategy,
        prompt_tokens: int,
        current_spend: float,
        require_context: int = 0,
    ) -> object:
        """Select the best ModelConfig for the given strategy and constraints."""
        raise NotImplementedError("Implemented in Phase 3")

    def build_fallback_chain(self, primary: object) -> list[object]:
        """Return an ordered list of ModelConfigs to try if the primary fails."""
        raise NotImplementedError("Implemented in Phase 3")

    def estimate_cost(
        self,
        model: object,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate the USD cost for the given token counts."""
        raise NotImplementedError("Implemented in Phase 3")
