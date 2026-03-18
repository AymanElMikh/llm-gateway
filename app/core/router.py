"""Cost-aware model router.

CostRouter selects the optimal LLM model based on the requested routing
strategy (cheapest / fastest / quality / balanced / auto) and the current
monthly spend. It also builds ordered fallback chains and estimates costs.
Pure logic — no I/O, no async, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.models.request import RoutingStrategy

# ── Model registry ────────────────────────────────────────────────────────────

_BUDGET_THRESHOLD = 0.80  # switch to CHEAPEST when spend exceeds this fraction


@dataclass(frozen=True)
class ModelConfig:
    """Immutable descriptor for a single LLM model."""

    provider: str
    model_id: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: int
    max_context_tokens: int
    quality_score: float  # 0.0–1.0, higher is better


MODELS: list[ModelConfig] = [
    ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.00060,
        avg_latency_ms=800,
        max_context_tokens=128_000,
        quality_score=0.75,
    ),
    ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_1k_input=0.00250,
        cost_per_1k_output=0.01000,
        avg_latency_ms=1200,
        max_context_tokens=128_000,
        quality_score=0.95,
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        avg_latency_ms=700,
        max_context_tokens=200_000,
        quality_score=0.78,
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        cost_per_1k_input=0.00300,
        cost_per_1k_output=0.01500,
        avg_latency_ms=1100,
        max_context_tokens=200_000,
        quality_score=0.93,
    ),
    ModelConfig(
        provider="ollama",
        model_id="mistral:7b",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=2000,
        max_context_tokens=32_000,
        quality_score=0.60,
    ),
]


# ── Router ────────────────────────────────────────────────────────────────────


class CostRouter:
    """Selects LLM models based on cost, latency, quality, and budget."""

    def __init__(self, settings: Settings) -> None:
        """Initialise with application settings (reads monthly_budget_usd)."""
        self._monthly_budget_usd = settings.monthly_budget_usd

    def select(
        self,
        strategy: RoutingStrategy,
        prompt_tokens: int,
        current_spend: float,
        require_context: int = 0,
    ) -> ModelConfig:
        """Return the best ModelConfig for the given strategy and constraints.

        Args:
            strategy: Routing strategy to apply.
            prompt_tokens: Number of tokens in the prompt (unused by current
                strategies but available for future cost-pre-estimation).
            current_spend: Total USD spent this month so far.
            require_context: Minimum context window required in tokens.
                Models with max_context_tokens < require_context are excluded.

        Raises:
            ValueError: If no eligible model satisfies the context requirement.
        """
        eligible = [m for m in MODELS if m.max_context_tokens >= require_context]
        if not eligible:
            raise ValueError(
                f"No model supports a context window of {require_context} tokens."
            )

        if strategy == RoutingStrategy.CHEAPEST:
            return self._select_cheapest(eligible)
        if strategy == RoutingStrategy.FASTEST:
            return self._select_fastest(eligible)
        if strategy == RoutingStrategy.QUALITY:
            return self._select_quality(eligible)
        if strategy == RoutingStrategy.BALANCED:
            return self._select_balanced(eligible)
        if strategy == RoutingStrategy.AUTO:
            return self._select_auto(eligible, current_spend)

        raise ValueError(f"Unknown routing strategy: {strategy!r}")

    def build_fallback_chain(self, primary: ModelConfig) -> list[ModelConfig]:
        """Return models to try if primary fails, sorted by cost ascending.

        The primary model is excluded from the chain.
        """
        return sorted(
            [m for m in MODELS if m is not primary],
            key=lambda m: m.cost_per_1k_input,
        )

    def estimate_cost(
        self,
        model: ModelConfig,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return the estimated USD cost for the given token counts."""
        return (input_tokens / 1000) * model.cost_per_1k_input + (
            output_tokens / 1000
        ) * model.cost_per_1k_output

    # ── Private strategy implementations ─────────────────────────────────────

    def _select_cheapest(self, eligible: list[ModelConfig]) -> ModelConfig:
        """Return the model with the lowest input cost."""
        return min(eligible, key=lambda m: m.cost_per_1k_input)

    def _select_fastest(self, eligible: list[ModelConfig]) -> ModelConfig:
        """Return the model with the lowest average latency."""
        return min(eligible, key=lambda m: m.avg_latency_ms)

    def _select_quality(self, eligible: list[ModelConfig]) -> ModelConfig:
        """Return the model with the highest quality score."""
        return max(eligible, key=lambda m: m.quality_score)

    def _select_balanced(self, eligible: list[ModelConfig]) -> ModelConfig:
        """Return the model with the highest weighted score.

        Scoring: 40% quality + 30% cost (inverted) + 30% latency (inverted).
        All dimensions are normalised to [0, 1] within the eligible pool.
        """
        max_cost = max(m.cost_per_1k_input for m in eligible)
        max_latency = max(m.avg_latency_ms for m in eligible)

        def score(m: ModelConfig) -> float:
            quality_norm = m.quality_score
            cost_norm = 1.0 - (m.cost_per_1k_input / max_cost) if max_cost > 0 else 1.0
            latency_norm = (
                1.0 - (m.avg_latency_ms / max_latency) if max_latency > 0 else 1.0
            )
            return 0.4 * quality_norm + 0.3 * cost_norm + 0.3 * latency_norm

        return max(eligible, key=score)

    def _select_auto(
        self, eligible: list[ModelConfig], current_spend: float
    ) -> ModelConfig:
        """Use BALANCED normally; switch to CHEAPEST when budget is 80% consumed."""
        if current_spend > _BUDGET_THRESHOLD * self._monthly_budget_usd:
            return self._select_cheapest(eligible)
        return self._select_balanced(eligible)
