"""Tests for CostRouter — pure logic, no mocking needed."""

from __future__ import annotations

import pytest

from app.core.router import MODELS, CostRouter, ModelConfig
from app.models.request import RoutingStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def router(settings) -> CostRouter:
    """CostRouter wired with test settings (monthly_budget_usd=50.0)."""
    return CostRouter(settings)


# ── Strategy: CHEAPEST ────────────────────────────────────────────────────────


def test_cheapest_returns_ollama(router):
    """CHEAPEST selects mistral:7b — the only zero-cost model."""
    result = router.select(RoutingStrategy.CHEAPEST, prompt_tokens=100, current_spend=0)
    assert result.model_id == "mistral:7b"
    assert result.provider == "ollama"


def test_cheapest_has_lowest_cost(router):
    """CHEAPEST result has the minimum cost_per_1k_input in the registry."""
    result = router.select(RoutingStrategy.CHEAPEST, prompt_tokens=100, current_spend=0)
    assert result.cost_per_1k_input == min(m.cost_per_1k_input for m in MODELS)


# ── Strategy: FASTEST ─────────────────────────────────────────────────────────


def test_fastest_returns_haiku(router):
    """FASTEST selects claude-haiku-4-5 — 700ms, lowest latency in registry."""
    result = router.select(RoutingStrategy.FASTEST, prompt_tokens=100, current_spend=0)
    assert result.model_id == "claude-haiku-4-5"
    assert result.provider == "anthropic"


def test_fastest_has_lowest_latency(router):
    """FASTEST result has the minimum avg_latency_ms in the registry."""
    result = router.select(RoutingStrategy.FASTEST, prompt_tokens=100, current_spend=0)
    assert result.avg_latency_ms == min(m.avg_latency_ms for m in MODELS)


# ── Strategy: QUALITY ─────────────────────────────────────────────────────────


def test_quality_returns_gpt4o(router):
    """QUALITY selects gpt-4o — quality_score 0.95, highest in registry."""
    result = router.select(RoutingStrategy.QUALITY, prompt_tokens=100, current_spend=0)
    assert result.model_id == "gpt-4o"
    assert result.provider == "openai"


def test_quality_has_highest_score(router):
    """QUALITY result has the maximum quality_score in the registry."""
    result = router.select(RoutingStrategy.QUALITY, prompt_tokens=100, current_spend=0)
    assert result.quality_score == max(m.quality_score for m in MODELS)


# ── Strategy: BALANCED ────────────────────────────────────────────────────────


def test_balanced_returns_a_model(router):
    """BALANCED returns a valid ModelConfig from the registry."""
    result = router.select(RoutingStrategy.BALANCED, prompt_tokens=100, current_spend=0)
    assert isinstance(result, ModelConfig)
    assert result in MODELS


def test_balanced_is_not_extreme(router):
    """BALANCED does not return the most expensive or slowest model exclusively."""
    result = router.select(RoutingStrategy.BALANCED, prompt_tokens=100, current_spend=0)
    # The balanced winner must trade off quality, cost, and latency —
    # it should never be the pure worst on all dimensions simultaneously.
    worst_latency = max(m.avg_latency_ms for m in MODELS)
    worst_quality = min(m.quality_score for m in MODELS)
    assert not (
        result.avg_latency_ms == worst_latency and result.quality_score == worst_quality
    )


# ── Strategy: AUTO ────────────────────────────────────────────────────────────


def test_auto_within_budget_matches_balanced(router):
    """AUTO delegates to BALANCED when spend is below 80% of budget."""
    auto_result = router.select(
        RoutingStrategy.AUTO, prompt_tokens=100, current_spend=5.0
    )
    balanced_result = router.select(
        RoutingStrategy.BALANCED, prompt_tokens=100, current_spend=5.0
    )
    assert auto_result == balanced_result


def test_auto_budget_exhausted_matches_cheapest(router):
    """AUTO delegates to CHEAPEST when spend exceeds 80% of monthly budget."""
    # settings.monthly_budget_usd = 50.0 → threshold = 40.0
    auto_result = router.select(
        RoutingStrategy.AUTO, prompt_tokens=100, current_spend=45.0
    )
    cheapest_result = router.select(
        RoutingStrategy.CHEAPEST, prompt_tokens=100, current_spend=45.0
    )
    assert auto_result == cheapest_result
    assert auto_result.model_id == "mistral:7b"


def test_auto_at_exact_threshold_uses_balanced(router):
    """AUTO uses BALANCED when spend equals exactly 80% (not strictly greater)."""
    threshold = 0.80 * 50.0  # = 40.0
    auto_result = router.select(
        RoutingStrategy.AUTO, prompt_tokens=100, current_spend=threshold
    )
    balanced_result = router.select(
        RoutingStrategy.BALANCED, prompt_tokens=100, current_spend=threshold
    )
    assert auto_result == balanced_result


# ── Context window filtering ──────────────────────────────────────────────────


def test_context_filter_excludes_small_models(router):
    """require_context=50000 excludes mistral:7b (32k context window)."""
    result = router.select(
        RoutingStrategy.CHEAPEST,
        prompt_tokens=100,
        current_spend=0,
        require_context=50_000,
    )
    assert result.model_id != "mistral:7b"
    assert result.max_context_tokens >= 50_000


def test_context_filter_allows_large_context_models(router):
    """require_context=150000 only keeps Anthropic models (200k context)."""
    result = router.select(
        RoutingStrategy.CHEAPEST,
        prompt_tokens=100,
        current_spend=0,
        require_context=150_000,
    )
    assert result.provider == "anthropic"
    assert result.max_context_tokens >= 150_000


def test_context_filter_raises_when_no_model_qualifies(router):
    """ValueError is raised when require_context exceeds all models."""
    with pytest.raises(ValueError, match="No model supports"):
        router.select(
            RoutingStrategy.CHEAPEST,
            prompt_tokens=100,
            current_spend=0,
            require_context=999_999,
        )


# ── Fallback chain ────────────────────────────────────────────────────────────


def test_fallback_chain_excludes_primary(router):
    """build_fallback_chain never includes the primary model."""
    primary = MODELS[0]  # gpt-4o-mini
    chain = router.build_fallback_chain(primary)
    assert primary not in chain


def test_fallback_chain_contains_all_others(router):
    """build_fallback_chain contains every model except the primary."""
    primary = MODELS[0]
    chain = router.build_fallback_chain(primary)
    assert len(chain) == len(MODELS) - 1
    for m in MODELS:
        if m is not primary:
            assert m in chain


def test_fallback_chain_sorted_by_cost_ascending(router):
    """Fallback chain is ordered cheapest first."""
    primary = MODELS[1]  # gpt-4o (expensive)
    chain = router.build_fallback_chain(primary)
    costs = [m.cost_per_1k_input for m in chain]
    assert costs == sorted(costs)


# ── estimate_cost ─────────────────────────────────────────────────────────────


def test_estimate_cost_gpt4o_mini(router):
    """estimate_cost returns correct USD for gpt-4o-mini."""
    model = next(m for m in MODELS if m.model_id == "gpt-4o-mini")
    # 1000 input tokens + 500 output tokens
    cost = router.estimate_cost(model, input_tokens=1000, output_tokens=500)
    expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.00060
    assert abs(cost - expected) < 1e-10


def test_estimate_cost_zero_for_ollama(router):
    """estimate_cost returns 0.0 for ollama (free local model)."""
    model = next(m for m in MODELS if m.model_id == "mistral:7b")
    cost = router.estimate_cost(model, input_tokens=10_000, output_tokens=5_000)
    assert cost == 0.0


def test_estimate_cost_zero_tokens(router):
    """estimate_cost returns 0.0 for zero token counts."""
    model = MODELS[0]
    assert router.estimate_cost(model, input_tokens=0, output_tokens=0) == 0.0
