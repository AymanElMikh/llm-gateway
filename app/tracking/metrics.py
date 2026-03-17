"""Prometheus metric definitions for the LLM Gateway.

All metric objects are module-level singletons. Import them directly:

    from app.tracking.metrics import REQUEST_COUNT, REQUEST_LATENCY
"""

from prometheus_client import Counter, Histogram

REQUEST_COUNT: Counter = Counter(
    "gateway_requests_total",
    "Total number of requests dispatched to an LLM provider",
    labelnames=["provider", "model", "status"],
)

REQUEST_LATENCY: Histogram = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency in seconds",
    labelnames=["provider", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

CACHE_HITS: Counter = Counter(
    "gateway_cache_hits_total",
    "Total number of semantic cache hits",
)

CACHE_MISSES: Counter = Counter(
    "gateway_cache_misses_total",
    "Total number of semantic cache misses",
)

TOKENS_USED: Counter = Counter(
    "gateway_tokens_total",
    "Total number of tokens processed",
    labelnames=["provider", "model", "type"],  # type: input | output
)

COST_USD: Counter = Counter(
    "gateway_cost_usd_total",
    "Cumulative estimated cost in USD",
    labelnames=["provider", "model"],
)
