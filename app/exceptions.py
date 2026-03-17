"""Custom exception hierarchy for the LLM Gateway.

All gateway exceptions inherit from GatewayError so callers can
catch the base class when they don't care about the specific type.
"""


class GatewayError(Exception):
    """Base class for all LLM Gateway errors."""


class ProviderError(GatewayError):
    """Base class for errors originating from an LLM provider."""

    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        """Initialise with an optional provider name and HTTP status code."""
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class FatalProviderError(ProviderError):
    """Provider error that should not be retried (e.g. timeout after all attempts)."""


class RateLimitError(ProviderError):
    """Provider returned HTTP 429 — Too Many Requests."""


class ProviderUnavailableError(ProviderError):
    """Provider returned a 5xx error or is otherwise unreachable."""


class AllProvidersFailedError(GatewayError):
    """All providers in the fallback chain have failed."""


class CacheError(GatewayError):
    """Error reading from or writing to the semantic cache."""
