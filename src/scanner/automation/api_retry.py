"""API retry utilities with exponential backoff and circuit breaker.

Provides:
- retry_api_call(): Decorator/wrapper for API calls with backoff
- CircuitBreaker: Tracks failure rates and trips to avoid cascading errors
"""
import time
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Auth error codes that should NOT be retried
_AUTH_ERROR_CODES = {401, 403}


class CircuitBreaker:
    """Tracks API call success/failure and trips when failure rate exceeds threshold.

    When tripped, all calls are rejected for a cooldown period to avoid
    cascading failures on a degraded upstream service.
    """

    def __init__(
        self,
        name: str = "default",
        window_size: int = 10,
        failure_threshold: float = 0.5,
        cooldown_seconds: float = 60.0,
    ):
        self.name = name
        self.window_size = window_size
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._results: deque = deque(maxlen=window_size)
        self._tripped_at: Optional[float] = None
        self._total_trips: int = 0

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently open (tripped)."""
        if self._tripped_at is None:
            return False
        elapsed = time.monotonic() - self._tripped_at
        if elapsed >= self.cooldown_seconds:
            # Cooldown expired — reset
            self._tripped_at = None
            logger.info(f"Circuit breaker '{self.name}' reset after {self.cooldown_seconds}s cooldown")
            return False
        return True

    def record_success(self) -> None:
        """Record a successful API call."""
        self._results.append(True)

    def record_failure(self) -> None:
        """Record a failed API call and check if breaker should trip."""
        self._results.append(False)
        if len(self._results) >= self.window_size:
            failure_rate = 1.0 - (sum(self._results) / len(self._results))
            if failure_rate >= self.failure_threshold and self._tripped_at is None:
                self._tripped_at = time.monotonic()
                self._total_trips += 1
                logger.warning(
                    f"Circuit breaker '{self.name}' TRIPPED: "
                    f"{failure_rate:.0%} failure rate in last {self.window_size} calls "
                    f"(cooldown {self.cooldown_seconds}s)"
                )

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for observability."""
        failures = sum(1 for r in self._results if not r)
        total = len(self._results)
        return {
            "name": self.name,
            "is_tripped": self.is_tripped,
            "failure_rate": (failures / total) if total > 0 else 0.0,
            "window_size": self.window_size,
            "total_trips": self._total_trips,
            "cooldown_remaining": (
                max(0, self.cooldown_seconds - (time.monotonic() - self._tripped_at))
                if self._tripped_at else 0
            ),
        }


# Global circuit breaker for OANDA API
_oanda_breaker = CircuitBreaker(name="oanda", window_size=10, failure_threshold=0.5, cooldown_seconds=60.0)


def get_oanda_breaker() -> CircuitBreaker:
    """Get the global OANDA circuit breaker instance."""
    return _oanda_breaker


def retry_api_call(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    **kwargs: Any,
) -> T:
    """Execute an API call with retry logic and circuit breaker protection.

    Args:
        func: The callable to execute (should return a requests.Response or similar).
        *args: Positional args passed to func.
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each retry).
        backoff_max: Maximum delay between retries.
        circuit_breaker: Optional CircuitBreaker to check/update.
        **kwargs: Keyword args passed to func.

    Returns:
        The return value of func on success.

    Raises:
        The last exception if all retries exhausted.
        RuntimeError if circuit breaker is tripped.
    """
    breaker = circuit_breaker or _oanda_breaker

    # Check circuit breaker
    if breaker.is_tripped:
        status = breaker.get_status()
        raise RuntimeError(
            f"Circuit breaker '{breaker.name}' is open "
            f"({status['cooldown_remaining']:.0f}s remaining). "
            f"Skipping API call to prevent cascading failures."
        )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)

            # Check for HTTP error status codes
            if hasattr(result, "status_code"):
                if result.status_code in _AUTH_ERROR_CODES:
                    # Auth errors fail fast — no retry
                    breaker.record_failure()
                    raise ValueError(
                        f"Auth error {result.status_code}: "
                        f"{getattr(result, 'text', 'no details')[:200]}"
                    )
                if result.status_code >= 500:
                    # Server error — should retry
                    raise ValueError(f"Server error {result.status_code}")
                if result.status_code >= 400:
                    # Client error — don't retry
                    breaker.record_success()  # API is responding, just rejecting
                    return result

            breaker.record_success()
            return result

        except ValueError as e:
            if "Auth error" in str(e):
                raise  # Don't retry auth errors
            last_error = e
            breaker.record_failure()

        except Exception as e:
            last_error = e
            breaker.record_failure()

        # Calculate backoff (skip if this was the last attempt)
        if attempt < max_retries:
            delay = min(backoff_base * (2 ** attempt), backoff_max)
            logger.debug(
                f"API retry {attempt + 1}/{max_retries} for {func.__name__}: "
                f"waiting {delay:.1f}s after {last_error}"
            )
            time.sleep(delay)

    # All retries exhausted
    raise last_error or RuntimeError("All retries exhausted")
