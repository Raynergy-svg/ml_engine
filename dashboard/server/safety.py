"""AXIOM data layer — structural READ-ONLY + PRACTICE-ONLY guarantees.

This module is the single safety boundary between the dashboard and the bot's
OANDA client. Two invariants, both enforced structurally (not by convention):

1. PRACTICE ONLY. ``assert_practice`` re-derives the environment from
   ``ScannerConfig`` and refuses to proceed unless it is ``"practice"`` — the
   same Hard NO the runtime enforces. The underlying ``OandaPracticeClient`` is
   itself hard-pinned to ``api-fxpractice`` (no live URL constant exists in it),
   so there is no reachable live path; this assert is defense-in-depth.

2. READ ONLY. ``ReadOnlyOandaClient`` is an allowlist proxy: only the v20 *read*
   methods are reachable. Any mutating method (``create_market_order``,
   ``close_position``, ``close_trade``) — or any method not on the allowlist —
   raises ``ReadOnlyViolation`` *before* the call reaches the network. The
   dashboard literally cannot place, modify, or close an order.
"""
from __future__ import annotations

import logging
import weakref
from typing import Any, Callable

logger = logging.getLogger("axiom.safety")

# v20 READ endpoints the dashboard is permitted to call. Anything not here is
# refused — including any future method, so a new mutating method can't sneak in.
READ_ALLOWLIST = frozenset({
    "get_account_summary",
    "get_open_positions",
    "get_pricing",
    "get_price_quote",
    "get_candles",
    "get_instruments",
    "get_order_book",
    "get_position_book",
    "get_trades",
    "get_transactions",
    "get_transactions_since",
})

# Explicitly named for a louder error message (these are the money-moving paths).
MUTATING_DENYLIST = frozenset({
    "create_market_order",
    "close_position",
    "close_trade",
    "_request",  # raw HTTP escape hatch — deny it too; reads go through typed methods
})


class ReadOnlyViolation(RuntimeError):
    """Raised when the dashboard attempts a non-read OANDA operation."""


class ConfigNotPractice(RuntimeError):
    """Raised when the resolved OANDA environment is not 'practice'."""


def assert_practice() -> str:
    """Re-derive the OANDA environment from ScannerConfig; refuse if not practice.

    Returns the environment string on success. Defense-in-depth: the client is
    already practice-pinned, but the dashboard asserts the doctrine independently
    so a misconfiguration fails closed here too.
    """
    from src.scanner.config import ScannerConfig

    env = getattr(ScannerConfig(), "oanda_environment", "practice")
    if env != "practice":
        raise ConfigNotPractice(
            f"HARD LINE: AXIOM refuses to run against oanda_environment={env!r}; "
            "the dashboard is PRACTICE-ONLY."
        )
    return env


# The wrapped client is held OUTSIDE the proxy instance (module-private weak map),
# so it is NOT an instance attribute: ``proxy._client`` / ``vars(proxy)`` /
# ``proxy.__dict__`` cannot reach the token-bearing, mutation-capable client. The
# only way to the raw client is importing this private map deliberately — accidental
# access (a careless ``proxy._client.create_market_order(...)``) is structurally
# impossible, not merely discouraged.
_WRAPPED: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


class ReadOnlyOandaClient:
    """Allowlist proxy around ``OandaPracticeClient`` exposing only read methods.

    Hardened: the underlying client is stored in a module-private weak map, never
    as an instance attribute. All private/underscore attribute access is refused,
    and only the read allowlist is reachable — there is no escape hatch to a
    mutating method through normal attribute access.
    """

    def __init__(self, client: Any):
        _WRAPPED[self] = client

    def __getattr__(self, name: str) -> Callable[..., Any]:
        # Block ALL private/dunder access first — this closes the ``_client`` /
        # ``_request`` / ``__dict__``-style escape hatches before the allowlist.
        if name.startswith("_"):
            raise ReadOnlyViolation(f"READ-ONLY: access to private attribute '{name}' is blocked.")
        if name in MUTATING_DENYLIST:
            raise ReadOnlyViolation(
                f"READ-ONLY: '{name}' is a mutating operation and is blocked "
                "by the AXIOM data layer. The dashboard never trades."
            )
        if name not in READ_ALLOWLIST:
            raise ReadOnlyViolation(
                f"READ-ONLY: '{name}' is not on the read allowlist and is blocked."
            )
        client = _WRAPPED.get(self)
        if client is None:
            raise ReadOnlyViolation("READ-ONLY: wrapped client is unavailable.")
        attr = getattr(client, name)
        if not callable(attr):
            raise ReadOnlyViolation(f"READ-ONLY: '{name}' is not a callable read method.")
        return attr

    def __repr__(self) -> str:  # never leak the token-bearing client repr
        return "<ReadOnlyOandaClient (practice, read-only)>"


def build_readonly_client() -> "ReadOnlyOandaClient | None":
    """Build the practice client from env and wrap it read-only.

    Returns ``None`` (never raises) if the token/account is missing or the client
    cannot be constructed — the caller renders an explicit 'not connected' state
    rather than fabricating data.
    """
    try:
        assert_practice()
        from src.utils.oanda_practice import OandaPracticeClient

        client = OandaPracticeClient.from_env()
        return ReadOnlyOandaClient(client)
    except ConfigNotPractice:
        raise  # a non-practice config is fatal — do not swallow it
    except Exception as exc:  # noqa: BLE001 — missing token / network / init failure
        logger.warning("OANDA practice client unavailable (live views degrade): %s", exc)
        return None
