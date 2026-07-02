"""Crypto-momentum LiveGate — the only authorized path from shadow to live.

Mirrors :mod:`src.equity.live_gate` byte-for-byte (same class, same typed
"LIVE" confirmation contract, same ship-gate enforcement) with crypto-scoped
default paths. Construction does NOT arm the gate, and nothing in this
codebase ever calls :meth:`LiveGate.arm` for the crypto lane — the shadow
runner (:mod:`src.crypto.momentum_shadow`, ``scripts/run_crypto_momentum_shadow.py``)
only ever calls :meth:`LiveGate.is_armed`.

Structural reason arming can never silently happen: H2/H4 crypto XS-momentum
does not pass the ship gate (`docs/experiment-crypto-edge-hunt-round2-2026-06-29.md`
§7 — significance FAILS at both N=15 and N=3; `clears_ex_history = FALSE`). No
``trained_data/crypto/SHIP_GATE.json`` exists, and none should be fabricated to
force an arm — :func:`src.equity.live_gate.enforce_ship_gate` refuses to arm
without a real ``gate_pass=True`` artifact. This module exists purely so the
crypto lane has the SAME gated live-promotion contract as the equity harvester
if the signal is ever revisited and clears the gate on fresh out-of-sample
evidence — not because arming is anticipated.

No broker/exchange client exists for crypto in this repo (research/backtest +
this shadow lane only) — even an armed gate has no execution path to route
through. Real orders require BOTH an armed gate AND a broker integration that
does not exist.
"""
from __future__ import annotations

from pathlib import Path

from src.equity.live_gate import (  # noqa: F401  re-exported for callers
    ArmResult,
    LIVE_CONFIRMATION_TOKEN,
    LiveGate,
    LiveGateConfig,
    LiveGateError,
    LiveGateState,
    MAX_INITIAL_NAV_FRACTION,
    MAX_PORTFOLIO_RISK_FRACTION,
    enforce_ship_gate,
)

CRYPTO_SHIP_GATE_PATH_DEFAULT = "trained_data/crypto/SHIP_GATE.json"
CRYPTO_STATE_PATH_DEFAULT = "trained_data/crypto/live_gate_state.json"
CRYPTO_AUDIT_PATH_DEFAULT = "trained_data/crypto/live_gate_audit.jsonl"


def make_crypto_live_gate(
    config: LiveGateConfig | None = None,
    *,
    state_path: Path | None = None,
    audit_path: Path | None = None,
) -> LiveGate:
    """Construct a crypto-scoped :class:`LiveGate` (always starts disarmed).

    Uses ``trained_data/crypto/`` state/audit paths so this lane's arm state
    is independent of the equity harvester's ``trained_data/equity/`` gate.
    """
    return LiveGate(
        config or LiveGateConfig(),
        state_path=Path(state_path or CRYPTO_STATE_PATH_DEFAULT),
        audit_path=Path(audit_path or CRYPTO_AUDIT_PATH_DEFAULT),
    )


__all__ = [
    "ArmResult",
    "CRYPTO_AUDIT_PATH_DEFAULT",
    "CRYPTO_SHIP_GATE_PATH_DEFAULT",
    "CRYPTO_STATE_PATH_DEFAULT",
    "LIVE_CONFIRMATION_TOKEN",
    "LiveGate",
    "LiveGateConfig",
    "LiveGateError",
    "LiveGateState",
    "MAX_INITIAL_NAV_FRACTION",
    "MAX_PORTFOLIO_RISK_FRACTION",
    "enforce_ship_gate",
    "make_crypto_live_gate",
]
