"""Crypto cash-and-carry LiveGate -- the only authorized path from shadow to live.

Mirrors :mod:`src.equity.live_gate` byte-for-byte (same class, same typed
"LIVE" confirmation contract, same ship-gate enforcement) with carry-scoped
default paths, independent of the crypto momentum lane's gate
(:mod:`src.crypto.crypto_live_gate`). Construction does NOT arm the gate, and
nothing in this codebase ever calls :meth:`LiveGate.arm` for the carry lane --
the shadow runner (:mod:`src.crypto.carry_shadow`,
``scripts/run_crypto_carry_shadow.py``) only ever calls :meth:`LiveGate.is_armed`.

Structural reason arming can never silently happen: the frozen cash-and-carry
backtest does not pass the ship gate
(`docs/prereg-crypto-cash-and-carry-shadow-2026-07-06.md` section 5 --
turnover cost dominates the funding premium; `clears_ex_history = FALSE`). No
``trained_data/crypto_carry/SHIP_GATE.json`` exists, and none should be
fabricated to force an arm -- :func:`src.equity.live_gate.enforce_ship_gate`
refuses to arm without a real ``gate_pass=True`` artifact. This module exists
purely so the carry lane has the SAME gated live-promotion contract as every
other lane if the construction is ever revisited (e.g. with holding-period
discipline to reduce the turnover that killed the naive daily backtest) and
clears the gate on fresh out-of-sample evidence -- not because arming is
anticipated.

No broker/exchange client exists for crypto in this repo (research/backtest +
this shadow lane only) -- even an armed gate has no execution path to route
through. Real orders require BOTH an armed gate AND a broker integration that
does not exist. Beyond that: even a hypothetically-armed, hypothetically-
integrated deployment of this specific strategy carries a real
exchange-solvency / liquidation tail (see the module docstring of
:mod:`src.crypto.carry_shadow`) that this repo's practice-account,
research-only posture is not equipped to take on -- arming this lane for real
money is explicitly out of scope of anything built here.
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

CRYPTO_CARRY_SHIP_GATE_PATH_DEFAULT = "trained_data/crypto_carry/SHIP_GATE.json"
CRYPTO_CARRY_STATE_PATH_DEFAULT = "trained_data/crypto_carry/live_gate_state.json"
CRYPTO_CARRY_AUDIT_PATH_DEFAULT = "trained_data/crypto_carry/live_gate_audit.jsonl"


def make_crypto_carry_live_gate(
    config: LiveGateConfig | None = None,
    *,
    state_path: Path | None = None,
    audit_path: Path | None = None,
) -> LiveGate:
    """Construct a carry-scoped :class:`LiveGate` (always starts disarmed).

    Uses ``trained_data/crypto_carry/`` state/audit paths so this lane's arm
    state is independent of the crypto momentum lane's, the equity
    harvester's, and Track B's gates.
    """
    return LiveGate(
        config or LiveGateConfig(),
        state_path=Path(state_path or CRYPTO_CARRY_STATE_PATH_DEFAULT),
        audit_path=Path(audit_path or CRYPTO_CARRY_AUDIT_PATH_DEFAULT),
    )


__all__ = [
    "ArmResult",
    "CRYPTO_CARRY_AUDIT_PATH_DEFAULT",
    "CRYPTO_CARRY_SHIP_GATE_PATH_DEFAULT",
    "CRYPTO_CARRY_STATE_PATH_DEFAULT",
    "LIVE_CONFIRMATION_TOKEN",
    "LiveGate",
    "LiveGateConfig",
    "LiveGateError",
    "LiveGateState",
    "MAX_INITIAL_NAV_FRACTION",
    "MAX_PORTFOLIO_RISK_FRACTION",
    "enforce_ship_gate",
    "make_crypto_carry_live_gate",
]
