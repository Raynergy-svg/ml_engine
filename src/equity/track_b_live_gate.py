"""Track B (SEC filing-text research-alpha) LiveGate — the only authorized
path from shadow to live.

Mirrors :mod:`src.equity.live_gate` byte-for-byte (same class, same typed
"LIVE" confirmation contract, same ship-gate enforcement) with track_b-scoped
default paths, exactly the way :mod:`src.crypto.crypto_live_gate` mirrors it
for the crypto momentum shadow lane. Construction does NOT arm the gate, and
nothing in this codebase ever calls :meth:`LiveGate.arm` for the track_b
lane — the shadow runner (:mod:`src.equity.track_b_shadow`,
``scripts/run_track_b_shadow.py``) only ever calls :meth:`LiveGate.is_armed`.

Structural reason arming can never silently happen: the frozen research
harness's own binding verdict for this signal is ``overall_verdict=
INSUFFICIENT`` (see ``trained_data/research/track_b_postcutoff_2026_07_02/
harness_result_primary_2bps.json`` and
``docs/adversarial-review-no-edge-verdicts-2026-07-02.md`` — N=48 scored
filings vs ~405 needed for 80% power). No ``trained_data/equity/track_b/
SHIP_GATE.json`` exists, and none should be fabricated to force an arm —
:func:`src.equity.live_gate.enforce_ship_gate` refuses to arm without a real
``gate_pass=True`` artifact. This module exists purely so the track_b lane has
the SAME gated live-promotion contract as the equity harvester and the crypto
momentum lane if the signal is ever revisited and clears the gate on fresh
out-of-sample evidence (e.g. after scoring the ~405 filings the power
calculation calls for) — not because arming is anticipated.
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

TRACK_B_SHIP_GATE_PATH_DEFAULT = "trained_data/equity/track_b/SHIP_GATE.json"
TRACK_B_STATE_PATH_DEFAULT = "trained_data/equity/track_b/live_gate_state.json"
TRACK_B_AUDIT_PATH_DEFAULT = "trained_data/equity/track_b/live_gate_audit.jsonl"


def make_track_b_live_gate(
    config: LiveGateConfig | None = None,
    *,
    state_path: Path | None = None,
    audit_path: Path | None = None,
) -> LiveGate:
    """Construct a track_b-scoped :class:`LiveGate` (always starts disarmed).

    Uses ``trained_data/equity/track_b/`` state/audit paths so this lane's arm
    state is independent of the equity harvester's ``trained_data/equity/``
    gate and the crypto momentum lane's ``trained_data/crypto/`` gate.
    """
    return LiveGate(
        config or LiveGateConfig(),
        state_path=Path(state_path or TRACK_B_STATE_PATH_DEFAULT),
        audit_path=Path(audit_path or TRACK_B_AUDIT_PATH_DEFAULT),
    )


__all__ = [
    "ArmResult",
    "TRACK_B_AUDIT_PATH_DEFAULT",
    "TRACK_B_SHIP_GATE_PATH_DEFAULT",
    "TRACK_B_STATE_PATH_DEFAULT",
    "LIVE_CONFIRMATION_TOKEN",
    "LiveGate",
    "LiveGateConfig",
    "LiveGateError",
    "LiveGateState",
    "MAX_INITIAL_NAV_FRACTION",
    "MAX_PORTFOLIO_RISK_FRACTION",
    "enforce_ship_gate",
    "make_track_b_live_gate",
]
