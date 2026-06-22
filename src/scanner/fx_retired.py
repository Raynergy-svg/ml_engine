"""US-020 — FX directional path retirement (non-destructive).

This module is the single, machine-checkable entry point for the
retirement of the FX directional engine path and its 15 specialist
agents. **Nothing under** ``src/scanner/`` **is hard-deleted** — the
code is preserved verbatim in git history (and on disk) so prior
backtests, journals, and audit trails remain reproducible. What this
module does is:

1. Enumerate the 15 retired directional agents (the canonical list,
   sourced from :class:`src.scanner.agents._team.ScannerAgentTeam`'s
   ``_BASE_WEIGHTS``) and the ``ScannerConfig`` flag names that toggle
   them.
2. Provide :func:`disable_fx_directional_path` so any caller that
   builds a ``ScannerConfig`` for legacy reasons can flip every
   directional agent off in one call (atomic in-place mutation;
   idempotent).
3. Provide :func:`enforce_harvester_ship_gate` — the ship-gate guard.
   This module refuses to be used as a live runtime unless
   ``trained_data/backtests/SHIP_GATE.json`` exists, contains
   ``gate_pass == True``, and its ``universe_hash`` matches the
   harvester universe the operator is deploying. Missing / false /
   mismatched → :class:`FXPathRetiredError`. Behaviour is byte-for-byte
   identical to :func:`src.equity.control_loop._enforce_ship_gate` so
   the operator sees one contract, not two.

Why retired
-----------

Two months of structured search closed the loop on FX directional
prediction at the data scale Buddy operates at:

* ``docs/fx-edge-search-final-verdict-2026-06-18.md`` — own-data 22-yr
  walk-forward daily direction = 50.3% balanced; majors ≈ random walk;
  high published claims provably artefacts of leakage / 2022-trend.
* ``docs/equity-harvester-verdict-2026-06-18.md`` and
  ``docs/harvester-verdict-2026-06-18.md`` — the harvester (vol-managed
  beta + drawdown circuit-breaker) clears the pre-registered ship gate
  on the single-stock universe (net Sharpe 0.92, maxDD 0.229, 2010-26).
* ``tasks/prd-equity-harvester-bot.md`` (US-020) — operator decision
  on 2026-06-18 to retire the FX directional path **non-destructively**
  and re-point the autonomous machinery onto the harvester.

What this module does NOT do
----------------------------

* Does not delete any FX module, model, journal, or backtest artifact.
* Does not raise from import time — calling :func:`disable_fx_directional_path`
  is purely cosmetic on a fresh ``ScannerConfig``; the runtime entry
  is the harvester (``src/equity/control_loop.py``).
* Does not affect agents the equity harvester uses (the equity
  layer's :class:`src.equity.risk_agents.RiskAgentLayer` is a separate,
  fail-closed deterministic agent stack — see US-008).

Hard constraints (mirrors CLAUDE.md "What we never do")
-------------------------------------------------------

* Atomic state writes (``tempfile.mkstemp`` + :func:`os.replace`).
* No bare ``except`` — every recoverable error logs + surfaces.
* No mocks in tests — real ``ScannerConfig`` + real disk via
  ``tmp_path`` (see ``tests/test_fx_path_retired.py``).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# Canonical ship-gate location (matches every other equity surface).
SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"


# ---------------------------------------------------------------------------
# Canonical roster of the 15 retired directional agents
# ---------------------------------------------------------------------------
#
# Source of truth: ``src.scanner.agents._team.ScannerAgentTeam._BASE_WEIGHTS``.
# Listed here as plain strings (not imported) so this module stays a
# leaf — importing _team would drag the live FX runtime into the
# retirement guard, which is the opposite of the intent.
#
# If the FX roster ever changes, both lists below must change in
# lock-step. A canary test asserts the two stay in sync against
# ``_BASE_WEIGHTS`` so drift cannot ship silently.

FX_DIRECTIONAL_AGENT_NAMES: Tuple[str, ...] = (
    "trend",
    "mean_reversion",
    "volatility",
    "risk_sentinel",
    "uncertainty",
    "execution_quality",
    "news_risk",
    "multi_timeframe",
    "pair_performance",
    "momentum",
    "session_timing",
    "support_resistance",
    "order_flow",
    "trader_readiness",
    "devil_advocate",
)


# ``ScannerConfig`` field names that toggle each agent on/off.
# ``enable_devil_advocate`` (legacy alias) is also flipped for safety.
FX_DIRECTIONAL_AGENT_FLAGS: Tuple[str, ...] = (
    "enable_trend_agent",
    "enable_mean_reversion_agent",
    "enable_volatility_agent",
    "enable_risk_sentinel_agent",
    "enable_uncertainty_agent",
    "enable_execution_quality_agent",
    "enable_news_risk_agent",
    "enable_multi_timeframe_agent",
    "enable_pair_performance_agent",
    "enable_momentum_agent",
    "enable_session_timing_agent",
    "enable_support_resistance_agent",
    "enable_order_flow_agent",
    "enable_trader_readiness_agent",
    "enable_devil_advocate_agent",
    "enable_devil_advocate",  # legacy alias — kept off in lock-step
)


class FXPathRetiredError(RuntimeError):
    """Raised when a caller tries to start the retired FX path as a runtime.

    Surfaced (never swallowed). The operator sees the exact reason the
    guard refused — missing file, malformed JSON, ``gate_pass != True``,
    blank ``universe_hash``, or mismatched ``universe_hash``.
    """


# ---------------------------------------------------------------------------
# Config helper — flip every directional agent off in one call
# ---------------------------------------------------------------------------


def disable_fx_directional_path(config: Any) -> Tuple[str, ...]:
    """Set every FX directional agent toggle on ``config`` to ``False``.

    Idempotent. Returns the tuple of flag names that were actually
    flipped (i.e. the flag existed on the dataclass *and* was True
    before). Flags that don't exist on the passed object are skipped
    silently — this lets the helper work with both the real
    ``ScannerConfig`` and lightweight ``getattr``-shaped duck-types.

    The function does not persist anything to disk; the caller decides
    whether to write the mutated config out via its own atomic path.

    Parameters
    ----------
    config:
        Any object that supports :func:`setattr`. Typically a
        :class:`src.scanner.config.ScannerConfig` instance, but any
        attribute-bag works.
    """
    flipped = []
    for flag in FX_DIRECTIONAL_AGENT_FLAGS:
        if not hasattr(config, flag):
            continue
        try:
            previous = getattr(config, flag)
        except AttributeError as exc:
            logger.warning(
                "disable_fx_directional_path: cannot read %s on %r: %s",
                flag,
                type(config).__name__,
                exc,
            )
            continue
        if previous is False:
            continue
        setattr(config, flag, False)
        flipped.append(flag)
    if flipped:
        logger.info(
            "FX directional path retired: flipped %d agent toggles "
            "to False on %s (see tasks/prd-equity-harvester-bot.md US-020)",
            len(flipped),
            type(config).__name__,
        )
    return tuple(flipped)


# ---------------------------------------------------------------------------
# Ship-gate guard — module refuses to run unless the gate clears
# ---------------------------------------------------------------------------


def _load_ship_gate(path: Path) -> Mapping[str, Any]:
    """Read the ship-gate JSON file. Raises :class:`FXPathRetiredError`."""
    if not path.exists():
        raise FXPathRetiredError(
            f"FX path retired and harvester ship gate refuses to start: "
            f"file not found at {path} "
            "(see tasks/prd-equity-harvester-bot.md US-005, US-020)"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FXPathRetiredError(
            f"FX path retired and harvester ship gate refuses to start: "
            f"cannot read {path}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FXPathRetiredError(
            f"FX path retired and harvester ship gate refuses to start: "
            f"malformed JSON at {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FXPathRetiredError(
            f"FX path retired and harvester ship gate refuses to start: "
            f"{path} payload is not a JSON object (got {type(payload).__name__})"
        )
    return payload


def enforce_harvester_ship_gate(
    expected_universe_hash: str,
    ship_gate_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Refuse to run the retired FX path unless the harvester gate clears.

    Contract (byte-for-byte identical to
    :func:`src.equity.control_loop._enforce_ship_gate`):

    * File exists at ``ship_gate_path`` (defaults to
      ``trained_data/backtests/SHIP_GATE.json``).
    * Payload is a JSON object.
    * ``payload["gate_pass"]`` is **exactly** :data:`True`.
    * ``payload["universe_hash"]`` is a non-blank string.
    * ``payload["universe_hash"]`` equals ``expected_universe_hash``.

    On any failure → :class:`FXPathRetiredError` with a precise reason.
    On success → returns the loaded payload (read-only mapping).
    """
    if not isinstance(expected_universe_hash, str) or not expected_universe_hash:
        raise FXPathRetiredError(
            "FX path retired guard misuse: expected_universe_hash must "
            "be a non-empty string"
        )
    path = Path(ship_gate_path) if ship_gate_path is not None else Path(
        SHIP_GATE_PATH_DEFAULT
    )
    payload = _load_ship_gate(path)
    if payload.get("gate_pass") is not True:
        raise FXPathRetiredError(
            "FX path retired and harvester ship gate refuses to start: "
            f"gate_pass={payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the harvester runtime)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise FXPathRetiredError(
            "FX path retired and harvester ship gate refuses to start: "
            f"universe_hash missing or blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise FXPathRetiredError(
            "FX path retired and harvester ship gate refuses to start: "
            f"universe_hash mismatch (gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )
    return payload


# ---------------------------------------------------------------------------
# Atomic write helper — for callers that persist retirement state
# ---------------------------------------------------------------------------


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomic JSON write — ``tempfile.mkstemp`` + :func:`os.replace`.

    Mirrors the pattern used by every other equity-surface persistence
    helper. Useful if a caller wants to journal the retirement state
    next to ``SHIP_GATE.json``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            logger.warning(
                "atomic_write_json: could not unlink %s: %s",
                tmp_name,
                cleanup_exc,
            )
        raise FXPathRetiredError(
            f"atomic_write_json failed for {path}: {exc}"
        ) from exc
    return path


__all__ = [
    "FX_DIRECTIONAL_AGENT_NAMES",
    "FX_DIRECTIONAL_AGENT_FLAGS",
    "FXPathRetiredError",
    "SHIP_GATE_PATH_DEFAULT",
    "atomic_write_json",
    "disable_fx_directional_path",
    "enforce_harvester_ship_gate",
]
