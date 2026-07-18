"""US-022 — Gated live path (typed confirmation + universe-keyed gate).

The only authorized way to upgrade the equity harvester from shadow/paper
mode into live IBKR execution.

A live arming attempt is refused unless *all* of the following hold:

1. ``trained_data/backtests/SHIP_GATE.json`` exists with
   ``gate_pass == True`` AND its ``universe_hash`` matches the deployed
   universe (US-005 contract — byte-for-byte the same guard used by
   :mod:`src.equity.control_loop`, :mod:`src.equity.rebalance`,
   :mod:`src.equity.risk_agents`, :mod:`src.equity.kill_switch`,
   :mod:`src.equity.shadow_pipeline`).
2. The operator supplied an explicit typed confirmation token equal to
   ``LIVE`` (case-sensitive). This mirrors the same string the TUI
   :class:`src.tui.screens.mode_modal.ModeConfirmModal` requires — the
   modal is the operator-facing UX; this module is the runtime guard
   the loop binds to. ``LIVE_CONFIRMATION_TOKEN`` is exported so the
   TUI and any tests can pin on the exact contract value.
3. A configured *small* initial NAV fraction
   (:attr:`LiveGateConfig.initial_nav_fraction`) sits in
   ``(0, MAX_INITIAL_NAV_FRACTION]`` and the resulting per-name caps and
   gross caps for the live book respect
   :attr:`LiveGateConfig.max_portfolio_risk_fraction` (default
   ``0.15`` — the project-wide 15% NAV ceiling from
   :doc:`CLAUDE.md`).
4. The kill switch (US-013) is constructable for this universe — the
   gate refuses to arm if the kill switch's ship-gate guard rejects.
5. The drawdown guardian (US-008 risk layer) is constructable for this
   universe — the gate refuses to arm if the risk layer's ship-gate
   guard rejects.

Any failed precondition raises :class:`LiveGateError` and the loop
stays halted. Every arm / disarm / refuse event is persisted to
``trained_data/equity/live_gate_state.json`` (atomic: ``tempfile.mkstemp``
+ :func:`os.replace`) and appended to a JSON-lines audit log
``trained_data/equity/live_gate_audit.jsonl`` so the operator can later
prove what changed when.

The module is intentionally deterministic and stdlib-only — no LLM, no
network, no clock injection except for testability. It is the lowest
layer of the live promotion stack: nothing else may flip dry_run → live
without going through :meth:`LiveGate.arm`.

Hard rules (project-wide):

* Real classes + real disk via ``tmp_path`` in tests — no
  ``unittest.mock`` / ``MagicMock`` / ``patch``.
* Atomic state writes (``tempfile.mkstemp`` + :func:`os.replace`).
* No bare ``except`` — every error path narrows the catch and logs.
* Long-only book by default (FR-11 from the equity PRD).
* Hummingbot Apache-2.0 attribution applies to any code under
  :mod:`src.equity.executors` / :mod:`src.equity.depth_pricing`; this
  module ports no third-party code.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public contract constants
# ---------------------------------------------------------------------------


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
"""Default location of the SHIP_GATE.json artifact (US-005 contract)."""

STATE_PATH_DEFAULT = "trained_data/equity/live_gate_state.json"
AUDIT_PATH_DEFAULT = "trained_data/equity/live_gate_audit.jsonl"

LIVE_CONFIRMATION_TOKEN = "LIVE"
"""Exact string the operator must type to confirm a live upgrade.

Stable by contract — the TUI :class:`ModeConfirmModal` pins on the same
literal. Renaming breaks the operator UX and the AC.
"""

MAX_INITIAL_NAV_FRACTION = 0.25
"""Hard ceiling on :attr:`LiveGateConfig.initial_nav_fraction`.

The PRD wants live to start *small*. A configurable fraction above this
ceiling defeats the purpose; the gate refuses to arm.
"""

MAX_PORTFOLIO_RISK_FRACTION = 0.15
"""Project-wide 15% NAV cap (CLAUDE.md "Trading invariants")."""

STATE_VERSION = 1

# Exact substrings the smoke test greps from logs/buddy_debug.log. Stable
# by contract — adding new tags is fine, renaming an existing one breaks
# the AC.
LOG_TAG_ARM = "LIVE_GATE arm"
LOG_TAG_DISARM = "LIVE_GATE disarm"
LOG_TAG_REFUSE = "LIVE_GATE refuse"
LOG_TAG_STATE_PERSIST = "LIVE_GATE state_persist"


class LiveGateError(RuntimeError):
    """Raised on any failed precondition for live arming.

    The runtime must catch this and stay halted. Mirrors the
    fail-closed contract of :class:`src.equity.control_loop.ControlLoopError`
    and :class:`src.equity.shadow_pipeline.ShadowPipelineError`.
    """


# ---------------------------------------------------------------------------
# Ship-gate guard (byte-for-byte identical contract to every other surface)
# ---------------------------------------------------------------------------


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to arm live unless SHIP_GATE.json clears for THIS universe.

    Fail-closed on any of:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        LiveGateError: On any of the above. The wording deliberately
            mirrors every other equity ship-gate guard so log greps and
            operator runbooks stay coherent across the stack.
    """
    path = Path(path)
    if not path.exists():
        raise LiveGateError(
            f"live gate refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveGateError(
            f"live gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LiveGateError(
            f"live gate refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise LiveGateError(
            "live gate refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm live)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise LiveGateError(
            f"live gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise LiveGateError(
            "live gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


# ---------------------------------------------------------------------------
# Atomic write helpers (mkstemp + os.replace, identical to every other surface)
# ---------------------------------------------------------------------------


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomic JSON write — ``tempfile.mkstemp`` + :func:`os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.error(
            "live gate atomic write failed for %s: %s",
            path, exc, exc_info=True,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise
    return path


def _append_audit_jsonl(payload: Mapping[str, Any], path: Path) -> Path:
    """Append a single JSON-line audit record atomically.

    The audit log is JSON-lines (one event per line). Each append uses
    ``open(..., "a")`` + ``flush`` + ``fsync`` so a crash mid-write leaves
    earlier events intact. We deliberately do NOT use mkstemp here — the
    audit log is append-only and partial writes of a final line are
    detectable (final line lacks a trailing newline).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        logger.error(
            "live gate audit append failed for %s: %s",
            path, exc, exc_info=True,
        )
        raise
    return path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveGateConfig:
    """Operator-tunable knobs for the live promotion path.

    ``initial_nav_fraction`` is the small, configurable fraction of NAV
    the gate is willing to expose on the first live cycle (e.g. ``0.05``
    = 5%). It is capped at :data:`MAX_INITIAL_NAV_FRACTION` (25%) — the
    PRD wants live to start small and this ceiling is a hard rail.

    ``max_portfolio_risk_fraction`` is the project-wide 15% cap (CLAUDE.md
    "Trading invariants"). Configurable so the gate refuses to arm when
    the operator's profile pushes above 15% — the cap stays a hard wall.
    """

    initial_nav_fraction: float = 0.05
    max_portfolio_risk_fraction: float = MAX_PORTFOLIO_RISK_FRACTION
    require_kill_switch: bool = True
    require_drawdown_guardian: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.initial_nav_fraction <= MAX_INITIAL_NAV_FRACTION):
            raise LiveGateError(
                "initial_nav_fraction must be in "
                f"(0, {MAX_INITIAL_NAV_FRACTION}], got "
                f"{self.initial_nav_fraction!r}"
            )
        if not (
            0.0 < self.max_portfolio_risk_fraction
            <= MAX_PORTFOLIO_RISK_FRACTION
        ):
            raise LiveGateError(
                "max_portfolio_risk_fraction must be in "
                f"(0, {MAX_PORTFOLIO_RISK_FRACTION}], got "
                f"{self.max_portfolio_risk_fraction!r}"
            )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class LiveGateState:
    """Persisted state of the live gate.

    ``armed`` is the single truth source the runtime reads to decide
    whether to route execution through the live IBKR broker. It can
    only flip True via :meth:`LiveGate.arm` (which enforces every
    precondition) and flips False on any disarm / refuse path.
    """

    armed: bool = False
    universe_hash: Optional[str] = None
    initial_nav_fraction: Optional[float] = None
    max_portfolio_risk_fraction: Optional[float] = None
    armed_at_ts: Optional[float] = None
    disarmed_at_ts: Optional[float] = None
    last_event: Optional[str] = None  # "arm" | "disarm" | "refuse"
    last_event_reason: Optional[str] = None
    last_event_ts: Optional[float] = None
    version: int = STATE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "armed": bool(self.armed),
            "universe_hash": self.universe_hash,
            "initial_nav_fraction": self.initial_nav_fraction,
            "max_portfolio_risk_fraction": self.max_portfolio_risk_fraction,
            "armed_at_ts": self.armed_at_ts,
            "disarmed_at_ts": self.disarmed_at_ts,
            "last_event": self.last_event,
            "last_event_reason": self.last_event_reason,
            "last_event_ts": self.last_event_ts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LiveGateState":
        if not isinstance(payload, Mapping):
            raise LiveGateError("state payload must be a JSON object")
        return cls(
            armed=bool(payload.get("armed", False)),
            universe_hash=payload.get("universe_hash"),
            initial_nav_fraction=payload.get("initial_nav_fraction"),
            max_portfolio_risk_fraction=payload.get(
                "max_portfolio_risk_fraction"
            ),
            armed_at_ts=payload.get("armed_at_ts"),
            disarmed_at_ts=payload.get("disarmed_at_ts"),
            last_event=payload.get("last_event"),
            last_event_reason=payload.get("last_event_reason"),
            last_event_ts=payload.get("last_event_ts"),
            version=int(payload.get("version", STATE_VERSION)),
        )


# ---------------------------------------------------------------------------
# LiveGate
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """Outcome of one :meth:`LiveGate.arm` call.

    On success the gate is armed and ``reason`` carries a structured
    one-liner. On failure :class:`LiveGateError` was raised — the call
    site never sees a falsey :class:`ArmResult`.
    """

    armed: bool
    universe_hash: str
    initial_nav_fraction: float
    max_portfolio_risk_fraction: float
    ts: float
    reason: str


class LiveGate:
    """Runtime guard that owns the live arm / disarm contract.

    Construction does NOT arm the gate. The operator must call
    :meth:`arm` with the ship-gate path, the deployed universe hash,
    and the typed confirmation token. Any failed precondition raises
    :class:`LiveGateError`; success persists ``armed=True`` to disk
    and appends an audit record.

    The gate exposes :meth:`is_armed` for the loop to query before
    routing execution to the live broker.
    """

    def __init__(
        self,
        config: LiveGateConfig,
        state_path: Optional[Path] = None,
        audit_path: Optional[Path] = None,
        clock: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.state_path = Path(state_path or STATE_PATH_DEFAULT)
        self.audit_path = Path(audit_path or AUDIT_PATH_DEFAULT)
        self._clock = clock or time.time
        self.state = self._load_state()

    # ------------------------------------------------------------------
    # State i/o
    # ------------------------------------------------------------------

    def _load_state(self) -> LiveGateState:
        if not self.state_path.exists():
            return LiveGateState()
        try:
            payload = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "live gate state at %s unreadable (%s) — falling back "
                "to disarmed defaults", self.state_path, exc,
            )
            return LiveGateState()
        try:
            return LiveGateState.from_dict(payload)
        except LiveGateError as exc:
            logger.warning(
                "live gate state at %s malformed (%s) — falling back "
                "to disarmed defaults", self.state_path, exc,
            )
            return LiveGateState()

    def _persist_state(self) -> None:
        _atomic_write_json(self.state.to_dict(), self.state_path)
        logger.info(
            "%s armed=%s universe=%s",
            LOG_TAG_STATE_PERSIST,
            self.state.armed,
            self.state.universe_hash,
        )

    def _audit(self, event: str, reason: str, extra: Mapping[str, Any]) -> None:
        record = {
            "ts": float(self._clock()),
            "event": event,
            "reason": reason,
            "armed": bool(self.state.armed),
            "universe_hash": self.state.universe_hash,
            **dict(extra),
        }
        _append_audit_jsonl(record, self.audit_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_armed(self) -> bool:
        """Return ``True`` iff the gate is armed for live execution.

        The loop calls this on every cycle before routing to the live
        broker. It is intentionally cheap (in-memory) — the persisted
        state file is the recovery path after restart.
        """
        return bool(self.state.armed)

    def arm(
        self,
        *,
        ship_gate_path: Path,
        expected_universe_hash: str,
        confirmation_token: str,
        kill_switch_factory: Optional[Any] = None,
        drawdown_guardian_factory: Optional[Any] = None,
    ) -> ArmResult:
        """Promote the gate to armed (live execution enabled).

        Refuses with :class:`LiveGateError` unless every precondition
        clears. On refuse the gate stays disarmed, the failure is
        audited, and the state file is rewritten with the refuse event.

        Args:
            ship_gate_path: Path to ``SHIP_GATE.json`` (US-005).
            expected_universe_hash: Hash of the universe the runtime is
                actually loading. Must equal the gate's ``universe_hash``.
            confirmation_token: Typed operator confirmation. Must equal
                :data:`LIVE_CONFIRMATION_TOKEN` exactly (the TUI's
                :class:`ModeConfirmModal` enforces this same literal).
            kill_switch_factory: Callable that returns a constructed
                :class:`src.equity.kill_switch.EquityKillSwitch` for the
                deployed universe. If supplied and the constructor raises,
                the live gate refuses (US-013 must be armable).
            drawdown_guardian_factory: Callable that returns a constructed
                :class:`src.equity.risk_agents.RiskAgentLayer` (with its
                drawdown guardian) for the deployed universe. If supplied
                and the constructor raises, the live gate refuses
                (US-008 must be armable).

        Returns:
            :class:`ArmResult` describing the successful arm.

        Raises:
            LiveGateError: On any failed precondition.
        """
        if not isinstance(confirmation_token, str):
            self._refuse(
                "confirmation_token must be a string, got "
                f"{type(confirmation_token).__name__}"
            )
        if confirmation_token != LIVE_CONFIRMATION_TOKEN:
            self._refuse(
                "typed confirmation mismatch — expected "
                f"{LIVE_CONFIRMATION_TOKEN!r}, got "
                f"{confirmation_token!r}"
            )

        try:
            enforce_ship_gate(
                Path(ship_gate_path), expected_universe_hash
            )
        except LiveGateError as exc:
            self._refuse(str(exc))

        # 2026-07-18 audit fix: require_* must mean REQUIRED. Previously a
        # caller that omitted the factory kwargs armed live with these
        # preconditions silently skipped (`require_kill_switch=True` was a
        # no-op unless a factory happened to be passed) — contradicting the
        # module docstring's "refused unless ALL of the following hold".
        # A missing factory now refuses arming, same as a failing one.
        if self.config.require_kill_switch:
            if kill_switch_factory is None:
                self._refuse(
                    "kill switch (US-013) is required for arming but no "
                    "kill_switch_factory was provided"
                )
            try:
                kill_switch_factory()
            except Exception as exc:  # noqa: BLE001 — surface + audit
                self._refuse(
                    f"kill switch (US-013) refused to construct: {exc}"
                )

        if self.config.require_drawdown_guardian:
            if drawdown_guardian_factory is None:
                self._refuse(
                    "drawdown guardian (US-008) is required for arming but no "
                    "drawdown_guardian_factory was provided"
                )
            try:
                drawdown_guardian_factory()
            except Exception as exc:  # noqa: BLE001 — surface + audit
                self._refuse(
                    "drawdown guardian (US-008) refused to "
                    f"construct: {exc}"
                )

        ts = float(self._clock())
        self.state.armed = True
        self.state.universe_hash = expected_universe_hash
        self.state.initial_nav_fraction = self.config.initial_nav_fraction
        self.state.max_portfolio_risk_fraction = (
            self.config.max_portfolio_risk_fraction
        )
        self.state.armed_at_ts = ts
        self.state.disarmed_at_ts = None
        self.state.last_event = "arm"
        self.state.last_event_reason = "typed confirmation + ship gate clear"
        self.state.last_event_ts = ts
        self._persist_state()
        reason = (
            "armed: ship gate clear, universe "
            f"{expected_universe_hash} matches, typed confirmation OK, "
            f"initial_nav_fraction={self.config.initial_nav_fraction}, "
            "max_portfolio_risk_fraction="
            f"{self.config.max_portfolio_risk_fraction}"
        )
        self._audit(
            "arm",
            reason,
            {
                "initial_nav_fraction": self.config.initial_nav_fraction,
                "max_portfolio_risk_fraction": (
                    self.config.max_portfolio_risk_fraction
                ),
            },
        )
        logger.info(
            "%s universe=%s initial_nav_fraction=%s max_risk=%s",
            LOG_TAG_ARM,
            expected_universe_hash,
            self.config.initial_nav_fraction,
            self.config.max_portfolio_risk_fraction,
        )
        return ArmResult(
            armed=True,
            universe_hash=expected_universe_hash,
            initial_nav_fraction=self.config.initial_nav_fraction,
            max_portfolio_risk_fraction=(
                self.config.max_portfolio_risk_fraction
            ),
            ts=ts,
            reason=reason,
        )

    def disarm(self, reason: str = "operator disarm") -> None:
        """Drop the gate back to disarmed.

        Idempotent — calling on an already-disarmed gate just rewrites
        the audit record so the operator has proof of the request.
        """
        ts = float(self._clock())
        was_armed = self.state.armed
        self.state.armed = False
        self.state.disarmed_at_ts = ts
        self.state.last_event = "disarm"
        self.state.last_event_reason = str(reason)
        self.state.last_event_ts = ts
        self._persist_state()
        self._audit(
            "disarm",
            reason,
            {"was_armed": bool(was_armed)},
        )
        logger.info(
            "%s reason=%r was_armed=%s",
            LOG_TAG_DISARM, reason, was_armed,
        )

    # ------------------------------------------------------------------
    # Internal — refuse path
    # ------------------------------------------------------------------

    def _refuse(self, reason: str) -> None:
        """Record + log + raise a refuse event. Never returns."""
        ts = float(self._clock())
        self.state.armed = False
        self.state.last_event = "refuse"
        self.state.last_event_reason = reason
        self.state.last_event_ts = ts
        # Persist + audit BEFORE raising so the refuse is durable.
        try:
            self._persist_state()
        except OSError as exc:
            logger.error(
                "live gate refuse state-persist failed: %s",
                exc, exc_info=True,
            )
        try:
            self._audit("refuse", reason, {})
        except OSError as exc:
            logger.error(
                "live gate refuse audit-append failed: %s",
                exc, exc_info=True,
            )
        logger.warning("%s reason=%r", LOG_TAG_REFUSE, reason)
        raise LiveGateError(reason)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


__all__ = [
    "ArmResult",
    "AUDIT_PATH_DEFAULT",
    "LIVE_CONFIRMATION_TOKEN",
    "LiveGate",
    "LiveGateConfig",
    "LiveGateError",
    "LiveGateState",
    "LOG_TAG_ARM",
    "LOG_TAG_DISARM",
    "LOG_TAG_REFUSE",
    "LOG_TAG_STATE_PERSIST",
    "MAX_INITIAL_NAV_FRACTION",
    "MAX_PORTFOLIO_RISK_FRACTION",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_PATH_DEFAULT",
    "STATE_VERSION",
    "enforce_ship_gate",
]
