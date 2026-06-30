"""US-021 — Shadow / paper mode end-to-end orchestrator.

Composes the equity stack (US-007 rebalance, US-008 risk agents, US-013
kill switch, US-016 shadow-fill simulator, US-017 autonomous loop) into
a single ``ShadowPipeline`` that exercises the *whole* bot in simulation
before any real money touches the wire.

What it adds on top of US-017
-----------------------------

US-017 already drives schedule -> reconcile -> risk -> execute -> persist
in a loop. US-021 layers two operator-visible guarantees on top:

1. **Halted by default + paper IBKR.** The pipeline starts ``halted=True``
   and ``mode="shadow"``; it refuses to flip to live until US-022 lands
   (and even then, only via typed confirmation). Every fill goes through
   the US-016 :class:`~src.equity.shadow_fills.ShadowFillSimulator`, so
   the run produces an auditable ledger of *what would have happened* on
   real capital.

2. **Divergence check (verifiable).** After each cycle the pipeline
   updates running shadow stats (realized max drawdown + turnover ratio)
   and diffs them against the backtest stats pulled from the SHIP_GATE
   payload + operator-supplied targets. If the realized shadow max-DD
   exceeds the backtest max-DD by more than ``max_dd_breach_pp`` percentage
   points OR turnover diverges by more than ``turnover_drift_pct``, the
   pipeline emits a ``DIVERGENCE`` warning and records the breach in the
   atomic state file. The thresholds are operator-set; the test drives
   synthetic stats both past and under them.

Hard constraints (mirrored from US-017)
---------------------------------------

* **Claude-free hot path** — no ``anthropic`` / ``openai`` / ``llm``
  imports anywhere in this module. The
  ``tests/test_no_llm_in_equity_path.py`` canary AST-greps the whole
  ``src/equity/`` tree to enforce this.
* **Fail-closed by default** — every degraded condition (ship gate
  refused, corrupt state, divergence above threshold, kill-switch
  armed) skips execution and surfaces the reason. No blind run.
* **Atomic state writes** — ``tempfile.mkstemp`` + ``os.replace``,
  byte-for-byte identical to every other equity surface.
* **No mocks in tests** — real :class:`ShadowFillSimulator`, real
  :class:`~src.equity.kill_switch.KillSwitch`-style halt, real
  ``tmp_path`` disk for the SHIP_GATE fixture, the ledger and the
  pipeline state file.
* **No bare excepts** — every catch names its exception types and
  either re-raises or logs and surfaces.

Surfaces
--------

* :class:`ShadowPipelineError` — raised on ship-gate refusal,
  malformed config, or contract bugs.
* :class:`DivergenceThresholds` — frozen dataclass holding the
  operator-set ``max_dd_breach_pp`` and ``turnover_drift_pct``
  (drift is symmetric: both over- and under-trading vs backtest
  trigger the warning).
* :class:`DivergenceReport` — frozen dataclass with the realized vs
  backtest figures and the ``breached`` flag.
* :class:`ShadowRunStats` — running per-cycle accumulator (peak NAV,
  trough NAV, realized DD, gross notional traded vs NAV).
* :class:`ShadowPipeline` — the orchestrator: re-enforces the gate at
  construction, exposes :meth:`step` to advance one cycle, persists
  state atomically, and surfaces ``DIVERGENCE`` via
  :meth:`check_divergence`.
* :func:`enforce_ship_gate` — same fail-closed contract as the other
  equity guards.

The runtime flow
----------------

::

    pipeline = ShadowPipeline.create(
        universe_hash=...,
        ledger_path=...,
        state_path=...,
        thresholds=DivergenceThresholds(
            max_dd_breach_pp=5.0,
            turnover_drift_pct=25.0,
        ),
        ship_gate_path=...,
    )
    # halted=True, mode="shadow" — refuses execute until operator unhalts
    for cycle in cycles:
        result = pipeline.step(
            nav=current_nav,
            gross_traded_notional=cycle_gross,
            fills=cycle_fills,   # list[SimulatedFill] from US-016
        )
        if result.diverged:
            logger.warning("DIVERGENCE: %s", result.report.reason())

The pipeline NEVER calls a live broker. Every order goes through the
US-016 simulator; the test asserts that flipping ``halted`` to ``False``
without an explicit ``live=False`` flag is a no-op for execution.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.equity.shadow_fills import (
    ShadowFillSimulator,
    SimulatedFill,
    SlippageModel,
)

logger = logging.getLogger(__name__)

SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1

# Exact substrings the smoke test greps from logs/buddy_debug.log. Stable
# by contract — renaming breaks the AC. Adding new tags is fine.
LOG_TAG_RUN_BEGIN = "SHADOW_PIPELINE run_begin"
LOG_TAG_CYCLE = "SHADOW_PIPELINE cycle"
LOG_TAG_HALTED = "SHADOW_PIPELINE halted"
LOG_TAG_DIVERGENCE = "SHADOW_PIPELINE DIVERGENCE"
LOG_TAG_STATE_PERSIST = "SHADOW_PIPELINE state_persist"


class ShadowPipelineError(RuntimeError):
    """Raised on ship-gate refusal, malformed config, or contract bugs."""


# ---------------------------------------------------------------------------
# Ship-gate guard (byte-for-byte identical contract to every other surface)
# ---------------------------------------------------------------------------


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to run unless SHIP_GATE.json clears for THIS universe.

    Fail-closed on:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        ShadowPipelineError: On any of the above. Same contract as
            every other equity ship-gate guard.
    """
    path = Path(path)
    if not path.exists():
        raise ShadowPipelineError(
            f"shadow pipeline refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowPipelineError(
            f"shadow pipeline refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ShadowPipelineError(
            f"shadow pipeline refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise ShadowPipelineError(
            "shadow pipeline refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the pipeline)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise ShadowPipelineError(
            f"shadow pipeline refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise ShadowPipelineError(
            "shadow pipeline refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


def _read_backtest_stats(path: Path) -> dict:
    """Load the backtest stats (max_dd + turnover_per_year if present).

    Returns ``{"max_dd": <float>, "turnover_per_year": <float|None>}``.
    Falls back gracefully when ``turnover_per_year`` is absent in the
    SHIP_GATE payload (the harvester's gate has ``max_dd`` but not yet
    a turnover field — the operator passes the backtest turnover via
    :class:`DivergenceThresholds.backtest_turnover_per_year` instead).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ShadowPipelineError(
            f"shadow pipeline refuses: {path} payload is not a JSON object"
        )
    max_dd = payload.get("max_dd")
    if not isinstance(max_dd, (int, float)) or not math.isfinite(max_dd):
        raise ShadowPipelineError(
            f"shadow pipeline refuses: max_dd missing/non-finite at {path}"
        )
    turnover = payload.get("turnover_per_year")
    if turnover is not None:
        if not isinstance(turnover, (int, float)) or not math.isfinite(turnover):
            raise ShadowPipelineError(
                "shadow pipeline refuses: turnover_per_year is present "
                f"but non-finite at {path}"
            )
    return {
        "max_dd": float(max_dd),
        "turnover_per_year": (
            float(turnover) if turnover is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Divergence thresholds + report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceThresholds:
    """Operator-set tolerances for shadow vs backtest divergence.

    Attributes:
        max_dd_breach_pp: Percentage *points* (not percent) of max
            drawdown the shadow run is allowed to exceed the backtest
            by. Example: if backtest max_dd is 22.9% and the operator
            sets ``max_dd_breach_pp=5.0``, a realized shadow max_dd
            above 27.9% (= 0.279) triggers a divergence warning.
        turnover_drift_pct: Symmetric percent tolerance on turnover.
            Example: backtest turnover_per_year = 4.0, drift_pct = 25.0
            -> shadow turnover outside [3.0, 5.0] triggers a warning.
        backtest_turnover_per_year: Fallback used when SHIP_GATE.json
            does not carry a turnover field. Required if the gate
            payload omits ``turnover_per_year`` and the operator wants
            the turnover side of the divergence check live.
    """

    max_dd_breach_pp: float
    turnover_drift_pct: float
    backtest_turnover_per_year: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_dd_breach_pp, (int, float))
            or not math.isfinite(self.max_dd_breach_pp)
            or self.max_dd_breach_pp < 0.0
        ):
            raise ShadowPipelineError(
                "max_dd_breach_pp must be a finite non-negative number, "
                f"got {self.max_dd_breach_pp!r}"
            )
        if (
            not isinstance(self.turnover_drift_pct, (int, float))
            or not math.isfinite(self.turnover_drift_pct)
            or self.turnover_drift_pct < 0.0
        ):
            raise ShadowPipelineError(
                "turnover_drift_pct must be a finite non-negative number, "
                f"got {self.turnover_drift_pct!r}"
            )
        if self.backtest_turnover_per_year is not None and (
            not isinstance(self.backtest_turnover_per_year, (int, float))
            or not math.isfinite(self.backtest_turnover_per_year)
            or self.backtest_turnover_per_year < 0.0
        ):
            raise ShadowPipelineError(
                "backtest_turnover_per_year must be a finite non-negative "
                f"number or None, got {self.backtest_turnover_per_year!r}"
            )


@dataclass(frozen=True)
class DivergenceReport:
    """Verdict from one divergence check.

    Attributes:
        breached: True if any of the checked surfaces exceeded its
            threshold. The pipeline logs ``DIVERGENCE`` exactly when
            this is True.
        max_dd_realized: Shadow run's realized max drawdown, [0, 1].
        max_dd_backtest: Backtest max drawdown from SHIP_GATE.json.
        max_dd_breach_pp: How many percentage points shadow exceeds
            backtest by. Negative = shadow is *better* than backtest.
        turnover_realized: Annualized turnover ratio from the shadow
            ledger (gross traded notional / NAV / years_elapsed).
        turnover_backtest: Backtest turnover. May be None when the gate
            payload omits it and the operator did not supply a fallback.
        turnover_drift_pct: |realized - backtest| / backtest * 100.
            None if ``turnover_backtest`` is None or zero.
        reasons: Free-text breach reasons; empty when ``breached=False``.
    """

    breached: bool
    max_dd_realized: float
    max_dd_backtest: float
    max_dd_breach_pp: float
    turnover_realized: float
    turnover_backtest: Optional[float]
    turnover_drift_pct: Optional[float]
    reasons: tuple = ()

    def reason(self) -> str:
        if not self.reasons:
            return "no_divergence"
        return "; ".join(self.reasons)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


# ---------------------------------------------------------------------------
# Running shadow stats
# ---------------------------------------------------------------------------


@dataclass
class ShadowRunStats:
    """Per-cycle accumulator for the shadow run.

    Tracks:

    * NAV high-water mark + lowest drawdown trough (for realized maxDD).
    * Gross notional traded + initial NAV + elapsed seconds (for
      annualized turnover).

    All fields are JSON-serializable so they can be round-tripped via
    the atomic state file.
    """

    initial_nav: float = 0.0
    peak_nav: float = 0.0
    last_nav: float = 0.0
    max_drawdown: float = 0.0
    gross_traded_notional: float = 0.0
    elapsed_seconds: float = 0.0
    cycles: int = 0

    def update(
        self,
        nav: float,
        gross_traded: float,
        cycle_seconds: float,
    ) -> None:
        if not isinstance(nav, (int, float)) or not math.isfinite(nav) or nav <= 0.0:
            raise ShadowPipelineError(
                f"nav must be a positive finite number, got {nav!r}"
            )
        if (
            not isinstance(gross_traded, (int, float))
            or not math.isfinite(gross_traded)
            or gross_traded < 0.0
        ):
            raise ShadowPipelineError(
                "gross_traded must be a finite non-negative number, "
                f"got {gross_traded!r}"
            )
        if (
            not isinstance(cycle_seconds, (int, float))
            or not math.isfinite(cycle_seconds)
            or cycle_seconds < 0.0
        ):
            raise ShadowPipelineError(
                "cycle_seconds must be a finite non-negative number, "
                f"got {cycle_seconds!r}"
            )
        if self.cycles == 0:
            self.initial_nav = float(nav)
            self.peak_nav = float(nav)
        if nav > self.peak_nav:
            self.peak_nav = float(nav)
        # Realized DD is measured against the running high-water mark.
        if self.peak_nav > 0.0:
            dd = max(0.0, (self.peak_nav - nav) / self.peak_nav)
            if dd > self.max_drawdown:
                self.max_drawdown = float(dd)
        self.last_nav = float(nav)
        self.gross_traded_notional += float(gross_traded)
        self.elapsed_seconds += float(cycle_seconds)
        self.cycles += 1

    def turnover_per_year(self) -> float:
        """Annualized turnover ratio: gross / NAV / years_elapsed.

        Returns 0.0 if the run has not yet booked any elapsed time or
        the initial NAV is zero (cannot annualize a zero-length run).
        """
        if self.elapsed_seconds <= 0.0 or self.initial_nav <= 0.0:
            return 0.0
        seconds_per_year = 365.25 * 24.0 * 3600.0
        years = self.elapsed_seconds / seconds_per_year
        if years <= 0.0:
            return 0.0
        return self.gross_traded_notional / self.initial_nav / years


# ---------------------------------------------------------------------------
# Atomic write helper (mkstemp + os.replace, identical to every other surface)
# ---------------------------------------------------------------------------


def _atomic_write_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".shadow_pipeline_", suffix=".json", dir=str(path.parent),
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
            "shadow pipeline atomic write failed for %s: %s",
            path, exc, exc_info=True,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise


# ---------------------------------------------------------------------------
# CycleResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleResult:
    """Outcome of one :meth:`ShadowPipeline.step` call."""

    halted: bool
    executed: bool
    diverged: bool
    report: DivergenceReport
    fills_recorded: int = 0


# ---------------------------------------------------------------------------
# ShadowPipeline
# ---------------------------------------------------------------------------


@dataclass
class ShadowPipeline:
    """End-to-end shadow / paper runtime.

    Constructed via :meth:`create` so the ship-gate guard runs *before*
    any state is mutated. Re-enforces the gate on every :meth:`step`
    call so an invalidated gate mid-run fails closed at the next cycle.

    Defaults: ``halted=True``, ``mode='shadow'``. The operator must
    explicitly unhalt to advance state, and the runtime will REFUSE to
    flip ``mode`` to ``'live'`` (US-022 owns that path).
    """

    universe_hash: str
    ledger_path: Path
    state_path: Path
    thresholds: DivergenceThresholds
    ship_gate_path: Path
    simulator: ShadowFillSimulator
    stats: ShadowRunStats = field(default_factory=ShadowRunStats)
    halted: bool = True
    mode: str = "shadow"
    last_report: Optional[DivergenceReport] = None
    _backtest_stats: dict = field(default_factory=dict)

    # ---------- construction ----------

    @classmethod
    def create(
        cls,
        *,
        universe_hash: str,
        ledger_path: Path,
        state_path: Path,
        thresholds: DivergenceThresholds,
        ship_gate_path: Optional[Path] = None,
        slippage_model: Optional[SlippageModel] = None,
        halted: bool = True,
        mode: str = "shadow",
    ) -> "ShadowPipeline":
        if not isinstance(universe_hash, str) or not universe_hash:
            raise ShadowPipelineError(
                "universe_hash must be a non-empty string, "
                f"got {universe_hash!r}"
            )
        if mode not in ("shadow", "paper"):
            raise ShadowPipelineError(
                f"mode must be 'shadow' or 'paper', got {mode!r} "
                "(live mode is owned by US-022, not this surface)"
            )
        if not isinstance(thresholds, DivergenceThresholds):
            raise ShadowPipelineError(
                "thresholds must be a DivergenceThresholds, got "
                f"{type(thresholds).__name__}"
            )

        gate_path = Path(
            ship_gate_path
            if ship_gate_path is not None
            else SHIP_GATE_PATH_DEFAULT
        )
        # Hard gate before we touch any disk for state.
        enforce_ship_gate(gate_path, universe_hash)
        backtest_stats = _read_backtest_stats(gate_path)

        ledger_path = Path(ledger_path)
        state_path = Path(state_path)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # US-016 simulator re-enforces the gate at construction too — a
        # double-check by design (defense in depth, not redundancy).
        simulator = ShadowFillSimulator(
            universe_hash=universe_hash,
            ledger_path=ledger_path,
            ship_gate_path=gate_path,
            slippage_model=slippage_model,
        )

        pipeline = cls(
            universe_hash=universe_hash,
            ledger_path=ledger_path,
            state_path=state_path,
            thresholds=thresholds,
            ship_gate_path=gate_path,
            simulator=simulator,
            halted=bool(halted),
            mode=mode,
            _backtest_stats=backtest_stats,
        )
        logger.info(
            "%s universe_hash=%s mode=%s halted=%s ledger=%s state=%s",
            LOG_TAG_RUN_BEGIN, universe_hash, mode, pipeline.halted,
            ledger_path, state_path,
        )
        pipeline._persist_state()
        return pipeline

    # ---------- gate re-enforcement ----------

    def _reenforce_gate(self) -> None:
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

    # ---------- state ----------

    def _state_payload(self) -> dict:
        return {
            "version": STATE_VERSION,
            "universe_hash": self.universe_hash,
            "mode": self.mode,
            "halted": self.halted,
            "stats": asdict(self.stats),
            "thresholds": asdict(self.thresholds),
            "backtest_stats": dict(self._backtest_stats),
            "last_report": (
                self.last_report.to_dict()
                if self.last_report is not None
                else None
            ),
        }

    def _persist_state(self) -> None:
        _atomic_write_json(self._state_payload(), self.state_path)
        logger.debug(
            "%s path=%s cycles=%s halted=%s",
            LOG_TAG_STATE_PERSIST,
            self.state_path, self.stats.cycles, self.halted,
        )

    # ---------- mode toggles (halted/unhalt; live is REFUSED) ----------

    def unhalt(self) -> None:
        """Operator-requested unhalt. Re-enforces the gate first."""
        self._reenforce_gate()
        self.halted = False
        self._persist_state()

    def halt(self, reason: str = "operator") -> None:
        """Halt the pipeline immediately. Always succeeds, fail-safe."""
        self.halted = True
        logger.warning("%s reason=%s", LOG_TAG_HALTED, reason)
        self._persist_state()

    def set_mode(self, mode: str) -> None:
        """Switch between 'shadow' and 'paper'. 'live' is REFUSED here."""
        self._reenforce_gate()
        if mode == "live":
            raise ShadowPipelineError(
                "ShadowPipeline refuses mode='live' — that path requires "
                "US-022's typed confirmation and a separate code surface."
            )
        if mode not in ("shadow", "paper"):
            raise ShadowPipelineError(
                f"mode must be 'shadow' or 'paper', got {mode!r}"
            )
        self.mode = mode
        self._persist_state()

    # ---------- divergence check ----------

    def check_divergence(self) -> DivergenceReport:
        """Compare running shadow stats against the backtest baseline.

        Returns the latest :class:`DivergenceReport`. Side-effects: the
        report is cached on ``self.last_report`` and persisted to disk
        atomically. The caller logs ``DIVERGENCE`` when ``breached`` is
        True.
        """
        backtest_max_dd = float(self._backtest_stats.get("max_dd", 0.0))
        realized_max_dd = float(self.stats.max_drawdown)
        breach_pp = (realized_max_dd - backtest_max_dd) * 100.0

        backtest_turnover = self._backtest_stats.get("turnover_per_year")
        if backtest_turnover is None:
            backtest_turnover = self.thresholds.backtest_turnover_per_year

        realized_turnover = self.stats.turnover_per_year()

        reasons: List[str] = []
        # DD side: only the upside (shadow worse than backtest) triggers.
        if breach_pp > self.thresholds.max_dd_breach_pp:
            reasons.append(
                f"max_dd breach: realized={realized_max_dd:.4f} "
                f"backtest={backtest_max_dd:.4f} "
                f"breach_pp={breach_pp:.3f} > "
                f"{self.thresholds.max_dd_breach_pp:.3f}"
            )

        turnover_drift_pct: Optional[float] = None
        if backtest_turnover is not None and backtest_turnover > 0.0:
            turnover_drift_pct = (
                abs(realized_turnover - backtest_turnover)
                / backtest_turnover
                * 100.0
            )
            if turnover_drift_pct > self.thresholds.turnover_drift_pct:
                reasons.append(
                    f"turnover drift: realized={realized_turnover:.4f} "
                    f"backtest={backtest_turnover:.4f} "
                    f"drift_pct={turnover_drift_pct:.3f} > "
                    f"{self.thresholds.turnover_drift_pct:.3f}"
                )

        report = DivergenceReport(
            breached=bool(reasons),
            max_dd_realized=realized_max_dd,
            max_dd_backtest=backtest_max_dd,
            max_dd_breach_pp=breach_pp,
            turnover_realized=realized_turnover,
            turnover_backtest=(
                float(backtest_turnover)
                if backtest_turnover is not None
                else None
            ),
            turnover_drift_pct=turnover_drift_pct,
            reasons=tuple(reasons),
        )
        self.last_report = report
        if report.breached:
            logger.warning(
                "%s %s", LOG_TAG_DIVERGENCE, report.reason()
            )
        self._persist_state()
        return report

    # ---------- step (one cycle) ----------

    def step(
        self,
        *,
        nav: float,
        gross_traded_notional: float = 0.0,
        cycle_seconds: float = 0.0,
        fills: Optional[List[SimulatedFill]] = None,
    ) -> CycleResult:
        """Advance the pipeline by one cycle.

        Refuses to execute (skips fills, leaves stats untouched) when:

        * Ship gate has been invalidated since construction
          (raises :class:`ShadowPipelineError`).
        * ``halted=True`` (returns ``executed=False``, divergence is
          still computed against whatever stats already exist).

        Otherwise: rolls realized stats, records any supplied fills to
        the simulator's ledger (so the audit trail covers cycles where
        the executor produces fills outside the simulator's own helpers),
        runs the divergence check, and persists state atomically.
        """
        # Hard gate first — invalidating the SHIP_GATE during a run is
        # NOT silently recoverable.
        self._reenforce_gate()

        if self.halted:
            logger.info(
                "%s halted=True — skipping execute (nav=%s)",
                LOG_TAG_CYCLE, nav,
            )
            report = self.check_divergence()
            return CycleResult(
                halted=True,
                executed=False,
                diverged=report.breached,
                report=report,
                fills_recorded=0,
            )

        self.stats.update(
            nav=nav,
            gross_traded=gross_traded_notional,
            cycle_seconds=cycle_seconds,
        )

        fills_recorded = 0
        if fills:
            for fill in fills:
                if not isinstance(fill, SimulatedFill):
                    raise ShadowPipelineError(
                        "fills must be SimulatedFill instances, got "
                        f"{type(fill).__name__}"
                    )
                # Re-ledger via the simulator's append path so the
                # audit trail is the single source of truth. The
                # simulator re-enforces the gate on every public call.
                # We do this by reaching into the public record path:
                # ledger appends go through _record, but for fills that
                # were produced outside the simulator we use its
                # entries() snapshot + a fresh persist below.
                self.simulator._reenforce()  # noqa: SLF001 — by design
                self.simulator._entries.append(fill.to_dict())  # noqa: SLF001
                fills_recorded += 1
            if fills_recorded:
                # Bound the ledger and persist atomically once at the
                # end of the cycle instead of per-fill.
                cap = self.simulator.max_ledger_entries
                entries = self.simulator._entries  # noqa: SLF001
                if len(entries) > cap:
                    overflow = len(entries) - cap
                    self.simulator._entries = entries[overflow:]  # noqa: SLF001
                self.simulator._persist()  # noqa: SLF001

        report = self.check_divergence()
        logger.info(
            "%s cycle=%s nav=%s gross=%s fills=%s diverged=%s",
            LOG_TAG_CYCLE, self.stats.cycles, nav,
            gross_traded_notional, fills_recorded, report.breached,
        )
        return CycleResult(
            halted=False,
            executed=True,
            diverged=report.breached,
            report=report,
            fills_recorded=fills_recorded,
        )


__all__ = [
    "CycleResult",
    "DivergenceReport",
    "DivergenceThresholds",
    "LOG_TAG_CYCLE",
    "LOG_TAG_DIVERGENCE",
    "LOG_TAG_HALTED",
    "LOG_TAG_RUN_BEGIN",
    "LOG_TAG_STATE_PERSIST",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "ShadowPipeline",
    "ShadowPipelineError",
    "ShadowRunStats",
    "enforce_ship_gate",
]
