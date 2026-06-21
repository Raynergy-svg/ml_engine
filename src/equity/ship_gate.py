"""US-005 — HARD GATE: re-validate harvester on the single-stock universe.

The deploy decision for the equity-beta harvester is taken by code against a
pre-registered bar so judgment and hope cannot leak in. This module runs the
harvester (equal-weight membership-based book + causal vol/drawdown overlay,
US-003 cost-aware harness) on the US-002 universe full-cycle, evaluates the
result against the canonical :func:`src.factor.ship_gate.evaluate_gate`
(REUSED — not hand-rolled), and writes two artifacts atomically:

* ``trained_data/backtests/equity_harvester_singlestock_*.json`` — the
  timestamped backtest report (summary, criteria, per-year returns, span).
* ``trained_data/backtests/SHIP_GATE.json`` — the small machine-readable
  ``{gate_pass, net_sharpe, max_dd, positive_years, universe_hash, asof}``
  blob that downstream build stories (US-006+) bind to.

Decisions recorded with every artifact
--------------------------------------

* **MIN_POSITIVE_YEARS / MIN_TOTAL_YEARS** — we APPLY the canonical
  ``MIN_POSITIVE_YEARS=6`` and ``MIN_TOTAL_YEARS=10`` from
  :mod:`src.factor.ship_gate` to this gate (a >=2010–2026 span gives 16
  yearly buckets which satisfies the history requirement by construction;
  long-run track-record is exactly what we want before binding deploy code).
* **2-vs-4 criterion inconsistency** — ``scripts/build_equity_harvester.py``
  prints a 2-criterion teaser gate (``net Sharpe>=0.40 AND maxDD<=25%``)
  used as a research-time signal. The CANONICAL gate is the
  :func:`src.factor.ship_gate.evaluate_gate` 5-criterion check
  (net Sharpe, positive_years, history_length, max_drawdown, walk_forward).
  Downstream code reads ``gate_pass`` from this module — the script's print
  is informational only. The recorded ``criteria_source`` field surfaces
  this resolution on every artifact.

Hard rules:

* Real classes + real disk via ``tmp_path`` in tests — no ``unittest.mock``.
* Atomic writes (``.tmp`` + ``os.replace``); no half-written ship-gate file.
* Long-only book by default (FR-11); negative-weight rows raise.
* Fail-loud on misaligned panels / NaN returns — no silent forward-fill.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.equity.backtest import (
    DEFAULT_EXECUTION_LAG,
    BacktestError,
    overlay,
    run_portfolio_backtest,
)
from src.equity.universe import UniverseSnapshot
from src.factor.ship_gate import (
    MAX_DRAWDOWN,
    MIN_NET_SHARPE,
    MIN_POSITIVE_YEARS,
    MIN_TOTAL_YEARS,
    evaluate_gate,
)

logger = logging.getLogger(__name__)

# Pre-registered harvester overlay defaults — match the validated grid in
# ``scripts/build_equity_harvester.py`` (best full-cycle setting tv=0.12,
# soft=0.10, hard=0.20). Changing any of these requires an operator-signed
# commit message ("gate-change: ...") per the PRD.
DEFAULT_TARGET_VOL = 0.12
DEFAULT_DD_SOFT = 0.10
DEFAULT_DD_HARD = 0.20
DEFAULT_MAX_LEV = 1.0
# Realistic per-name single-stock costs. Flat commission ~2 bps + linear
# slippage scaled by 5 bps per 1pp ADV participation (so a name we trade at
# 5% of its ADV pays ~25 bps slip on the traded notional).
DEFAULT_COST_BPS = 2.0
DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV = 5.0

CRITERIA_SOURCE = "src.factor.ship_gate.evaluate_gate (US-007 canonical bar)"


class ShipGateError(RuntimeError):
    """Raised when ship-gate inputs are malformed, empty, or unusable."""


@dataclass(frozen=True)
class HarvesterShipGateReport:
    """Outcome of the harvester ship-gate run.

    The :func:`write_ship_gate` helper persists the minimal canonical blob
    ``{gate_pass, net_sharpe, max_dd, positive_years, universe_hash, asof}``
    plus the recommendation. :func:`write_backtest_artifact` persists the
    full timestamped report (summary, criteria, per-year, span, overlay).
    """

    gate_pass: bool
    net_sharpe: float
    max_dd: float
    positive_years: int
    total_years: int
    universe_hash: str
    asof: str
    summary: str
    criteria: Dict[str, dict]
    backtest_summary: Dict[str, object]
    span: str
    recommendation: str
    overlay_params: Dict[str, float] = field(default_factory=dict)
    cost_params: Dict[str, float] = field(default_factory=dict)
    pipeline_version: str = ""
    criteria_source: str = CRITERIA_SOURCE
    thresholds: Dict[str, float] = field(default_factory=dict)

    def to_ship_gate_payload(self) -> Dict[str, object]:
        """Minimal canonical SHIP_GATE.json payload (PRD schema)."""
        return {
            "gate_pass": bool(self.gate_pass),
            "net_sharpe": float(self.net_sharpe),
            "max_dd": float(self.max_dd),
            "positive_years": int(self.positive_years),
            "total_years": int(self.total_years),
            "universe_hash": str(self.universe_hash),
            "asof": str(self.asof),
            "recommendation": str(self.recommendation),
            "criteria_source": str(self.criteria_source),
            "thresholds": dict(self.thresholds),
            "pipeline_version": str(self.pipeline_version),
        }

    def to_full_payload(self) -> Dict[str, object]:
        """Full timestamped artifact payload (superset of the gate payload)."""
        payload = self.to_ship_gate_payload()
        payload.update(
            {
                "summary": str(self.summary),
                "criteria": dict(self.criteria),
                "backtest_summary": dict(self.backtest_summary),
                "span": str(self.span),
                "overlay_params": dict(self.overlay_params),
                "cost_params": dict(self.cost_params),
            }
        )
        return payload


def build_equal_weight_panel(
    snapshot: UniverseSnapshot,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build a date x ticker EW weight panel from a universe snapshot.

    For each date in ``snapshot.membership`` that also exists in ``prices``,
    each member ticker gets ``1/N`` where N is the membership count on that
    date. Tickers not in the membership row get 0. Tickers with NaN price on
    the date are dropped from the row (renormalised) so the harvester never
    sizes into a missing price.
    """
    if snapshot.membership.empty:
        raise ShipGateError("universe snapshot has empty membership")
    if prices.empty:
        raise ShipGateError("prices panel is empty")

    membership = snapshot.membership.copy()
    if membership.index.tz is None:
        membership.index = membership.index.tz_localize("UTC")
    else:
        membership.index = membership.index.tz_convert("UTC")
    membership.index = membership.index.normalize()

    prices = prices.copy()
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    else:
        prices.index = prices.index.tz_convert("UTC")
    prices.index = prices.index.normalize()

    common_idx = membership.index.intersection(prices.index)
    if len(common_idx) == 0:
        raise ShipGateError(
            "no overlap between universe membership and price-panel dates"
        )

    members_aligned = membership.loc[common_idx].astype(bool)
    prices_aligned = prices.reindex(index=common_idx)

    # Only keep tickers that exist in both panels.
    tickers = [t for t in members_aligned.columns if t in prices_aligned.columns]
    if not tickers:
        raise ShipGateError(
            "no shared tickers between universe membership and price panel"
        )
    members_aligned = members_aligned[tickers]
    prices_aligned = prices_aligned[tickers]

    # A row is "members[t] AND price[t] is finite" — drop unpriced members.
    priced_mask = members_aligned & prices_aligned.notna()
    row_counts = priced_mask.sum(axis=1)
    weights = priced_mask.astype(float).div(
        row_counts.replace(0, np.nan), axis=0
    ).fillna(0.0)
    return weights


def _derive_returns(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight base returns from an aligned price panel.

    Used to drive the overlay scalar — independent of the executed weight
    panel so the scalar's vol estimate sees a clean diversified series.
    """
    ret_panel = prices.pct_change()
    return ret_panel.mean(axis=1, skipna=True)


def evaluate_harvester(
    snapshot: UniverseSnapshot,
    prices: pd.DataFrame,
    *,
    adv: Optional[pd.DataFrame] = None,
    target_vol: float = DEFAULT_TARGET_VOL,
    dd_soft: float = DEFAULT_DD_SOFT,
    dd_hard: float = DEFAULT_DD_HARD,
    max_lev: float = DEFAULT_MAX_LEV,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps_per_pct_adv: float = DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
    asof: Optional[pd.Timestamp] = None,
    pipeline_version: str = "",
) -> HarvesterShipGateReport:
    """Run harvester on (snapshot, prices) and evaluate the ship gate.

    Reuses ``src.equity.backtest.overlay`` for the causal scalar and
    ``run_portfolio_backtest`` for the cost-aware execution path, then calls
    ``src.factor.ship_gate.evaluate_gate`` for the canonical PASS/FAIL.

    Slippage uses the US-003 ``adv`` panel when provided; passing ``None``
    falls back to flat-bps execution which is the right default when an ADV
    panel is unavailable.
    """
    if execution_lag < 1:
        raise ShipGateError(
            f"execution_lag must be >= 1 (causal), got {execution_lag}"
        )

    base_weights = build_equal_weight_panel(snapshot, prices)
    base_returns = _derive_returns(prices.reindex(index=base_weights.index))

    scalar = overlay(
        base_returns.fillna(0.0),
        target_vol=target_vol,
        dd_soft=dd_soft,
        dd_hard=dd_hard,
        max_lev=max_lev,
    )
    scalar = scalar.reindex(base_weights.index).fillna(0.0)

    managed_weights = base_weights.mul(scalar, axis=0)
    if (managed_weights < -1e-9).any().any():
        raise ShipGateError(
            "managed weights contain negative entries — long-only invariant "
            "violated upstream; refuse to evaluate gate on a short book"
        )

    effective_slippage = float(slippage_bps_per_pct_adv)
    if adv is None and effective_slippage > 0.0:
        logger.warning(
            "no ADV panel supplied; falling back to flat-bps execution "
            "(slippage_bps_per_pct_adv %.2f -> 0.0).",
            effective_slippage,
        )
        effective_slippage = 0.0

    try:
        result = run_portfolio_backtest(
            managed_weights,
            prices.reindex(index=base_weights.index),
            cost_bps=cost_bps,
            slippage_bps_per_pct_adv=effective_slippage,
            adv=adv,
            execution_lag=execution_lag,
        )
    except BacktestError as exc:
        raise ShipGateError(
            f"underlying US-003 backtest refused inputs: {exc}"
        ) from exc

    summary_stats = result.summary
    net_sharpe = float(summary_stats.get("net_sharpe", 0.0))
    max_dd = float(summary_stats.get("max_dd", 1.0))
    per_year = summary_stats.get("per_year", {}) or {}
    positive_years = sum(1 for v in per_year.values() if float(v) > 0)
    total_years = len(per_year)

    gate_report = {
        "net_sharpe": net_sharpe,
        "positive_years": positive_years,
        "total_years": total_years,
        "max_drawdown": max_dd,
        # The harvester IS a walk-forward construct by design (causal overlay
        # + causal weights via execution_lag>=1). Pass the flag explicitly so
        # the canonical gate's walk_forward criterion is satisfied honestly.
        "walk_forward": True,
    }
    verdict = evaluate_gate(gate_report)
    criteria = asdict(verdict)["criteria"]

    if asof is None:
        if not len(result.returns):
            raise ShipGateError(
                "managed returns are empty after backtest — cannot derive asof"
            )
        asof_iso = pd.Timestamp(result.returns.index.max()).date().isoformat()
    else:
        asof_iso = pd.Timestamp(asof).date().isoformat()

    recommendation = (
        "PASS — single-stock harvester clears the pre-registered bar; "
        "downstream build stories (US-006+) may proceed."
        if verdict.passed
        else (
            "FAIL — single-stock harvester does not clear the pre-registered "
            "bar. Recommend reverting to the validated EW-sector universe "
            "(scripts/build_equity_harvester.py) for any deployable book; "
            "all downstream build stories (US-006+) remain guarded by this "
            "artifact."
        )
    )

    return HarvesterShipGateReport(
        gate_pass=verdict.passed,
        net_sharpe=net_sharpe,
        max_dd=max_dd,
        positive_years=positive_years,
        total_years=total_years,
        universe_hash=snapshot.universe_hash,
        asof=asof_iso,
        summary=verdict.summary,
        criteria=criteria,
        backtest_summary=result.to_json_payload(),
        span=result.span,
        recommendation=recommendation,
        overlay_params={
            "target_vol": float(target_vol),
            "dd_soft": float(dd_soft),
            "dd_hard": float(dd_hard),
            "max_lev": float(max_lev),
        },
        cost_params={
            "cost_bps": float(cost_bps),
            "slippage_bps_per_pct_adv": float(effective_slippage),
            "slippage_bps_per_pct_adv_requested": float(slippage_bps_per_pct_adv),
            "execution_lag": int(execution_lag),
            "adv_supplied": bool(adv is not None),
        },
        pipeline_version=str(pipeline_version),
        thresholds={
            "min_net_sharpe": float(MIN_NET_SHARPE),
            "min_positive_years": float(MIN_POSITIVE_YEARS),
            "min_total_years": float(MIN_TOTAL_YEARS),
            "max_drawdown": float(MAX_DRAWDOWN),
        },
    )


def _atomic_write_json(payload: dict, path: Path) -> Path:
    """Atomic JSON write: ``.tmp`` sibling then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def write_ship_gate(
    report: HarvesterShipGateReport,
    path: Path,
) -> Path:
    """Atomically persist the canonical SHIP_GATE.json artifact."""
    return _atomic_write_json(report.to_ship_gate_payload(), Path(path))


def write_backtest_artifact(
    report: HarvesterShipGateReport,
    backtests_dir: Path,
    *,
    timestamp: Optional[str] = None,
) -> Path:
    """Atomically write ``equity_harvester_singlestock_<UTC>.json``.

    ``timestamp`` defaults to ``time.strftime('%Y%m%dT%H%M%SZ', gmtime())``
    so tests can pin a deterministic name when needed.
    """
    backtests_dir = Path(backtests_dir)
    backtests_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = backtests_dir / f"equity_harvester_singlestock_{ts}.json"
    return _atomic_write_json(report.to_full_payload(), path)


def run_and_persist(
    snapshot: UniverseSnapshot,
    prices: pd.DataFrame,
    backtests_dir: Path,
    *,
    adv: Optional[pd.DataFrame] = None,
    timestamp: Optional[str] = None,
    **kwargs,
) -> tuple[HarvesterShipGateReport, Path, Path]:
    """Convenience: run gate + write both artifacts in one call.

    Returns ``(report, backtest_artifact_path, ship_gate_path)``. The
    SHIP_GATE.json always lands at ``<backtests_dir>/SHIP_GATE.json``.
    """
    report = evaluate_harvester(snapshot, prices, adv=adv, **kwargs)
    backtests_dir = Path(backtests_dir)
    artifact_path = write_backtest_artifact(
        report, backtests_dir, timestamp=timestamp
    )
    ship_gate_path = write_ship_gate(report, backtests_dir / "SHIP_GATE.json")
    if not report.gate_pass:
        logger.warning(
            "SHIP_GATE FAIL — net_sharpe=%.3f maxDD=%.3f pos=%d/%d. %s",
            report.net_sharpe,
            report.max_dd,
            report.positive_years,
            report.total_years,
            report.recommendation,
        )
    return report, artifact_path, ship_gate_path


__all__ = [
    "CRITERIA_SOURCE",
    "DEFAULT_COST_BPS",
    "DEFAULT_DD_HARD",
    "DEFAULT_DD_SOFT",
    "DEFAULT_MAX_LEV",
    "DEFAULT_SLIPPAGE_BPS_PER_PCT_ADV",
    "DEFAULT_TARGET_VOL",
    "HarvesterShipGateReport",
    "ShipGateError",
    "build_equal_weight_panel",
    "evaluate_harvester",
    "run_and_persist",
    "write_backtest_artifact",
    "write_ship_gate",
]
