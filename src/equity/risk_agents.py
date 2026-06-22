"""US-008 — Lean deterministic portfolio-risk agent layer for the harvester.

This is the harvester's risk gate stack. It is intentionally TINY and TYPED:
six lean, pure functions plus a thin orchestrator. Every function returns the
existing :class:`src.scanner.agents._team.AgentVerdict` shape so downstream
telemetry stays uniform across the FX scanner and the equity harvester.

Gates (in evaluation order)
---------------------------

1. ``drawdown_guardian`` — runs EVERY cycle. Reads ``peak_nav`` and ``nav``
   off :class:`PortfolioState`. Below ``dd_soft`` is a pass-through; in the
   ``[dd_soft, dd_hard)`` band the verdict carries a ``degross_factor`` in
   ``(0, 1)``; at/above ``dd_hard`` it sets ``block_trade=True`` and the
   orchestrator trips the circuit-breaker (halt). The semantics mirror the
   causal overlay in :func:`src.equity.backtest.overlay` but applied as a
   runtime gate, not a backtest scalar.
2. ``vol_targeter`` — annualised realised vol from the recent equity-curve
   tail vs ``target_vol``; returns a multiplier in ``[0, max_lev]`` via the
   metadata so the orchestrator can compose it with the drawdown factor.
3. ``gross_exposure_cap`` + ``per_name_exposure_cap`` — hard caps on
   ``sum(|w|)`` and ``max(|w|)`` against ``RiskAgentConfig``. Mirrors the
   :func:`src.factor.portfolio.enforce_guards` caps so the runtime gate and
   the backtest can NEVER disagree on what is in-bounds.
4. ``concentration_check`` — Herfindahl + top-K share against a cap, plus
   max pairwise correlation when a correlation matrix is supplied.
5. ``adv_participation`` — for each name with a non-zero traded notional
   (``|target_w - current_w| * NAV``), require participation rate
   ``trade_notional / adv`` <= ``adv_participation_cap``. ``adv`` is a Series
   mapping ticker -> 20-day average dollar-volume. Names traded against zero
   / missing / negative ADV BLOCK the cycle (fail-loud — silent participation
   into an illiquid name is exactly the failure mode this gate exists for).
6. ``rebalance_gate`` — cooldown enforcement: at least ``cooldown_days`` must
   have elapsed since ``state.last_rebalance_asof``, AND no plan can be
   in-flight (``state.active_plan_open``).

Composite
---------

:class:`RiskAgentLayer` holds the config + the ship-gate guard. Construction
is fail-closed (missing/malformed/false/hash-mismatch SHIP_GATE.json all
raise). :meth:`evaluate` runs the six gates in order and returns a
:class:`RiskDecision` with the verdicts, ``block_trade``, the composite
``degross_factor``, and a halt flag. :meth:`write_state` persists
``PortfolioState`` atomically via ``tempfile.mkstemp`` + :func:`os.replace`.

Hard rules
----------

* Claude-free / deterministic hot path. No network, no LLM, no random.
* Real classes + real disk via ``tmp_path`` in tests. No ``unittest.mock``.
* No bare ``except``. Narrow exceptions, logged + surfaced.
* Hummingbot Apache-2.0 attribution (PRD §notes): no source copied verbatim.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.scanner.agents._team import AgentVerdict

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1
ANNUALIZATION_FACTOR = 252  # daily harvester


class RiskAgentError(RuntimeError):
    """Raised on malformed inputs or ship-gate refusal."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAgentConfig:
    """Thresholds for the lean risk-agent layer.

    Defaults mirror the harvester's validated grid in
    :mod:`src.equity.ship_gate` (``DEFAULT_TARGET_VOL=0.12``,
    ``DEFAULT_DD_SOFT=0.10``, ``DEFAULT_DD_HARD=0.20``, ``max_lev=1.0``).
    """

    gross_cap: float = 1.0
    per_name_cap: float = 0.10
    target_vol: float = 0.12
    max_lev: float = 1.0
    vol_lookback: int = 21
    dd_soft: float = 0.10
    dd_hard: float = 0.20
    concentration_cap: float = 0.30  # top-K share of gross
    concentration_top_k: int = 3
    max_pairwise_correlation: float = 0.95
    adv_participation_cap: float = 0.05  # <=5% of 20-day ADV per rebalance
    adv_lookback_days: int = 20  # informational; ADV is precomputed externally
    cooldown_days: int = 1

    def __post_init__(self) -> None:
        if self.gross_cap <= 0:
            raise RiskAgentError(f"gross_cap must be > 0, got {self.gross_cap}")
        if self.per_name_cap <= 0 or self.per_name_cap > self.gross_cap + 1e-9:
            raise RiskAgentError(
                f"per_name_cap must be in (0, gross_cap], got "
                f"{self.per_name_cap} vs gross_cap={self.gross_cap}"
            )
        if self.target_vol <= 0:
            raise RiskAgentError(
                f"target_vol must be > 0, got {self.target_vol}"
            )
        if self.max_lev <= 0:
            raise RiskAgentError(f"max_lev must be > 0, got {self.max_lev}")
        if self.vol_lookback < 2:
            raise RiskAgentError(
                f"vol_lookback must be >= 2, got {self.vol_lookback}"
            )
        if not (0 <= self.dd_soft < self.dd_hard):
            raise RiskAgentError(
                f"require 0 <= dd_soft < dd_hard, got soft={self.dd_soft}, "
                f"hard={self.dd_hard}"
            )
        if not (0 < self.concentration_cap <= 1.0):
            raise RiskAgentError(
                f"concentration_cap must be in (0, 1], got "
                f"{self.concentration_cap}"
            )
        if self.concentration_top_k < 1:
            raise RiskAgentError(
                f"concentration_top_k must be >= 1, got "
                f"{self.concentration_top_k}"
            )
        if not (0 < self.max_pairwise_correlation <= 1.0):
            raise RiskAgentError(
                f"max_pairwise_correlation must be in (0, 1], got "
                f"{self.max_pairwise_correlation}"
            )
        if not (0 < self.adv_participation_cap <= 1.0):
            raise RiskAgentError(
                f"adv_participation_cap must be in (0, 1], got "
                f"{self.adv_participation_cap}"
            )
        if self.cooldown_days < 0:
            raise RiskAgentError(
                f"cooldown_days must be >= 0, got {self.cooldown_days}"
            )


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------


@dataclass
class PortfolioState:
    """The live portfolio snapshot the risk layer reads on every cycle.

    Mutable on purpose: the runtime updates ``nav``, ``peak_nav``,
    ``recent_returns`` (deque-style tail), ``current_weights``, and
    ``last_rebalance_asof`` as the executor reports fills. :meth:`to_dict` is
    the atomic-write payload; :meth:`from_dict` is the restart path.
    """

    nav: float
    peak_nav: float
    current_weights: Dict[str, float] = field(default_factory=dict)
    recent_returns: List[float] = field(default_factory=list)
    last_rebalance_asof: Optional[str] = None
    active_plan_open: bool = False
    halted: bool = False
    version: int = STATE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "nav": float(self.nav),
            "peak_nav": float(self.peak_nav),
            "current_weights": {
                str(k): float(v) for k, v in self.current_weights.items()
            },
            "recent_returns": [float(r) for r in self.recent_returns],
            "last_rebalance_asof": self.last_rebalance_asof,
            "active_plan_open": bool(self.active_plan_open),
            "halted": bool(self.halted),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioState":
        if not isinstance(payload, Mapping):
            raise RiskAgentError("state payload must be a JSON object")
        try:
            nav = float(payload["nav"])
            peak = float(payload["peak_nav"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RiskAgentError(
                f"state payload missing nav/peak_nav: {exc}"
            ) from exc
        weights = payload.get("current_weights") or {}
        if not isinstance(weights, Mapping):
            raise RiskAgentError("state.current_weights must be a mapping")
        recent = payload.get("recent_returns") or []
        if not isinstance(recent, list):
            raise RiskAgentError("state.recent_returns must be a list")
        last = payload.get("last_rebalance_asof")
        return cls(
            nav=nav,
            peak_nav=peak,
            current_weights={str(k): float(v) for k, v in weights.items()},
            recent_returns=[float(r) for r in recent],
            last_rebalance_asof=str(last) if last else None,
            active_plan_open=bool(payload.get("active_plan_open", False)),
            halted=bool(payload.get("halted", False)),
            version=int(payload.get("version", STATE_VERSION)),
        )

    def drawdown(self) -> float:
        """Current drawdown in ``[0, 1]`` against the high-watermark."""
        if self.peak_nav <= 0 or self.nav < 0:
            return 1.0
        return max(0.0, 1.0 - (self.nav / self.peak_nav))


# ---------------------------------------------------------------------------
# Composite decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskDecision:
    """Composite output of :meth:`RiskAgentLayer.evaluate`."""

    verdicts: List[AgentVerdict]
    block_trade: bool
    degross_factor: float
    halt: bool
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_trade": bool(self.block_trade),
            "degross_factor": float(self.degross_factor),
            "halt": bool(self.halt),
            "reasons": list(self.reasons),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------------------
# Gate implementations (lean, pure)
# ---------------------------------------------------------------------------


_BASE_WEIGHTS: Dict[str, float] = {
    "drawdown_guardian": 1.30,
    "vol_targeter": 1.00,
    "gross_exposure_cap": 1.20,
    "per_name_exposure_cap": 1.10,
    "concentration_check": 1.00,
    "adv_participation": 1.15,
    "rebalance_gate": 0.90,
}


def _verdict(
    name: str,
    *,
    passed: bool,
    score: float,
    reason: str,
    reason_code: str,
    block_trade: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentVerdict:
    return AgentVerdict(
        name=name,
        score=max(0.0, min(1.0, float(score))),
        passed=bool(passed),
        weight=float(_BASE_WEIGHTS.get(name, 1.0)),
        reason=str(reason),
        reason_code=str(reason_code),
        block_trade=bool(block_trade),
        metadata=dict(metadata or {}),
    )


def drawdown_guardian(
    state: PortfolioState, config: RiskAgentConfig
) -> AgentVerdict:
    """Runs EVERY cycle. Below ``dd_soft``: pass. In ``[soft, hard)``: degross
    (verdict carries ``degross_factor`` in metadata). At/above ``dd_hard``:
    block + trip the circuit-breaker (``halt=True`` in metadata).
    """
    dd = state.drawdown()
    if dd >= config.dd_hard:
        return _verdict(
            "drawdown_guardian",
            passed=False,
            score=0.0,
            reason=(
                f"drawdown {dd:.4f} >= dd_hard {config.dd_hard:.4f} — "
                "tripping circuit-breaker (halt)"
            ),
            reason_code="drawdown_hard_breach_halt",
            block_trade=True,
            metadata={
                "drawdown": float(dd),
                "dd_soft": float(config.dd_soft),
                "dd_hard": float(config.dd_hard),
                "degross_factor": 0.0,
                "halt": True,
            },
        )
    if dd >= config.dd_soft:
        # Linear de-gross between soft (factor=1) and hard (factor=0).
        span = max(config.dd_hard - config.dd_soft, 1e-12)
        factor = max(0.0, min(1.0, 1.0 - (dd - config.dd_soft) / span))
        return _verdict(
            "drawdown_guardian",
            passed=True,
            score=float(factor),
            reason=(
                f"drawdown {dd:.4f} in soft band — degrossing to "
                f"factor={factor:.4f}"
            ),
            reason_code="drawdown_soft_band_degross",
            block_trade=False,
            metadata={
                "drawdown": float(dd),
                "dd_soft": float(config.dd_soft),
                "dd_hard": float(config.dd_hard),
                "degross_factor": float(factor),
                "halt": False,
            },
        )
    return _verdict(
        "drawdown_guardian",
        passed=True,
        score=1.0,
        reason=f"drawdown {dd:.4f} below soft band {config.dd_soft:.4f}",
        reason_code="drawdown_ok",
        metadata={
            "drawdown": float(dd),
            "dd_soft": float(config.dd_soft),
            "dd_hard": float(config.dd_hard),
            "degross_factor": 1.0,
            "halt": False,
        },
    )


def vol_targeter(
    state: PortfolioState, config: RiskAgentConfig
) -> AgentVerdict:
    """Annualised realised vol from the recent-returns tail.

    Insufficient history (< ``vol_lookback``) emits a pass-through factor of
    1.0 with reason_code ``vol_warmup`` — the harvester's overlay handles the
    warm-up window symmetrically (US-003).
    """
    tail = [float(r) for r in state.recent_returns[-config.vol_lookback:]]
    if len(tail) < config.vol_lookback:
        return _verdict(
            "vol_targeter",
            passed=True,
            score=1.0,
            reason=(
                f"insufficient history "
                f"({len(tail)} < {config.vol_lookback}) — warm-up factor 1.0"
            ),
            reason_code="vol_warmup",
            metadata={
                "realized_vol": 0.0,
                "target_vol": float(config.target_vol),
                "vol_factor": 1.0,
            },
        )
    arr = np.array(tail, dtype=float)
    if not np.all(np.isfinite(arr)):
        return _verdict(
            "vol_targeter",
            passed=False,
            score=0.0,
            reason="recent returns contain NaN/inf — refuse to size",
            reason_code="vol_inputs_nonfinite",
            block_trade=True,
            metadata={"target_vol": float(config.target_vol)},
        )
    sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    annualised = sigma * math.sqrt(ANNUALIZATION_FACTOR)
    if annualised <= 1e-9:
        # Zero realised vol -> cap at max_lev (don't divide by ~0).
        factor = float(config.max_lev)
    else:
        factor = max(0.0, min(config.max_lev, config.target_vol / annualised))
    return _verdict(
        "vol_targeter",
        passed=True,
        score=float(min(1.0, factor / config.max_lev)),
        reason=(
            f"realised_vol={annualised:.4f} target={config.target_vol:.4f} "
            f"factor={factor:.4f}"
        ),
        reason_code="vol_target_ok",
        metadata={
            "realized_vol": float(annualised),
            "target_vol": float(config.target_vol),
            "vol_factor": float(factor),
            "lookback": int(config.vol_lookback),
        },
    )


def _coerce_weights(
    weights: Mapping[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for ticker, raw in weights.items():
        if not isinstance(ticker, str) or not ticker:
            raise RiskAgentError(f"weight key must be non-empty str, got {ticker!r}")
        try:
            w = float(raw)
        except (TypeError, ValueError) as exc:
            raise RiskAgentError(
                f"weight for {ticker!r} not numeric: {raw!r}"
            ) from exc
        if not math.isfinite(w):
            raise RiskAgentError(
                f"weight for {ticker!r} non-finite: {w!r}"
            )
        out[ticker] = w
    return out


def gross_exposure_cap(
    target_weights: Mapping[str, float], config: RiskAgentConfig
) -> AgentVerdict:
    """Hard cap on ``sum(|w|)`` vs ``config.gross_cap``."""
    w = _coerce_weights(target_weights)
    gross = float(sum(abs(v) for v in w.values()))
    if gross > config.gross_cap + 1e-9:
        return _verdict(
            "gross_exposure_cap",
            passed=False,
            score=0.0,
            reason=(
                f"gross {gross:.4f} exceeds cap {config.gross_cap:.4f}"
            ),
            reason_code="gross_cap_exceeded",
            block_trade=True,
            metadata={"gross": gross, "gross_cap": float(config.gross_cap)},
        )
    return _verdict(
        "gross_exposure_cap",
        passed=True,
        score=float(1.0 - gross / max(config.gross_cap, 1e-12)),
        reason=f"gross {gross:.4f} within cap {config.gross_cap:.4f}",
        reason_code="gross_cap_ok",
        metadata={"gross": gross, "gross_cap": float(config.gross_cap)},
    )


def per_name_exposure_cap(
    target_weights: Mapping[str, float], config: RiskAgentConfig
) -> AgentVerdict:
    """Hard cap on ``max(|w_t|)`` vs ``config.per_name_cap``."""
    w = _coerce_weights(target_weights)
    if not w:
        return _verdict(
            "per_name_exposure_cap",
            passed=True,
            score=1.0,
            reason="empty target — nothing to size",
            reason_code="per_name_cap_empty",
            metadata={"max_name": "", "max_weight": 0.0},
        )
    max_ticker, max_weight = max(w.items(), key=lambda kv: abs(kv[1]))
    if abs(max_weight) > config.per_name_cap + 1e-9:
        return _verdict(
            "per_name_exposure_cap",
            passed=False,
            score=0.0,
            reason=(
                f"per-name |w| {abs(max_weight):.4f} on {max_ticker} exceeds "
                f"cap {config.per_name_cap:.4f}"
            ),
            reason_code="per_name_cap_exceeded",
            block_trade=True,
            metadata={
                "max_name": str(max_ticker),
                "max_weight": float(max_weight),
                "per_name_cap": float(config.per_name_cap),
            },
        )
    return _verdict(
        "per_name_exposure_cap",
        passed=True,
        score=float(1.0 - abs(max_weight) / max(config.per_name_cap, 1e-12)),
        reason=(
            f"per-name |w| {abs(max_weight):.4f} on {max_ticker} within cap "
            f"{config.per_name_cap:.4f}"
        ),
        reason_code="per_name_cap_ok",
        metadata={
            "max_name": str(max_ticker),
            "max_weight": float(max_weight),
            "per_name_cap": float(config.per_name_cap),
        },
    )


def concentration_check(
    target_weights: Mapping[str, float],
    config: RiskAgentConfig,
    *,
    correlations: Optional[pd.DataFrame] = None,
) -> AgentVerdict:
    """Top-K concentration of gross + max pairwise correlation."""
    w = _coerce_weights(target_weights)
    if not w:
        return _verdict(
            "concentration_check",
            passed=True,
            score=1.0,
            reason="empty target — nothing to concentrate",
            reason_code="concentration_empty",
            metadata={"top_k_share": 0.0},
        )
    gross = float(sum(abs(v) for v in w.values()))
    if gross <= 1e-12:
        return _verdict(
            "concentration_check",
            passed=True,
            score=1.0,
            reason="gross is zero — concentration trivially zero",
            reason_code="concentration_zero_gross",
            metadata={"top_k_share": 0.0},
        )
    weights_desc = sorted((abs(v) for v in w.values()), reverse=True)
    top_share = float(sum(weights_desc[: config.concentration_top_k]) / gross)
    # Skip top-K check when the book is smaller than top_K: by construction
    # top-K share = 1.0 there, which is "the book IS its own top-K" — not
    # concentration risk. Per-name cap already bounds single-name dominance.
    n_names = sum(1 for v in weights_desc if v > 1e-12)
    check_topk = n_names > config.concentration_top_k
    if check_topk and top_share > config.concentration_cap + 1e-9:
        return _verdict(
            "concentration_check",
            passed=False,
            score=0.0,
            reason=(
                f"top-{config.concentration_top_k} share {top_share:.4f} "
                f"exceeds cap {config.concentration_cap:.4f}"
            ),
            reason_code="concentration_topk_exceeded",
            block_trade=True,
            metadata={
                "top_k_share": top_share,
                "top_k": int(config.concentration_top_k),
                "concentration_cap": float(config.concentration_cap),
            },
        )

    max_corr = 0.0
    max_pair = ("", "")
    if correlations is not None:
        if not isinstance(correlations, pd.DataFrame):
            raise RiskAgentError(
                f"correlations must be DataFrame, got {type(correlations)!r}"
            )
        held = [t for t, v in w.items() if abs(v) > 1e-12]
        held_in_corr = [t for t in held if t in correlations.columns]
        if len(held_in_corr) >= 2:
            sub = correlations.loc[held_in_corr, held_in_corr]
            arr = sub.to_numpy(dtype=float, copy=True)
            np.fill_diagonal(arr, 0.0)
            if not np.all(np.isfinite(arr)):
                return _verdict(
                    "concentration_check",
                    passed=False,
                    score=0.0,
                    reason="correlation matrix contains NaN/inf",
                    reason_code="concentration_corr_nonfinite",
                    block_trade=True,
                    metadata={"top_k_share": top_share},
                )
            i, j = np.unravel_index(int(np.argmax(np.abs(arr))), arr.shape)
            max_corr = float(abs(arr[i, j]))
            max_pair = (str(held_in_corr[i]), str(held_in_corr[j]))
            if max_corr > config.max_pairwise_correlation + 1e-9:
                return _verdict(
                    "concentration_check",
                    passed=False,
                    score=0.0,
                    reason=(
                        f"pairwise corr {max_corr:.4f} on "
                        f"{max_pair[0]}/{max_pair[1]} exceeds "
                        f"{config.max_pairwise_correlation:.4f}"
                    ),
                    reason_code="concentration_corr_exceeded",
                    block_trade=True,
                    metadata={
                        "top_k_share": top_share,
                        "max_pairwise_corr": max_corr,
                        "max_corr_pair": list(max_pair),
                    },
                )

    score_share = top_share if check_topk else 0.0
    score = float(
        max(0.0, 1.0 - score_share / max(config.concentration_cap, 1e-12))
    )
    return _verdict(
        "concentration_check",
        passed=True,
        score=score,
        reason=(
            f"top-{config.concentration_top_k} share {top_share:.4f} within "
            f"cap (checked={check_topk}); max pairwise corr {max_corr:.4f}"
        ),
        reason_code="concentration_ok",
        metadata={
            "top_k_share": top_share,
            "max_pairwise_corr": max_corr,
            "max_corr_pair": list(max_pair),
            "n_names": int(n_names),
            "topk_checked": bool(check_topk),
        },
    )


def adv_participation(
    target_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
    nav: float,
    adv: Optional[Mapping[str, float]],
    config: RiskAgentConfig,
) -> AgentVerdict:
    """Per-name participation cap of trade delta vs 20-day dollar-ADV.

    For each name with non-zero traded notional (``|target_w - current_w| *
    NAV``), participation = ``trade_notional / adv[ticker]``. Names traded
    against zero / missing / negative / non-finite ADV BLOCK the cycle.
    """
    if not isinstance(nav, (int, float)) or not math.isfinite(float(nav)):
        raise RiskAgentError(f"nav must be a finite number, got {nav!r}")
    if float(nav) <= 0:
        return _verdict(
            "adv_participation",
            passed=False,
            score=0.0,
            reason=f"NAV {nav!r} is not positive — cannot size",
            reason_code="adv_nav_nonpositive",
            block_trade=True,
            metadata={"nav": float(nav)},
        )
    target = _coerce_weights(target_weights)
    current = _coerce_weights(current_weights)
    tickers = set(target.keys()) | set(current.keys())
    deltas: Dict[str, float] = {}
    for t in tickers:
        delta = abs(target.get(t, 0.0) - current.get(t, 0.0))
        if delta > 1e-12:
            deltas[t] = delta
    if not deltas:
        return _verdict(
            "adv_participation",
            passed=True,
            score=1.0,
            reason="no traded delta — nothing to participate",
            reason_code="adv_no_delta",
            metadata={"max_participation": 0.0, "n_names_traded": 0},
        )

    if adv is None:
        return _verdict(
            "adv_participation",
            passed=False,
            score=0.0,
            reason="ADV panel is None but trades are pending — refuse",
            reason_code="adv_panel_missing",
            block_trade=True,
            metadata={"n_names_traded": len(deltas)},
        )

    worst_ticker = ""
    worst_part = 0.0
    breach_ticker: Optional[str] = None
    for ticker, delta in deltas.items():
        notional = float(delta) * float(nav)
        raw_adv = adv.get(ticker) if isinstance(adv, Mapping) else None
        try:
            adv_val = float(raw_adv) if raw_adv is not None else None
        except (TypeError, ValueError) as exc:
            raise RiskAgentError(
                f"adv[{ticker!r}] not numeric: {raw_adv!r}"
            ) from exc
        if adv_val is None or not math.isfinite(adv_val) or adv_val <= 0:
            return _verdict(
                "adv_participation",
                passed=False,
                score=0.0,
                reason=(
                    f"ADV for {ticker} is missing/non-positive/non-finite "
                    f"({raw_adv!r}) — cannot trade"
                ),
                reason_code="adv_value_invalid",
                block_trade=True,
                metadata={
                    "blocked_ticker": ticker,
                    "blocked_adv": raw_adv,
                    "blocked_delta": float(delta),
                },
            )
        part = notional / adv_val
        if part > worst_part:
            worst_part = float(part)
            worst_ticker = str(ticker)
        if part > config.adv_participation_cap + 1e-9 and breach_ticker is None:
            breach_ticker = ticker

    if breach_ticker is not None:
        return _verdict(
            "adv_participation",
            passed=False,
            score=0.0,
            reason=(
                f"participation {worst_part:.4f} on {worst_ticker} exceeds "
                f"cap {config.adv_participation_cap:.4f}"
            ),
            reason_code="adv_participation_exceeded",
            block_trade=True,
            metadata={
                "max_participation": float(worst_part),
                "worst_ticker": str(worst_ticker),
                "cap": float(config.adv_participation_cap),
                "n_names_traded": len(deltas),
            },
        )
    return _verdict(
        "adv_participation",
        passed=True,
        score=float(
            1.0 - worst_part / max(config.adv_participation_cap, 1e-12)
        ),
        reason=(
            f"max participation {worst_part:.4f} on {worst_ticker} within "
            f"cap {config.adv_participation_cap:.4f}"
        ),
        reason_code="adv_participation_ok",
        metadata={
            "max_participation": float(worst_part),
            "worst_ticker": str(worst_ticker),
            "cap": float(config.adv_participation_cap),
            "n_names_traded": len(deltas),
        },
    )


def rebalance_gate(
    state: PortfolioState, asof: pd.Timestamp, config: RiskAgentConfig
) -> AgentVerdict:
    """Cooldown + plan-in-flight gate."""
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tz is None:
        asof_ts = asof_ts.tz_localize("UTC")
    if state.active_plan_open:
        return _verdict(
            "rebalance_gate",
            passed=False,
            score=0.0,
            reason="active rebalance plan in flight — block new cycle",
            reason_code="rebalance_plan_active",
            block_trade=True,
            metadata={"active_plan_open": True},
        )
    if state.last_rebalance_asof is None:
        return _verdict(
            "rebalance_gate",
            passed=True,
            score=1.0,
            reason="no prior rebalance — first cycle allowed",
            reason_code="rebalance_first_cycle",
            metadata={"cooldown_days": int(config.cooldown_days)},
        )
    try:
        last_ts = pd.Timestamp(state.last_rebalance_asof)
    except (TypeError, ValueError) as exc:
        raise RiskAgentError(
            f"state.last_rebalance_asof unparseable: "
            f"{state.last_rebalance_asof!r}"
        ) from exc
    if last_ts.tz is None:
        last_ts = last_ts.tz_localize("UTC")
    elapsed_days = (asof_ts - last_ts) / pd.Timedelta(days=1)
    if elapsed_days < float(config.cooldown_days) - 1e-9:
        return _verdict(
            "rebalance_gate",
            passed=False,
            score=0.0,
            reason=(
                f"elapsed {elapsed_days:.4f}d < cooldown "
                f"{config.cooldown_days}d"
            ),
            reason_code="rebalance_cooldown_active",
            block_trade=True,
            metadata={
                "elapsed_days": float(elapsed_days),
                "cooldown_days": int(config.cooldown_days),
            },
        )
    return _verdict(
        "rebalance_gate",
        passed=True,
        score=1.0,
        reason=(
            f"elapsed {elapsed_days:.4f}d >= cooldown "
            f"{config.cooldown_days}d"
        ),
        reason_code="rebalance_ok",
        metadata={
            "elapsed_days": float(elapsed_days),
            "cooldown_days": int(config.cooldown_days),
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskAgentLayer:
    """Composite layer: ship-gate guard + 6 lean gates + composite decision.

    Construct via :func:`RiskAgentLayer.create` (does ship-gate guard).
    Calling :meth:`evaluate` always runs ``drawdown_guardian`` first and
    UNCONDITIONALLY (PRD AC: "drawdown guardian runs EVERY cycle"). The
    composite ``degross_factor`` is the product of the drawdown factor and
    the vol-targeter factor, clipped to ``[0, max_lev]``.
    """

    config: RiskAgentConfig
    ship_gate_path: Path
    universe_hash: str

    @classmethod
    def create(
        cls,
        config: RiskAgentConfig,
        ship_gate_path: Path,
        universe_hash: str,
    ) -> "RiskAgentLayer":
        if not isinstance(config, RiskAgentConfig):
            raise RiskAgentError(
                f"config must be RiskAgentConfig, got {type(config)!r}"
            )
        if not isinstance(universe_hash, str) or not universe_hash:
            raise RiskAgentError(
                "universe_hash must be a non-empty string"
            )
        _enforce_ship_gate(Path(ship_gate_path), universe_hash)
        return cls(
            config=config,
            ship_gate_path=Path(ship_gate_path),
            universe_hash=str(universe_hash),
        )

    def evaluate(
        self,
        *,
        asof: pd.Timestamp,
        target_weights: Mapping[str, float],
        state: PortfolioState,
        adv: Optional[Mapping[str, float]] = None,
        correlations: Optional[pd.DataFrame] = None,
    ) -> RiskDecision:
        """Run all gates and synthesise the composite decision."""
        verdicts: List[AgentVerdict] = []
        reasons: List[str] = []

        dd = drawdown_guardian(state, self.config)
        verdicts.append(dd)
        dd_factor = float(dd.metadata.get("degross_factor", 0.0))
        halt = bool(dd.metadata.get("halt", False))
        block = bool(dd.block_trade)
        if dd.block_trade:
            reasons.append(dd.reason_code)

        vt = vol_targeter(state, self.config)
        verdicts.append(vt)
        vol_factor = float(vt.metadata.get("vol_factor", 1.0))
        if vt.block_trade:
            block = True
            reasons.append(vt.reason_code)

        gx = gross_exposure_cap(target_weights, self.config)
        verdicts.append(gx)
        if gx.block_trade:
            block = True
            reasons.append(gx.reason_code)

        pn = per_name_exposure_cap(target_weights, self.config)
        verdicts.append(pn)
        if pn.block_trade:
            block = True
            reasons.append(pn.reason_code)

        cc = concentration_check(
            target_weights, self.config, correlations=correlations
        )
        verdicts.append(cc)
        if cc.block_trade:
            block = True
            reasons.append(cc.reason_code)

        ap = adv_participation(
            target_weights,
            state.current_weights,
            state.nav,
            adv,
            self.config,
        )
        verdicts.append(ap)
        if ap.block_trade:
            block = True
            reasons.append(ap.reason_code)

        rg = rebalance_gate(state, asof, self.config)
        verdicts.append(rg)
        if rg.block_trade:
            block = True
            reasons.append(rg.reason_code)

        composite = max(
            0.0, min(self.config.max_lev, dd_factor * vol_factor)
        )
        if halt:
            composite = 0.0
        return RiskDecision(
            verdicts=verdicts,
            block_trade=block,
            degross_factor=float(composite),
            halt=halt,
            reasons=reasons,
        )

    def write_state(self, state: PortfolioState, path: Path) -> Path:
        """Atomic JSON write — ``tempfile.mkstemp`` + :func:`os.replace`."""
        return _atomic_write_json(state.to_dict(), Path(path))


# ---------------------------------------------------------------------------
# Internals — ship-gate guard + atomic write
# ---------------------------------------------------------------------------


def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to construct unless the ship gate clears for THIS universe.

    Same fail-closed semantics as :func:`src.equity.harvester_strategy.
    _enforce_ship_gate`: missing file, malformed JSON, ``gate_pass`` not
    exactly ``True``, universe-hash mismatch all raise.
    """
    if not path.exists():
        raise RiskAgentError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RiskAgentError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RiskAgentError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    gate_pass = payload.get("gate_pass")
    if gate_pass is not True:
        raise RiskAgentError(
            f"ship gate refuses: gate_pass={gate_pass!r} at {path} "
            "(must be exactly True to unblock the risk layer)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise RiskAgentError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise RiskAgentError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomic JSON write — ``tempfile.mkstemp`` then :func:`os.replace`."""
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
    except OSError as exc:
        logger.error(
            "atomic write failed for %s: %s", path, exc, exc_info=True
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise
    return path


__all__ = [
    "ANNUALIZATION_FACTOR",
    "PortfolioState",
    "RiskAgentConfig",
    "RiskAgentError",
    "RiskAgentLayer",
    "RiskDecision",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "adv_participation",
    "concentration_check",
    "drawdown_guardian",
    "gross_exposure_cap",
    "per_name_exposure_cap",
    "rebalance_gate",
    "vol_targeter",
]


# Hummingbot Apache-2.0 attribution (PRD §notes): no source has been copied
# verbatim from Hummingbot into this module. The risk-gate math here is
# original to this repo. No freqtrade GPL.
