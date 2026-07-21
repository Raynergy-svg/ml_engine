"""Offline execution-cost model (P2, docs/ENGINEERING_BRAIN.md).

"Fill-quality / execution-cost model ... legitimate ML that pays regardless
of alpha, and the honest way to keep the harness cost model calibrated."
Builds realistic spread + slippage estimates from two read-only sources:

1. Historical ``ORDER_FILL`` transactions in
   ``trained_data/oanda/transactions.jsonl`` -- each fill carries OANDA's
   full order-book ladder (``fullPrice.bids``/``asks``, 7 depth levels) plus
   the actual fill ``price``, ``halfSpreadCost``, and requested-vs-filled
   ``units``. This is the ONLY source that can measure slippage (fill price
   vs mid at fill time) because it is the only source with an actual fill.
2. Captured ticks under ``trained_data/ticks/`` (once
   ``src.data.tick_capture`` has accumulated data) -- higher-resolution,
   fresher top-of-book spread samples than the (comparatively rare) fills,
   used to refresh the SPREAD half of the estimate when enough samples exist.

Advisory only, offline, read-only: this module never imports a broker
*client* or ``execution.py``/``engine.py``, never places an order, and never
mutates the live risk gate or backtest-harness cost defaults. It does import
``src.brokers.registry`` for its static pip-value table (a metadata lookup --
no network, no client, no order path; verified by AST import-graph trace: it
resolves only to ``instrument.py``). It writes one artifact
(``trained_data/cost_model/execution_cost_estimates.json``) that those
systems MAY choose to read later -- adoption is a separate, explicit decision
(P2 is "surface, don't start").

Degrades honestly: with too few fill samples or no tick coverage, an
instrument reports ``source="insufficient_data"`` with ``None`` estimates and
an explanatory note rather than fabricating a number.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSACTIONS_PATH = REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl"
DEFAULT_TICK_ROOT = REPO_ROOT / "trained_data" / "ticks"
DEFAULT_MODEL_PATH = REPO_ROOT / "trained_data" / "cost_model" / "execution_cost_estimates.json"

MIN_FILL_SAMPLES = 3      # below this, no fill-derived estimate is trustworthy
MIN_TICK_SAMPLES = 200    # below this, ticks don't override the fills-derived spread


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically (tmp + rename) per repo JSON-safety convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _pip_value(instrument: str) -> Optional[float]:
    """Price size of one pip for ``instrument``, or None if unknown.

    Deliberately returns None (never a guessed default) for instruments not
    in the registry -- an assumed pip size would be fabrication, same
    discipline as ``trend_journal_sync.py``'s entry_spread handling.
    """
    try:
        from src.brokers.registry import get_default_registry
        inst = get_default_registry().get_optional(instrument)
        return float(inst.pip_value) if inst is not None and inst.pip_value else None
    except Exception:
        return None


@dataclass(frozen=True)
class FillQuality:
    """Realized execution quality derived from one ORDER_FILL transaction."""

    instrument: str
    time: str  # ISO8601
    side: str  # "buy" | "sell"
    fill_price: float
    mid: float
    spread_raw: float          # top-of-book ask - bid, raw price units
    slippage_raw: float        # signed cost vs mid; positive = adverse to the trader
    half_spread_cost: Optional[float]
    notional_units: float
    requested_units: Optional[float]
    partial_fill: bool


@dataclass(frozen=True)
class CostEstimate:
    """Advisory execution-cost estimate for one instrument."""

    instrument: str
    source: str  # "fills" | "ticks+fills" | "insufficient_data"
    spread_raw: Optional[float]
    spread_pips: Optional[float]
    slippage_raw: Optional[float]
    slippage_pips: Optional[float]
    half_spread_cost_per_unit: Optional[float]
    sample_count_fills: int
    sample_count_ticks: int
    confidence: float  # 0..1 heuristic, purely a function of sample size
    as_of: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def iter_order_fills(transactions_path: Optional[Path] = None) -> Iterator[Dict[str, Any]]:
    """Yield ORDER_FILL transaction dicts, tolerant of corrupt lines."""
    path = transactions_path or DEFAULT_TRANSACTIONS_PATH
    if not path.exists():
        return
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "ORDER_FILL":
                yield obj


def fill_quality_from_transaction(tx: Dict[str, Any]) -> Optional[FillQuality]:
    """Derive realized spread/slippage from one ORDER_FILL. None if malformed."""
    try:
        full_price = tx.get("fullPrice") or {}
        bids = full_price.get("bids") or []
        asks = full_price.get("asks") or []
        if not bids or not asks:
            return None
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        mid = (bid + ask) / 2.0
        fill_price = float(tx["price"])
        units = float(tx.get("units", 0) or 0)
        requested_units = tx.get("requestedUnits")
        requested_units_f = float(requested_units) if requested_units is not None else None
        side = "buy" if units > 0 else "sell"
        spread_raw = ask - bid
        # Signed so positive always means "cost the trader money": a buy that
        # fills above mid, or a sell that fills below mid, is adverse.
        slippage_raw = (fill_price - mid) if side == "buy" else (mid - fill_price)
        half_spread_cost = tx.get("halfSpreadCost")
        half_spread_cost_f = abs(float(half_spread_cost)) if half_spread_cost is not None else None
        partial_fill = (
            requested_units_f is not None and abs(requested_units_f) > 0
            and abs(abs(units) - abs(requested_units_f)) > 1e-6
        )
        return FillQuality(
            instrument=tx["instrument"],
            time=tx.get("time", ""),
            side=side,
            fill_price=fill_price,
            mid=mid,
            spread_raw=spread_raw,
            slippage_raw=slippage_raw,
            half_spread_cost=half_spread_cost_f,
            notional_units=abs(units),
            requested_units=requested_units_f,
            partial_fill=partial_fill,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def load_fill_qualities(transactions_path: Optional[Path] = None) -> List[FillQuality]:
    out: List[FillQuality] = []
    for tx in iter_order_fills(transactions_path):
        fq = fill_quality_from_transaction(tx)
        if fq is not None:
            out.append(fq)
    return out


def fill_stats_by_instrument(qualities: List[FillQuality]) -> Dict[str, Dict[str, Any]]:
    """Group fill qualities by instrument and compute summary stats."""
    by_inst: Dict[str, List[FillQuality]] = {}
    for fq in qualities:
        by_inst.setdefault(fq.instrument, []).append(fq)

    stats: Dict[str, Dict[str, Any]] = {}
    for inst, rows in by_inst.items():
        spreads = np.array([r.spread_raw for r in rows], dtype=float)
        slippages = np.array([r.slippage_raw for r in rows], dtype=float)
        half_costs = [r.half_spread_cost / r.notional_units for r in rows
                      if r.half_spread_cost is not None and r.notional_units > 0]
        stats[inst] = {
            "n": len(rows),
            "spread_mean": float(np.mean(spreads)),
            "spread_median": float(np.median(spreads)),
            "slippage_mean": float(np.mean(slippages)),
            "slippage_median": float(np.median(slippages)),
            "slippage_std": float(np.std(slippages)) if len(slippages) > 1 else 0.0,
            "half_spread_cost_per_unit_mean": float(np.mean(half_costs)) if half_costs else None,
            "partial_fill_rate": float(np.mean([r.partial_fill for r in rows])),
        }
    return stats


def tick_spread_stats(
    instrument: str,
    tick_root: Optional[Path] = None,
    lookback_days: int = 30,
) -> Optional[Dict[str, Any]]:
    """Spread distribution from captured ticks over the last ``lookback_days``.

    Returns None (not a fabricated zero) when there is no tick coverage for
    this instrument yet -- tick_capture.py is newly activated, so sparse or
    empty coverage during the initial accumulation window is expected and
    must be reported honestly, not silently backfilled.
    """
    root = (tick_root or DEFAULT_TICK_ROOT) / instrument
    if not root.exists():
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    frames = []
    for pq_path in root.glob("*/*/*.parquet"):
        try:
            day = int(pq_path.stem)
            month = int(pq_path.parent.name)
            year = int(pq_path.parent.parent.name)
            part_date = datetime(year, month, day, tzinfo=timezone.utc)
        except (ValueError, OSError):
            continue
        if part_date < cutoff - timedelta(days=1):
            continue
        try:
            frames.append(pd.read_parquet(pq_path, columns=["bid", "ask"]))
        except Exception as read_err:
            logger.warning("Skipping unreadable tick partition %s: %s", pq_path, read_err)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return None
    spread = (df["ask"] - df["bid"]).to_numpy(dtype=float)
    spread = spread[np.isfinite(spread) & (spread >= 0)]
    if spread.size == 0:
        return None
    return {
        "n": int(spread.size),
        "spread_mean": float(np.mean(spread)),
        "spread_median": float(np.median(spread)),
        "spread_p90": float(np.percentile(spread, 90)),
    }


def estimate_execution_cost(
    instrument: str,
    *,
    transactions_path: Optional[Path] = None,
    tick_root: Optional[Path] = None,
    lookback_days: int = 30,
    fill_qualities: Optional[List[FillQuality]] = None,
    min_fill_samples: int = MIN_FILL_SAMPLES,
    min_tick_samples: int = MIN_TICK_SAMPLES,
) -> CostEstimate:
    """Build one instrument's advisory cost estimate.

    Slippage always comes from fills (ticks carry no fill-price signal).
    Spread prefers ticks when there are enough fresh samples, else falls back
    to the fills' own spread distribution. Confidence is a simple, honest
    function of sample size -- not a fitted/tuned parameter.
    """
    as_of = datetime.now(timezone.utc).isoformat()
    qualities = fill_qualities if fill_qualities is not None else load_fill_qualities(transactions_path)
    inst_fills = [q for q in qualities if q.instrument == instrument]
    fstats = fill_stats_by_instrument(inst_fills).get(instrument)
    tstats = tick_spread_stats(instrument, tick_root=tick_root, lookback_days=lookback_days)

    notes: List[str] = []
    pip = _pip_value(instrument)
    if pip is None:
        notes.append("unknown pip_value for instrument; raw price units only")

    n_fills = fstats["n"] if fstats else 0
    n_ticks = tstats["n"] if tstats else 0

    if n_fills < min_fill_samples and (tstats is None or n_ticks < min_tick_samples):
        notes.append(
            f"insufficient data: {n_fills} fill(s) (need {min_fill_samples}), "
            f"{n_ticks} tick(s) (need {min_tick_samples})"
        )
        return CostEstimate(
            instrument=instrument, source="insufficient_data",
            spread_raw=None, spread_pips=None, slippage_raw=None, slippage_pips=None,
            half_spread_cost_per_unit=None, sample_count_fills=n_fills, sample_count_ticks=n_ticks,
            confidence=0.0, as_of=as_of, notes=notes,
        )

    # Spread: prefer fresh tick coverage when it clears the sample bar.
    if tstats is not None and n_ticks >= min_tick_samples:
        spread_raw = tstats["spread_mean"]
        source = "ticks+fills" if n_fills >= min_fill_samples else "ticks"
    elif fstats is not None:
        spread_raw = fstats["spread_mean"]
        source = "fills"
    else:
        spread_raw = None
        source = "insufficient_data"

    if fstats is not None and n_fills >= min_fill_samples:
        slippage_raw = fstats["slippage_mean"]
        half_spread_cost_per_unit = fstats["half_spread_cost_per_unit_mean"]
    else:
        slippage_raw = None
        half_spread_cost_per_unit = None
        if n_fills:
            notes.append(f"only {n_fills} fill(s) (< {min_fill_samples}); slippage not estimated")
        else:
            notes.append("no ORDER_FILL history for this instrument; slippage not estimated")

    # Confidence: pure function of sample size, capped at 1.0. Sample sizes
    # required to sound quantitatively rigorous are calibration knobs, not
    # tunable free parameters here -- this is a coverage heuristic, not a fit.
    confidence = min(1.0, (n_fills / (min_fill_samples * 10.0)) * 0.6 + (n_ticks / (min_tick_samples * 10.0)) * 0.4)

    return CostEstimate(
        instrument=instrument,
        source=source,
        spread_raw=spread_raw,
        spread_pips=(spread_raw / pip) if (spread_raw is not None and pip) else None,
        slippage_raw=slippage_raw,
        slippage_pips=(slippage_raw / pip) if (slippage_raw is not None and pip) else None,
        half_spread_cost_per_unit=half_spread_cost_per_unit,
        sample_count_fills=n_fills,
        sample_count_ticks=n_ticks,
        confidence=round(confidence, 3),
        as_of=as_of,
        notes=notes,
    )


def build_cost_model(
    instruments: Optional[List[str]] = None,
    *,
    transactions_path: Optional[Path] = None,
    tick_root: Optional[Path] = None,
    lookback_days: int = 30,
) -> Dict[str, CostEstimate]:
    """Build (not persist) the full per-instrument cost model."""
    qualities = load_fill_qualities(transactions_path)
    if instruments is None:
        instruments = sorted({q.instrument for q in qualities})
    return {
        inst: estimate_execution_cost(
            inst, transactions_path=transactions_path, tick_root=tick_root,
            lookback_days=lookback_days, fill_qualities=qualities,
        )
        for inst in instruments
    }


def save_cost_model(model: Dict[str, CostEstimate], path: Optional[Path] = None) -> Path:
    """Persist estimates, merging into any existing artifact.

    A caller building only a subset of instruments (e.g. ``--instrument
    GBP_USD``) must not silently erase every other instrument's last-known
    estimate -- that would be a worse dead-write than the ones this repo's
    JSON safety gates exist to prevent.
    """
    out_path = path or DEFAULT_MODEL_PATH
    existing_estimates: Dict[str, Any] = {}
    if out_path.exists():
        try:
            existing_estimates = (json.loads(out_path.read_text()) or {}).get("estimates", {})
        except (json.JSONDecodeError, OSError):
            existing_estimates = {}
    existing_estimates.update({inst: est.to_dict() for inst, est in model.items()})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "advisory_only": True,
        "estimates": existing_estimates,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def validate_against_holdout(
    qualities: List[FillQuality],
    holdout_frac: float = 0.2,
    min_fill_samples: int = MIN_FILL_SAMPLES,
) -> Dict[str, Dict[str, Any]]:
    """Chronological train/holdout split per instrument: does the train-mean
    slippage predict held-out realized slippage better than assuming zero?

    Reports MAE for both the model and a naive zero-slippage baseline so the
    honest answer ("model doesn't beat naive on n=4") is visible, not hidden.
    Instruments with too few fills to split are reported with
    ``holdout_n=0`` rather than a fabricated comparison.
    """
    by_inst: Dict[str, List[FillQuality]] = {}
    for fq in qualities:
        by_inst.setdefault(fq.instrument, []).append(fq)

    results: Dict[str, Dict[str, Any]] = {}
    for inst, rows in by_inst.items():
        rows_sorted = sorted(rows, key=lambda r: r.time)
        n = len(rows_sorted)
        split = int(n * (1.0 - holdout_frac))
        train, holdout = rows_sorted[:split], rows_sorted[split:]
        if len(train) < min_fill_samples or not holdout:
            results[inst] = {"holdout_n": 0, "reason": f"insufficient data (n={n})"}
            continue
        train_mean_slippage = float(np.mean([r.slippage_raw for r in train]))
        actual = np.array([r.slippage_raw for r in holdout], dtype=float)
        model_mae = float(np.mean(np.abs(actual - train_mean_slippage)))
        naive_mae = float(np.mean(np.abs(actual)))  # naive baseline: predict zero slippage
        results[inst] = {
            "holdout_n": len(holdout),
            "train_n": len(train),
            "train_mean_slippage_raw": train_mean_slippage,
            "model_mae_raw": model_mae,
            "naive_zero_mae_raw": naive_mae,
            "model_beats_naive": model_mae < naive_mae,
        }
    return results
