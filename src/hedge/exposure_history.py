"""Exposure HISTORY capture — the persistence bridge between the hedge layer
and future model training. SHADOW / ANALYSIS-ONLY (same contract as the rest
of ``src.hedge``): reads on-disk artifacts, appends JSONL rows, never touches
a broker, a gate, the halt, or the environment pin.

Why this module exists (2026-07-08 training-infra reconciliation): the
risk-target pre-registration (docs/prereg-risk-target-vol-drawdown-2026-07-08.md)
had to EXCLUDE exposure / hedge / cost features because
``portfolio_exposure.py`` is a point-in-time function with no historical
snapshot store to replay against — the same reason tick-derived features were
deferred until ``tick_capture`` accumulates months of history. This module is
the exposure-plane equivalent of ``tick_capture``: it snapshots the live OANDA
FX trend-lane book into currency/correlation exposure buckets once per capture
tick, so that after enough history accumulates there is an honestly-joinable
time series for the P2 training round. No fabricated historical join, no
backfill — history starts the day this runs and grows forward only.

Inputs (all read-only, all already produced by the live lane):
  - ``trained_data/oanda/account_state.json`` — written every cycle by
    ``snapshot_account_state`` (positions: instrument, net_units, stop_loss).
  - ``trained_data/ticks/{PAIR}/YYYY/MM/DD.parquet`` — live tick capture
    (started 2026-07-07); the latest mid per pair supplies conversion rates.

Output: append-only ``trained_data/hedge/exposure_history.jsonl``. One row
per distinct account-state snapshot (deduped on the source file's mtime).

Exposure basis: per-position we record BOTH
  - ``notional_home`` — |units| x base_to_home_rate (the roadmap §5.1
    currency-bucket decomposition basis; used for the netting below), and
  - ``risk_home_r`` — |units| x |mid - stop_loss| x quote_to_home_rate (the
    same R-based dollar-risk basis the runtime exposure/bucket-cap gates use),
    ``None`` with a recorded reason when no stop or no mid is available.
Bucket nets are computed on the NOTIONAL basis via the Phase-2 engine
(``net_currency_exposure`` / ``net_correlation_bucket_exposure``) — reused,
not forked.

Fail-closed contract (mirrors the package): a missing/corrupt/stale account
state refuses the whole capture; a position whose rate cannot be derived is
skipped WITH a recorded reason (never silently zeroed); if every open
position fails to resolve, the row is refused rather than appending an
all-fail husk. A genuinely flat book DOES append (zero exposure is real
information).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.equity.fx_rates import base_to_home_rate
from src.equity.trend_risk_gates import quote_to_home_rate
from src.hedge.hedged_shadow_lane import _append_jsonl
from src.hedge.portfolio_exposure import (
    ExposurePosition,
    net_correlation_bucket_exposure,
    net_currency_exposure,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCOUNT_STATE_PATH = REPO_ROOT / "trained_data" / "oanda" / "account_state.json"
TICKS_ROOT = REPO_ROOT / "trained_data" / "ticks"
EXPOSURE_HISTORY_PATH = REPO_ROOT / "trained_data" / "hedge" / "exposure_history.jsonl"

EXPOSURE_HISTORY_SCHEMA_VERSION = "2026-07-08-v1"

# Freshness bounds. account_state is rewritten every live cycle (hourly loop),
# so 2h staleness means the lane is down — refuse rather than log a zombie
# book. Tick staleness of 1h bounds how old a conversion rate may be; over
# weekends the stream goes quiet, positions fail rate-resolution and the
# capture refuses (documented behavior, not a bug — a static weekend book adds
# no training-relevant variance).
DEFAULT_MAX_STATE_AGE_S = 2 * 3600
DEFAULT_MAX_TICK_AGE_S = 3600


@dataclass(frozen=True)
class PositionExposureRecord:
    """One decomposed open position as persisted into the history row."""
    instrument: str
    direction: str
    net_units: float
    notional_home: Optional[float]
    risk_home_r: Optional[float]
    stop_loss: Optional[float]
    unrealized_pl: Optional[float]
    fail_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "net_units": self.net_units,
            "notional_home": self.notional_home,
            "risk_home_r": self.risk_home_r,
            "stop_loss": self.stop_loss,
            "unrealized_pl": self.unrealized_pl,
            "fail_reasons": list(self.fail_reasons),
        }


def _safe_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _load_account_state(path: Path, *, max_age_s: float,
                        now: datetime) -> Optional[Tuple[dict, datetime]]:
    """Read + freshness-check the account snapshot. None = refuse capture."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError as exc:
        logger.warning("exposure_history: account state unreadable at %s: %s", path, exc)
        return None
    if (now - mtime).total_seconds() > max_age_s:
        logger.warning("exposure_history: account state STALE (%s, mtime %s, max %ss) — refusing",
                       path, mtime.isoformat(), max_age_s)
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("exposure_history: account state corrupt at %s: %s", path, exc)
        return None
    if not isinstance(state, dict):
        logger.warning("exposure_history: account state not a dict at %s — refusing", path)
        return None
    return state, mtime


def _latest_tick_mid(pair_dir: Path, *, max_age_s: float,
                     now: datetime) -> Optional[float]:
    """Latest fresh mid for one pair from today's (or yesterday's, for the
    UTC-midnight boundary) tick parquet. None = no fresh rate (fail-closed)."""
    import pandas as pd  # local: keep module import light for non-capture users

    for day_offset in (0, 1):
        day = (now - timedelta(days=day_offset)).astimezone(timezone.utc)
        f = pair_dir / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f, columns=["mid"])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("exposure_history: unreadable tick file %s: %s", f, exc)
            continue
        if df.empty:
            continue
        ts = df.index[-1]
        ts = ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts
        if (now - ts.to_pydatetime()).total_seconds() > max_age_s:
            return None  # newest data for this pair is already too old
        mid = _safe_float(df["mid"].iloc[-1])
        return mid if mid and mid > 0 else None
    return None


def _build_last_prices(ticks_root: Path, *, max_age_s: float,
                       now: datetime) -> Dict[str, float]:
    """Fresh mid per captured pair — the ``last_prices`` dict the fx_rates /
    trend_risk_gates conversion helpers expect."""
    prices: Dict[str, float] = {}
    if not ticks_root.is_dir():
        return prices
    for pair_dir in sorted(p for p in ticks_root.iterdir() if p.is_dir()
                           and not p.name.startswith("_")):
        mid = _latest_tick_mid(pair_dir, max_age_s=max_age_s, now=now)
        if mid is not None:
            prices[pair_dir.name] = mid
    return prices


def _decompose_position(pos: dict, last_prices: Dict[str, float]
                        ) -> Optional[PositionExposureRecord]:
    """One raw account-state position -> exposure record. None = not a
    position at all (zero units / malformed instrument-free entry)."""
    instrument = pos.get("instrument")
    units = _safe_float(pos.get("net_units"))
    if not instrument or not units:
        return None
    direction = "long" if units > 0 else "short"
    reasons: List[str] = []

    notional_home = None
    b2h = base_to_home_rate(instrument, last_prices)
    if b2h is None:
        reasons.append(f"no_base_to_home_rate:{instrument}")
    else:
        notional_home = abs(units) * b2h

    risk_home_r = None
    stop_loss = _safe_float(pos.get("stop_loss"))
    mid = _safe_float(last_prices.get(instrument))
    if stop_loss is None:
        reasons.append(f"no_stop_loss:{instrument}")
    elif mid is None:
        reasons.append(f"no_mid_for_r_distance:{instrument}")
    else:
        q2h = quote_to_home_rate(instrument, last_prices)
        if q2h is None:
            reasons.append(f"no_quote_to_home_rate:{instrument}")
        else:
            risk_home_r = abs(units) * abs(mid - stop_loss) * q2h

    return PositionExposureRecord(
        instrument=instrument, direction=direction, net_units=units,
        notional_home=notional_home, risk_home_r=risk_home_r,
        stop_loss=stop_loss, unrealized_pl=_safe_float(pos.get("unrealized_pl")),
        fail_reasons=tuple(reasons))


def _last_recorded_mtime(out_path: Path) -> Optional[str]:
    """source_mtime_utc of the newest history row (dedupe key)."""
    try:
        with open(out_path, "rb") as fh:
            tail = fh.read().splitlines()
    except OSError:
        return None
    for raw in reversed(tail):
        if not raw.strip():
            continue
        try:
            return json.loads(raw).get("source_mtime_utc")
        except (json.JSONDecodeError, AttributeError):
            return None  # corrupt tail row: don't guess, allow the append
    return None


def capture_exposure_snapshot(
    *,
    account_state_path: Path = ACCOUNT_STATE_PATH,
    ticks_root: Path = TICKS_ROOT,
    out_path: Path = EXPOSURE_HISTORY_PATH,
    max_state_age_s: float = DEFAULT_MAX_STATE_AGE_S,
    max_tick_age_s: float = DEFAULT_MAX_TICK_AGE_S,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """Capture one exposure-history row. Returns the appended row, or None
    when the capture refused (stale/corrupt state, duplicate snapshot, or a
    non-flat book in which not a single position resolved)."""
    now = now or datetime.now(timezone.utc)

    loaded = _load_account_state(account_state_path, max_age_s=max_state_age_s, now=now)
    if loaded is None:
        return None
    state, mtime = loaded
    mtime_iso = mtime.isoformat()

    if _last_recorded_mtime(out_path) == mtime_iso:
        logger.debug("exposure_history: snapshot %s already recorded — skipping", mtime_iso)
        return None

    last_prices = _build_last_prices(ticks_root, max_age_s=max_tick_age_s, now=now)

    raw_positions = state.get("positions") or []
    records: List[PositionExposureRecord] = []
    for pos in raw_positions:
        if not isinstance(pos, dict):
            continue
        rec = _decompose_position(pos, last_prices)
        if rec is not None:
            records.append(rec)

    resolved = [r for r in records if r.notional_home is not None]
    if records and not resolved:
        logger.warning("exposure_history: %d open positions, 0 resolved a home rate "
                       "(reasons: %s) — refusing an all-fail row",
                       len(records), [r.fail_reasons for r in records])
        return None

    exposure_positions = [
        ExposurePosition(asset_class="fx", instrument=r.instrument,
                         direction=r.direction, risk_home=r.notional_home,
                         source="oanda_account_state")
        for r in resolved
    ]
    currency_net = net_currency_exposure(exposure_positions)
    correlation_net = net_correlation_bucket_exposure(exposure_positions)

    rates_used = {
        inst: last_prices[inst]
        for inst in sorted({r.instrument for r in records} & set(last_prices))
    }
    row = {
        "schema_version": EXPOSURE_HISTORY_SCHEMA_VERSION,
        "source": "oanda_fx_trend_lane",
        "captured_at_utc": now.isoformat(),
        "source_mtime_utc": mtime_iso,
        "nav": _safe_float(state.get("nav")),
        "margin_used": _safe_float(state.get("margin_used")),
        "margin_available": _safe_float(state.get("margin_available")),
        "open_trade_count": state.get("open_trade_count"),
        "unrealized_pl": _safe_float(state.get("unrealized_pl")),
        "exposure_basis": "notional_home",
        "positions": [r.to_dict() for r in records],
        "net_currency_notional_home": dict(currency_net.net),
        "net_correlation_buckets_notional_home": dict(correlation_net.net),
        "resolved_notional_total": currency_net.resolved_risk_total,
        "fail_reasons": sorted({*currency_net.fail_reasons, *correlation_net.fail_reasons,
                                *(reason for r in records for reason in r.fail_reasons)}),
        "skipped_instruments": sorted({r.instrument for r in records
                                       if r.notional_home is None}),
        "rates_used": rates_used,
        # Hard-rule attestation (roadmap §11): this row is analysis output only.
        "paper_only": True,
        "runtime_allowed": False,
    }
    # _append_jsonl logs-and-swallows write OSErrors (but its mkdir can still
    # raise); catch here and verify the row actually landed, so a failed write
    # is reported as a refusal, not a phantom success (verifier gap G1).
    try:
        _append_jsonl(out_path, row)
    except OSError as exc:
        logger.error("exposure_history: cannot create/append %s: %s", out_path, exc)
        return None
    if _last_recorded_mtime(out_path) != mtime_iso:
        logger.error("exposure_history: append did NOT persist to %s — "
                     "reporting capture failure", out_path)
        return None
    logger.info("exposure_history: captured %d positions (%d resolved) -> %s "
                "[nav=%s, buckets=%d, fails=%d]",
                len(records), len(resolved), out_path, row["nav"],
                len(row["net_currency_notional_home"]), len(row["fail_reasons"]))
    return row
