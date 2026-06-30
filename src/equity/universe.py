"""US-002 — Point-in-time single-stock universe builder.

Builds a date-indexed membership table for a liquid US large-cap universe.
Membership is **point-in-time**: a ticker only appears on dates D where, as of
D, it (a) has been listed for at least ``min_history_days`` sessions and (b)
ranked in the top-N by 20-day average dollar-volume on the most recent
reconstitution date ≤ D.

Reconstitution rules (documented; deterministic):

* **Cadence**: monthly, evaluated on the first US business day of each calendar
  month (``rebalance_freq='MS'``). Configurable.
* **Eligibility on date R**:
  - ``min_history_days`` business sessions of price data already exist (the
    bot would not yet have trained features on a brand-new IPO).
  - ``ADV20$ = mean(close * volume) over the trailing 20 sessions`` is finite
    and > 0 on R.
* **Selection on R**: rank eligible tickers by ADV20$ descending; take top
  ``top_n`` deterministically (tie-break by ticker symbol, ascending).
* **Carry-forward**: between reconstitution dates the set is constant. A name
  that delists/loses data mid-month stays in the named set until the next
  reconstitution but downstream `members_on(D)` only returns it if its price
  series covered D (so a delisting cleanly removes it without future leak).

The deployed snapshot is serialized atomically (``.tmp`` + ``os.replace``) and
fingerprinted via ``universe_hash`` = SHA-256 of the sorted
``(date_iso, ticker)`` tuples. US-005 (SHIP_GATE) binds to this hash so the
runtime harvester refuses to run if the universe drifted from what was
validated.

No mocks, no bare excepts, no future leak — see the unit test in
``tests/test_equity_universe.py`` which asserts a known later-IPO ticker is
**excluded** from membership on a pre-IPO date.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import pandas as pd

from . import EQUITY_PIPELINE_VERSION

logger = logging.getLogger(__name__)


DEFAULT_TOP_N = 100
DEFAULT_LOOKBACK_DAYS = 20
DEFAULT_MIN_HISTORY_DAYS = 252
DEFAULT_REBALANCE_FREQ = "MS"  # Month Start; pandas date_range freq alias.


class UniverseError(RuntimeError):
    """Raised when universe construction inputs are malformed or unusable."""


@dataclass(frozen=True)
class UniverseRules:
    """Documented reconstitution rules — persisted with every snapshot."""

    top_n: int = DEFAULT_TOP_N
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    ranking_metric: str = "adv20_dollar"  # mean(close*volume) over `lookback_days`
    tie_break: str = "ticker_ascending"

    def to_dict(self) -> Dict[str, object]:
        return {
            "top_n": int(self.top_n),
            "lookback_days": int(self.lookback_days),
            "min_history_days": int(self.min_history_days),
            "rebalance_freq": str(self.rebalance_freq),
            "ranking_metric": self.ranking_metric,
            "tie_break": self.tie_break,
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    """Date-indexed point-in-time membership of the equity universe.

    Attributes:
        membership: DataFrame indexed by date (UTC midnight) with one bool
            column per ticker. ``True`` means the ticker was *eligible AND*
            ranked in the top-N at the most recent reconstitution on/before
            that date AND had price data for the date itself.
        rules: the reconstitution rules used.
        built_at: UTC ISO-8601 timestamp at construction.
        pipeline_version: ``EQUITY_PIPELINE_VERSION`` at construction.
        universe_hash: SHA-256 of the sorted ``(date_iso, ticker)`` membership
            tuples. Deterministic across runs given identical inputs.
    """

    membership: pd.DataFrame
    rules: UniverseRules
    built_at: str
    pipeline_version: str
    universe_hash: str
    reconstitution_dates: Tuple[str, ...] = field(default_factory=tuple)

    def members_on(self, date) -> List[str]:
        """Return the sorted tickers in the universe on ``date`` (UTC date)."""
        ts = _to_utc_midnight(date)
        if ts not in self.membership.index:
            # No row for date D → snap to the most recent date <= D within
            # the membership window; if none, return empty (pre-history).
            idx = self.membership.index
            prior = idx[idx <= ts]
            if len(prior) == 0:
                return []
            ts = prior.max()
        row = self.membership.loc[ts]
        return sorted([t for t, present in row.items() if bool(present)])

    def as_records(self) -> List[Dict[str, object]]:
        """Yield one record per (date, ticker) pair for serialization."""
        records: List[Dict[str, object]] = []
        for ts, row in self.membership.iterrows():
            date_iso = pd.Timestamp(ts).date().isoformat()
            for ticker in sorted(row.index):
                if bool(row[ticker]):
                    records.append({"date": date_iso, "ticker": ticker})
        return records

    def to_json(self) -> str:
        payload = {
            "pipeline_version": self.pipeline_version,
            "built_at": self.built_at,
            "universe_hash": self.universe_hash,
            "rules": self.rules.to_dict(),
            "reconstitution_dates": list(self.reconstitution_dates),
            "members": self.as_records(),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def save(self, path: Path) -> Path:
        """Atomically persist the snapshot to ``path`` (JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "UniverseSnapshot":
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise UniverseError(
                f"Universe snapshot unreadable at {path}: {exc}"
            ) from exc
        try:
            rules = UniverseRules(**payload["rules"])
            members = payload["members"]
            recon = tuple(payload.get("reconstitution_dates", ()))
            built_at = payload["built_at"]
            pipeline_version = payload["pipeline_version"]
            uhash = payload["universe_hash"]
        except (KeyError, TypeError) as exc:
            raise UniverseError(
                f"Universe snapshot schema mismatch at {path}: {exc}"
            ) from exc

        tickers = sorted({m["ticker"] for m in members})
        dates = sorted({m["date"] for m in members})
        if not dates:
            # An empty snapshot is legal but produces an empty frame.
            membership = pd.DataFrame(columns=tickers)
            membership.index = pd.DatetimeIndex([], tz="UTC", name="date")
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=True), name="date")
            membership = pd.DataFrame(False, index=idx, columns=tickers)
            for record in members:
                ts = pd.Timestamp(record["date"], tz="UTC").normalize()
                membership.at[ts, record["ticker"]] = True

        recomputed = compute_universe_hash(membership)
        if recomputed != uhash:
            raise UniverseError(
                f"Universe snapshot at {path}: hash mismatch "
                f"(stored={uhash}, recomputed={recomputed})"
            )
        return cls(
            membership=membership,
            rules=rules,
            built_at=built_at,
            pipeline_version=pipeline_version,
            universe_hash=uhash,
            reconstitution_dates=recon,
        )


def _to_utc_midnight(date) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _require_ohlcv_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise UniverseError(f"{ticker}: prices must be a DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise UniverseError(f"{ticker}: prices index must be DatetimeIndex")
    cols = {c.lower() for c in df.columns}
    if "close" not in cols or "volume" not in cols:
        raise UniverseError(
            f"{ticker}: prices must have 'close' and 'volume' columns"
        )
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index = out.index.normalize()
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def _build_dollar_volume_panel(
    prices_by_ticker: Mapping[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(close, dollar_volume)`` wide panels aligned on the union of dates."""
    if not prices_by_ticker:
        raise UniverseError("prices_by_ticker is empty")
    closes: Dict[str, pd.Series] = {}
    dvols: Dict[str, pd.Series] = {}
    for ticker, raw in prices_by_ticker.items():
        df = _require_ohlcv_frame(raw, ticker)
        closes[ticker] = df["close"].astype(float)
        dvols[ticker] = (df["close"].astype(float) * df["volume"].astype(float))
    close_panel = pd.DataFrame(closes).sort_index()
    dvol_panel = pd.DataFrame(dvols).sort_index()
    return close_panel, dvol_panel


def _reconstitution_dates(
    trading_dates: pd.DatetimeIndex,
    freq: str,
) -> pd.DatetimeIndex:
    """Pick reconstitution dates: first trading date on/after each period start."""
    if len(trading_dates) == 0:
        return pd.DatetimeIndex([], tz="UTC", name="date")
    start = trading_dates.min()
    end = trading_dates.max()
    try:
        period_starts = pd.date_range(
            start=start.normalize(),
            end=end.normalize(),
            freq=freq,
            tz="UTC",
        )
    except (ValueError, TypeError) as exc:
        raise UniverseError(
            f"Invalid rebalance_freq={freq!r}: {exc}"
        ) from exc

    if len(period_starts) == 0 or period_starts[0] > start:
        period_starts = period_starts.insert(0, start.normalize())

    picked: List[pd.Timestamp] = []
    seen = set()
    for ps in period_starts:
        eligible = trading_dates[trading_dates >= ps]
        if len(eligible) == 0:
            continue
        d = eligible.min()
        if d not in seen:
            picked.append(d)
            seen.add(d)
    return pd.DatetimeIndex(sorted(picked), tz="UTC", name="date")


def _eligible_tickers_on(
    recon_date: pd.Timestamp,
    close_panel: pd.DataFrame,
    dvol_panel: pd.DataFrame,
    rules: UniverseRules,
) -> List[str]:
    """Return tickers ranked by ADV20$ that pass eligibility on ``recon_date``."""
    window_end = recon_date
    window = close_panel.loc[:window_end]
    if len(window) < rules.min_history_days:
        # Pre-history for everyone — no one qualifies.
        return []

    last_idx = window.index.max()
    history_len = window.notna().sum()  # per-ticker count of finite closes
    eligible_mask = history_len >= rules.min_history_days

    dvol_window = dvol_panel.loc[:window_end].tail(rules.lookback_days)
    if len(dvol_window) < rules.lookback_days:
        # Not enough lookback yet at this reconstitution point.
        return []

    adv = dvol_window.mean(axis=0, skipna=False)

    # Also require the ticker to have a finite close on the recon date itself,
    # so we never "carry forward" through gaps from delisted names.
    last_close = close_panel.loc[last_idx]

    candidates = []
    for ticker in adv.index:
        if not eligible_mask.get(ticker, False):
            continue
        value = adv[ticker]
        if not pd.notna(value) or value <= 0:
            continue
        if not pd.notna(last_close.get(ticker)):
            continue
        candidates.append((ticker, float(value)))

    if not candidates:
        return []

    # Sort by ADV desc, ticker asc tie-break.
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in candidates[: rules.top_n]]


def build_universe(
    prices_by_ticker: Mapping[str, pd.DataFrame],
    *,
    top_n: int = DEFAULT_TOP_N,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    asof: Optional[pd.Timestamp] = None,
) -> UniverseSnapshot:
    """Construct a point-in-time UniverseSnapshot.

    Args:
        prices_by_ticker: mapping ticker → daily OHLCV DataFrame. Must include
            ``close`` and ``volume`` columns and a DatetimeIndex.
        top_n: max universe size at each reconstitution.
        lookback_days: trailing window for the ADV20$ ranking metric.
        min_history_days: minimum business-day history before a name is eligible.
        rebalance_freq: pandas date-range frequency alias (e.g. 'MS' monthly,
            'QS' quarterly).
        asof: optional cutoff; rows after this date are dropped (default: union
            of all input dates).

    Returns:
        UniverseSnapshot with ``membership`` indexed by every trading date in
        the union of inputs (up to ``asof``) and ``universe_hash`` set.

    Raises:
        UniverseError: on empty input, malformed frames, or invalid frequencies.
    """
    if top_n <= 0:
        raise UniverseError(f"top_n must be > 0, got {top_n}")
    if lookback_days <= 0:
        raise UniverseError(f"lookback_days must be > 0, got {lookback_days}")
    if min_history_days < 0:
        raise UniverseError(
            f"min_history_days must be >= 0, got {min_history_days}"
        )

    rules = UniverseRules(
        top_n=top_n,
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        rebalance_freq=rebalance_freq,
    )

    close_panel, dvol_panel = _build_dollar_volume_panel(prices_by_ticker)

    if asof is not None:
        cutoff = _to_utc_midnight(asof)
        close_panel = close_panel.loc[:cutoff]
        dvol_panel = dvol_panel.loc[:cutoff]

    trading_dates = close_panel.index
    if len(trading_dates) == 0:
        raise UniverseError("no trading dates after applying `asof` filter")

    recon_dates = _reconstitution_dates(trading_dates, rebalance_freq)

    selected_per_recon: Dict[pd.Timestamp, List[str]] = {}
    for rd in recon_dates:
        selected_per_recon[rd] = _eligible_tickers_on(
            rd, close_panel, dvol_panel, rules
        )

    tickers = sorted(close_panel.columns)
    membership = pd.DataFrame(
        False, index=trading_dates, columns=tickers
    )

    if not recon_dates.empty:
        current_set: List[str] = []
        recon_iter = iter(recon_dates)
        next_recon = next(recon_iter, None)
        for ts in trading_dates:
            while next_recon is not None and ts >= next_recon:
                current_set = selected_per_recon[next_recon]
                next_recon = next(recon_iter, None)
            if not current_set:
                continue
            close_row = close_panel.loc[ts]
            for ticker in current_set:
                if pd.notna(close_row.get(ticker)):
                    membership.at[ts, ticker] = True

    universe_hash = compute_universe_hash(membership)
    built_at = datetime.now(timezone.utc).isoformat()
    recon_iso = tuple(
        pd.Timestamp(d).date().isoformat() for d in recon_dates
    )

    return UniverseSnapshot(
        membership=membership,
        rules=rules,
        built_at=built_at,
        pipeline_version=EQUITY_PIPELINE_VERSION,
        universe_hash=universe_hash,
        reconstitution_dates=recon_iso,
    )


def compute_universe_hash(membership: pd.DataFrame) -> str:
    """SHA-256 over the sorted ``(date_iso, ticker)`` membership tuples."""
    sha = hashlib.sha256()
    if membership.empty:
        sha.update(b"empty-universe")
        return sha.hexdigest()
    sha.update(b"v1\n")
    for ts in sorted(membership.index):
        date_iso = pd.Timestamp(ts).date().isoformat()
        row = membership.loc[ts]
        members = sorted(
            ticker for ticker in row.index if bool(row[ticker])
        )
        if not members:
            continue
        sha.update(date_iso.encode("ascii"))
        sha.update(b"|")
        sha.update(",".join(members).encode("ascii"))
        sha.update(b"\n")
    return sha.hexdigest()
