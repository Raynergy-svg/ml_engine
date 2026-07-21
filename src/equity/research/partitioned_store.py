"""Predicate-pushed equity panels and immutable PIT-universe snapshots."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.evidence.canonical import canonical_bytes
from src.training.performance.partition_store import (
    ImmutableMonthlyParquetStore,
    bounded_batches,
    file_sha256,
)


class EquityResearchStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.monthly = ImmutableMonthlyParquetStore(self.root / "monthly")

    def partition_prices(self, prices: pd.DataFrame) -> tuple[Path, ...]:
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("price panel requires a DatetimeIndex")
        outputs: list[Path] = []
        for ticker in prices.columns:
            frame = prices[[ticker]].rename(columns={ticker: "adjusted_close"})
            outputs.extend(self.monthly.write("prices", str(ticker), frame))
        return tuple(outputs)

    def read_prices(
        self,
        *,
        tickers: Sequence[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        long = self.monthly.query(
            "prices",
            entities=tickers,
            start=start,
            end=end,
            columns=("adjusted_close",),
        )
        if long.empty:
            return pd.DataFrame(columns=list(tickers))
        return long.pivot(index="event_time", columns="entity", values="adjusted_close").sort_index()

    def write_pit_universe(self, universe: pd.DataFrame, *, as_of: str) -> Path:
        if "ticker" not in universe.columns:
            raise ValueError("PIT universe requires a ticker column")
        ordered = universe.sort_values("ticker").reset_index(drop=True)
        identity = hashlib.sha256(
            canonical_bytes(
                {
                    "as_of": as_of,
                    "tickers": ordered["ticker"].astype(str).tolist(),
                    "columns": list(ordered.columns),
                    "content_digest": hashlib.sha256(
                        pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
                    ).hexdigest(),
                }
            )
        ).hexdigest()
        directory = self.root / "universes" / f"as_of={as_of}"
        destination = directory / f"{identity}.parquet"
        if destination.exists():
            return destination
        existing = sorted(directory.glob("*.parquet")) if directory.exists() else []
        if existing:
            raise ValueError(
                f"PIT universe for {as_of} is already frozen as {existing[0].name}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".parquet.tmp")
        ordered.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
        destination.with_suffix(".manifest.json").write_bytes(
            canonical_bytes(
                {
                    "schema": "axiom_pit_universe_v1",
                    "as_of": as_of,
                    "rows": len(ordered),
                    "sha256": file_sha256(destination),
                }
            )
        )
        return destination

    def read_pit_universe(
        self, *, as_of: str, tickers: Sequence[str] | None = None
    ) -> pd.DataFrame:
        candidates = sorted((self.root / "universes").glob("as_of=*"))
        eligible = [path for path in candidates if path.name.removeprefix("as_of=") <= as_of]
        if not eligible:
            raise FileNotFoundError(f"no PIT universe exists on or before {as_of}")
        files = sorted(eligible[-1].glob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"expected one immutable PIT universe in {eligible[-1]}")
        filters = [("ticker", "in", list(tickers))] if tickers else None
        return pd.read_parquet(files[0], filters=filters)

    def write_rebalance_partition(
        self, frame: pd.DataFrame, *, rebalance_date: str
    ) -> Path:
        """Persist one bounded rebalance cross-section, never a full-history panel."""
        if "ticker" not in frame.columns:
            raise ValueError("rebalance partition requires a ticker column")
        ordered = frame.sort_values("ticker").reset_index(drop=True)
        identity = hashlib.sha256(
            canonical_bytes(
                {
                    "rebalance_date": rebalance_date,
                    "tickers": ordered["ticker"].astype(str).tolist(),
                    "rows": ordered.to_dict(orient="records"),
                }
            )
        ).hexdigest()
        directory = self.root / "rebalances" / f"date={rebalance_date}"
        destination = directory / f"{identity}.parquet"
        if destination.exists():
            return destination
        directory.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".parquet.tmp")
        ordered.to_parquet(tmp, index=False)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, destination)
        return destination

    def read_rebalance_partition(
        self, *, rebalance_date: str, tickers: Sequence[str] | None = None
    ) -> pd.DataFrame:
        files = sorted(
            (self.root / "rebalances" / f"date={rebalance_date}").glob("*.parquet")
        )
        if len(files) != 1:
            raise ValueError(f"expected one rebalance partition for {rebalance_date}")
        filters = [("ticker", "in", list(tickers))] if tickers else None
        return pd.read_parquet(files[0], filters=filters)


def iter_filing_worker_batches(
    items: Iterable[object], *, batch_size: int = 32
) -> Iterable[tuple[object, ...]]:
    return bounded_batches(items, batch_size=batch_size)
