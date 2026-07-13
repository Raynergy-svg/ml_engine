"""Canonical forward-only capture plane for AXIOM's seven priority data families."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from src.evidence.contracts import StrictContract
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import TrustStore

from .contracts import StorageDomain, StorageFormat
from .platform import (
    ControlStateConflict,
    DataIntegrityError,
    DataPlatform,
    PartitionPayload,
    _validate_container,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "axiom-data"
DEFAULT_CONTROL_ROOT = REPO_ROOT / "trained_data" / "forward_capture"
DEFAULT_P2_MIN_TRADING_DAYS = 60

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=240)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CaptureFamily(str, Enum):
    TICK_SPREAD = "tick_spread"
    EXPOSURE = "exposure"
    TRADE_ENTRY_CONTEXT = "trade_entry_context"
    FILL_LADDER = "fill_ladder"
    CRYPTO_MARKET = "crypto_market"
    TRACK_B_FILING = "track_b_filing"
    TRACK_B_REBALANCE = "track_b_rebalance"
    HEDGE_FORWARD = "hedge_forward"


class CaptureMode(str, Enum):
    FORWARD = "forward"
    BACKFILL = "backfill"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("capture timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        raise ValueError("source event time is required")
    if len(text) == 10:
        return datetime.combine(date.fromisoformat(text), time.min, tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return _utc(parsed)


class ForwardCaptureRecord(StrictContract):
    capture_id: Sha256Digest
    family: CaptureFamily
    source_key: Identifier
    source_event_time_utc: datetime
    observed_at_utc: datetime
    captured_at_utc: datetime
    eligible_from_utc: datetime | None
    mode: CaptureMode
    source_payload_digest: Sha256Digest
    context_complete: bool
    training_eligible: bool
    payload: dict[str, JsonValue]

    _event_utc = field_validator("source_event_time_utc")(_utc)
    _observed_utc = field_validator("observed_at_utc")(_utc)
    _captured_utc = field_validator("captured_at_utc")(_utc)
    _eligible_utc = field_validator("eligible_from_utc")(
        lambda value: None if value is None else _utc(value)
    )

    @model_validator(mode="after")
    def enforce_forward_boundary(self) -> "ForwardCaptureRecord":
        if self.source_event_time_utc > self.observed_at_utc:
            raise ValueError("source_event_time_utc cannot be later than observation")
        if self.observed_at_utc > self.captured_at_utc:
            raise ValueError("observed_at_utc cannot be later than capture")
        expected_eligible = self.observed_at_utc if self.mode is CaptureMode.FORWARD else None
        if self.eligible_from_utc != expected_eligible:
            raise ValueError("eligible_from_utc must equal observed_at for forward data and null for backfill")
        expected_training = self.mode is CaptureMode.FORWARD and self.context_complete
        if self.training_eligible != expected_training:
            raise ValueError("training_eligible must require forward mode and complete context")
        return self


class P2ReadinessReport(StrictContract):
    generated_at_utc: datetime
    minimum_trading_days: int = Field(ge=1)
    required_pairs: tuple[Identifier, ...] = Field(min_length=1)
    exposure_trading_days: int = Field(ge=0)
    tick_trading_days_by_pair: dict[Identifier, int]
    exposure_first_date: date | None = None
    exposure_last_date: date | None = None
    tick_first_date_by_pair: dict[Identifier, date | None]
    tick_last_date_by_pair: dict[Identifier, date | None]
    join_key: str
    ready: bool
    blocking_reasons: tuple[str, ...]

    _generated_utc = field_validator("generated_at_utc")(_utc)


_FAMILY_DOMAIN: dict[CaptureFamily, StorageDomain] = {
    CaptureFamily.TICK_SPREAD: StorageDomain.FX,
    CaptureFamily.EXPOSURE: StorageDomain.PORTFOLIO,
    CaptureFamily.TRADE_ENTRY_CONTEXT: StorageDomain.OUTCOMES,
    CaptureFamily.FILL_LADDER: StorageDomain.OUTCOMES,
    CaptureFamily.CRYPTO_MARKET: StorageDomain.CRYPTO,
    CaptureFamily.TRACK_B_FILING: StorageDomain.FILINGS,
    CaptureFamily.TRACK_B_REBALANCE: StorageDomain.OUTCOMES,
    CaptureFamily.HEDGE_FORWARD: StorageDomain.PORTFOLIO,
}


class ForwardCaptureService:
    """Deduplicated capture facade over the Phase-C immutable data platform."""

    def __init__(
        self,
        platform: DataPlatform,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.platform = platform
        self._writer = platform.ingestion_writer()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @classmethod
    def default(cls) -> "ForwardCaptureService":
        data_root = Path(os.getenv("AXIOM_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        return cls(DataPlatform(data_root, trust_store=TrustStore()))

    @staticmethod
    def _stream_id(family: CaptureFamily, source_key: str) -> str:
        return f"forward-capture:{family.value}:{source_key}"

    def records(self, family: CaptureFamily, source_key: str) -> list[ForwardCaptureRecord]:
        domain = _FAMILY_DOMAIN[family]
        rows = self.platform.iter_history(domain, self._stream_id(family, source_key))
        records: list[ForwardCaptureRecord] = []
        for row in rows:
            try:
                records.append(
                    ForwardCaptureRecord.model_validate_json(
                        json.dumps(row, allow_nan=False),
                        strict=True,
                    )
                )
            except (ValueError, TypeError) as exc:
                raise DataIntegrityError(f"invalid forward capture record: {exc}") from exc
        return records

    def _capture(
        self,
        family: CaptureFamily,
        source_key: str,
        payload: Mapping[str, JsonValue],
        *,
        source_event_time: datetime | date | str,
        observed_at: datetime | None = None,
        mode: CaptureMode = CaptureMode.FORWARD,
        context_complete: bool = True,
        dedupe_key: str | None = None,
    ) -> ForwardCaptureRecord | None:
        captured_at = _utc(self._clock())
        observed = _utc(observed_at or captured_at)
        event_time = _parse_time(source_event_time)
        payload_dict = dict(payload)
        payload_digest = content_digest(payload_dict)
        identity = {
            "family": family.value,
            "source_key": source_key,
            "dedupe_key": dedupe_key or payload_digest,
            "source_payload_digest": payload_digest,
        }
        capture_id = content_digest(identity)
        record = ForwardCaptureRecord(
            capture_id=capture_id,
            family=family,
            source_key=source_key,
            source_event_time_utc=event_time,
            observed_at_utc=observed,
            captured_at_utc=captured_at,
            eligible_from_utc=observed if mode is CaptureMode.FORWARD else None,
            mode=mode,
            source_payload_digest=payload_digest,
            context_complete=context_complete,
            training_eligible=mode is CaptureMode.FORWARD and context_complete,
            payload=payload_dict,
        )
        domain = _FAMILY_DOMAIN[family]
        stream_id = self._stream_id(family, source_key)
        for _attempt in range(2):
            existing = self.records(family, source_key)
            if any(item.capture_id == capture_id for item in existing):
                return None
            try:
                self._writer.append_history(
                    domain,
                    stream_id,
                    record.model_dump(mode="json", round_trip=True),
                    expected_sequence=len(existing),
                )
                return record
            except ControlStateConflict:
                continue
        raise ControlStateConflict(f"forward capture stream remained contended: {stream_id}")

    def _publish_parquet_version(
        self,
        *,
        domain: StorageDomain,
        dataset_id: str,
        partition_id: str,
        data: bytes,
        source_id: str,
        event_time: datetime,
        row_count: int,
        metadata: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        digest = sha256_bytes(data)
        raw = self._writer.put_raw(
            domain,
            data,
            source_id=source_id,
            retrieved_at=_utc(self._clock()),
            media_type=StorageFormat.PARQUET.media_type,
            metadata=metadata,
        )
        _, schema_digest = _validate_container(
            data,
            StorageFormat.PARQUET,
            expected_row_count=row_count,
        )
        version = self._writer.publish_normalized(
            domain,
            dataset_id=dataset_id,
            version_id=f"content-{digest}",
            storage_format=StorageFormat.PARQUET,
            schema_digest=schema_digest,
            source_raw_digests=(raw.object.digest,),
            created_at=event_time,
            partitions={partition_id: PartitionPayload(data=data, row_count=row_count)},
        )
        partition = version.partitions[0]
        return {
            "dataset_id": dataset_id,
            "normalized_version_digest": content_digest(version),
            "partition_digest": partition.digest,
            "partition_uri": partition.uri,
            "row_count": row_count,
        }

    def capture_tick_partition(
        self,
        pair: str,
        observations: Sequence[Mapping[str, JsonValue]],
        parquet_bytes: bytes,
        *,
        observed_at: datetime | None = None,
    ) -> ForwardCaptureRecord | None:
        if not observations:
            return None
        times = [_parse_time(item["time"]) for item in observations]
        spreads = []
        for item in observations:
            try:
                bid, ask = float(item["bid"]), float(item["ask"])
            except (KeyError, TypeError, ValueError):
                continue
            spread = ask - bid
            if math.isfinite(spread) and spread >= 0:
                spreads.append(spread)
        event_time = max(times)
        observed = _utc(observed_at or self._clock())
        mode = (
            CaptureMode.FORWARD
            if 0 <= (observed - event_time).total_seconds() <= 3600
            else CaptureMode.BACKFILL
        )
        ref = self._publish_parquet_version(
            domain=StorageDomain.FX,
            dataset_id=f"fx-ticks-{pair}",
            partition_id=f"pair={pair}/event_date={event_time.date().isoformat()}/batch={sha256_bytes(parquet_bytes)}",
            data=parquet_bytes,
            source_id="oanda-practice-pricing-stream",
            event_time=event_time,
            row_count=len(observations),
            metadata={"pair": pair, "first_event_time": min(times).isoformat(), "last_event_time": event_time.isoformat()},
        )
        payload: dict[str, JsonValue] = {
            **ref,
            "pair": pair,
            "first_event_time_utc": min(times).isoformat(),
            "last_event_time_utc": event_time.isoformat(),
            "tick_count": len(observations),
            "spread_count": len(spreads),
            "spread_min": min(spreads) if spreads else None,
            "spread_max": max(spreads) if spreads else None,
            "spread_mean": (sum(spreads) / len(spreads)) if spreads else None,
        }
        return self._capture(
            CaptureFamily.TICK_SPREAD,
            pair,
            payload,
            source_event_time=event_time,
            observed_at=observed,
            mode=mode,
            context_complete=bool(spreads),
            dedupe_key=sha256_bytes(parquet_bytes),
        )

    def capture_exposure_snapshot(self, row: Mapping[str, JsonValue]) -> ForwardCaptureRecord | None:
        required = ("captured_at_utc", "source_mtime_utc", "net_currency_notional_home")
        complete = all(row.get(field) is not None for field in required)
        observed = _parse_time(row.get("captured_at_utc"))
        age_seconds = (_utc(self._clock()) - observed).total_seconds()
        mode = CaptureMode.FORWARD if 0 <= age_seconds <= 2 * 3600 else CaptureMode.BACKFILL
        return self._capture(
            CaptureFamily.EXPOSURE,
            str(row.get("source") or "portfolio"),
            row,
            source_event_time=str(row.get("source_mtime_utc") or row.get("captured_at_utc")),
            observed_at=observed,
            mode=mode,
            context_complete=complete,
            dedupe_key=str(row.get("source_mtime_utc") or content_digest(dict(row))),
        )

    def capture_trade_entry_context(self, entry: Mapping[str, JsonValue]) -> ForwardCaptureRecord | None:
        gates = entry.get("gates")
        agents = entry.get("agents")
        features = entry.get("ridge_features") or entry.get("features")
        complete = (
            isinstance(gates, dict)
            and bool(gates)
            and isinstance(agents, dict)
            and agents.get("agent_votes") is not None
            and features is not None
            and entry.get("spread_pips") is not None
            and entry.get("expected_price") is not None
        )
        payload = dict(entry)
        payload["context_complete"] = complete
        event_time = _parse_time(entry.get("timestamp"))
        observed = _utc(self._clock())
        age_seconds = (observed - event_time).total_seconds()
        mode = CaptureMode.FORWARD if 0 <= age_seconds <= 24 * 3600 else CaptureMode.BACKFILL
        return self._capture(
            CaptureFamily.TRADE_ENTRY_CONTEXT,
            str(entry.get("pair") or "unknown"),
            payload,
            source_event_time=event_time,
            observed_at=observed,
            mode=mode,
            context_complete=complete,
            dedupe_key=str(entry.get("trade_id") or content_digest(payload)),
        )

    def capture_fill_transaction(self, transaction: Mapping[str, JsonValue]) -> ForwardCaptureRecord | None:
        if transaction.get("type") != "ORDER_FILL":
            return None
        full_price = transaction.get("fullPrice")
        bids = full_price.get("bids") if isinstance(full_price, dict) else None
        asks = full_price.get("asks") if isinstance(full_price, dict) else None
        complete = bool(bids) and bool(asks)

        def first_price(levels: Any) -> float | None:
            if not isinstance(levels, list) or not levels or not isinstance(levels[0], dict):
                return None
            try:
                value = float(levels[0].get("price"))
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None

        bid = first_price(bids)
        ask = first_price(asks)
        reference_mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        try:
            fill_price = float(transaction.get("price"))
        except (TypeError, ValueError):
            fill_price = None
        requested_raw = transaction.get("requestedPrice")
        try:
            requested_price = float(requested_raw) if requested_raw is not None else None
        except (TypeError, ValueError):
            requested_price = None
        payload = {
            "transaction_id": transaction.get("id"),
            "instrument": transaction.get("instrument"),
            "time": transaction.get("time"),
            "reason": transaction.get("reason"),
            "price": transaction.get("price"),
            "units": transaction.get("units"),
            "requested_units": transaction.get("requestedUnits"),
            "half_spread_cost": transaction.get("halfSpreadCost"),
            "pl": transaction.get("pl"),
            "financing": transaction.get("financing"),
            "full_price": full_price,
            "ladder_complete": complete,
            "top_bid": bid,
            "top_ask": ask,
            "reference_mid": reference_mid,
            "requested_price": requested_price,
            "slippage_vs_ladder_mid_price": (
                fill_price - reference_mid
                if fill_price is not None and reference_mid is not None
                else None
            ),
            "slippage_vs_requested_price": (
                fill_price - requested_price
                if fill_price is not None and requested_price is not None
                else None
            ),
            "slippage_units": "raw_price_units",
        }
        event_time = _parse_time(transaction.get("time"))
        observed = _utc(self._clock())
        # TransactionLedger intentionally fetches a bounded history on first
        # run. Old fills remain useful evidence, but must never masquerade as
        # forward observations from capture activation.
        mode = (
            CaptureMode.FORWARD
            if (observed - event_time).total_seconds() <= 24 * 3600
            else CaptureMode.BACKFILL
        )
        return self._capture(
            CaptureFamily.FILL_LADDER,
            str(transaction.get("instrument") or "unknown"),
            payload,
            source_event_time=event_time,
            observed_at=observed,
            mode=mode,
            context_complete=complete,
            dedupe_key=str(transaction.get("id") or content_digest(payload)),
        )

    def capture_crypto_partition(
        self,
        *,
        venue: str,
        symbol: str,
        data_kind: str,
        interval: str,
        parquet_bytes: bytes,
        first_event_time: datetime,
        last_event_time: datetime,
        row_count: int,
        is_initial_backfill: bool,
    ) -> ForwardCaptureRecord | None:
        dataset_id = f"crypto-{venue}-{data_kind}-{symbol}-{interval}"
        ref = self._publish_parquet_version(
            domain=StorageDomain.CRYPTO,
            dataset_id=dataset_id,
            partition_id=f"symbol={symbol}/through={last_event_time.date().isoformat()}",
            data=parquet_bytes,
            source_id=f"{venue}-{data_kind}",
            event_time=_utc(last_event_time),
            row_count=row_count,
            metadata={"venue": venue, "symbol": symbol, "data_kind": data_kind, "interval": interval},
        )
        payload = {
            **ref,
            "venue": venue,
            "symbol": symbol,
            "data_kind": data_kind,
            "interval": interval,
            "first_event_time_utc": _utc(first_event_time).isoformat(),
            "last_event_time_utc": _utc(last_event_time).isoformat(),
            "contains_historical_backfill": is_initial_backfill,
        }
        return self._capture(
            CaptureFamily.CRYPTO_MARKET,
            f"{venue}:{data_kind}:{symbol}:{interval}",
            payload,
            source_event_time=last_event_time,
            mode=CaptureMode.BACKFILL if is_initial_backfill else CaptureMode.FORWARD,
            context_complete=True,
            dedupe_key=sha256_bytes(parquet_bytes),
        )

    def capture_track_b_filing(
        self,
        *,
        ticker: str,
        cik: str,
        accession: str,
        form: str,
        filed: str,
        as_of: str,
        document_url: str,
        text: str,
        observed_at: datetime | None = None,
    ) -> ForwardCaptureRecord | None:
        observed = _utc(observed_at or self._clock())
        filed_time = _parse_time(filed)
        age_days = (observed.date() - filed_time.date()).days
        mode = CaptureMode.FORWARD if 0 <= age_days <= 7 else CaptureMode.BACKFILL
        raw = self._writer.put_raw(
            StorageDomain.FILINGS,
            text.encode("utf-8"),
            source_id="sec-edgar-primary-document",
            retrieved_at=observed,
            media_type="text/plain; charset=utf-8",
            metadata={
                "ticker": ticker,
                "cik": cik,
                "accession": accession,
                "form": form,
                "filed": filed,
                "as_of": as_of,
                "document_url": document_url,
            },
        )
        payload = {
            "ticker": ticker,
            "cik": cik,
            "accession": accession,
            "form": form,
            "filed": filed,
            "as_of": as_of,
            "document_url": document_url,
            "document_digest": raw.object.digest,
            "document_uri": raw.object.uri,
        }
        return self._capture(
            CaptureFamily.TRACK_B_FILING,
            accession,
            payload,
            source_event_time=filed_time,
            observed_at=observed,
            mode=mode,
            context_complete=True,
            dedupe_key=accession,
        )

    def capture_track_b_rebalance(self, row: Mapping[str, JsonValue]) -> ForwardCaptureRecord | None:
        complete = bool(row.get("asof_date")) and isinstance(row.get("book"), dict)
        observed = _parse_time(row.get("cycle_ts"))
        age_seconds = (_utc(self._clock()) - observed).total_seconds()
        mode = CaptureMode.FORWARD if 0 <= age_seconds <= 48 * 3600 else CaptureMode.BACKFILL
        return self._capture(
            CaptureFamily.TRACK_B_REBALANCE,
            "track_b",
            row,
            source_event_time=str(row.get("asof_date")),
            observed_at=observed,
            mode=mode,
            context_complete=complete,
            dedupe_key=str(row.get("asof_date")),
        )

    def capture_hedge_forward(self, row: Mapping[str, JsonValue]) -> ForwardCaptureRecord | None:
        complete = (
            isinstance(row.get("raw"), dict)
            and isinstance(row.get("hedged"), dict)
            and row.get("asof_date") is not None
        )
        observed = _parse_time(row.get("cycle_ts"))
        age_seconds = (_utc(self._clock()) - observed).total_seconds()
        mode = CaptureMode.FORWARD if 0 <= age_seconds <= 48 * 3600 else CaptureMode.BACKFILL
        return self._capture(
            CaptureFamily.HEDGE_FORWARD,
            str(row.get("strategy") or "unknown"),
            row,
            source_event_time=str(row.get("asof_date")),
            observed_at=observed,
            mode=mode,
            context_complete=complete,
            dedupe_key=f"{row.get('strategy')}:{row.get('asof_date')}",
        )

    def evaluate_p2_readiness(
        self,
        required_pairs: Sequence[str],
        *,
        minimum_trading_days: int = DEFAULT_P2_MIN_TRADING_DAYS,
    ) -> P2ReadinessReport:
        if not required_pairs:
            raise ValueError("at least one P2 tick pair is required")

        def eligible_dates(records: Sequence[ForwardCaptureRecord]) -> list[date]:
            return sorted(
                {
                    record.observed_at_utc.date()
                    for record in records
                    if record.training_eligible
                    and record.mode is CaptureMode.FORWARD
                    and record.observed_at_utc.weekday() < 5
                }
            )

        exposure_records: list[ForwardCaptureRecord] = []
        portfolio_history = self.platform._domain_root(StorageDomain.PORTFOLIO) / "history"
        if portfolio_history.exists():
            # Exposure has one declared source key today. Keep the explicit key
            # so unrelated portfolio history can never inflate this gate.
            exposure_records = self.records(CaptureFamily.EXPOSURE, "oanda_fx_trend_lane")
        exposure_dates = eligible_dates(exposure_records)

        tick_dates: dict[str, list[date]] = {}
        for pair in required_pairs:
            tick_dates[pair] = eligible_dates(self.records(CaptureFamily.TICK_SPREAD, pair))

        reasons: list[str] = []
        if len(exposure_dates) < minimum_trading_days:
            reasons.append(
                f"exposure_forward_days:{len(exposure_dates)}<{minimum_trading_days}"
            )
        for pair, days in tick_dates.items():
            if len(days) < minimum_trading_days:
                reasons.append(f"tick_forward_days:{pair}:{len(days)}<{minimum_trading_days}")
        return P2ReadinessReport(
            generated_at_utc=_utc(self._clock()),
            minimum_trading_days=minimum_trading_days,
            required_pairs=tuple(required_pairs),
            exposure_trading_days=len(exposure_dates),
            tick_trading_days_by_pair={pair: len(days) for pair, days in tick_dates.items()},
            exposure_first_date=exposure_dates[0] if exposure_dates else None,
            exposure_last_date=exposure_dates[-1] if exposure_dates else None,
            tick_first_date_by_pair={pair: days[0] if days else None for pair, days in tick_dates.items()},
            tick_last_date_by_pair={pair: days[-1] if days else None for pair, days in tick_dates.items()},
            join_key="observed_at_utc UTC trading date; source_event_time_utc retained for causal lag",
            ready=not reasons,
            blocking_reasons=tuple(reasons),
        )


_DEFAULT_SERVICE: ForwardCaptureService | None = None
_DEFAULT_SERVICE_LOCK = threading.Lock()


def get_default_forward_capture() -> ForwardCaptureService:
    global _DEFAULT_SERVICE
    if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("AXIOM_DATA_ROOT"):
        raise RuntimeError(
            "default forward capture is disabled under pytest; inject an explicit temporary service"
        )
    with _DEFAULT_SERVICE_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = ForwardCaptureService.default()
        return _DEFAULT_SERVICE


def capture_best_effort(method: str, *args: Any, **kwargs: Any) -> None:
    """Invoke a capture hook without changing the calling lane's control flow."""
    if os.getenv("PYTEST_CURRENT_TEST") and not os.getenv("AXIOM_DATA_ROOT"):
        return
    try:
        service = get_default_forward_capture()
        getattr(service, method)(*args, **kwargs)
    except Exception:
        logger.exception("forward capture hook failed: %s", method)
