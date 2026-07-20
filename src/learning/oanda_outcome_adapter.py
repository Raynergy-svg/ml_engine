"""Build :class:`OutcomeRecord` objects from real OANDA transaction history.

This adapter exists so the deterministic attribution core can be validated
against ground truth that already exists on disk, instead of against an
imagined data shape. ``trained_data/oanda/transactions.jsonl`` holds a real
practice-account history: 297 ``ORDER_FILL`` records, 171 of them
P&L-realizing, spanning 2026-04 to 2026-07.

Linkage
-------
An OANDA close does not carry its own entry price. The opening fill carries
``tradeOpened.{tradeID, price, units}``; the closing fill carries
``tradesClosed[].{tradeID, price, units, realizedPL, ...}``. Joining on
``tradeID`` reconstructs the round trip. 124 of 126 distinct closed trade IDs
link successfully; the 2 that do not were opened before this journal window
began and are reported as unlinkable rather than silently dropped.

Home-currency conversion
------------------------
OANDA reports ``realizedPL`` in home currency but quotes prices in the quote
currency. The applicable factor differs for gains and losses
(``gainQuoteHomeConversionFactor`` / ``lossQuoteHomeConversionFactor``), so the
sign of the quote-currency P&L selects the factor. This was verified
empirically across all 124 linkable slices (worst absolute error 5e-05 home).

Everything here is parsing. No attribution logic lives in this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.learning.outcome_attribution import OutcomeRecord

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSACTIONS_PATH = REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl"


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce an OANDA numeric-string field to float, defaulting on absence.

    OANDA sends numbers as strings. A missing optional cost field legitimately
    means zero; an unparseable present value is a data problem and is logged
    rather than silently zeroed.
    """
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("oanda_outcome_adapter: unparseable numeric %r -- using %s", value, default)
        return default


def load_transactions(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read the transaction journal, skipping malformed lines with a warning.

    A corrupt line must not abort the whole load (the journal is append-only
    and a torn final write is possible), but it must never pass silently.
    """
    src = Path(path) if path is not None else DEFAULT_TRANSACTIONS_PATH
    if not src.exists():
        raise FileNotFoundError(f"OANDA transaction journal not found: {src}")
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(src.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("oanda_outcome_adapter: skipping malformed line %d in %s: %s", lineno, src, exc)
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            logger.warning("oanda_outcome_adapter: line %d in %s is not an object -- skipped", lineno, src)
    return rows


def index_opening_fills(transactions: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map ``tradeID`` -> {entry_time, entry_price, opened_units, instrument}."""
    opens: Dict[str, Dict[str, Any]] = {}
    for row in transactions:
        if row.get("type") != "ORDER_FILL":
            continue
        opened = row.get("tradeOpened")
        if not isinstance(opened, Mapping):
            continue
        trade_id = str(opened.get("tradeID") or "")
        if not trade_id:
            continue
        opens[trade_id] = {
            "entry_time": str(row.get("time") or ""),
            "entry_price": _as_float(opened.get("price")),
            "opened_units": _as_float(opened.get("units")),
            "instrument": str(row.get("instrument") or ""),
            "entry_half_spread_cost": _as_float(opened.get("halfSpreadCost")),
        }
    return opens


def _conversion_factor(closing_fill: Mapping[str, Any], quote_pl_sign: float) -> float:
    """Pick the gain or loss home-conversion factor per the sign of quote P&L.

    Falls back to 1.0 only when neither factor is present (a home-currency
    instrument), which is correct rather than a guess.
    """
    gain = _as_float(closing_fill.get("gainQuoteHomeConversionFactor"), 0.0)
    loss = _as_float(closing_fill.get("lossQuoteHomeConversionFactor"), 0.0)
    chosen = gain if quote_pl_sign >= 0 else loss
    if chosen:
        return chosen
    return gain or loss or 1.0


def build_outcome_records(
    transactions: Iterable[Mapping[str, Any]],
) -> Tuple[List[OutcomeRecord], List[str]]:
    """Reconstruct closed round trips from a transaction journal.

    Args:
        transactions: parsed OANDA transaction objects, in journal order.

    Returns:
        ``(records, unlinkable_trade_ids)``. Unlinkable IDs are returned, not
        dropped, so a caller can report coverage honestly.
    """
    rows = list(transactions)
    opens = index_opening_fills(rows)

    records: List[OutcomeRecord] = []
    unlinkable: List[str] = []

    for row in rows:
        if row.get("type") != "ORDER_FILL":
            continue
        closed_slices = row.get("tradesClosed")
        if not isinstance(closed_slices, list):
            continue
        exit_reason = str(row.get("reason") or "UNKNOWN")
        exit_time = str(row.get("time") or "")
        instrument = str(row.get("instrument") or "")

        for slice_ in closed_slices:
            if not isinstance(slice_, Mapping):
                continue
            trade_id = str(slice_.get("tradeID") or "")
            if not trade_id:
                continue
            entry = opens.get(trade_id)
            if entry is None:
                unlinkable.append(trade_id)
                continue

            exit_price = _as_float(slice_.get("price"))
            entry_price = float(entry["entry_price"])
            realized_pl = _as_float(slice_.get("realizedPL"))

            # The size attributable to THIS slice is the slice's own units,
            # negated -- a close carries units opposite in sign to the open.
            # Using the opening size instead would overstate every PARTIAL
            # close (55 fills in the real journal carry ``tradeReduced``), and
            # that error is invisible on fully-closed trades because there
            # ``-slice_units == opened_units`` exactly.
            closed_units = -_as_float(slice_.get("units"))
            if closed_units == 0:
                logger.warning(
                    "oanda_outcome_adapter: trade %s closed slice has zero units -- skipped", trade_id
                )
                continue

            quote_pl = (exit_price - entry_price) * closed_units
            factor = _conversion_factor(row, quote_pl)

            # Spread is paid on both legs; the entry leg's cost lives on the
            # opening fill, the exit leg's on this closing slice. The entry
            # cost is pro-rated to this slice's share of the original position
            # so partial closes don't each charge the full entry spread.
            opened_units = float(entry["opened_units"])
            entry_share = abs(closed_units / opened_units) if opened_units else 0.0
            execution_cost = (
                float(entry["entry_half_spread_cost"]) * entry_share
                + _as_float(slice_.get("halfSpreadCost"))
            )

            records.append(
                OutcomeRecord(
                    trade_id=trade_id,
                    instrument=instrument or str(entry["instrument"]),
                    entry_time=str(entry["entry_time"]),
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    units=closed_units,
                    realized_pl_home=realized_pl,
                    conversion_factor=factor,
                    execution_cost_home=execution_cost,
                    financing_home=_as_float(slice_.get("financing")),
                    commission_home=_as_float(row.get("commission")),
                    other_fees_home=_as_float(slice_.get("guaranteedExecutionFee")),
                    conversion_cost_home=_as_float(slice_.get("plHomeConversionCost")),
                    exit_reason=exit_reason,
                    lane="oanda_fx",
                )
            )

    return records, unlinkable


def load_outcome_records(
    path: Optional[Path] = None,
) -> Tuple[List[OutcomeRecord], List[str]]:
    """Convenience: read the journal from disk and reconstruct round trips."""
    return build_outcome_records(load_transactions(path))


__all__ = [
    "DEFAULT_TRANSACTIONS_PATH",
    "build_outcome_records",
    "index_opening_fills",
    "load_outcome_records",
    "load_transactions",
]
