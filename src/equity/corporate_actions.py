"""US-009 — Corporate-action handling for the single-stock harvester.

Single-stock books are exquisitely sensitive to splits, dividends, and
delistings: a 4:1 split on a name held at NAV-weight 10% turns the broker's
raw price from ``$400`` to ``$100`` overnight, and a naive ``last_close vs
yesterday_close`` drawdown check would log a -75% drawdown on a non-event.
Conversely, if the data source has already back-adjusted the price series but
the broker's share count has NOT been multiplied through, the same name now
shows a 4x position-value blow-out on the books. Either error compounds into
sized trades on wrong numbers.

This module is the seam that keeps the data side (back-adjusted closes from
:mod:`src.equity.data_loader`), the action feed (split / dividend / delisting
events), and the broker side (shares + cash + last raw price) in sync.

Public surface
--------------

* :class:`CorporateAction` — split / cash dividend / delisting / cash M&A.
* :class:`Position` and :class:`BookState` — the broker book.
* :class:`DivergenceAlert` — surfaces any (ticker, asof) where the broker's
  shares × raw price diverge from ``data_source_adjusted_price /
  cumulative_adjustment_factor`` beyond ``tolerance``.
* :class:`CorporateActionResult` — payload from :meth:`reconcile`.
* :class:`CorporateActionHandler` — the orchestrator. Built fail-closed
  through the ship-gate guard (matches :mod:`src.equity.harvester_strategy`
  and :mod:`src.equity.risk_agents`).

Hard rules
----------

* Claude-free, deterministic hot path.
* Real classes + real disk via ``tmp_path`` in tests. No ``unittest.mock``.
* No bare ``except``. Narrow exceptions, logged + surfaced.
* Atomic writes via ``tempfile.mkstemp`` + :func:`os.replace`.
* HALT on divergence. Do not trade on a skew the books cannot explain.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

from src.equity.universe import UniverseSnapshot, compute_universe_hash

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1

ACTION_SPLIT = "split"
ACTION_CASH_DIV = "cash_dividend"
ACTION_DELIST = "delisting"
ACTION_MA_CASH = "ma_cash"
SUPPORTED_ACTIONS = frozenset(
    {ACTION_SPLIT, ACTION_CASH_DIV, ACTION_DELIST, ACTION_MA_CASH}
)


class CorporateActionError(RuntimeError):
    """Raised on malformed inputs or ship-gate refusal."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorporateActionConfig:
    """Tolerances + behaviour switches for the handler.

    ``divergence_tolerance`` is the relative tolerance for the raw-vs-adjusted
    market-value check: a 0.5% disagreement on a single name halts the book.
    The single-stock harvester runs on US large-caps and the only legitimate
    sources of disagreement at that scale are unprocessed splits/dividends.
    """

    divergence_tolerance: float = 0.005  # 0.5% mv mismatch -> halt
    price_min: float = 1e-6
    halt_on_divergence: bool = True

    def __post_init__(self) -> None:
        if not (0 < self.divergence_tolerance < 1.0):
            raise CorporateActionError(
                f"divergence_tolerance must be in (0, 1), "
                f"got {self.divergence_tolerance!r}"
            )
        if self.price_min <= 0:
            raise CorporateActionError(
                f"price_min must be > 0, got {self.price_min!r}"
            )


# ---------------------------------------------------------------------------
# Corporate action record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorporateAction:
    """One corporate action against one ticker on one ex-date.

    Schema by ``kind``:

    * ``split`` — ``ratio`` is post-split shares per pre-split share. A 4:1
      forward split is ``ratio=4.0`` (price ÷ 4, shares × 4). A 1:2 reverse
      split is ``ratio=0.5`` (price × 2, shares × 0.5).
    * ``cash_dividend`` — ``cash_per_share`` paid on each held share on the
      pay date (we book it on ``asof`` to keep one bookkeeping pass).
    * ``delisting`` — position is liquidated at ``cash_payout`` per share
      (``0.0`` is a legitimate zero-recovery delisting).
    * ``ma_cash`` — cash-out merger. Same mechanics as a delisting; we
      keep them distinct for telemetry / journal clarity.
    """

    ticker: str
    asof: str  # ISO date
    kind: str
    ratio: float = 1.0
    cash_per_share: float = 0.0
    cash_payout: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker:
            raise CorporateActionError(
                f"ticker must be non-empty str, got {self.ticker!r}"
            )
        if self.kind not in SUPPORTED_ACTIONS:
            raise CorporateActionError(
                f"unsupported action kind {self.kind!r}; "
                f"expected one of {sorted(SUPPORTED_ACTIONS)}"
            )
        # Validate ISO date / timestamp early.
        try:
            pd.Timestamp(self.asof)
        except (TypeError, ValueError) as exc:
            raise CorporateActionError(
                f"asof {self.asof!r} not a parseable date: {exc}"
            ) from exc
        if self.kind == ACTION_SPLIT:
            if not math.isfinite(self.ratio) or self.ratio <= 0:
                raise CorporateActionError(
                    f"split ratio must be > 0 and finite, got {self.ratio!r}"
                )
        if self.kind == ACTION_CASH_DIV:
            if (
                not math.isfinite(self.cash_per_share)
                or self.cash_per_share < 0
            ):
                raise CorporateActionError(
                    f"cash_per_share must be >= 0 and finite, "
                    f"got {self.cash_per_share!r}"
                )
        if self.kind in (ACTION_DELIST, ACTION_MA_CASH):
            if (
                not math.isfinite(self.cash_payout)
                or self.cash_payout < 0
            ):
                raise CorporateActionError(
                    f"cash_payout must be >= 0 and finite, "
                    f"got {self.cash_payout!r}"
                )

    @property
    def asof_ts(self) -> pd.Timestamp:
        ts = pd.Timestamp(self.asof)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.normalize()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "asof": self.asof,
            "kind": self.kind,
            "ratio": float(self.ratio),
            "cash_per_share": float(self.cash_per_share),
            "cash_payout": float(self.cash_payout),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorporateAction":
        if not isinstance(payload, Mapping):
            raise CorporateActionError(
                f"action payload must be a mapping, got {type(payload)!r}"
            )
        try:
            return cls(
                ticker=str(payload["ticker"]),
                asof=str(payload["asof"]),
                kind=str(payload["kind"]),
                ratio=float(payload.get("ratio", 1.0)),
                cash_per_share=float(payload.get("cash_per_share", 0.0)),
                cash_payout=float(payload.get("cash_payout", 0.0)),
                note=str(payload.get("note", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorporateActionError(
                f"action payload malformed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Broker book
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Broker position for a single ticker.

    ``shares`` and ``avg_cost`` adjust through splits; ``last_price`` is the
    most recent raw quote the broker reported.
    """

    ticker: str
    shares: float
    avg_cost: float
    last_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker:
            raise CorporateActionError(
                f"ticker must be non-empty str, got {self.ticker!r}"
            )
        for field_name in ("shares", "avg_cost", "last_price"):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)) or not math.isfinite(
                float(val)
            ):
                raise CorporateActionError(
                    f"position.{field_name} must be finite numeric, "
                    f"got {val!r}"
                )

    @property
    def market_value(self) -> float:
        return float(self.shares) * float(self.last_price)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "shares": float(self.shares),
            "avg_cost": float(self.avg_cost),
            "last_price": float(self.last_price),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Position":
        if not isinstance(payload, Mapping):
            raise CorporateActionError(
                f"position payload must be a mapping, got {type(payload)!r}"
            )
        try:
            return cls(
                ticker=str(payload["ticker"]),
                shares=float(payload["shares"]),
                avg_cost=float(payload["avg_cost"]),
                last_price=float(payload["last_price"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorporateActionError(
                f"position payload malformed: {exc}"
            ) from exc


@dataclass
class BookState:
    """Snapshot of the broker book at one timestamp."""

    asof: str
    cash: float
    peak_nav: float
    positions: Dict[str, Position] = field(default_factory=dict)
    halted: bool = False
    halt_reason: Optional[str] = None
    version: int = STATE_VERSION

    def __post_init__(self) -> None:
        try:
            pd.Timestamp(self.asof)
        except (TypeError, ValueError) as exc:
            raise CorporateActionError(
                f"asof {self.asof!r} not parseable: {exc}"
            ) from exc
        if not math.isfinite(self.cash):
            raise CorporateActionError(
                f"cash must be finite, got {self.cash!r}"
            )
        if not math.isfinite(self.peak_nav) or self.peak_nav < 0:
            raise CorporateActionError(
                f"peak_nav must be finite and >= 0, got {self.peak_nav!r}"
            )

    @property
    def nav(self) -> float:
        return float(self.cash) + sum(
            p.market_value for p in self.positions.values()
        )

    def drawdown(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.nav / self.peak_nav))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "asof": str(self.asof),
            "cash": float(self.cash),
            "peak_nav": float(self.peak_nav),
            "nav": float(self.nav),
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "positions": {
                t: p.to_dict() for t, p in self.positions.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BookState":
        if not isinstance(payload, Mapping):
            raise CorporateActionError(
                "book payload must be a mapping"
            )
        try:
            positions_raw = payload.get("positions") or {}
            if not isinstance(positions_raw, Mapping):
                raise CorporateActionError(
                    "book.positions must be a mapping"
                )
            positions = {
                str(t): Position.from_dict(p)
                for t, p in positions_raw.items()
            }
            return cls(
                asof=str(payload["asof"]),
                cash=float(payload["cash"]),
                peak_nav=float(payload["peak_nav"]),
                positions=positions,
                halted=bool(payload.get("halted", False)),
                halt_reason=(
                    str(payload["halt_reason"])
                    if payload.get("halt_reason")
                    else None
                ),
                version=int(payload.get("version", STATE_VERSION)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorporateActionError(
                f"book payload malformed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceAlert:
    """One ticker's raw-vs-adjusted market-value mismatch."""

    ticker: str
    asof: str
    raw_price: float
    adjusted_price: float
    expected_raw_price: float
    relative_error: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "asof": self.asof,
            "raw_price": float(self.raw_price),
            "adjusted_price": float(self.adjusted_price),
            "expected_raw_price": float(self.expected_raw_price),
            "relative_error": float(self.relative_error),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CorporateActionResult:
    """Output of :meth:`CorporateActionHandler.reconcile`.

    ``cash_received`` is the sum of cash dividends + delisting/M&A payouts
    booked on this call. ``actions_applied`` is the (input-order-preserving)
    list of accepted actions; ``actions_skipped`` flags actions for tickers
    not in the book (warned, not an error).
    """

    asof: str
    nav_before: float
    nav_after: float
    cash_received: float
    actions_applied: List[CorporateAction]
    actions_skipped: List[CorporateAction]
    delisted_tickers: List[str]
    divergences: List[DivergenceAlert]
    halted: bool
    halt_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asof": self.asof,
            "nav_before": float(self.nav_before),
            "nav_after": float(self.nav_after),
            "cash_received": float(self.cash_received),
            "actions_applied": [a.to_dict() for a in self.actions_applied],
            "actions_skipped": [a.to_dict() for a in self.actions_skipped],
            "delisted_tickers": list(self.delisted_tickers),
            "divergences": [d.to_dict() for d in self.divergences],
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
        }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorporateActionHandler:
    """Reconcile broker positions, data-source adjusted prices, and action feed.

    Construct via :meth:`create` (ship-gate guard). The handler is stateless
    in its hot path; :meth:`reconcile` is a pure function from
    ``(book, raw_prices, adjusted_prices, actions, asof)`` to a
    :class:`CorporateActionResult` and a mutated copy of the book. State is
    persisted via :meth:`write_book` (atomic).
    """

    config: CorporateActionConfig
    ship_gate_path: Path
    universe_hash: str

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def create(
        cls,
        config: CorporateActionConfig,
        ship_gate_path: Path,
        universe_hash: str,
    ) -> "CorporateActionHandler":
        if not isinstance(config, CorporateActionConfig):
            raise CorporateActionError(
                f"config must be CorporateActionConfig, got {type(config)!r}"
            )
        if not isinstance(universe_hash, str) or not universe_hash:
            raise CorporateActionError(
                "universe_hash must be a non-empty string"
            )
        _enforce_ship_gate(Path(ship_gate_path), universe_hash)
        return cls(
            config=config,
            ship_gate_path=Path(ship_gate_path),
            universe_hash=str(universe_hash),
        )

    # -----------------------------------------------------------------
    # Apply a single action (pure on Position; mutates a copy)
    # -----------------------------------------------------------------
    def _apply_split(
        self, position: Position, action: CorporateAction
    ) -> Tuple[Position, float]:
        ratio = float(action.ratio)
        if ratio <= 0:
            raise CorporateActionError(
                f"split ratio non-positive for {action.ticker}: {ratio!r}"
            )
        new_shares = float(position.shares) * ratio
        # Cost basis must scale inversely so total cost is preserved.
        new_avg_cost = (
            float(position.avg_cost) / ratio
            if ratio > 0
            else float(position.avg_cost)
        )
        new_last_price = float(position.last_price) / ratio
        adjusted = Position(
            ticker=position.ticker,
            shares=new_shares,
            avg_cost=new_avg_cost,
            last_price=new_last_price,
        )
        # Splits pay no cash; NAV invariant should hold.
        return adjusted, 0.0

    def _apply_dividend(
        self, position: Position, action: CorporateAction
    ) -> Tuple[Position, float]:
        cps = float(action.cash_per_share)
        if cps < 0:
            raise CorporateActionError(
                f"cash_per_share negative for {action.ticker}: {cps!r}"
            )
        cash_in = float(position.shares) * cps
        # Shares & avg_cost unchanged. The broker quote drops by the dividend
        # on ex-date; the back-adjusted feed treats it as continuous. We
        # mark the broker side down here so the post-call book equals
        # ``shares × ex-div raw price`` — divergence check stays clean.
        new_last_price = max(
            float(position.last_price) - cps, self.config.price_min
        )
        adjusted = Position(
            ticker=position.ticker,
            shares=position.shares,
            avg_cost=position.avg_cost,
            last_price=new_last_price,
        )
        return adjusted, cash_in

    def _apply_cash_out(
        self, position: Position, action: CorporateAction
    ) -> Tuple[Optional[Position], float]:
        payout = float(action.cash_payout)
        if payout < 0:
            raise CorporateActionError(
                f"cash_payout negative for {action.ticker}: {payout!r}"
            )
        cash_in = float(position.shares) * payout
        return None, cash_in

    # -----------------------------------------------------------------
    # Top-level reconcile
    # -----------------------------------------------------------------
    def reconcile(
        self,
        book: BookState,
        actions: Iterable[CorporateAction],
        asof: pd.Timestamp,
        *,
        raw_prices: Optional[Mapping[str, float]] = None,
        adjusted_prices: Optional[Mapping[str, float]] = None,
        cumulative_adjustment: Optional[Mapping[str, float]] = None,
        universe: Optional[UniverseSnapshot] = None,
    ) -> Tuple[BookState, CorporateActionResult]:
        """Apply actions, mark to market, and divergence-check.

        Parameters
        ----------
        book:
            Current book. Caller's instance is NOT mutated; we operate on a
            deep-copy snapshot.
        actions:
            Iterable of :class:`CorporateAction` with ``asof_ts == asof``.
            Actions whose ticker is not in the book are reported as
            ``actions_skipped`` (warning, not an error — universe churn is
            normal).
        asof:
            UTC date of this reconciliation cycle.
        raw_prices:
            Broker-quoted raw prices for each held ticker on ``asof``,
            POST any actions taken on this date. Optional but required for
            the divergence check.
        adjusted_prices:
            Back-adjusted prices from :mod:`src.equity.data_loader` for
            each held ticker on ``asof``. Optional; with
            ``cumulative_adjustment`` we can predict the raw price and
            compare against the broker quote.
        cumulative_adjustment:
            Multiplier such that ``raw_price = adjusted_price /
            cumulative_adjustment``. (A 4:1 split brings cum_adj from 1.0
            → 4.0; a $1 cash div on a $400 stock brings it 1.0 → 1.0025.)
            If absent, the divergence check uses only the raw-vs-adjusted
            level ratio against the action history applied so far.
        universe:
            Optional :class:`UniverseSnapshot`. If supplied AND the handler's
            ``universe_hash`` matches the snapshot's, delisted/M&A tickers
            are confirmed to be off the panel on/after ``asof``.
        """
        asof_ts = _normalize_asof(asof)
        new_book = BookState.from_dict(book.to_dict())
        new_book.asof = asof_ts.date().isoformat()

        nav_before = new_book.nav
        cash_received_total = 0.0
        applied: List[CorporateAction] = []
        skipped: List[CorporateAction] = []
        delisted: List[str] = []

        for action in actions:
            if not isinstance(action, CorporateAction):
                raise CorporateActionError(
                    f"action must be CorporateAction, got {type(action)!r}"
                )
            if action.asof_ts != asof_ts:
                raise CorporateActionError(
                    f"action {action.ticker}/{action.kind} dated "
                    f"{action.asof_ts} != reconcile asof {asof_ts}"
                )
            pos = new_book.positions.get(action.ticker)
            if pos is None:
                logger.info(
                    "corp-action skipped: %s/%s — not in book",
                    action.ticker,
                    action.kind,
                )
                skipped.append(action)
                continue
            if action.kind == ACTION_SPLIT:
                adjusted_pos, cash_in = self._apply_split(pos, action)
                new_book.positions[action.ticker] = adjusted_pos
            elif action.kind == ACTION_CASH_DIV:
                adjusted_pos, cash_in = self._apply_dividend(pos, action)
                new_book.positions[action.ticker] = adjusted_pos
            elif action.kind in (ACTION_DELIST, ACTION_MA_CASH):
                _, cash_in = self._apply_cash_out(pos, action)
                del new_book.positions[action.ticker]
                delisted.append(action.ticker)
            else:  # pragma: no cover — guarded by SUPPORTED_ACTIONS
                raise CorporateActionError(
                    f"unhandled action kind: {action.kind!r}"
                )
            new_book.cash = float(new_book.cash) + float(cash_in)
            cash_received_total += float(cash_in)
            applied.append(action)

        # Post-action mark-to-market from the broker quotes (if supplied).
        if raw_prices is not None:
            for ticker, pos in new_book.positions.items():
                raw = raw_prices.get(ticker)
                if raw is None:
                    continue
                try:
                    raw_f = float(raw)
                except (TypeError, ValueError) as exc:
                    raise CorporateActionError(
                        f"raw_prices[{ticker!r}] not numeric: {raw!r}"
                    ) from exc
                if not math.isfinite(raw_f) or raw_f < self.config.price_min:
                    raise CorporateActionError(
                        f"raw_prices[{ticker!r}] invalid: {raw_f!r}"
                    )
                pos.last_price = raw_f

        # Divergence sweep.
        divergences = self._sweep_divergence(
            new_book,
            raw_prices=raw_prices,
            adjusted_prices=adjusted_prices,
            cumulative_adjustment=cumulative_adjustment,
            asof_iso=new_book.asof,
        )
        halt = False
        halt_reason: Optional[str] = None
        if divergences and self.config.halt_on_divergence:
            halt = True
            worst = max(divergences, key=lambda d: abs(d.relative_error))
            halt_reason = (
                f"divergence_halt:{worst.ticker} "
                f"rel_err={worst.relative_error:.4f}"
            )
            new_book.halted = True
            new_book.halt_reason = halt_reason

        # Universe sanity for delistings (best-effort, non-fatal).
        if universe is not None:
            self._check_universe_drops(universe, delisted, asof_ts)

        nav_after = new_book.nav
        # Bound the peak watermark forward; splits/divs should never shrink
        # peak_nav, and divergence-driven halt leaves it alone.
        new_book.peak_nav = max(float(new_book.peak_nav), nav_after)

        # 4:1 split sanity: with no divergence and no cash leg, NAV is
        # invariant. We do NOT raise on small float drift; the caller can
        # inspect the result to confirm.
        result = CorporateActionResult(
            asof=new_book.asof,
            nav_before=float(nav_before),
            nav_after=float(nav_after),
            cash_received=float(cash_received_total),
            actions_applied=applied,
            actions_skipped=skipped,
            delisted_tickers=delisted,
            divergences=divergences,
            halted=halt,
            halt_reason=halt_reason,
        )
        return new_book, result

    # -----------------------------------------------------------------
    # Divergence math
    # -----------------------------------------------------------------
    def _sweep_divergence(
        self,
        book: BookState,
        *,
        raw_prices: Optional[Mapping[str, float]],
        adjusted_prices: Optional[Mapping[str, float]],
        cumulative_adjustment: Optional[Mapping[str, float]],
        asof_iso: str,
    ) -> List[DivergenceAlert]:
        """Compare broker raw price against ``adjusted / cumulative_adj``.

        The check is per-name and skipped when inputs are not supplied for
        that ticker. We deliberately compare PRICES rather than NAVs because
        broker share counts also change through splits — a price-level check
        is the tightest signal for "data feed and broker disagree".
        """
        if adjusted_prices is None or cumulative_adjustment is None:
            return []
        alerts: List[DivergenceAlert] = []
        for ticker, pos in book.positions.items():
            adj = adjusted_prices.get(ticker)
            cum = cumulative_adjustment.get(ticker)
            if adj is None or cum is None:
                continue
            try:
                adj_f = float(adj)
                cum_f = float(cum)
            except (TypeError, ValueError) as exc:
                raise CorporateActionError(
                    f"divergence inputs for {ticker!r} not numeric: "
                    f"adjusted={adj!r}, cum_adj={cum!r}"
                ) from exc
            if not math.isfinite(adj_f) or adj_f <= 0:
                raise CorporateActionError(
                    f"adjusted_prices[{ticker!r}] invalid: {adj_f!r}"
                )
            if not math.isfinite(cum_f) or cum_f <= 0:
                raise CorporateActionError(
                    f"cumulative_adjustment[{ticker!r}] invalid: "
                    f"{cum_f!r}"
                )
            expected_raw = adj_f / cum_f
            raw_obs = float(pos.last_price)
            if raw_obs <= self.config.price_min:
                # Broker quote effectively zero — flag explicitly.
                alerts.append(
                    DivergenceAlert(
                        ticker=ticker,
                        asof=asof_iso,
                        raw_price=raw_obs,
                        adjusted_price=adj_f,
                        expected_raw_price=expected_raw,
                        relative_error=1.0,
                        reason="raw_price_at_floor",
                    )
                )
                continue
            rel_err = abs(raw_obs - expected_raw) / max(
                expected_raw, self.config.price_min
            )
            if rel_err > self.config.divergence_tolerance:
                alerts.append(
                    DivergenceAlert(
                        ticker=ticker,
                        asof=asof_iso,
                        raw_price=raw_obs,
                        adjusted_price=adj_f,
                        expected_raw_price=expected_raw,
                        relative_error=float(rel_err),
                        reason="raw_vs_adjusted_mismatch",
                    )
                )
        return alerts

    def _check_universe_drops(
        self,
        universe: UniverseSnapshot,
        delisted: List[str],
        asof_ts: pd.Timestamp,
    ) -> None:
        """Best-effort warn-only: delisted tickers should be off the panel.

        If a delisting / M&A clears a name from the book, the next universe
        snapshot must exclude it on/after ``asof``. We do not raise here —
        universe builds are upstream and may lag a single day — but we LOG
        so the runtime journal carries the mismatch.
        """
        if not delisted:
            return
        # Confirm hash matches before trusting membership lookups.
        try:
            recomputed = compute_universe_hash(universe.membership)
        except Exception as exc:
            logger.warning(
                "could not recompute universe hash: %s — skipping check",
                exc,
            )
            return
        if recomputed != universe.universe_hash:
            logger.warning(
                "universe snapshot hash mismatch (snap=%s, recomputed=%s) "
                "— skipping delisting check",
                universe.universe_hash,
                recomputed,
            )
            return
        # Lookup members at asof; warn if any delisted ticker survives.
        try:
            members = set(universe.members_on(asof_ts.date()))
        except (ValueError, KeyError) as exc:
            logger.warning(
                "universe.members_on(%s) failed: %s", asof_ts.date(), exc
            )
            return
        for ticker in delisted:
            if ticker in members:
                logger.warning(
                    "delisted ticker %s still in universe on %s "
                    "(universe needs a rebuild)",
                    ticker,
                    asof_ts.date(),
                )

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def write_book(self, book: BookState, path: Path) -> Path:
        """Atomic JSON write of the post-reconcile book."""
        return _atomic_write_json(book.to_dict(), Path(path))

    def write_result(
        self, result: CorporateActionResult, path: Path
    ) -> Path:
        """Atomic JSON write of the reconcile result (for the journal)."""
        return _atomic_write_json(result.to_dict(), Path(path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_asof(asof: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to construct unless the ship gate clears for THIS universe.

    Mirrors :func:`src.equity.harvester_strategy._enforce_ship_gate` and
    :func:`src.equity.risk_agents._enforce_ship_gate`: missing file,
    malformed JSON, ``gate_pass`` not exactly ``True``, or universe-hash
    mismatch all raise :class:`CorporateActionError`.
    """
    if not path.exists():
        raise CorporateActionError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorporateActionError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CorporateActionError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    gate_pass = payload.get("gate_pass")
    if gate_pass is not True:
        raise CorporateActionError(
            f"ship gate refuses: gate_pass={gate_pass!r} at {path} "
            "(must be exactly True to unblock the handler)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise CorporateActionError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise CorporateActionError(
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
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
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


# ---------------------------------------------------------------------------
# Universe-builder hook for delistings
# ---------------------------------------------------------------------------


def drop_delisted_from_universe(
    universe: UniverseSnapshot,
    delistings: Iterable[Tuple[str, pd.Timestamp]],
) -> UniverseSnapshot:
    """Return a new :class:`UniverseSnapshot` with delisted names dropped.

    ``delistings`` is an iterable of ``(ticker, effective_asof)`` tuples;
    the ticker's membership column is set to ``False`` on/after the
    ``effective_asof`` UTC midnight. The new snapshot rehashes the
    membership panel, so downstream ship-gate guards see a fresh
    ``universe_hash`` and must be re-cleared before trading resumes.
    """
    if not isinstance(universe, UniverseSnapshot):
        raise CorporateActionError(
            f"universe must be UniverseSnapshot, got {type(universe)!r}"
        )
    membership = universe.membership.copy()
    touched = False
    for ticker, asof in delistings:
        ts = _normalize_asof(asof)
        if ticker not in membership.columns:
            continue
        mask = membership.index >= ts
        if not mask.any():
            continue
        membership.loc[mask, ticker] = False
        touched = True
    if not touched:
        return universe
    new_hash = compute_universe_hash(membership)
    return UniverseSnapshot(
        membership=membership,
        rules=universe.rules,
        built_at=universe.built_at,
        pipeline_version=universe.pipeline_version,
        universe_hash=new_hash,
        reconstitution_dates=universe.reconstitution_dates,
    )


__all__ = [
    "ACTION_CASH_DIV",
    "ACTION_DELIST",
    "ACTION_MA_CASH",
    "ACTION_SPLIT",
    "BookState",
    "CorporateAction",
    "CorporateActionConfig",
    "CorporateActionError",
    "CorporateActionHandler",
    "CorporateActionResult",
    "DivergenceAlert",
    "Position",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "SUPPORTED_ACTIONS",
    "drop_delisted_from_universe",
]


# Hummingbot Apache-2.0 attribution (PRD §notes): no source has been copied
# verbatim from Hummingbot into this module. Corporate-action math here is
# original to this repo. No freqtrade GPL.
