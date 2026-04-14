"""
Journal Reader -- Trade journal data reader for the Buddy TUI dashboard.

Reads from trained_data/trade_journal_rl.json, parses entries into typed
dataclasses, and computes display-ready performance metrics.

All methods are safe: missing files, bad JSON, and empty data produce
graceful defaults (empty lists, zeroed stats) -- never exceptions.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default journal path (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_JOURNAL = "trained_data/trade_journal_rl.json"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentVote:
    """Single agent's vote on a trade entry."""

    name: str
    score: float
    passed: bool
    weight: float
    reason: str
    reason_code: str
    confidence_delta: float


@dataclass(frozen=True)
class TradeOutcome:
    """Outcome of a closed trade."""

    trade_won: bool
    realized_pl: float
    pnl_pips: float
    exit_reason: str
    exit_price: float
    exit_time: str
    duration_minutes: float
    mae_pips: float
    mfe_pips: float
    rr_ratio: float


@dataclass(frozen=True)
class JournalEntry:
    """Single trade journal entry -- immutable after parse."""

    trade_id: str
    pair: str
    direction: str
    confidence: float
    entry_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    rr_ratio: float
    lots: float
    timestamp: str
    broker: str
    asset_class: str

    # Gate results
    gate_summary: str
    momentum_passed: bool
    confidence_passed: bool
    risk_passed: bool

    # Agent consensus
    weighted_vote_score: float
    agent_votes: list[AgentVote]

    # Regime context
    volatility_regime: str
    atr_pips: float

    # Outcome (None if trade still open)
    outcome: TradeOutcome | None

    @property
    def is_closed(self) -> bool:
        return self.outcome is not None

    @property
    def is_win(self) -> bool:
        return self.outcome is not None and self.outcome.trade_won

    @property
    def realized_pl(self) -> float:
        if self.outcome is None:
            return 0.0
        return self.outcome.realized_pl


@dataclass
class JournalStats:
    """Aggregated performance statistics for display."""

    total_trades: int = 0
    closed_trades: int = 0
    open_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0

    best_trade_pl: float = 0.0
    best_trade_id: str = ""
    worst_trade_pl: float = 0.0
    worst_trade_id: str = ""

    current_streak: int = 0  # positive = win streak, negative = loss streak
    max_win_streak: int = 0
    max_loss_streak: int = 0

    avg_rr_ratio: float = 0.0
    avg_duration_min: float = 0.0
    avg_confidence: float = 0.0

    sharpe_approx: float = 0.0

    # Per-exit-reason breakdown
    exit_reasons: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning default for None/invalid."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_agent_vote(raw: dict[str, Any]) -> AgentVote:
    """Parse a single agent vote dict into an AgentVote."""
    return AgentVote(
        name=str(raw.get("name") or "unknown"),
        score=_safe_float(raw.get("score")),
        passed=bool(raw.get("passed", False)),
        weight=_safe_float(raw.get("weight"), default=1.0),
        reason=str(raw.get("reason") or ""),
        reason_code=str(raw.get("reason_code") or ""),
        confidence_delta=_safe_float(raw.get("confidence_delta")),
    )


def _parse_outcome(raw: dict[str, Any] | None) -> TradeOutcome | None:
    """Parse an outcome dict into a TradeOutcome, or None if absent/empty."""
    if not raw or not isinstance(raw, dict):
        return None
    # Must have at least realized_pl to count as a real outcome
    if "realized_pl" not in raw:
        return None
    return TradeOutcome(
        trade_won=bool(raw.get("trade_won", False)),
        realized_pl=_safe_float(raw.get("realized_pl")),
        pnl_pips=_safe_float(raw.get("pnl_pips")),
        exit_reason=str(raw.get("exit_reason") or "unknown"),
        exit_price=_safe_float(raw.get("exit_price", raw.get("close_price"))),
        exit_time=str(raw.get("exit_time", raw.get("close_time")) or ""),
        duration_minutes=_safe_float(raw.get("duration_minutes")),
        mae_pips=_safe_float(raw.get("mae_pips")),
        mfe_pips=_safe_float(raw.get("mfe_pips")),
        rr_ratio=_safe_float(raw.get("rr_ratio", raw.get("risk_reward"))),
    )


def _parse_entry(raw: dict[str, Any]) -> JournalEntry | None:
    """Parse a single raw journal dict into a JournalEntry.

    Returns None if the dict is missing critical fields (trade_id, pair).
    All numeric fields use _safe_float to tolerate None values from
    upstream writers.
    """
    trade_id = raw.get("trade_id")
    pair = raw.get("pair")
    if not trade_id or not pair:
        logger.debug("Skipping journal entry with missing trade_id or pair")
        return None

    gates = raw.get("gates") or {}
    agents_block = raw.get("agents") or {}
    regime = raw.get("regime") or {}

    agent_reasons = agents_block.get("agent_reasons") or []
    parsed_votes: list[AgentVote] = []
    for ar in agent_reasons:
        if isinstance(ar, dict):
            parsed_votes.append(_parse_agent_vote(ar))

    return JournalEntry(
        trade_id=str(trade_id),
        pair=str(pair),
        direction=str(raw.get("direction") or "UNKNOWN"),
        confidence=_safe_float(raw.get("confidence")),
        entry_price=_safe_float(raw.get("entry_price")),
        sl_price=_safe_float(raw.get("sl_price")),
        tp_price=_safe_float(raw.get("tp_price")),
        sl_pips=_safe_float(raw.get("sl_pips")),
        tp_pips=_safe_float(raw.get("tp_pips")),
        rr_ratio=_safe_float(raw.get("rr_ratio", raw.get("risk_reward"))),
        lots=_safe_float(raw.get("lots")),
        timestamp=str(raw.get("timestamp") or ""),
        broker=str(raw.get("broker") or "unknown"),
        asset_class=str(raw.get("asset_class") or "FX"),
        gate_summary=str(gates.get("gate_summary") or ""),
        momentum_passed=bool(gates.get("momentum_passed", False)),
        confidence_passed=bool(gates.get("confidence_passed", False)),
        risk_passed=bool(gates.get("risk_passed", False)),
        weighted_vote_score=_safe_float(agents_block.get("weighted_vote_score")),
        agent_votes=parsed_votes,
        volatility_regime=str(regime.get("volatility_regime") or "UNKNOWN"),
        atr_pips=_safe_float(regime.get("atr_pips")),
        outcome=_parse_outcome(raw.get("outcome")),
    )


# ---------------------------------------------------------------------------
# JournalReader
# ---------------------------------------------------------------------------


class JournalReader:
    """Reads and processes trade journal data for TUI display.

    Thread-safe: all methods are pure readers with no side effects.
    All methods return safe defaults on error (empty lists, zeroed stats).

    Usage::

        reader = JournalReader()
        entries = reader.load()
        stats = reader.get_stats()
        curve = reader.get_pnl_curve()
    """

    def __init__(self, journal_path: str | Path | None = None) -> None:
        if journal_path is not None:
            self._path = Path(journal_path)
        else:
            project_root = Path(os.environ.get("BUDDY_ROOT", Path(__file__).resolve().parents[2]))
            self._path = project_root / _DEFAULT_JOURNAL

        self._entries: list[JournalEntry] | None = None

    # ------------------------------------------------------------------
    # Core: load
    # ------------------------------------------------------------------

    def load(self) -> list[JournalEntry]:
        """Parse the journal JSON and return typed JournalEntry list.

        Safe: returns empty list on missing file, bad JSON, or unexpected
        structure. Logs warnings but never raises.

        Results are cached after first load. Call ``reload()`` to force a
        fresh read.
        """
        if self._entries is not None:
            return self._entries

        self._entries = self._read_from_disk()
        return self._entries

    def reload(self) -> list[JournalEntry]:
        """Force a fresh read from disk, discarding cached entries."""
        self._entries = None
        return self.load()

    def _read_from_disk(self) -> list[JournalEntry]:
        """Internal: read and parse the journal file."""
        if not self._path.exists():
            logger.warning("Journal file not found: %s", self._path)
            return []

        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read journal file: %s", exc)
            return []

        try:
            raw_data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Journal file contains invalid JSON: %s", exc)
            return []

        if not isinstance(raw_data, list):
            logger.warning(
                "Journal root is %s, expected list -- returning empty",
                type(raw_data).__name__,
            )
            return []

        entries: list[JournalEntry] = []
        for idx, raw_entry in enumerate(raw_data):
            if not isinstance(raw_entry, dict):
                logger.debug("Skipping non-dict entry at index %d", idx)
                continue
            try:
                parsed = _parse_entry(raw_entry)
                if parsed is not None:
                    entries.append(parsed)
            except (TypeError, ValueError, KeyError) as exc:
                logger.debug("Failed to parse entry %d: %s", idx, exc)
                continue

        logger.info("Loaded %d journal entries from %s", len(entries), self._path)
        return entries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> JournalStats:
        """Compute aggregated performance statistics.

        Only closed trades (those with an outcome) contribute to win/loss
        metrics. Open trades are counted but excluded from P/L calculations.
        """
        entries = self.load()
        stats = JournalStats()

        if not entries:
            return stats

        stats.total_trades = len(entries)
        closed = [e for e in entries if e.is_closed]
        stats.closed_trades = len(closed)
        stats.open_trades = stats.total_trades - stats.closed_trades

        if not closed:
            return stats

        # Win / loss counts
        winners = [e for e in closed if e.is_win]
        losers = [e for e in closed if not e.is_win]
        stats.wins = len(winners)
        stats.losses = len(losers)
        stats.win_rate = stats.wins / stats.closed_trades if stats.closed_trades > 0 else 0.0

        # Gross profit / loss
        win_pls = [e.realized_pl for e in winners]
        loss_pls = [e.realized_pl for e in losers]  # these are negative values

        stats.gross_profit = sum(win_pls) if win_pls else 0.0
        stats.gross_loss = abs(sum(loss_pls)) if loss_pls else 0.0
        stats.net_pnl = stats.gross_profit - stats.gross_loss

        # Averages
        stats.avg_win = stats.gross_profit / stats.wins if stats.wins > 0 else 0.0
        stats.avg_loss = stats.gross_loss / stats.losses if stats.losses > 0 else 0.0

        # Expectancy: avg_win * win_rate - avg_loss * loss_rate
        loss_rate = 1.0 - stats.win_rate
        stats.expectancy = (stats.avg_win * stats.win_rate) - (stats.avg_loss * loss_rate)

        # Profit factor: gross_profit / gross_loss
        stats.profit_factor = (
            stats.gross_profit / stats.gross_loss
            if stats.gross_loss > 0
            else float("inf") if stats.gross_profit > 0 else 0.0
        )

        # Best / worst trade
        all_pls = [(e.realized_pl, e.trade_id) for e in closed]
        best = max(all_pls, key=lambda x: x[0])
        worst = min(all_pls, key=lambda x: x[0])
        stats.best_trade_pl = best[0]
        stats.best_trade_id = best[1]
        stats.worst_trade_pl = worst[0]
        stats.worst_trade_id = worst[1]

        # Streaks (chronological order, using the entries list order)
        stats.current_streak = self._compute_current_streak(closed)
        stats.max_win_streak, stats.max_loss_streak = self._compute_max_streaks(closed)

        # Averages: R:R, duration, confidence
        rr_values = [e.rr_ratio for e in closed if e.rr_ratio > 0]
        stats.avg_rr_ratio = sum(rr_values) / len(rr_values) if rr_values else 0.0

        durations = [
            e.outcome.duration_minutes
            for e in closed
            if e.outcome is not None and e.outcome.duration_minutes > 0
        ]
        stats.avg_duration_min = sum(durations) / len(durations) if durations else 0.0

        conf_values = [e.confidence for e in closed if e.confidence > 0]
        stats.avg_confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0

        # Sharpe approximation: mean(returns) / std(returns)
        stats.sharpe_approx = self._compute_sharpe(closed)

        # Exit reason distribution
        for e in closed:
            if e.outcome is not None:
                reason = e.outcome.exit_reason
                stats.exit_reasons[reason] = stats.exit_reasons.get(reason, 0) + 1

        return stats

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 20) -> list[JournalEntry]:
        """Return the last N journal entries (most recent last)."""
        entries = self.load()
        if n <= 0:
            return []
        return entries[-n:]

    def get_by_pair(self, pair: str) -> list[JournalEntry]:
        """Return all entries for a given currency pair.

        Matches case-insensitively and normalizes separators
        (EUR/USD, EUR_USD, EURUSD all match EUR_USD).
        """
        entries = self.load()
        normalized = pair.upper().replace("/", "_").replace(" ", "_")
        # Also handle the 6-char no-separator form: EURUSD -> EUR_USD
        if "_" not in normalized and len(normalized) == 6:
            normalized = normalized[:3] + "_" + normalized[3:]

        return [e for e in entries if e.pair.upper() == normalized]

    def get_closed(self) -> list[JournalEntry]:
        """Return only closed trades (those with an outcome)."""
        return [e for e in self.load() if e.is_closed]

    def get_open(self) -> list[JournalEntry]:
        """Return only open trades (those without an outcome)."""
        return [e for e in self.load() if not e.is_closed]

    # ------------------------------------------------------------------
    # Agent accuracy
    # ------------------------------------------------------------------

    def get_agent_accuracy(self) -> dict[str, float]:
        """Compute per-agent win rate when that agent voted 'passed'.

        Returns a dict of {agent_name: win_rate} where win_rate is 0.0-1.0.
        Only closed trades contribute. Agents that never passed get 0.0.
        """
        closed = self.get_closed()
        if not closed:
            return {}

        # Collect: for each agent, how many times it passed, and of those how many won
        agent_passed_count: dict[str, int] = {}
        agent_passed_wins: dict[str, int] = {}

        for entry in closed:
            for vote in entry.agent_votes:
                if vote.passed:
                    agent_passed_count[vote.name] = agent_passed_count.get(vote.name, 0) + 1
                    if entry.is_win:
                        agent_passed_wins[vote.name] = agent_passed_wins.get(vote.name, 0) + 1

        result: dict[str, float] = {}
        for agent_name, count in sorted(agent_passed_count.items()):
            wins = agent_passed_wins.get(agent_name, 0)
            result[agent_name] = wins / count if count > 0 else 0.0

        return result

    # ------------------------------------------------------------------
    # P/L curve
    # ------------------------------------------------------------------

    def get_pnl_curve(self) -> list[float]:
        """Compute cumulative P/L curve for sparkline display.

        Returns a list of running-total P/L values, one per closed trade
        in chronological order. Empty list if no closed trades.
        """
        closed = self.get_closed()
        if not closed:
            return []

        cumulative: list[float] = []
        running = 0.0
        for entry in closed:
            running += entry.realized_pl
            cumulative.append(round(running, 2))

        return cumulative

    def get_pnl_curve_pips(self) -> list[float]:
        """Cumulative P/L curve in pips (alternative to dollar P/L)."""
        closed = self.get_closed()
        if not closed:
            return []

        cumulative: list[float] = []
        running = 0.0
        for entry in closed:
            if entry.outcome is not None:
                running += entry.outcome.pnl_pips
            cumulative.append(round(running, 1))

        return cumulative

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_current_streak(closed: list[JournalEntry]) -> int:
        """Compute current streak from the end of the closed list.

        Positive = consecutive wins, negative = consecutive losses.
        """
        if not closed:
            return 0

        streak = 0
        last_won = closed[-1].is_win

        for entry in reversed(closed):
            if entry.is_win == last_won:
                streak += 1
            else:
                break

        return streak if last_won else -streak

    @staticmethod
    def _compute_max_streaks(closed: list[JournalEntry]) -> tuple[int, int]:
        """Return (max_win_streak, max_loss_streak) across all closed trades."""
        if not closed:
            return 0, 0

        max_win = 0
        max_loss = 0
        current = 0
        prev_won: bool | None = None

        for entry in closed:
            if entry.is_win:
                current = current + 1 if prev_won else 1
                max_win = max(max_win, current)
            else:
                current = current + 1 if prev_won is False else 1
                max_loss = max(max_loss, current)
            prev_won = entry.is_win

        return max_win, max_loss

    @staticmethod
    def _compute_sharpe(closed: list[JournalEntry]) -> float:
        """Approximate Sharpe ratio from trade returns.

        Sharpe = mean(returns) / std(returns).
        Returns 0.0 if fewer than 2 trades or zero variance.
        """
        if len(closed) < 2:
            return 0.0

        returns = [e.realized_pl for e in closed]
        n = len(returns)
        mean_r = sum(returns) / n
        variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)

        if variance <= 0:
            return 0.0

        std_r = math.sqrt(variance)
        if std_r == 0:
            return 0.0

        return round(mean_r / std_r, 4)
