"""Dataclasses for the Trade Homework System.

Three core types:
    HomeworkEntry  — frozen record of a closed trade + Buddy's analysis + review state
    Heuristic      — predicate + lesson_template that pattern-matches over (trade, outcome)
    TrainingSignal — payload emitted to RL queue when operator grades a homework entry

See spec §3 (Data Model) and §4.5 (Training Signal Payload).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class HomeworkEntry:
    """A closed trade + Buddy's structured analysis + review state.

    Frozen because review state transitions write a NEW entry to history rather
    than mutating the original. The pending → history move replaces the entry.
    """
    # Identity
    homework_id: str
    trade_id: str
    generated_at: str

    # Trade snapshot (denormalized for self-containment)
    pair: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    rr_ratio: float
    confidence: float
    weighted_vote_score: float
    regime: str
    agent_verdicts: List[Dict[str, Any]]

    # Outcome (from OANDA backfill)
    close_time: str
    close_price: float
    realized_pl: float
    close_reason: str  # TP | SL | MANUAL
    duration_minutes: int
    mfe_pips: float
    mae_pips: float

    # Analysis (Buddy's homework)
    analysis_markdown: str
    proposed_lesson: str
    confidence_in_analysis: float
    agents_to_reinforce: List[str]
    agents_to_penalize: List[str]

    # Review state (defaults; updated via HomeworkReviewer transition)
    schema_version: int = 1
    status: str = "pending"
    operator_grade: Optional[str] = None
    operator_note: Optional[str] = None
    operator_edits: Optional[Dict[str, Any]] = None
    reviewed_at: Optional[str] = None


@dataclass
class Heuristic:
    """A pattern-matching rule that runs over (trade, outcome) and proposes a lesson.

    The predicate is a closure: takes a TradeView and an OutcomeView (lightweight
    dicts with attribute-style access) and returns bool. Multiple heuristics may
    fire on one trade; highest-confidence wins.
    """
    id: str  # e.g. "A1"
    name: str  # e.g. "setup_adx_trend_mismatch"
    category: str  # A | B | C | D | E | F
    predicate: Callable[[Any, Any], bool]
    lesson_template: str  # Python f-string-like with {trade.x} / {outcome.y} / {atr_pips} placeholders
    confidence: float  # 0.0 - 1.0
    source: str  # citation, e.g. "Bellafiore One Good Trade Ch.4"


@dataclass
class TrainingSignal:
    """Payload emitted by HomeworkReviewer.transition() when operator grades a homework.

    See spec §4.5. Approved entries apply deltas as Buddy proposed.
    Edited entries apply operator's edited deltas instead.
    Rejected entries discard deltas; record heuristic in rejected_heuristics_log.jsonl.
    """
    homework_id: str
    trade_id: str
    outcome: str  # TP | SL | MANUAL
    agent_weight_deltas: Dict[str, float]
    regime_prior_deltas: Dict[str, Dict[str, float]]
    heuristic_fired: Optional[str]
    operator_action: str  # approved | edited | rejected
    operator_note: Optional[str] = None
    regime: str = "UNKNOWN"  # which regime row in agent_weights.json the deltas target
