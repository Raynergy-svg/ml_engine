"""Trade journal shape adapter — bridges the two journal schemas.

The trade journal at `trained_data/trade_journal_rl.json` is written by Buddy's
production loop in a *flat* shape: `outcome` is a string ("win"/"loss"), and
`close_reason`, `close_time`, `realized_pl` are promoted to the trade's top
level. Test fixtures and the future schema use a *nested* shape: `outcome` is
a dict with `close_reason`, `close_time`, `realized_pl`, etc.

`HomeworkGenerator.generate(trade, outcome)` expects the *nested* shape. This
module provides `normalize_trade(raw_trade)` which returns a `(trade, outcome)`
pair in the nested shape — or `None` if the trade is not closed yet.

Used by:
    - buddy_scanner.py (`homework --generate-batch` CLI)
    - src/tui/screens/inbox_screen.py (Phase 96 Task 5 — homework rendering)
"""
from __future__ import annotations

from typing import Optional


def normalize_trade(trade: dict) -> Optional[tuple[dict, dict]]:
    """Return a (trade, outcome) pair in the shape the HomeworkGenerator expects.

    Two journal shapes exist in the wild:

    1. **Nested-shape** (test fixtures, future schema): `outcome` is a dict
       with keys like `close_reason`, `close_time`, `realized_pl`, …; `agents`
       is a list of `{name, passed, score, weight, …}` dicts.

    2. **Flat-shape** (legacy production journal at trained_data/...): `outcome`
       is a string ("win"/"loss"); `close_reason`, `close_time`, `realized_pl`
       are promoted to the trade's top level; `agents` is a dict containing
       `agent_reasons` (the actual list of per-agent verdicts).

    Returns None if the trade is not closed (no `close_reason`).
    """
    if not isinstance(trade, dict):
        return None

    outcome_raw = trade.get("outcome")

    # Shape 1: nested outcome dict.
    if isinstance(outcome_raw, dict):
        if not outcome_raw.get("close_reason"):
            return None
        return trade, outcome_raw

    # Shape 2: flat journal — synthesize nested outcome and re-shape agents.
    close_reason = trade.get("close_reason")
    if not close_reason:
        return None

    outcome = {
        "close_time": trade.get("close_time", ""),
        "close_price": float(trade.get("close_price", trade.get("fill_price", 0.0)) or 0.0),
        "realized_pl": float(trade.get("realized_pl", 0.0) or 0.0),
        "close_reason": close_reason,
        "duration_minutes": int(trade.get("duration_minutes", 0) or 0),
        "mfe_pips": float(trade.get("mfe_pips", 0.0) or 0.0),
        "mae_pips": float(trade.get("mae_pips", 0.0) or 0.0),
    }

    # Build a shallow-copied trade with `agents` reshaped to list-of-dicts and
    # weighted_vote_score promoted to top-level (the generator reads it from
    # the trade record, not from the agents block).
    agents_raw = trade.get("agents")
    agents_list: list[dict] = []
    weighted_vote_score = float(trade.get("weighted_vote_score", 0.0) or 0.0)
    if isinstance(agents_raw, dict):
        agents_list = list(agents_raw.get("agent_reasons", []) or [])
        if "weighted_vote_score" in agents_raw:
            weighted_vote_score = float(agents_raw.get("weighted_vote_score") or 0.0)
    elif isinstance(agents_raw, list):
        agents_list = agents_raw

    # Regime field in the production journal is a dict; the generator stores
    # it as a string. Reduce to volatility_regime label.
    regime_raw = trade.get("regime")
    if isinstance(regime_raw, dict):
        regime_label = str(regime_raw.get("volatility_regime", "UNKNOWN"))
    else:
        regime_label = str(regime_raw or "UNKNOWN")

    normalized_trade = dict(trade)
    normalized_trade["agents"] = agents_list
    normalized_trade["weighted_vote_score"] = weighted_vote_score
    normalized_trade["regime"] = regime_label

    return normalized_trade, outcome
