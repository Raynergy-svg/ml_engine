"""HEURISTIC_CATALOG — pattern-matching rules for trade homework.

Organized into 6 categories per spec §4.3:
  A — Setup Validity              (Steenbarger, Bellafiore)
  B — Risk Calibration            (Kelly, Seykota, Phase 91)
  C — Agent Consensus Quality     (Phase 91+93 promoted rules)
  D — Execution Quality           (Bellafiore, Steenbarger)
  E — Regime / Context Drift      (Raschke, Phase 91 staleness)
  F — Meta-Patterns               (López de Prado meta-labeling)

Predicates take a TradeView and an OutcomeView (SimpleNamespace-like). Adding
new heuristics: append a Heuristic(...) entry. Generator picks them up at
import time. Each entry MUST have a `source` field for auditability.
"""
from __future__ import annotations

from typing import Any, List

from src.scanner.automation.homework.types import Heuristic


def _agent(trade: Any, name: str) -> dict:
    """Look up an agent verdict by name; returns empty dict if absent."""
    for v in getattr(trade, "agent_verdicts", []) or []:
        if v.get("name") == name:
            return v
    return {}


# ---------------- Category A — Setup Validity ----------------

A1 = Heuristic(
    id="A1",
    name="setup_adx_trend_mismatch",
    category="A",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and t.direction in ("LONG", "SHORT")
        and getattr(t, "adx", 99.0) < 10.0
    ),
    lesson_template=(
        "ADX={adx:.0f} too low for a directional trade — no trend present. "
        "Suggest hard-veto direction when ADX < 10 in {regime} regime."
    ),
    confidence=0.85,
    source="Bellafiore One Good Trade Ch.4",
)

A2 = Heuristic(
    id="A2",
    name="setup_volatility_regime_mismatch",
    category="A",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and (
            (t.regime == "LOW" and t.direction in ("LONG", "SHORT") and getattr(t, "adx", 99.0) < 15.0)
            or (t.regime == "HIGH" and _agent(t, "mean_reversion").get("passed") is True)
        )
    ),
    lesson_template=(
        "Strategy mismatched to {regime} regime. Directional in LOW (ranging) "
        "or mean-reversion in HIGH (trending) is the wrong setup type."
    ),
    confidence=0.80,
    source="Raschke Street Smarts Ch.3",
)

A3 = Heuristic(
    id="A3",
    name="setup_session_mismatch",
    category="A",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and getattr(t, "session", None) == "TOKYO"
        and t.pair in {"EUR_GBP", "EUR_USD", "GBP_USD"}
    ),
    lesson_template="EUR/GBP-family pair traded in Tokyo session — illiquid, wide spreads.",
    confidence=0.65,
    source="Bellafiore One Good Trade Ch.5",
)

A4 = Heuristic(
    id="A4",
    name="setup_rsi_neutral",
    category="A",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and t.direction in ("LONG", "SHORT")
        and 45.0 <= getattr(t, "rsi", 50.0) <= 55.0
    ),
    lesson_template=(
        "RSI={rsi:.1f} in neutral zone (45-55) — no momentum bias either direction. "
        "Setup quality was low at entry."
    ),
    confidence=0.70,
    source="Steenbarger Trading Psychology 2.0 Ch.7",
)


# ---------------- Category B — Risk Calibration ----------------

B1 = Heuristic(
    id="B1",
    name="risk_rr_below_breakeven",
    category="B",
    predicate=lambda t, o: t.rr_ratio < 1.2,
    lesson_template=(
        "R:R {rr_ratio:.2f} below 1.2 minimum (Phase 91 rule). "
        "Even 50% win-rate produces near-zero expectancy."
    ),
    confidence=0.90,
    source="Kelly criterion / Phase 91 trading.md",
)

B2 = Heuristic(
    id="B2",
    name="risk_sl_too_tight_for_atr",
    category="B",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and getattr(t, "atr_pips", 0) > 0
        and (t.sl_pips / t.atr_pips) < 1.0
    ),
    lesson_template=(
        "SL/ATR = {sl_to_atr:.2f} below 1.0 — stop placed inside normal noise. "
        "Whipsaw was nearly guaranteed."
    ),
    confidence=0.85,
    source="Seykota / Market Wizards",
)

B3 = Heuristic(
    id="B3",
    name="risk_low_regime_sl_violation",
    category="B",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and t.regime == "LOW"
        and getattr(t, "atr_pips", 0) > 0
        and (t.sl_pips / t.atr_pips) < 1.2
    ),
    lesson_template=(
        "LOW regime + sl_mult={sl_to_atr:.2f} < 1.2 violates Phase 91 promoted rule. "
        "Ranging markets need wider stops, not tighter."
    ),
    confidence=0.95,
    source="Phase 91 trading.md (LOW regime sl_mult >= 1.2)",
)

B4 = Heuristic(
    id="B4",
    name="risk_correlated_double_exposure",
    category="B",
    predicate=lambda t, o: bool(getattr(t, "correlated_open_at_entry", False)),
    lesson_template=(
        "Another correlated pair was already open at entry. "
        "Effective leverage doubled vs intended."
    ),
    confidence=0.75,
    source="Raschke Street Smarts Ch.6",
)


# ---------------- Category C — Agent Consensus Quality ----------------

C1 = Heuristic(
    id="C1",
    name="consensus_trend_veto_unhonored",
    category="C",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and t.direction in ("LONG", "SHORT")
        and _agent(t, "trend").get("passed") is False
    ),
    lesson_template=(
        "Trend agent voted NO (score={trend_score:.2f}) but trade executed anyway. "
        "Phase 91 rule: trend.passed=False is a hard veto on directional trades."
    ),
    confidence=0.90,
    source="Phase 91 trading.md (trend hard-veto rule)",
)

C2 = Heuristic(
    id="C2",
    name="consensus_mr_composite_match",
    category="C",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and _agent(t, "mean_reversion").get("passed") is False
        and (t.gate_details or {}).get("model_disagreement", 0.0) > 0.25
    ),
    lesson_template=(
        "MR voted NO + model_disagreement={disagree:.2f} > 0.25. "
        "Phase 93 composite veto fingerprint — would have caught this trade."
    ),
    confidence=0.90,
    source="Phase 93 trading.md (MR composite veto)",
)

C3 = Heuristic(
    id="C3",
    name="consensus_disagreement_at_floor",
    category="C",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and abs(
            (t.gate_details or {}).get("model_disagreement", 0.0)
            - (t.gate_details or {}).get("disagreement_hard_floor", 0.50)
        ) < 0.03
    ),
    lesson_template=(
        "model_disagreement was within 0.03 of hard_floor — boundary case. "
        "Consider tightening disagreement_hard_floor by 0.02."
    ),
    confidence=0.70,
    source="Phase 91 trading.md (disagreement boundary)",
)

C4 = Heuristic(
    id="C4",
    name="consensus_high_winner",
    category="C",
    predicate=lambda t, o: (
        o.close_reason == "TP"
        and t.weighted_vote_score > 0.75
        and sum(1 for v in t.agent_verdicts if v.get("passed")) >= 10
    ),
    lesson_template=(
        "TP + WVS={wvs:.2f} + {n_passed} agents passed. "
        "High-confluence pattern — reinforce."
    ),
    confidence=0.80,
    source="Bayesian voting / promoted-pattern reinforcement",
)

C5 = Heuristic(
    id="C5",
    name="consensus_single_agent_dragged",
    category="C",
    predicate=lambda t, o: False,  # Implementation deferred — needs agent contribution analysis
    lesson_template=(
        "One high-weight agent's vote dragged the consensus opposite the outcome. "
        "Audit that agent's regime weights."
    ),
    confidence=0.65,
    source="Bayesian voting integrity",
)


# ---------------- Category D — Execution Quality ----------------

D1 = Heuristic(
    id="D1",
    name="exec_mfe_zero_directional_loss",
    category="D",
    predicate=lambda t, o: (
        o.close_reason == "SL"
        and t.direction in ("LONG", "SHORT")
        and getattr(t, "atr_pips", 0) > 0
        and (o.mfe_pips / t.atr_pips) < 0.2
    ),
    lesson_template=(
        "MFE/ATR={mfe_to_atr:.2f} — price never moved in our favor. "
        "Entry was directionally wrong from tick 1. "
        "(04-15 catastrophic-streak fingerprint.)"
    ),
    confidence=0.85,
    source="Phase 91 04-15 streak forensics",
)

D2 = Heuristic(
    id="D2",
    name="exec_whipsaw_reversal",
    category="D",
    predicate=lambda t, o: bool(getattr(t, "whipsaw_reversal_within_2atr_60min", False)),
    lesson_template=(
        "SL hit, then price reversed past entry within 2× ATR over the next 60min. "
        "Stop was too tight; thesis was correct, exit premature."
    ),
    confidence=0.80,
    source="Seykota / Market Wizards",
)

D3 = Heuristic(
    id="D3",
    name="exec_slow_tp_widening_candidate",
    category="D",
    predicate=lambda t, o: (
        o.close_reason == "TP"
        and o.duration_minutes > 4 * max(1, getattr(t, "expected_hold_minutes", 60))
    ),
    lesson_template=(
        "TP hit slowly ({duration}min vs {expected_hold}min expected). "
        "Consider widening tp_mult to capture more."
    ),
    confidence=0.60,
    source="Bellafiore One Good Trade Ch.6",
)

D4 = Heuristic(
    id="D4",
    name="exec_fast_sl_bad_timing",
    category="D",
    predicate=lambda t, o: o.close_reason == "SL" and o.duration_minutes < 5,
    lesson_template=(
        "Stopped in {duration}min. Likely news event, bad fill, or stale signal."
    ),
    confidence=0.70,
    source="Steenbarger Trading Psychology 2.0 Ch.4",
)

D5 = Heuristic(
    id="D5",
    name="exec_slippage_cost_winner",
    category="D",
    predicate=lambda t, o: (
        o.close_reason == "TP"
        and abs(getattr(t, "slippage_pips", 0.0)) > 0.3 * t.sl_pips
    ),
    lesson_template=(
        "Slippage {slippage:.1f} pips ate {slip_pct:.0f}% of risk. "
        "Review broker latency or use limit orders."
    ),
    confidence=0.55,
    source="Seykota / Market Wizards",
)


# ---------------- Category E — Regime / Context Drift ----------------

E1 = Heuristic(
    id="E1",
    name="context_stale_models",
    category="E",
    predicate=lambda t, o: (
        o.close_reason == "SL" and getattr(t, "oldest_age_days", 0.0) > 7.0
    ),
    lesson_template=(
        "oldest_age_days={age:.1f} > 7 — Phase 91 staleness threshold violated. "
        "Models predicting old regime into new market."
    ),
    confidence=0.85,
    source="Phase 91 trading.md (staleness block)",
)

E2 = Heuristic(
    id="E2",
    name="context_regime_transition",
    category="E",
    predicate=lambda t, o: (
        getattr(t, "entry_regime", t.regime) != getattr(o, "close_regime", t.regime)
    ),
    lesson_template=(
        "Entered in {entry_regime}, closed in {close_regime}. "
        "Regime shift mid-trade — was thesis still valid after the shift?"
    ),
    confidence=0.65,
    source="Raschke Street Smarts Ch.3",
)

E3 = Heuristic(
    id="E3",
    name="context_news_window",
    category="E",
    predicate=lambda t, o: bool(getattr(t, "news_window", False)),
    lesson_template=(
        "High-impact news (NFP/FOMC/CPI/ECB) during trade duration. "
        "Outcome may reflect news shock not setup quality."
    ),
    confidence=0.70,
    source="Steenbarger Trading Psychology 2.0 Ch.5",
)

E4 = Heuristic(
    id="E4",
    name="context_correlated_co_loss",
    category="E",
    predicate=lambda t, o: getattr(t, "correlated_co_losses_30min", 0) >= 2,
    lesson_template=(
        "{n_co_losses} correlated pairs closed at SL within 30min. "
        "Regime issue not single-trade issue — consider regime-pause heuristic."
    ),
    confidence=0.75,
    source="Raschke Street Smarts Ch.6",
)


# ---------------- Category F — Meta-Patterns ----------------

F1 = Heuristic(
    id="F1",
    name="meta_repeat_fingerprint",
    category="F",
    predicate=lambda t, o: getattr(t, "matching_recent_loss_count", 0) >= 3,
    lesson_template=(
        "Last {n} trades sharing this gate-combination all lost. "
        "Suggests structural hole in gate logic, not bad luck."
    ),
    confidence=0.85,
    source="López de Prado Advances in Financial ML Ch.3 (meta-labeling)",
)

F2 = Heuristic(
    id="F2",
    name="meta_lucky_winner",
    category="F",
    predicate=lambda t, o: (
        o.close_reason == "TP"
        and t.sl_pips > 0
        and (o.mae_pips / t.sl_pips) > 0.7
    ),
    lesson_template=(
        "TP, but MAE/SL={mae_to_sl:.2f} > 0.7 — trade nearly stopped before reversing. "
        "Profit depended on luck not skill — don't reinforce confidently."
    ),
    confidence=0.70,
    source="López de Prado Advances in Financial ML Ch.3",
)

F3 = Heuristic(
    id="F3",
    name="meta_underrepresented_setup",
    category="F",
    predicate=lambda t, o: getattr(t, "training_cluster_n_examples", 999) < 10,
    lesson_template=(
        "Setup-type cluster has {n_examples} training examples (< 10). "
        "Model was extrapolating; outcome carries low evidential weight."
    ),
    confidence=0.60,
    source="López de Prado Advances in Financial ML Ch.4 (sample weights)",
)


HEURISTIC_CATALOG: List[Heuristic] = [
    A1, A2, A3, A4,
    B1, B2, B3, B4,
    C1, C2, C3, C4, C5,
    D1, D2, D3, D4, D5,
    E1, E2, E3, E4,
    F1, F2, F3,
]
