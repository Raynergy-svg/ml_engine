"""
Scanner specialist agents.

Adds a lightweight deliberation layer on top of the existing scanner signal.
Each agent evaluates one aspect of the setup and emits a structured verdict
that can be combined into a weighted vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from .results import PairAnalysis


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if result != result:
            return default
        return result
    except Exception:
        return default


def _last_value(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in df.columns or df.empty:
        return default
    return _safe_float(df[column].iloc[-1], default)


@dataclass
class AgentDecisionContext:
    """Normalized context shared by specialist agents."""

    analysis: PairAnalysis
    df_raw: pd.DataFrame
    df_feat: pd.DataFrame
    gate_details: Dict[str, Any]
    config: Any


@dataclass
class AgentVerdict:
    """Outcome from one specialist agent."""

    name: str
    score: float
    passed: bool
    weight: float
    reason: str
    reason_code: str
    confidence_delta: float = 0.0
    block_trade: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": _clip01(self.score),
            "passed": bool(self.passed),
            "weight": float(self.weight),
            "reason": self.reason,
            "reason_code": self.reason_code,
            "confidence_delta": float(self.confidence_delta),
            "block_trade": bool(self.block_trade),
            "metadata": dict(self.metadata),
        }


class ScannerAgentTeam:
    """Specialist-agent layer for scanner opportunity assessment."""

    _BASE_WEIGHTS: Dict[str, float] = {
        "trend": 1.15,
        "mean_reversion": 0.90,
        "volatility": 1.00,
        "risk_sentinel": 1.25,
        "uncertainty": 1.10,
        "execution_quality": 1.05,
        "news_risk": 0.95,
        "multi_timeframe": 1.10,
        "pair_performance": 0.85,
        "momentum": 1.05,
        "session_timing": 0.80,
        "support_resistance": 1.00,
    }

    _WEIGHTS_FILE = "trained_data/models/agent_weights.json"

    def __init__(self, config: Any):
        self.config = config
        self._learned_weights: Dict[str, float] = self._load_learned_weights()

    # --- Persistent weight management ---

    def _load_learned_weights(self) -> Dict[str, float]:
        """Load learned agent weights from disk."""
        import json
        from pathlib import Path

        path = Path(self._WEIGHTS_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                return {str(k): float(v) for k, v in data.items()}
            except Exception:
                pass
        return {}

    def reload_learned_weights(self) -> None:
        """Reload agent weights from disk (after RL sync or weight decay).

        This ensures the team always uses the latest learned weights.
        Call before scanning if weights may have been updated since __init__.
        """
        self._learned_weights = self._load_learned_weights()

    def _save_learned_weights(self) -> None:
        """Persist learned agent weights to disk."""
        import json
        from pathlib import Path

        path = Path(self._WEIGHTS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._learned_weights, f, indent=2)

    def apply_weight_decay(self, decay_rate: float = 0.02) -> Dict[str, float]:
        """Decay learned weights toward base weights to prevent overfitting.

        Should be called once per scan cycle (e.g., from ContinuousScanner).
        Each call moves learned weights ``decay_rate`` fraction closer to the
        base weight.  This ensures recent RL adjustments don't persist
        indefinitely when the sample size is small.

        Args:
            decay_rate: Fraction to move toward base weight per call (0.0–1.0).

        Returns:
            Dict of agent name -> new weight after decay.
        """
        if not self._learned_weights:
            return {}

        decay_rate = max(0.0, min(1.0, decay_rate))
        changed = False
        for name in list(self._learned_weights):
            base = self._BASE_WEIGHTS.get(name, 1.0)
            current = self._learned_weights[name]
            if abs(current - base) < 1e-4:
                # Already at base – remove from learned dict
                del self._learned_weights[name]
                changed = True
                continue
            new_weight = current + decay_rate * (base - current)
            new_weight = round(new_weight, 4)
            self._learned_weights[name] = new_weight
            changed = True

        if changed:
            self._save_learned_weights()
        return dict(self._learned_weights)

    def update_weights_from_outcome(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
    ) -> Dict[str, float]:
        """Update agent weights based on trade outcome (RL feedback).

        Args:
            agent_verdicts: List of agent verdict dicts from the trade's PairAnalysis
            trade_won: True if trade hit TP, False if hit SL

        Returns:
            Dict of agent name -> new weight
        """
        boost = _safe_float(getattr(self.config, "weight_boost_on_win", 0.10), 0.10)
        penalty = _safe_float(getattr(self.config, "weight_penalty_on_loss", 0.15), 0.15)
        min_w = _safe_float(getattr(self.config, "min_agent_weight", 0.1), 0.1)
        max_w = _safe_float(getattr(self.config, "max_agent_weight", 2.0), 2.0)

        for verdict in agent_verdicts:
            name = str(verdict.get("name", ""))
            if not name:
                continue
            current = self._learned_weights.get(name, self._BASE_WEIGHTS.get(name, 1.0))
            if verdict.get("passed"):
                # Agent voted for the trade
                delta = boost if trade_won else -penalty
            else:
                # Agent voted against the trade
                delta = -boost * 0.5 if trade_won else penalty * 0.5
            new_weight = max(min_w, min(max_w, current + delta))
            self._learned_weights[name] = round(new_weight, 4)

        self._save_learned_weights()
        return dict(self._learned_weights)

    def evaluate(
        self,
        analysis: PairAnalysis,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        gate_details: Optional[Dict[str, Any]] = None,
    ) -> PairAnalysis:
        """Apply specialist agents to a scan result."""
        if analysis.error is not None or analysis.direction not in {"LONG", "SHORT"}:
            return analysis

        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details=gate_details or {},
            config=self.config,
        )

        verdicts: List[AgentVerdict] = []

        if getattr(self.config, "enable_trend_agent", True):
            verdicts.append(self._evaluate_trend(ctx))
        if getattr(self.config, "enable_mean_reversion_agent", True):
            verdicts.append(self._evaluate_mean_reversion(ctx))
        if getattr(self.config, "enable_volatility_agent", True):
            verdicts.append(self._evaluate_volatility(ctx))
        if getattr(self.config, "enable_risk_sentinel_agent", True):
            verdicts.append(self._evaluate_risk(ctx))
        if getattr(self.config, "enable_uncertainty_agent", True):
            verdicts.append(self._evaluate_uncertainty(ctx))
        if getattr(self.config, "enable_execution_quality_agent", True):
            verdicts.append(self._evaluate_execution_quality(ctx))
        if getattr(self.config, "enable_news_risk_agent", False):
            news_verdict = self._evaluate_news_risk(ctx)
            if news_verdict is not None:
                verdicts.append(news_verdict)
        if getattr(self.config, "enable_multi_timeframe_agent", False):
            verdicts.append(self._evaluate_multi_timeframe(ctx))
        if getattr(self.config, "enable_pair_performance_agent", False):
            perf_verdict = self._evaluate_pair_performance(ctx)
            if perf_verdict is not None:
                verdicts.append(perf_verdict)
        if self.config.enable_momentum_agent:
            verdicts.append(self._evaluate_momentum(ctx))
        if self.config.enable_session_timing_agent:
            verdicts.append(self._evaluate_session_timing(ctx))
        if self.config.enable_support_resistance_agent:
            sr_verdict = self._evaluate_support_resistance(ctx)
            if sr_verdict is not None:
                verdicts.append(sr_verdict)

        if not verdicts:
            return analysis

        total_weight = sum(max(v.weight, 0.0) for v in verdicts) or 1.0
        weighted_vote_score = sum(_clip01(v.score) * max(v.weight, 0.0) for v in verdicts) / total_weight

        regime_name = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        regime_thresholds = getattr(self.config, "regime_vote_thresholds", {}) or {}
        weighted_vote_threshold = max(
            _safe_float(getattr(self.config, "weighted_vote_threshold", 0.55), 0.55),
            _safe_float(regime_thresholds.get(regime_name), 0.0),
        )

        confidence_adjustment = sum(float(v.confidence_delta) * max(v.weight, 0.0) for v in verdicts) / total_weight
        block_trade = any(v.block_trade for v in verdicts)

        analysis.confidence = _clip01(float(analysis.confidence) + confidence_adjustment)
        analysis.weighted_vote_score = _clip01(weighted_vote_score)
        analysis.weighted_vote_threshold = _clip01(weighted_vote_threshold)
        analysis.agent_reasons = [v.to_dict() for v in verdicts]
        analysis.agent_reason_codes = list(dict.fromkeys(v.reason_code for v in verdicts if v.reason_code))
        analysis.why_trade = [v.reason for v in verdicts if v.passed and v.reason]
        analysis.why_no_trade = [v.reason for v in verdicts if (not v.passed or v.block_trade) and v.reason]

        uncertainty_meta = next((v.metadata for v in verdicts if v.name == "uncertainty"), {})
        analysis.uncertainty_score = _clip01(_safe_float(uncertainty_meta.get("uncertainty_score"), analysis.uncertainty_score))
        analysis.confidence_variance = _clip01(_safe_float(uncertainty_meta.get("confidence_variance"), analysis.confidence_variance))
        analysis.model_disagreement = _clip01(_safe_float(uncertainty_meta.get("model_disagreement"), analysis.model_disagreement))

        execution_meta = next((v.metadata for v in verdicts if v.name == "execution_quality"), {})
        analysis.execution_quality_score = _clip01(_safe_float(execution_meta.get("execution_quality_score"), analysis.execution_quality_score))
        analysis.execution_quality_passed = bool(execution_meta.get("execution_quality_passed", analysis.execution_quality_passed))
        analysis.spread_pips = _safe_float(execution_meta.get("spread_pips"), analysis.spread_pips)
        analysis.est_slippage_pips = _safe_float(execution_meta.get("est_slippage_pips"), analysis.est_slippage_pips)
        analysis.liquidity_score = _clip01(_safe_float(execution_meta.get("liquidity_score"), analysis.liquidity_score))

        if block_trade:
            analysis.blocked_by_circuit_breaker = True
            analysis.circuit_breakers_triggered = list(
                dict.fromkeys(v.reason_code for v in verdicts if v.block_trade and v.reason_code)
            )

        if analysis.weighted_vote_score < analysis.weighted_vote_threshold:
            analysis.why_no_trade.append(
                f"agent vote {analysis.weighted_vote_score:.2f} below {analysis.weighted_vote_threshold:.2f}"
            )

        if block_trade or analysis.weighted_vote_score < analysis.weighted_vote_threshold:
            analysis.gates_passed = False

        analysis.why_trade = list(dict.fromkeys(analysis.why_trade))
        analysis.why_no_trade = list(dict.fromkeys(analysis.why_no_trade))
        return analysis

    def _weight_for(self, name: str) -> float:
        # Prefer learned weights from RL feedback, fall back to base
        if name in self._learned_weights:
            return float(self._learned_weights[name])
        return float(self._BASE_WEIGHTS.get(name, 1.0))

    def _direction_matches(self, signal_direction: float, expected: str) -> bool:
        if expected == "LONG":
            return signal_direction > 0
        if expected == "SHORT":
            return signal_direction < 0
        return False

    def _evaluate_trend(self, ctx: AgentDecisionContext) -> AgentVerdict:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        adx = _last_value(ctx.df_feat, "adx", _safe_float(ctx.analysis.ridge_confidence))

        trend_signal = 0.0
        if close > sma_20 >= sma_50:
            trend_signal = 1.0
        elif close < sma_20 <= sma_50:
            trend_signal = -1.0

        aligned = self._direction_matches(trend_signal, ctx.analysis.direction)
        contra = trend_signal != 0.0 and not aligned
        adx_support = _clip01((adx - 18.0) / 22.0)

        score = 0.48 + adx_support * 0.24
        if aligned:
            score += 0.22
        elif contra:
            score -= 0.24

        if aligned:
            reason = f"trend aligned (ADX {adx:.0f})"
            reason_code = "trend_align"
            confidence_delta = 0.04
        elif contra:
            reason = f"trend conflicts with {ctx.analysis.direction.lower()} bias"
            reason_code = "trend_contra"
            confidence_delta = -0.06
        else:
            reason = f"trend neutral (ADX {adx:.0f})"
            reason_code = "trend_neutral"
            confidence_delta = 0.0

        return AgentVerdict(
            name="trend",
            score=_clip01(score),
            passed=score >= 0.55,
            weight=self._weight_for("trend"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
        )

    def _evaluate_mean_reversion(self, ctx: AgentDecisionContext) -> AgentVerdict:
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        direction = ctx.analysis.direction

        if direction == "LONG":
            support = _clip01((50.0 - rsi) / 20.0)
            oppose = _clip01((rsi - 58.0) / 20.0)
        else:
            support = _clip01((rsi - 50.0) / 20.0)
            oppose = _clip01((42.0 - rsi) / 20.0)

        score = _clip01(0.50 + support * 0.25 - oppose * 0.30)
        if support >= 0.40:
            reason = f"mean reversion supports {direction.lower()} (RSI {rsi:.1f})"
            reason_code = "mean_reversion_support"
            confidence_delta = 0.02
        elif oppose >= 0.45:
            reason = f"RSI stretched against {direction.lower()} (RSI {rsi:.1f})"
            reason_code = "mean_reversion_contra"
            confidence_delta = -0.04
        else:
            reason = f"RSI neutral (RSI {rsi:.1f})"
            reason_code = "mean_reversion_neutral"
            confidence_delta = 0.0

        return AgentVerdict(
            name="mean_reversion",
            score=score,
            passed=score >= 0.52,
            weight=self._weight_for("mean_reversion"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
        )

    def _evaluate_volatility(self, ctx: AgentDecisionContext) -> AgentVerdict:
        atr_pips = _safe_float(ctx.analysis.atr_pips)
        vol_pct = _clip01(_safe_float(ctx.analysis.volatility_percentile, 0.5))
        regime = str(getattr(ctx.analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        regime_score = {
            "LOW": 0.25,
            "NORMAL": 0.52,
            "HIGH": 0.72,
            "EXTREME": 0.85,
        }.get(regime, 0.55)

        min_atr = max(_safe_float(getattr(ctx.config, "min_atr_pips", 5.0), 5.0), 0.1)
        atr_score = _clip01(atr_pips / (min_atr * 1.5))
        score = _clip01(0.20 + atr_score * 0.45 + vol_pct * 0.20 + regime_score * 0.15)

        block_trade = bool(getattr(ctx.analysis, "volatility_gate_passed", True) is False) or atr_pips < (min_atr * 0.50)
        if block_trade:
            reason = f"volatility too weak ({atr_pips:.1f} pips)"
            reason_code = "volatility_block"
            confidence_delta = -0.07
        elif score >= 0.60:
            reason = f"volatility supportive ({regime.lower()}, {atr_pips:.1f} pips)"
            reason_code = "volatility_support"
            confidence_delta = 0.03
        else:
            reason = f"volatility mixed ({regime.lower()}, {atr_pips:.1f} pips)"
            reason_code = "volatility_mixed"
            confidence_delta = -0.01

        return AgentVerdict(
            name="volatility",
            score=score,
            passed=score >= 0.52 and not block_trade,
            weight=self._weight_for("volatility"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_risk(self, ctx: AgentDecisionContext) -> AgentVerdict:
        drawdown = _safe_float(ctx.analysis.drawdown)
        max_drawdown = max(_safe_float(getattr(ctx.config, "max_drawdown_pct", 0.025), 0.025), 1e-6)
        risk_ratio = drawdown / max_drawdown

        score = _clip01(1.0 - min(risk_ratio, 2.0) / 2.0)
        if bool(ctx.analysis.risk_passed):
            score = _clip01(score + 0.12)

        block_trade = (not bool(ctx.analysis.risk_passed)) and risk_ratio >= 1.0
        if block_trade:
            reason = f"risk sentinel blocked ({drawdown:.2%} drawdown)"
            reason_code = "risk_block"
            confidence_delta = -0.08
        elif score >= 0.60:
            reason = f"risk acceptable ({drawdown:.2%} drawdown)"
            reason_code = "risk_ok"
            confidence_delta = 0.02
        else:
            reason = f"risk elevated ({drawdown:.2%} drawdown)"
            reason_code = "risk_elevated"
            confidence_delta = -0.03

        return AgentVerdict(
            name="risk_sentinel",
            score=score,
            passed=score >= 0.55 and not block_trade,
            weight=self._weight_for("risk_sentinel"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_uncertainty(self, ctx: AgentDecisionContext) -> AgentVerdict:
        direction = ctx.analysis.direction
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        ret_5 = _last_value(ctx.df_feat, "returns", 0.0)
        if abs(ret_5) < 1e-6 and "close" in ctx.df_raw.columns and len(ctx.df_raw) >= 6:
            base = ctx.df_raw["close"]
            ret_5 = _safe_float((base.iloc[-1] / base.iloc[-6]) - 1.0, 0.0)

        heuristics: List[int] = []
        if close != sma_20:
            heuristics.append(1 if close > sma_20 else -1)
        if abs(rsi - 50.0) >= 3.0:
            heuristics.append(1 if rsi < 50.0 else -1)
        if abs(ret_5) >= 0.0001:
            heuristics.append(1 if ret_5 > 0 else -1)

        if direction == "SHORT":
            heuristics = [-1 * x for x in heuristics]

        oppose = sum(1 for h in heuristics if h < 0)
        total = len(heuristics)
        model_disagreement = (oppose / total) if total > 0 else 0.0
        confidence_uncertainty = _clip01((0.60 - confidence) / 0.10)
        confidence_variance = _clip01(abs(0.62 - confidence) / 0.20)
        uncertainty_score = _clip01(confidence_uncertainty * 0.60 + model_disagreement * 0.40)

        max_uncertainty = _clip01(_safe_float(getattr(ctx.config, "max_uncertainty_score", 0.40), 0.40))
        max_disagreement = _clip01(_safe_float(getattr(ctx.config, "max_model_disagreement", 0.50), 0.50))
        soft_blocking = bool(getattr(ctx.config, "soft_uncertainty_blocking", False))
        exceeds_threshold = uncertainty_score > max_uncertainty or model_disagreement > max_disagreement
        # Soft blocking: penalize confidence heavily instead of hard-blocking
        block_trade = exceeds_threshold and not soft_blocking

        if exceeds_threshold and soft_blocking:
            # Scale penalty by how far over the threshold we are
            overshoot = max(uncertainty_score - max_uncertainty, 0.0)
            reason = f"uncertainty high ({uncertainty_score:.2f}) [soft penalty]"
            reason_code = "uncertainty_soft_penalty"
            confidence_delta = -0.05 - (overshoot * 0.30)  # -5% to -14% penalty
        elif block_trade:
            reason = f"uncertainty high ({uncertainty_score:.2f})"
            reason_code = "uncertainty_block"
            confidence_delta = -0.10
        elif uncertainty_score <= max_uncertainty * 0.75:
            reason = f"uncertainty contained ({uncertainty_score:.2f})"
            reason_code = "uncertainty_ok"
            confidence_delta = 0.02
        else:
            reason = f"uncertainty borderline ({uncertainty_score:.2f})"
            reason_code = "uncertainty_watch"
            confidence_delta = -0.03

        return AgentVerdict(
            name="uncertainty",
            score=_clip01(1.0 - uncertainty_score),
            passed=not block_trade,
            weight=self._weight_for("uncertainty"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "uncertainty_score": uncertainty_score,
                "confidence_variance": confidence_variance,
                "model_disagreement": model_disagreement,
            },
        )

    def _evaluate_execution_quality(self, ctx: AgentDecisionContext) -> AgentVerdict:
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        volume_ratio = _last_value(ctx.df_feat, "volume_ratio_20", 1.0)
        liquidity_score = _clip01((volume_ratio - 0.35) / 1.15)
        if liquidity_score <= 0.0:
            liquidity_score = 0.45

        spread_pips = _safe_float(ctx.analysis.spread_pips)
        if spread_pips <= 0.0:
            spread_pips = max(0.1, min(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0) * 1.5, atr_pips * 0.04))

        slippage_pips = _safe_float(ctx.analysis.est_slippage_pips)
        if slippage_pips <= 0.0:
            slippage_pips = max(0.05, spread_pips * 0.45 + (1.0 - liquidity_score) * 0.9)

        max_spread = max(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0), 0.1)
        max_slippage = max(_safe_float(getattr(ctx.config, "max_slippage_pips", 2.0), 2.0), 0.1)
        min_liquidity = _clip01(_safe_float(getattr(ctx.config, "min_liquidity_score", 0.3), 0.3))

        spread_score = _clip01(1.0 - (spread_pips / (max_spread * 1.4)))
        slippage_score = _clip01(1.0 - (slippage_pips / (max_slippage * 1.4)))
        execution_quality_score = _clip01(spread_score * 0.35 + slippage_score * 0.30 + liquidity_score * 0.35)

        passed = spread_pips <= max_spread and slippage_pips <= max_slippage and liquidity_score >= min_liquidity
        block_trade = spread_pips > (max_spread * 1.25) or slippage_pips > (max_slippage * 1.25)

        if block_trade:
            reason = f"execution quality poor (spread {spread_pips:.1f} pips)"
            reason_code = "execution_block"
            confidence_delta = -0.08
        elif passed:
            reason = f"execution quality solid (liq {liquidity_score:.2f})"
            reason_code = "execution_ok"
            confidence_delta = 0.02
        else:
            reason = f"execution quality mixed (spread {spread_pips:.1f} pips)"
            reason_code = "execution_watch"
            confidence_delta = -0.02

        return AgentVerdict(
            name="execution_quality",
            score=execution_quality_score,
            passed=passed and not block_trade,
            weight=self._weight_for("execution_quality"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "execution_quality_score": execution_quality_score,
                "execution_quality_passed": passed and not block_trade,
                "spread_pips": spread_pips,
                "est_slippage_pips": slippage_pips,
                "liquidity_score": liquidity_score,
            },
        )

    def _evaluate_news_risk(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        try:
            from market_intelligence import fetch_forex_news
        except Exception:
            return None

        try:
            headlines = fetch_forex_news(ctx.analysis.pair, max_items=4) or []
        except Exception:
            headlines = []

        if not headlines:
            return AgentVerdict(
                name="news_risk",
                score=0.55,
                passed=True,
                weight=self._weight_for("news_risk"),
                reason="no material news headlines",
                reason_code="news_clear",
                confidence_delta=0.0,
            )

        keywords = (
            "nfp",
            "payroll",
            "cpi",
            "inflation",
            "fomc",
            "ecb",
            "boe",
            "boj",
            "rate decision",
            "central bank",
            "gdp",
        )
        joined = " ".join(str(h).lower() for h in headlines)
        hit_count = sum(1 for token in keywords if token in joined)
        block_trade = hit_count >= 2
        score = 0.35 if block_trade else 0.50
        reason = "headline risk elevated" if block_trade else "headline flow manageable"
        reason_code = "news_block" if block_trade else "news_watch"
        confidence_delta = -0.06 if block_trade else -0.01

        return AgentVerdict(
            name="news_risk",
            score=score,
            passed=not block_trade,
            weight=self._weight_for("news_risk"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={"headline_count": len(headlines)},
        )

    def _evaluate_multi_timeframe(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Multi-timeframe confluence: synthesize H4 and D1 signals from H1 candles.

        Aggregates H1 data to approximate higher timeframe trend direction,
        checking if the trade direction aligns across H1, H4, and D1.
        """
        df = ctx.df_raw
        direction = ctx.analysis.direction
        confluence_count = 0  # how many timeframes agree

        # H1 trend (already the primary): use SMA crossover
        if len(df) >= 20:
            h1_close = float(df["close"].iloc[-1])
            h1_sma20 = float(df["close"].iloc[-20:].mean())
            h1_sma50 = float(df["close"].iloc[-min(50, len(df)):].mean()) if len(df) >= 50 else h1_sma20
            if direction == "LONG" and h1_close > h1_sma20:
                confluence_count += 1
            elif direction == "SHORT" and h1_close < h1_sma20:
                confluence_count += 1

        # H4 trend: aggregate last 80 H1 candles (20 H4 candles) by groups of 4
        if len(df) >= 80:
            h4_closes = df["close"].iloc[-80:].values.reshape(-1, 4).mean(axis=1)
            h4_last = float(h4_closes[-1])
            h4_sma = float(h4_closes[-min(20, len(h4_closes)):].mean())
            if direction == "LONG" and h4_last > h4_sma:
                confluence_count += 1
            elif direction == "SHORT" and h4_last < h4_sma:
                confluence_count += 1
        elif len(df) >= 20:
            # Not enough data for H4 - treat as neutral agreement
            confluence_count += 1

        # D1 trend: aggregate last 120 H1 candles (5 daily candles) by groups of 24
        if len(df) >= 120:
            usable = len(df) - (len(df) % 24)
            if usable >= 120:
                d1_closes = df["close"].iloc[-usable:].values.reshape(-1, 24).mean(axis=1)
                d1_last = float(d1_closes[-1])
                d1_prev = float(d1_closes[-2]) if len(d1_closes) >= 2 else d1_last
                if direction == "LONG" and d1_last > d1_prev:
                    confluence_count += 1
                elif direction == "SHORT" and d1_last < d1_prev:
                    confluence_count += 1
                # else: no agreement
            else:
                confluence_count += 1  # Insufficient data, neutral
        elif len(df) >= 24:
            confluence_count += 1  # Insufficient data, neutral

        # Score based on confluence
        score = 0.30 + confluence_count * 0.20  # 0.30, 0.50, 0.70, 0.90
        passed = confluence_count >= 2
        confidence_delta = (confluence_count - 1.5) * 0.03  # -0.045 to +0.045

        if confluence_count >= 3:
            reason = f"full MTF confluence ({confluence_count}/3 timeframes agree)"
            reason_code = "mtf_full_confluence"
        elif confluence_count == 2:
            reason = f"partial MTF confluence ({confluence_count}/3)"
            reason_code = "mtf_partial"
        elif confluence_count == 1:
            reason = f"weak MTF support ({confluence_count}/3)"
            reason_code = "mtf_weak"
        else:
            reason = "no MTF confluence (all timeframes disagree)"
            reason_code = "mtf_none"

        return AgentVerdict(
            name="multi_timeframe",
            score=_clip01(score),
            passed=passed,
            weight=self._weight_for("multi_timeframe"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={"confluence_count": confluence_count, "timeframes": 3},
        )

    def _evaluate_momentum(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Momentum agent: MACD crossover + rate-of-change assessment.

        Evaluates whether price momentum supports the trade direction using
        MACD signal alignment and short-term rate of change.
        """
        direction = ctx.analysis.direction
        df = ctx.df_feat if not ctx.df_feat.empty else ctx.df_raw

        # MACD components (normalized by price in feature engineering)
        macd = _last_value(df, "macd_norm", 0.0)
        macd_signal = _last_value(df, "macd_signal_norm", 0.0)
        macd_hist = _last_value(df, "macd_hist_norm", macd - macd_signal)

        # Rate of change (5-bar, try roc_5 first then roc)
        roc = _last_value(df, "roc_5", 0.0)
        if abs(roc) < 1e-8:
            roc = _last_value(df, "roc", 0.0)
        if abs(roc) < 1e-8 and "close" in ctx.df_raw.columns and len(ctx.df_raw) >= 6:
            closes = ctx.df_raw["close"]
            prev = float(closes.iloc[-6])
            if prev != 0:
                roc = (float(closes.iloc[-1]) / prev - 1.0) * 100.0

        # Direction alignment
        if direction == "LONG":
            macd_aligned = macd_hist > 0
            roc_aligned = roc > 0
            macd_strength = _clip01(macd_hist / max(abs(macd) + 1e-6, 1e-6))
        else:
            macd_aligned = macd_hist < 0
            roc_aligned = roc < 0
            macd_strength = _clip01(-macd_hist / max(abs(macd) + 1e-6, 1e-6))

        # Composite score
        score = 0.45
        if macd_aligned:
            score += 0.25
        if roc_aligned:
            score += 0.15
        score += min(macd_strength * 0.15, 0.15)

        if macd_aligned and roc_aligned:
            reason = f"momentum aligned with {direction.lower()} (MACD hist {macd_hist:+.5f})"
            reason_code = "momentum_aligned"
            confidence_delta = 0.03
        elif macd_aligned or roc_aligned:
            reason = f"momentum mixed (MACD {'aligned' if macd_aligned else 'opposed'}, ROC {'aligned' if roc_aligned else 'opposed'})"
            reason_code = "momentum_mixed"
            confidence_delta = 0.0
        else:
            reason = f"momentum opposes {direction.lower()} bias"
            reason_code = "momentum_opposed"
            confidence_delta = -0.04

        return AgentVerdict(
            name="momentum",
            score=_clip01(score),
            passed=score >= 0.52,
            weight=self._weight_for("momentum"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "macd_hist": macd_hist,
                "roc": roc,
                "macd_aligned": macd_aligned,
                "roc_aligned": roc_aligned,
            },
        )

    def _evaluate_session_timing(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Session timing agent: forex session overlap awareness.

        Major pairs trade best during London/NY overlap (13:00-17:00 UTC).
        Scores higher during active sessions for the pair's currencies.
        """
        # Use last candle timestamp if available, otherwise system clock
        now = datetime.now(timezone.utc)
        if not ctx.df_raw.empty and ctx.df_raw.index.dtype.kind == 'M':
            try:
                last_ts = ctx.df_raw.index[-1]
                if hasattr(last_ts, 'to_pydatetime'):
                    now = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            except Exception:
                pass  # Fall back to system clock
        hour = now.hour

        # Define forex sessions (UTC hours)
        tokyo_active = 0 <= hour < 9
        london_active = 7 <= hour < 16
        ny_active = 13 <= hour < 22
        overlap_london_ny = 13 <= hour < 16

        pair = ctx.analysis.pair.upper()

        # Map currencies to their primary sessions
        jpy_pair = "JPY" in pair
        eur_gbp_pair = any(c in pair for c in ("EUR", "GBP", "CHF"))
        usd_cad_pair = any(c in pair for c in ("USD", "CAD"))
        aud_nzd_pair = any(c in pair for c in ("AUD", "NZD"))

        # Base score from session relevance
        score = 0.40
        active_sessions = []

        if overlap_london_ny:
            score += 0.35
            active_sessions.append("London/NY overlap")
        elif london_active:
            score += 0.25 if eur_gbp_pair else 0.15
            active_sessions.append("London")
        elif ny_active:
            score += 0.25 if usd_cad_pair else 0.15
            active_sessions.append("New York")
        elif tokyo_active:
            score += 0.25 if (jpy_pair or aud_nzd_pair) else 0.08
            active_sessions.append("Tokyo")
        else:
            active_sessions.append("off-hours")

        # Weekend/low-liquidity penalty
        weekday = now.weekday()
        if weekday >= 5:  # Saturday/Sunday
            score -= 0.25
            active_sessions.append("weekend")

        score = _clip01(score)
        passed = score >= 0.45

        if score >= 0.65:
            reason = f"optimal session timing ({', '.join(active_sessions)})"
            reason_code = "session_optimal"
            confidence_delta = 0.02
        elif passed:
            reason = f"acceptable session ({', '.join(active_sessions)})"
            reason_code = "session_ok"
            confidence_delta = 0.0
        else:
            reason = f"suboptimal session ({', '.join(active_sessions)})"
            reason_code = "session_weak"
            confidence_delta = -0.03

        return AgentVerdict(
            name="session_timing",
            score=score,
            passed=passed,
            weight=self._weight_for("session_timing"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "hour_utc": hour,
                "weekday": weekday,
                "active_sessions": active_sessions,
            },
        )

    def _evaluate_support_resistance(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Support/resistance agent: proximity to key price levels.

        Identifies recent swing highs/lows as S/R levels and checks if
        the current price is near a level that supports or opposes the trade.
        """
        df = ctx.df_raw
        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            return None
        if len(df) < 30:
            return None

        close = float(df["close"].iloc[-1])
        highs = df["high"].values[-60:] if len(df) >= 60 else df["high"].values
        lows = df["low"].values[-60:] if len(df) >= 60 else df["low"].values
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)

        # Estimate pip value for the pair
        pip_value = 0.0001
        if "JPY" in ctx.analysis.pair.upper():
            pip_value = 0.01

        atr_price = atr_pips * pip_value

        # Find swing highs/lows (simple 5-bar pivot)
        resistance_levels: List[float] = []
        support_levels: List[float] = []

        for i in range(2, len(highs) - 2):
            if highs[i] >= max(highs[i - 2], highs[i - 1], highs[i + 1], highs[i + 2]):
                resistance_levels.append(float(highs[i]))
            if lows[i] <= min(lows[i - 2], lows[i - 1], lows[i + 1], lows[i + 2]):
                support_levels.append(float(lows[i]))

        if not resistance_levels and not support_levels:
            return None

        # Find nearest S/R levels
        nearest_resistance = min(resistance_levels, key=lambda r: abs(r - close)) if resistance_levels else close + atr_price * 5
        nearest_support = min(support_levels, key=lambda s: abs(s - close)) if support_levels else close - atr_price * 5

        dist_to_resistance = (nearest_resistance - close) / atr_price if atr_price > 0 else 5.0
        dist_to_support = (close - nearest_support) / atr_price if atr_price > 0 else 5.0

        direction = ctx.analysis.direction
        score = 0.50

        if direction == "LONG":
            # Good: support nearby (bounce), resistance far (room to run)
            if dist_to_support < 1.5:
                score += 0.20  # Near support - good entry
            if dist_to_resistance > 2.0:
                score += 0.15  # Room to TP
            if dist_to_resistance < 0.5:
                score -= 0.25  # Resistance right above - bad
        else:  # SHORT
            # Good: resistance nearby (rejection), support far (room to run)
            if dist_to_resistance < 1.5:
                score += 0.20  # Near resistance - good entry
            if dist_to_support > 2.0:
                score += 0.15  # Room to TP
            if dist_to_support < 0.5:
                score -= 0.25  # Support right below - bad

        score = _clip01(score)
        passed = score >= 0.50

        if score >= 0.65:
            reason = f"S/R structure supports {direction.lower()} (R:{dist_to_resistance:.1f}x ATR, S:{dist_to_support:.1f}x ATR)"
            reason_code = "sr_support"
            confidence_delta = 0.03
        elif score >= 0.45:
            reason = f"S/R neutral (R:{dist_to_resistance:.1f}x ATR, S:{dist_to_support:.1f}x ATR)"
            reason_code = "sr_neutral"
            confidence_delta = 0.0
        else:
            reason = f"S/R opposes {direction.lower()} (price near {'resistance' if direction == 'LONG' else 'support'})"
            reason_code = "sr_opposed"
            confidence_delta = -0.04

        return AgentVerdict(
            name="support_resistance",
            score=score,
            passed=passed,
            weight=self._weight_for("support_resistance"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "dist_to_resistance_atr": round(dist_to_resistance, 2),
                "dist_to_support_atr": round(dist_to_support, 2),
            },
        )

    def _evaluate_pair_performance(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Check historical performance for this pair and adjust confidence.

        Reads ``trained_data/models/pair_performance.json`` (updated by
        :meth:`ExecutionManager._update_pair_performance`) and emits a verdict
        based on the pair's historical win rate and P/L.

        Returns ``None`` when no history exists for the pair (agent not counted).
        """
        import json
        from pathlib import Path

        perf_path = Path("trained_data/models/pair_performance.json")
        if not perf_path.exists():
            return None

        try:
            data = json.loads(perf_path.read_text())
        except Exception:
            return None

        pair = ctx.analysis.pair
        stats = data.get(pair)
        if not stats or stats.get("trades", 0) < 3:
            # Not enough history — skip this agent
            return None

        win_rate = _safe_float(stats.get("win_rate"), 0.5)
        total_pnl_pips = _safe_float(stats.get("total_pnl_pips"), 0.0)
        trades = int(stats.get("trades", 0))

        # Score: base 0.50, boost for good win rate, penalize for bad
        score = 0.50 + (win_rate - 0.50) * 0.60
        score = _clip01(score)

        # Confidence delta: up to +/- 5% based on historical edge
        confidence_delta = (win_rate - 0.50) * 0.10  # e.g., 60% WR -> +1%, 40% -> -1%
        confidence_delta = max(-0.05, min(0.05, confidence_delta))

        passed = win_rate >= 0.45

        if win_rate >= 0.60:
            reason = f"pair has strong history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_strong"
        elif win_rate >= 0.45:
            reason = f"pair has neutral history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_neutral"
        else:
            reason = f"pair has weak history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_weak"

        return AgentVerdict(
            name="pair_performance",
            score=score,
            passed=passed,
            weight=self._weight_for("pair_performance"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "win_rate": win_rate,
                "total_pnl_pips": total_pnl_pips,
                "trades_history": trades,
            },
        )
