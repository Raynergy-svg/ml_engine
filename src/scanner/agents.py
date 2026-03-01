"""
Scanner specialist agents.

Adds a lightweight deliberation layer on top of the existing scanner signal.
Each agent evaluates one aspect of the setup and emits a structured verdict
that can be combined into a weighted vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    }

    def __init__(self, config: Any):
        self.config = config

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
        block_trade = uncertainty_score > max_uncertainty or model_disagreement > max_disagreement

        if block_trade:
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
