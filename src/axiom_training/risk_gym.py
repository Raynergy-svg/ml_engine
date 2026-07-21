"""Risk-only training gym for AXIOM.

Strategy returns are exogenous: AXIOM cannot choose or alter direction. Its
only action is the fraction of a strategy's proposed risk to accept.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - minimal installations can still read reports
    gym = None
    spaces = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_POLICY_MODEL_PATH = REPO_ROOT / "trained_data" / "axiom" / "risk_policy_candidate.json"
RISK_POLICY_REPORT_PATH = REPO_ROOT / "trained_data" / "axiom" / "risk_policy_report.json"
STRATEGY_ORDER = (
    "oanda_fx_trend",
    "equity_harvester",
    "crypto_momentum",
    "track_b_filing_text",
    "crypto_cash_and_carry",
    "portfolio_exposure_shadow",
)
REGIME_ORDINAL = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}


@dataclass(frozen=True)
class RiskStep:
    strategy: str
    timestamp: str
    return_pct: float
    cost_pct: float = 0.0
    source: str = "unknown"
    portfolio_concentration: float = 0.0
    exposure_resolved_fraction: float = 1.0
    correlation_cluster_stress: float = 0.0
    regime_transition_risk: float = 0.0
    liquidity_stress: float = 0.0
    confidence_shortfall: float = 0.0
    model_disagreement: float = 0.0


@dataclass(frozen=True)
class RiskPolicyParams:
    target_vol_multiplier: float = 1.0
    drawdown_soft: float = 0.04
    drawdown_hard: float = 0.10
    loss_streak_decay: float = 0.72
    min_scale: float = 0.10
    max_scale: float = 1.00
    concentration_soft: float = 0.45
    concentration_hard: float = 0.80
    unresolved_exposure_scale: float = 0.75
    # These floors are neutral for legacy artifacts. Newly trained state-v2
    # candidates set conservative values below 1.0 in candidate_grid().
    correlation_stress_scale: float = 1.0
    regime_transition_scale: float = 1.0
    liquidity_stress_scale: float = 1.0
    uncertainty_stress_scale: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.min_scale <= self.max_scale <= 1.0:
            raise ValueError("risk scales must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.drawdown_soft < self.drawdown_hard <= 1.0:
            raise ValueError("drawdown thresholds must satisfy 0 <= soft < hard <= 1")
        if not 0.0 < self.loss_streak_decay <= 1.0:
            raise ValueError("loss_streak_decay must be in (0, 1]")
        if self.target_vol_multiplier <= 0.0:
            raise ValueError("target_vol_multiplier must be positive")
        if not 0.0 <= self.concentration_soft < self.concentration_hard <= 1.0:
            raise ValueError("concentration thresholds must satisfy 0 <= soft < hard <= 1")
        if not 0.0 <= self.unresolved_exposure_scale <= 1.0:
            raise ValueError("unresolved_exposure_scale must be in [0, 1]")
        for name in (
            "correlation_stress_scale", "regime_transition_scale",
            "liquidity_stress_scale", "uncertainty_stress_scale",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class RiskPolicy:
    """Auditable controller driven only by risk state."""

    def __init__(self, params: RiskPolicyParams):
        params.validate()
        self.params = params

    def scale(
        self,
        *,
        trailing_vol: float,
        target_vol: float,
        drawdown: float,
        loss_streak: int,
        portfolio_concentration: float = 0.0,
        exposure_resolved_fraction: float = 1.0,
        strategy_risk_multiplier: float = 1.0,
        correlation_cluster_stress: float = 0.0,
        regime_transition_risk: float = 0.0,
        liquidity_stress: float = 0.0,
        confidence_shortfall: float = 0.0,
        model_disagreement: float = 0.0,
    ) -> float:
        p = self.params
        if drawdown >= p.drawdown_hard:
            return 0.0
        effective_target = max(target_vol * p.target_vol_multiplier, 1e-8)
        vol_multiplier = min(1.0, effective_target / max(trailing_vol, effective_target * 0.25))
        if drawdown <= p.drawdown_soft:
            dd_multiplier = 1.0
        else:
            dd_multiplier = max(
                0.0,
                (p.drawdown_hard - drawdown) / (p.drawdown_hard - p.drawdown_soft),
            )
        streak_multiplier = p.loss_streak_decay ** max(0, int(loss_streak) - 1)
        concentration = float(np.clip(portfolio_concentration, 0.0, 1.0))
        if concentration <= p.concentration_soft:
            concentration_multiplier = 1.0
        elif concentration >= p.concentration_hard:
            concentration_multiplier = 0.0
        else:
            concentration_multiplier = (
                (p.concentration_hard - concentration)
                / (p.concentration_hard - p.concentration_soft)
            )
        resolved = float(np.clip(exposure_resolved_fraction, 0.0, 1.0))
        resolution_multiplier = resolved + (1.0 - resolved) * p.unresolved_exposure_scale
        strategy_multiplier = float(np.clip(strategy_risk_multiplier, 0.0, 1.0))

        def stress_multiplier(value: float, floor: float) -> float:
            stress = float(np.clip(value, 0.0, 1.0))
            return 1.0 - stress * (1.0 - floor)

        cluster_multiplier = stress_multiplier(
            correlation_cluster_stress, p.correlation_stress_scale,
        )
        transition_multiplier = stress_multiplier(
            regime_transition_risk, p.regime_transition_scale,
        )
        liquidity_multiplier = stress_multiplier(
            liquidity_stress, p.liquidity_stress_scale,
        )
        # Confidence shortfall and model disagreement are overlapping views of
        # uncertainty. Max avoids counting the same uncertainty twice.
        uncertainty = max(
            float(np.clip(confidence_shortfall, 0.0, 1.0)),
            float(np.clip(model_disagreement, 0.0, 1.0)),
        )
        uncertainty_multiplier = stress_multiplier(
            uncertainty, p.uncertainty_stress_scale,
        )
        raw = (
            p.max_scale * vol_multiplier * dd_multiplier * streak_multiplier
            * concentration_multiplier * resolution_multiplier * strategy_multiplier
            * cluster_multiplier * transition_multiplier * liquidity_multiplier
            * uncertainty_multiplier
        )
        return float(np.clip(max(p.min_scale, raw), 0.0, p.max_scale))


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _source_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _latest(pattern: str) -> Path | None:
    paths = sorted(REPO_ROOT.glob(pattern))
    return paths[-1] if paths else None


def _load_fx_feedback(path: Path) -> list[RiskStep]:
    rows: list[RiskStep] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    prior_regime: str | None = None
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        shaped = _finite(payload.get("shaped_reward"))
        if shaped is None:
            continue
        # One normalized risk unit is 20 pips. Capping prevents a legacy
        # outlier from defining the controller's entire volatility estimate.
        normalized = float(np.clip(shaped / 20.0, -3.0, 3.0) * 0.01)
        slippage_pips = max(0.0, _finite(payload.get("slippage_pips")) or 0.0)
        spread_pips = max(0.0, _finite(payload.get("spread_pips")) or 0.0)
        slippage = slippage_pips / 20.0 * 0.01
        regime = str(payload.get("regime") or "UNKNOWN").upper()
        transition = 0.0
        if regime in REGIME_ORDINAL and prior_regime in REGIME_ORDINAL:
            transition = abs(REGIME_ORDINAL[regime] - REGIME_ORDINAL[prior_regime]) / 3.0
        if regime in REGIME_ORDINAL:
            prior_regime = regime
        confidence = _finite(payload.get("confidence"))
        disagreement = _finite(payload.get("model_disagreement"))
        rows.append(RiskStep(
            strategy="oanda_fx_trend",
            timestamp=str(payload.get("timestamp") or f"fx-{index:06d}"),
            return_pct=normalized,
            cost_pct=slippage,
            source=_source_label(path),
            regime_transition_risk=float(np.clip(transition, 0.0, 1.0)),
            liquidity_stress=float(np.clip(max(slippage_pips, spread_pips) / 4.0, 0.0, 1.0)),
            confidence_shortfall=(
                float(np.clip(1.0 - confidence, 0.0, 1.0))
                if confidence is not None else 0.0
            ),
            model_disagreement=float(np.clip(disagreement or 0.0, 0.0, 1.0)),
        ))
    return rows


def _annual_steps(strategy: str, per_year: Mapping[str, Any], source: Path) -> list[RiskStep]:
    rows: list[RiskStep] = []
    for year, value in sorted(per_year.items(), key=lambda item: str(item[0])):
        result = _finite(value)
        if result is not None:
            rows.append(RiskStep(strategy, str(year), result, source=str(source.relative_to(REPO_ROOT))))
    return rows


def _load_equity() -> list[RiskStep]:
    path = _latest("trained_data/backtests/equity_harvester_singlestock_*.json")
    if path is None:
        path = _latest("trained_data/backtests/equity_harvester_*.json")
    payload = _read_json(path) if path else None
    if not isinstance(payload, Mapping) or path is None:
        return []
    summary = payload.get("backtest_summary", {}).get("summary", {})
    per_year = summary.get("per_year") if isinstance(summary, Mapping) else None
    if not isinstance(per_year, Mapping):
        best = payload.get("best_gate_pass", {})
        per_year = best.get("per_year") if isinstance(best, Mapping) else None
    return _annual_steps("equity_harvester", per_year or {}, path)


def _load_track_b() -> list[RiskStep]:
    path = REPO_ROOT / "trained_data/research/track_b_postcutoff_2026_07_02/harness_result_primary_2bps.json"
    payload = _read_json(path)
    full = payload.get("full", {}) if isinstance(payload, Mapping) else {}
    report = full.get("report", {}) if isinstance(full, Mapping) else {}
    per_year = report.get("per_year", {}) if isinstance(report, Mapping) else {}
    return _annual_steps("track_b_filing_text", per_year, path)


def _load_crypto_summary(strategy: str, pattern: str) -> list[RiskStep]:
    path = _latest(pattern)
    payload = _read_json(path) if path else None
    blocks = payload.get("blocks", {}) if isinstance(payload, Mapping) else {}
    rows: list[RiskStep] = []
    # full_all overlaps the other two blocks, so it is excluded to avoid split leakage.
    for key in ("full_IS_pre2024", "full_OOS_2024+"):
        block = blocks.get(key, {}) if isinstance(blocks, Mapping) else {}
        value = _finite(block.get("net_ann_ret")) if isinstance(block, Mapping) else None
        if value is not None and path is not None:
            rows.append(RiskStep(strategy, key, value, source=str(path.relative_to(REPO_ROOT))))
    return rows


def _load_exposure_history(path: Path) -> list[RiskStep]:
    """Turn successive hedge snapshots into an isolated portfolio-risk lane.

    The outcome is the observed NAV change between snapshots. The inputs are
    unsigned concentration and resolution coverage, never currency direction.
    """
    try:
        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError):
        return []
    rows: list[RiskStep] = []
    prior_nav: float | None = None
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        nav = _finite(payload.get("nav"))
        if nav is None or nav <= 0.0:
            continue
        if prior_nav is not None and prior_nav > 0.0:
            buckets = payload.get("net_correlation_buckets_notional_home")
            values = []
            if isinstance(buckets, Mapping):
                values = [abs(value) for raw in buckets.values() if (value := _finite(raw)) is not None]
            gross = sum(values)
            concentration = max(values, default=0.0) / gross if gross > 0.0 else 0.0
            shares = [value / gross for value in values] if gross > 0.0 else []
            hhi = sum(share * share for share in shares)
            cluster_stress = (
                (hhi - 1.0 / len(shares)) / (1.0 - 1.0 / len(shares))
                if len(shares) > 1 else (1.0 if shares else 0.0)
            )
            try:
                total = max(0, int(payload.get("open_trade_count", 0)))
                resolved = max(0, total - len(payload.get("skipped_instruments") or []))
            except (TypeError, ValueError):
                total = resolved = 0
            rows.append(RiskStep(
                strategy="portfolio_exposure_shadow",
                timestamp=str(payload.get("captured_at_utc") or len(rows)),
                return_pct=float(np.clip(nav / prior_nav - 1.0, -0.10, 0.10)),
                source=_source_label(path),
                portfolio_concentration=float(np.clip(concentration, 0.0, 1.0)),
                exposure_resolved_fraction=float(np.clip(resolved / total if total else 0.0, 0.0, 1.0)),
                correlation_cluster_stress=float(np.clip(cluster_stress, 0.0, 1.0)),
            ))
        prior_nav = nav
    return rows


def load_default_risk_steps() -> list[RiskStep]:
    """Load frozen outcomes from every strategy lane with usable history."""
    rows = _load_fx_feedback(REPO_ROOT / "logs" / "trade_feedback.jsonl")
    rows.extend(_load_equity())
    rows.extend(_load_track_b())
    rows.extend(_load_crypto_summary(
        "crypto_momentum", "trained_data/backtests/crypto_xs_momentum_*.json"
    ))
    rows.extend(_load_crypto_summary(
        "crypto_cash_and_carry", "trained_data/backtests/crypto_cash_and_carry_*.json"
    ))
    rows.extend(_load_exposure_history(
        REPO_ROOT / "trained_data" / "hedge" / "exposure_history.jsonl"
    ))
    return rows


def split_steps(
    steps: Sequence[RiskStep], *, holdout_fraction: float = 0.25,
) -> tuple[list[RiskStep], list[RiskStep]]:
    """Chronologically split each lane so the large FX lane cannot swallow others."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    train: list[RiskStep] = []
    holdout: list[RiskStep] = []
    for strategy in STRATEGY_ORDER:
        lane = [row for row in steps if row.strategy == strategy]
        if len(lane) < 2:
            continue
        n_holdout = max(1, int(math.ceil(len(lane) * holdout_fraction)))
        n_holdout = min(n_holdout, len(lane) - 1)
        train.extend(lane[:-n_holdout])
        holdout.extend(lane[-n_holdout:])
    return train, holdout


def split_steps_three_way(
    steps: Sequence[RiskStep],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[RiskStep], list[RiskStep], list[RiskStep]]:
    """Per-lane chronological train/validation/test split.

    Sparse research lanes with only two observations contribute their first
    non-overlapping block to training and their OOS block to final testing.
    They never get duplicated merely to make a validation metric look fuller.
    """
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation + test fractions must be less than 1")
    train: list[RiskStep] = []
    validation: list[RiskStep] = []
    test: list[RiskStep] = []
    for strategy in STRATEGY_ORDER:
        lane = [row for row in steps if row.strategy == strategy]
        if len(lane) < 2:
            continue
        if len(lane) < 5:
            train.extend(lane[:-1])
            test.append(lane[-1])
            continue
        n_validation = max(1, int(math.ceil(len(lane) * validation_fraction)))
        n_test = max(1, int(math.ceil(len(lane) * test_fraction)))
        if n_validation + n_test >= len(lane):
            n_validation = 1
            n_test = 1
        cut_validation = len(lane) - n_validation - n_test
        cut_test = len(lane) - n_test
        train.extend(lane[:cut_validation])
        validation.extend(lane[cut_validation:cut_test])
        test.extend(lane[cut_test:])
    return train, validation, test


def target_vol_by_strategy(steps: Sequence[RiskStep]) -> dict[str, float]:
    result: dict[str, float] = {}
    for strategy in STRATEGY_ORDER:
        values = np.asarray([row.return_pct for row in steps if row.strategy == strategy], dtype=float)
        if len(values) > 1:
            vol = float(np.std(values, ddof=1))
        elif len(values):
            vol = float(np.abs(values).mean())
        else:
            vol = 0.01
        result[strategy] = max(vol, 1e-4)
    return result


def strategy_risk_multipliers(steps: Sequence[RiskStep]) -> dict[str, float]:
    """Learn conservative training-only lane priors from downside severity.

    Lanes with fewer than five observations remain neutral. Larger lanes are
    shrunk toward neutral so a short bad patch cannot permanently suppress a
    strategy; even fully reliable/high-downside lanes are capped at 0.75.
    """
    all_downside = np.asarray(
        [min(0.0, float(row.return_pct) - float(row.cost_pct)) for row in steps],
        dtype=float,
    )
    baseline = float(np.sqrt(np.mean(np.square(all_downside)))) if len(all_downside) else 0.0
    baseline = max(baseline, 1e-6)
    result: dict[str, float] = {}
    for strategy in STRATEGY_ORDER:
        lane = [row for row in steps if row.strategy == strategy]
        if len(lane) < 5:
            result[strategy] = 1.0
            continue
        downside = np.asarray(
            [min(0.0, float(row.return_pct) - float(row.cost_pct)) for row in lane],
            dtype=float,
        )
        severity = float(np.sqrt(np.mean(np.square(downside))))
        excess = float(np.clip(severity / baseline - 1.0, 0.0, 1.0))
        reliability = min(1.0, len(lane) / 20.0)
        result[strategy] = float(np.clip(1.0 - 0.25 * excess * reliability, 0.75, 1.0))
    return result


def simulate_policy(
    steps: Sequence[RiskStep],
    policy: RiskPolicy | None,
    *,
    target_vols: Mapping[str, float],
    stress_multiplier: float = 1.0,
    turnover_cost_pct: float = 0.0001,
    strategy_multipliers: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Replay lanes independently and aggregate them with equal lane weight."""
    lane_reports: dict[str, dict[str, Any]] = {}
    all_scales: list[float] = []
    for strategy in STRATEGY_ORDER:
        lane = [row for row in steps if row.strategy == strategy]
        if not lane:
            continue
        equity = peak = 1.0
        prior_scale = 0.0
        loss_streak = 0
        realized: list[float] = []
        scales: list[float] = []
        policy_returns: list[float] = []
        max_drawdown = 0.0
        risk_violations = 0
        target_vol = float(target_vols.get(strategy, 0.01))
        strategy_multiplier = float((strategy_multipliers or {}).get(strategy, 1.0))
        for row in lane:
            drawdown = max(0.0, 1.0 - equity / peak)
            trailing = realized[-20:]
            trailing_vol = float(np.std(trailing, ddof=1)) if len(trailing) > 1 else target_vol
            scale = 1.0 if policy is None else policy.scale(
                trailing_vol=trailing_vol,
                target_vol=target_vol,
                drawdown=drawdown,
                loss_streak=loss_streak,
                portfolio_concentration=row.portfolio_concentration,
                exposure_resolved_fraction=row.exposure_resolved_fraction,
                strategy_risk_multiplier=strategy_multiplier,
                correlation_cluster_stress=row.correlation_cluster_stress,
                regime_transition_risk=row.regime_transition_risk,
                liquidity_stress=row.liquidity_stress,
                confidence_shortfall=row.confidence_shortfall,
                model_disagreement=row.model_disagreement,
            )
            if not 0.0 <= scale <= 1.0:
                risk_violations += 1
            raw_return = float(row.return_pct) * stress_multiplier
            pnl = scale * raw_return - abs(scale - prior_scale) * turnover_cost_pct - scale * row.cost_pct
            equity *= max(1e-6, 1.0 + pnl)
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, 1.0 - equity / peak)
            loss_streak = loss_streak + 1 if raw_return < 0.0 else 0
            prior_scale = scale
            realized.append(raw_return)
            policy_returns.append(pnl)
            scales.append(scale)
        downside = [value for value in policy_returns if value < 0.0]
        lane_reports[strategy] = {
            "observations": len(lane),
            "total_return": equity - 1.0,
            "max_drawdown": max_drawdown,
            "downside_deviation": float(np.std(downside, ddof=1)) if len(downside) > 1 else abs(downside[0]) if downside else 0.0,
            "average_scale": float(np.mean(scales)),
            "scale_std": float(np.std(scales)),
            "min_scale": float(min(scales)),
            "max_scale": float(max(scales)),
            "risk_violations": risk_violations,
        }
        all_scales.extend(scales)

    reports = list(lane_reports.values())
    return {
        "lane_count": len(reports),
        "observations": sum(item["observations"] for item in reports),
        "average_lane_return": float(np.mean([item["total_return"] for item in reports])) if reports else 0.0,
        "average_max_drawdown": float(np.mean([item["max_drawdown"] for item in reports])) if reports else 0.0,
        "average_downside_deviation": float(np.mean([item["downside_deviation"] for item in reports])) if reports else 0.0,
        "worst_lane_drawdown": max((item["max_drawdown"] for item in reports), default=0.0),
        "average_scale": float(np.mean(all_scales)) if all_scales else 0.0,
        "scale_std": float(np.std(all_scales)) if all_scales else 0.0,
        "risk_violations": sum(item["risk_violations"] for item in reports),
        "lanes": lane_reports,
    }


_GymBase = gym.Env if gym is not None else object


class AxiomRiskGym(_GymBase):
    """Gymnasium adapter whose sole action is accepted risk fraction [0, 1]."""

    metadata = {"render_modes": []}

    def __init__(
        self, steps: Sequence[RiskStep], *, target_vol: float | None = None,
        strategy_multipliers: Mapping[str, float] | None = None,
    ):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required to instantiate AxiomRiskGym")
        if len(steps) < 2:
            raise ValueError("risk gym requires at least two outcome steps")
        self.steps = list(steps)
        values = np.asarray([row.return_pct for row in self.steps], dtype=float)
        self.target_vol = float(target_vol or max(np.std(values, ddof=1), 1e-4))
        self.strategy_multipliers = dict(strategy_multipliers or {})
        # trailing-vol/target, drawdown, loss-streak/5, prior accepted risk,
        # portfolio concentration, fraction resolved, strategy risk prior,
        # cluster stress, regime transition, liquidity stress, confidence
        # shortfall, and model disagreement.
        self.observation_space = spaces.Box(
            low=np.asarray([0.0] * 12, dtype=np.float32),
            high=np.asarray([10.0] + [1.0] * 11, dtype=np.float32),
        )
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self._index = 0
        self._equity = self._peak = 1.0
        self._prior_scale = 0.0
        self._loss_streak = 0
        self._returns: list[float] = []

    def _observation(self) -> np.ndarray:
        trailing = self._returns[-20:]
        vol = float(np.std(trailing, ddof=1)) if len(trailing) > 1 else self.target_vol
        drawdown = max(0.0, 1.0 - self._equity / self._peak)
        return np.asarray([
            min(10.0, vol / self.target_vol),
            drawdown,
            min(1.0, self._loss_streak / 5.0),
            self._prior_scale,
            float(np.clip(self.steps[self._index].portfolio_concentration, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
            float(np.clip(self.steps[self._index].exposure_resolved_fraction, 0.0, 1.0))
            if self._index < len(self.steps) else 1.0,
            float(np.clip(self.strategy_multipliers.get(
                self.steps[self._index].strategy, 1.0), 0.0, 1.0))
            if self._index < len(self.steps) else 1.0,
            float(np.clip(self.steps[self._index].correlation_cluster_stress, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
            float(np.clip(self.steps[self._index].regime_transition_risk, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
            float(np.clip(self.steps[self._index].liquidity_stress, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
            float(np.clip(self.steps[self._index].confidence_shortfall, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
            float(np.clip(self.steps[self._index].model_disagreement, 0.0, 1.0))
            if self._index < len(self.steps) else 0.0,
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._index = 0
        self._equity = self._peak = 1.0
        self._prior_scale = 0.0
        self._loss_streak = 0
        self._returns = []
        return self._observation(), {}

    def step(self, action):
        scale = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        row = self.steps[self._index]
        before_dd = max(0.0, 1.0 - self._equity / self._peak)
        pnl = scale * row.return_pct - abs(scale - self._prior_scale) * 0.0001 - scale * row.cost_pct
        self._equity *= max(1e-6, 1.0 + pnl)
        self._peak = max(self._peak, self._equity)
        after_dd = max(0.0, 1.0 - self._equity / self._peak)
        reward = pnl - 2.5 * max(0.0, after_dd - before_dd)
        self._loss_streak = self._loss_streak + 1 if row.return_pct < 0.0 else 0
        self._returns.append(row.return_pct)
        self._prior_scale = scale
        self._index += 1
        terminated = self._index >= len(self.steps)
        info = {
            "strategy": row.strategy,
            "strategy_return": row.return_pct,
            "accepted_risk": scale,
            "equity": self._equity,
            "drawdown": after_dd,
        }
        return self._observation(), float(reward), terminated, False, info


def policy_payload(
    params: RiskPolicyParams,
    *,
    strategy_multipliers: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "model_type": "axiom_risk_controller_v1",
        "state_schema_version": 2,
        "features": [
            "trailing_volatility", "drawdown", "loss_streak", "prior_exposure",
            "portfolio_concentration", "exposure_resolved_fraction",
            "strategy_risk_prior", "correlation_cluster_stress",
            "regime_transition_risk", "liquidity_stress",
            "confidence_shortfall", "model_disagreement",
        ],
        "forbidden_features": [
            "direction", "pair", "strategy_direction_signal", "future_return",
        ],
        "action": "accepted_risk_fraction",
        "params": asdict(params),
        "strategy_risk_multipliers": dict(strategy_multipliers or {}),
        "runtime_allowed": False,
        "shadow_only": True,
    }
