"""Post-deploy critic for the meta-cybernetic change pipeline.

After a `ChangePackage` enters a deploy stage (shadow / canary / live),
the critic compares predicted vs realized metrics N cycles later. If
the realized metrics fall outside the scorecard's prediction envelope,
the critic recommends rollback. If they hold, it recommends promotion
to the next stage.

The critic does not act on the recommendation itself — it returns a
`PostDeployReview` and lets `MetaManager` choose between
`StagedDeployer.promote_next` and `StagedDeployer.rollback`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.scanner.automation.meta_types import (
    ChangePackage,
    DeployStage,
    PostDeployReview,
)

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_LEDGER = _PROJECT_ROOT / ".claude" / "meta" / "changes.jsonl"

# Tolerances for "realized close enough to predicted"
PNL_DELTA_TOLERANCE_R = 1.5
WIN_RATE_TOLERANCE = 0.10


MetricsSlicer = Callable[[ChangePackage, DeployStage], Dict[str, Any]]


def _default_metrics_slicer(package: ChangePackage, stage: DeployStage) -> Dict[str, Any]:
    """Best-effort default that pulls live metrics from the trade journal.

    Reports sample_size / win_rate / pnl_total_pips / r_mean over the
    post-deploy slice. When DeploymentRecord.regime_at_deploy is present,
    additionally emits a `regime_slices` breakdown keyed by the regime
    found in each trade's outcome dict — input for the 14-day deferred-G2
    review's per-regime distribution comparison.
    """
    try:
        from src.scanner.automation.safe_json import safe_json_read
        journal_path = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
        entries = safe_json_read(journal_path, default=[])
        if not isinstance(entries, list):
            return {}
        deploy_record = next(
            (d for d in package.deployments if d.stage == stage and d.rolled_back_at is None),
            None,
        )
        if deploy_record is None:
            return {}
        cutoff = deploy_record.deployed_at
        slice_ = [e for e in entries if str(e.get("timestamp", "")) >= cutoff]
        if not slice_:
            return {"sample_size": 0}

        def r_of(e):
            o = (e.get("outcome") or {}) if isinstance(e, dict) else {}
            try:
                return float(o.get("r_multiple", 0.0))
            except (TypeError, ValueError):
                return 0.0

        wins = sum(1 for e in slice_ if (e.get("outcome", {}) or {}).get("trade_won"))
        pnl_total = sum(float((e.get("outcome", {}) or {}).get("pnl_pips", 0)) for e in slice_)
        r_mean = sum(r_of(e) for e in slice_) / len(slice_)
        out: Dict[str, Any] = {
            "sample_size": len(slice_),
            "win_rate": wins / len(slice_),
            "pnl_total_pips": pnl_total,
            "r_mean": r_mean,
        }

        # Regime slicing — only when the deploy record was tagged with a regime.
        if deploy_record.regime_at_deploy:
            slices: Dict[str, Dict[str, Any]] = {}
            seen_regimes = {
                (e.get("outcome", {}) or {}).get("regime")
                for e in slice_
                if isinstance(e, dict)
            }
            for regime in seen_regimes:
                if not regime:
                    continue
                bucket = [
                    e for e in slice_
                    if (e.get("outcome", {}) or {}).get("regime") == regime
                ]
                if not bucket:
                    continue
                b_wins = sum(
                    1 for e in bucket
                    if (e.get("outcome", {}) or {}).get("trade_won")
                )
                b_r = sum(r_of(e) for e in bucket) / len(bucket)
                slices[regime] = {
                    "sample_size": len(bucket),
                    "win_rate": b_wins / len(bucket),
                    "r_mean": b_r,
                }
            out["regime_slices"] = slices
        return out
    except Exception as e:
        logger.debug("post_deploy_critic.default_slicer_failed err=%s", e)
        return {}


class PostDeployCritic:
    """Evaluates predicted vs realized for one deploy stage."""

    def __init__(
        self,
        metrics_slicer: Optional[MetricsSlicer] = None,
        ledger_path: Optional[Path] = None,
    ):
        self._slice = metrics_slicer or _default_metrics_slicer
        self._ledger = Path(ledger_path or _DEFAULT_LEDGER)

    def review(self, package: ChangePackage, stage: DeployStage) -> PostDeployReview:
        predicted = self._predicted_from_scorecard(package)
        actual = self._slice(package, stage) or {}
        delta = _delta(predicted, actual)
        passed = self._evaluate(predicted, actual, delta)
        review = PostDeployReview(
            stage=stage,
            predicted=predicted,
            actual=actual,
            delta=delta,
            passed=passed,
            notes="" if passed else "realized_outside_predicted_envelope",
        )
        package.post_deploy_reviews.append(review)
        package.touch(f"post_deploy_review stage={stage.value} passed={passed}")
        self._append_ledger(package, review)
        logger.info(
            "post_deploy_critic.review change_id=%s stage=%s passed=%s",
            package.change_id, stage.value, passed,
        )
        return review

    def _predicted_from_scorecard(self, package: ChangePackage) -> Dict[str, Any]:
        if package.scorecard is None or package.scorecard.trading_quality is None:
            return {}
        trading = package.scorecard.trading_quality.details or {}
        return {
            "win_rate": float(trading.get("win_rate", 0.0)),
            "pnl_delta_r": float(trading.get("pnl_delta_r", 0.0)),
            "sharpe": float(trading.get("sharpe", 0.0)),
        }

    def _evaluate(
        self,
        predicted: Dict[str, Any],
        actual: Dict[str, Any],
        delta: Dict[str, Any],
    ) -> bool:
        if not actual or actual.get("sample_size", 0) == 0:
            return True
        win_rate_drift = abs(float(delta.get("win_rate", 0.0)))
        pnl_drift = abs(float(delta.get("pnl_delta_r", 0.0)))
        if win_rate_drift > WIN_RATE_TOLERANCE:
            return False
        if pnl_drift > PNL_DELTA_TOLERANCE_R:
            return False
        return True

    def _append_ledger(self, package: ChangePackage, review: PostDeployReview) -> None:
        try:
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "change_id": package.change_id,
                "stage": review.stage.value,
                "predicted": review.predicted,
                "actual": review.actual,
                "delta": review.delta,
                "passed": review.passed,
                "reviewed_at": review.reviewed_at,
            }
            with open(self._ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("post_deploy_critic.ledger_failed err=%s", e)


def _delta(predicted: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in {*predicted.keys(), *actual.keys()}:
        try:
            p = float(predicted.get(key, 0.0))
            a = float(actual.get(key, 0.0))
            out[key] = a - p
        except (TypeError, ValueError):
            out[key] = None
    return out
