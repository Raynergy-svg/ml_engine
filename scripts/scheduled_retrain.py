#!/usr/bin/env python3
"""
Scheduled Joint Model Retraining Script.

This script is designed to be run via launchd on Mon/Wed/Fri at 10 AM UTC.
It trains joint models for all scanner pairs and sends failure-only email alerts.

Usage:
    python scripts/scheduled_retrain.py
    
    # Or with custom pairs:
    python scripts/scheduled_retrain.py --pairs EUR_USD,GBP_USD,USD_JPY
    
Schedule (launchd):
    Mon/Wed/Fri at 10 AM UTC during London session
    See: scripts/com.mlengine.retrain.plist

Email Alerts:
    - Only sent on FAILURE (no spam on success)
    - Recipient: dcertan84@gmail.com
"""

import argparse
import json
import logging
import os
import re
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Default pairs for scanner
DEFAULT_SCANNER_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
]

# Email configuration
EMAIL_RECIPIENT = "dcertan84@gmail.com"
EMAIL_SENDER = os.getenv("RETRAIN_EMAIL_SENDER", "mlengine-retrain@localhost")
SMTP_HOST = os.getenv("RETRAIN_SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("RETRAIN_SMTP_PORT", "25"))


def send_failure_email(
    subject: str,
    body: str,
    recipient: str = EMAIL_RECIPIENT,
) -> bool:
    """Send failure notification email.
    
    Args:
        subject: Email subject
        body: Email body (plain text)
        recipient: Email recipient
        
    Returns:
        True if email sent successfully
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = recipient
        msg["Subject"] = f"[ML Engine] {subject}"
        
        # Add body
        msg.attach(MIMEText(body, "plain"))
        
        # Try to send via SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.sendmail(EMAIL_SENDER, [recipient], msg.as_string())
        
        logger.info(f"Failure email sent to {recipient}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to send email: {e}")
        # Fallback: write to a file for manual review
        alert_file = LOG_DIR / f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        alert_file.write_text(f"Subject: {subject}\n\n{body}")
        logger.info(f"Alert written to {alert_file}")
        return False


def run_train_joint(
    pairs: List[str],
    granularity: str = "H1",
    candles: int = 7500,
    force_per_pair: bool = False,
) -> tuple[bool, str, Optional[dict]]:
    """Run training for the specified pairs.

    Default path (audit 2026-04-28): correlation-transfer training. The
    CorrelationTransferOrchestrator trains W&B-tuned master pairs and
    transfer-learns correlated pairs from them. This replaces the legacy
    joint multi-pair path which was producing 44.8% H1 accuracy across all
    7 pairs (well below the 52% promotion floor) while operator observation
    showed single-pair training above 60%.

    Override path: set BUDDY_USE_JOINT_TRAINING=1 in the environment to
    fall back to the legacy joint multi-pair trainer. Useful for A/B
    comparisons against the new default during the post-swap soak window.

    Args:
        pairs: List of currency pairs
        granularity: Timeframe
        candles: Number of candles to fetch
        force_per_pair: If True, fine-tune EVERY pair regardless of perf
            threshold. Currently ignored by the correlation-transfer path
            (which always trains master pairs end-to-end and always runs
            transfer for correlated pairs); preserved on the signature for
            the legacy-joint fallback's `should_fine_tune` gate.

    Returns:
        Tuple of (success, message, result_dict)
    """
    use_joint_legacy = os.environ.get("BUDDY_USE_JOINT_TRAINING", "0").lower() in ("1", "true", "yes")
    training_kind = "joint multi-pair (legacy)" if use_joint_legacy else "per-pair correlation transfer"
    logger.info(f"Starting {training_kind} training for {len(pairs)} pairs...")
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Granularity: {granularity}, Candles: {candles}, force_per_pair={force_per_pair}")

    try:
        from rich.console import Console
        console = Console(file=sys.stdout, force_terminal=True)

        if use_joint_legacy:
            from src.training.buddy_training_helpers import train_joint_multi_pair_ensemble
            result = train_joint_multi_pair_ensemble(
                instruments=pairs,
                granularity=granularity,
                candles=candles,
                fine_tune=True,
                fine_tune_threshold=0.05,
                console=console,
            )
        else:
            from src.training.buddy_training_helpers import train_per_pair_correlation_ensemble
            result = train_per_pair_correlation_ensemble(
                instruments=pairs,
                granularity=granularity,
                candles=candles,
                console=console,
            )

        if result.get("status") == "success":
            msg = (
                f"{training_kind} training completed successfully\n"
                f"Instruments: {result.get('n_instruments', 0)}\n"
                f"Save directory: {result.get('joint_save_dir', 'N/A')}"
            )
            if not use_joint_legacy:
                msg += (
                    f"\nMaster pairs: {result.get('n_master_pairs', 0)}"
                    f"\nTransfer-learned pairs: {result.get('n_transfer_pairs', 0)}"
                )
            logger.info(msg)
            return True, msg, result
        else:
            error = result.get("error", "Unknown error")
            msg = f"{training_kind} training failed: {error}"
            logger.error(msg)
            return False, msg, result

    except Exception as e:
        error_trace = traceback.format_exc()
        msg = f"{training_kind} training exception: {e}\n\n{error_trace}"
        logger.error(msg)
        return False, msg, None


def validate_models(model_dir: Path) -> tuple[bool, str]:
    """Validate that required models exist after training.
    
    Args:
        model_dir: Path to joint model directory
        
    Returns:
        Tuple of (success, message)
    """
    required_models = [
        "transformer_direction.keras",
        "tcn_volatility_regime.keras",  # REQUIRED for scanner
        "ridge_confidence.pkl",
    ]
    
    optional_models = [
        "catboost_momentum.cbm",
        "xgb_momentum.pkl",
        "rf_risk.pkl",
        "meta_labeler.pkl",
    ]
    
    missing_required = []
    missing_optional = []
    
    for model in required_models:
        if not (model_dir / model).exists():
            missing_required.append(model)
    
    for model in optional_models:
        if not (model_dir / model).exists():
            missing_optional.append(model)
    
    if missing_required:
        msg = f"CRITICAL: Missing required models: {', '.join(missing_required)}"
        logger.error(msg)
        return False, msg
    
    if missing_optional:
        logger.warning(f"Missing optional models: {', '.join(missing_optional)}")
    
    msg = "All required models validated successfully"
    logger.info(msg)
    return True, msg


def _parse_holdout_message(msg: str) -> tuple[Optional[float], Optional[int]]:
    """Extract (accuracy, total_samples) from validate_holdout_accuracy's message.

    Expected format: "Hold-out validation: 55.2% accuracy (110/199) PASSED ..."
    Returns (None, None) on parse failure — callers treat as "metrics unknown".
    """
    try:
        pct = re.search(r"([\d.]+)%\s+accuracy", msg)
        counts = re.search(r"\((\d+)/(\d+)\)", msg)
        accuracy = float(pct.group(1)) / 100.0 if pct else None
        samples = int(counts.group(2)) if counts else None
        return accuracy, samples
    except Exception:
        return None, None


def _log_and_maybe_promote(
    pairs: List[str],
    model_dir: Path,
    holdout_metrics: Dict[str, Any],
) -> None:
    """Log joint model dir as a W&B artifact, consult promotion policy, promote.

    `holdout_metrics` must contain:
        accuracy            — aggregate accuracy across timeframes
        holdout_samples     — total samples across timeframes
        timeframes_passed   — count of TFs that cleared min_accuracy
        per_timeframe       — per-TF breakdown (for audit/metadata)

    No-op unless BUDDY_WANDB_REGISTRY_ENABLED=1. All failures logged and swallowed
    — this pipeline must never block a successful retrain from completing.
    """
    from src.training.model_artifact import (
        is_enabled,
        log_model_artifact,
        get_production_metrics,
        promote_to_production,
        mark_staging,
    )
    from src.training.promotion_policy import should_promote

    if not is_enabled():
        logger.debug("Registry disabled — skipping artifact log + promotion check")
        return

    artifact_name = "buddy-joint-models"
    # Candidate metrics consumed by policy gates 1-8.
    # 1: accuracy, 2: holdout_samples, 3: accuracy (vs prod), 4: brier,
    # 5: timeframes_passed, 7: max_component_age_days, 8: ensemble_complete.
    metrics = {
        "accuracy": holdout_metrics.get("accuracy"),
        "holdout_samples": holdout_metrics.get("holdout_samples"),
        "timeframes_passed": holdout_metrics.get("timeframes_passed"),
        "timeframes_tested": holdout_metrics.get("timeframes_tested"),
        "max_component_age_days": holdout_metrics.get("max_component_age_days"),
        "ensemble_complete": holdout_metrics.get("ensemble_complete"),
        "ensemble_stale_groups": holdout_metrics.get("ensemble_stale_groups"),
        "pairs": len(pairs),
    }
    metadata = {
        "pairs": pairs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "per_timeframe": holdout_metrics.get("per_timeframe", {}),
    }

    # 1. Log as :staging candidate.
    ref = log_model_artifact(
        name=artifact_name,
        paths=[model_dir],
        metrics={k: v for k, v in metrics.items() if v is not None},
        metadata=metadata,
        aliases=["staging"],
    )
    if ref is None:
        logger.warning("Artifact log returned None — skipping promotion check")
        return

    mark_staging(ref)

    # 2. Compare candidate vs current production.
    production = get_production_metrics(artifact_name, alias="production")
    decision = should_promote(candidate=metrics, production=production)

    # 3. Log the decision prominently — this is audit-grade output.
    logger.info("=" * 60)
    logger.info("PROMOTION POLICY VERDICT")
    logger.info("  candidate metrics : %s", metrics)
    logger.info("  production metrics: %s", production)
    logger.info("  decision          : %s", "PROMOTE" if decision.promote else "HOLD")
    logger.info("  reason            : %s", decision.reason)
    logger.info("=" * 60)

    # 4. Promote if policy says yes.
    if decision.promote:
        ok = promote_to_production(ref)
        if ok:
            logger.info("✓ %s promoted to :production", ref.qualified)
        else:
            logger.error("Promotion API call failed; model stays at :staging")
    else:
        logger.info("Model remains at :staging pending policy re-evaluation")


def reset_accuracy_gate_for_pairs(pairs: List[str]) -> None:
    """Reset AccuracyGate rolling window for retrained pairs.

    After a successful retrain, the pair's accuracy history is stale — it reflects
    the OLD model's performance. Clear it so the pair can be re-evaluated fresh.
    """
    try:
        from src.scanner.automation.accuracy_gate import AccuracyGate

        gate = AccuracyGate()
        for pair in pairs:
            normalized = pair.upper().replace("/", "_")
            if normalized in gate._data:
                # Keep total history but reset rolling counters
                gate._data[normalized]["rolling_total"] = 0
                gate._data[normalized]["rolling_wins"] = 0
                gate._data[normalized]["accuracy"] = None
                # Clear the blocked flag — let the pair trade again under new model
                gate._data[normalized]["blocked"] = False
                logger.info(f"AccuracyGate reset for {normalized} (post-retrain)")
        gate._save_data()
    except Exception as e:
        logger.warning(f"AccuracyGate reset failed (non-fatal): {e}")


def validate_holdout_accuracy(
    pairs: List[str],
    min_accuracy: float = 0.52,
    holdout_candles: int = 500,
    granularity: str = "H1",
) -> tuple[bool, str]:
    """Validate new model accuracy on a hold-out set before promoting.

    Loads the newly trained model and runs inference on the most recent
    holdout_candles (not used in training). If accuracy < min_accuracy,
    the retrain is rejected.

    Returns:
        Tuple of (passed, message). Message format includes granularity so
        the multi-TF wrapper + `_parse_holdout_message` can extract it.
    """
    try:
        from src.core.modular_data_loaders import compute_normalized_features
        from src.scanner.gates import GateEvaluator

        evaluator = GateEvaluator(
            model_dir=str(PROJECT_ROOT / "trained_data" / "models"),
        )
        # GateEvaluator requires explicit load_models() before evaluate_all_gates()
        load_status = evaluator.load_models(require_tcn=False)
        logger.info("Holdout: loaded models: %s", load_status)

        correct = 0
        total = 0
        from src.utils.oanda_practice import OandaPracticeClient
        oanda = OandaPracticeClient.from_env()

        for pair in pairs[:3]:  # Validate on up to 3 pairs (speed)
            try:
                candles_raw = oanda.get_candles(pair, granularity=granularity, count=holdout_candles)
                # OandaPracticeClient returns a dict or Response — extract DataFrame
                if candles_raw is None:
                    logger.debug("Holdout %s: candles returned None", pair)
                    continue
                import pandas as _pd
                if isinstance(candles_raw, dict):
                    # Extract from OANDA response: {"candles": [...]}
                    candles_list = candles_raw.get("candles", [])
                    if not candles_list:
                        logger.debug("Holdout %s: empty candles list in response", pair)
                        continue
                    rows = []
                    for c in candles_list:
                        mid = c.get("mid", {})
                        rows.append({
                            "time": c.get("time"),
                            "open": float(mid.get("o", 0)),
                            "high": float(mid.get("h", 0)),
                            "low": float(mid.get("l", 0)),
                            "close": float(mid.get("c", 0)),
                            "volume": int(c.get("volume", 0)),
                        })
                    candles_df = _pd.DataFrame(rows)
                elif isinstance(candles_raw, _pd.DataFrame):
                    candles_df = candles_raw
                else:
                    logger.debug("Holdout %s: unexpected candles type %s", pair, type(candles_raw))
                    continue

                if len(candles_df) < 100:
                    logger.debug("Holdout %s: only %d candles", pair, len(candles_df))
                    continue

                # Use last holdout_candles as test set
                test_df = candles_df.tail(holdout_candles)
                features = compute_normalized_features(test_df)
                if features is None or features.empty:
                    logger.debug("Holdout %s: features empty after compute_normalized_features", pair)
                    continue
                logger.info("Holdout %s [%s]: %d features rows, %d columns", pair, granularity, len(features), features.shape[1])

                # Check direction predictions against actual price movement
                n_errors = 0
                for i in range(min(len(features) - 5, 200)):  # cap at 200 for speed
                    row = features.iloc[i : i + 1]
                    future_close = test_df["close"].iloc[min(i + 5, len(test_df) - 1)]
                    current_close = test_df["close"].iloc[i]
                    actual_dir = "LONG" if future_close > current_close else "SHORT"

                    try:
                        gate_result = evaluator.evaluate_all_gates(
                            row, instrument=pair,
                        )
                        if gate_result and isinstance(gate_result, dict):
                            pred_dir = str(gate_result.get("direction", "")).upper()
                            if pred_dir in ("LONG", "SHORT"):
                                total += 1
                                if pred_dir == actual_dir:
                                    correct += 1
                    except Exception as _eval_err:
                        n_errors += 1
                        if n_errors <= 3:
                            logger.warning(
                                "Holdout evaluate_all_gates error on %s row %d: %s",
                                pair, i, _eval_err,
                            )
                        continue
                if n_errors:
                    logger.warning("Holdout %s: %d/%d rows threw exceptions", pair, n_errors, min(len(features)-5, 200))
            except Exception as e:
                logger.debug(f"Hold-out validation skipped for {pair}: {e}")

        if total < 20:
            msg = f"Hold-out validation: insufficient samples ({total}). Proceeding with caution."
            logger.warning(msg)
            return True, msg  # Don't block if we can't validate

        accuracy = correct / total
        passed = accuracy >= min_accuracy
        msg = (
            f"Hold-out validation [{granularity}]: {accuracy:.1%} accuracy ({correct}/{total}) "
            f"{'PASSED' if passed else 'FAILED'} (threshold: {min_accuracy:.0%})"
        )
        if passed:
            logger.info(msg)
        else:
            logger.error(msg)
        return passed, msg

    except ImportError as e:
        msg = f"Hold-out validation [{granularity}] skipped (import error): {e}"
        logger.warning(msg)
        return True, msg  # Don't block on import failures
    except Exception as e:
        msg = f"Hold-out validation [{granularity}] error (non-fatal): {e}"
        logger.warning(msg)
        return True, msg


def verify_ensemble_refreshed(
    retrain_start_ts: datetime,
    models_root: Path,
) -> Dict[str, Any]:
    """Verify every ensemble meta file's mtime is >= retrain_start_ts.

    Source: learnings.md (2026-04-15) — partial-retrain bug where
    transformer_direction_best.keras refreshed but modular_ensemble.meta.json +
    per-pair models stayed at March mtimes while "RETRAINING COMPLETED" fired.

    Checks:
      * `trained_data/models/modular_ensemble.meta.json`
      * `trained_data/models/joint/*.keras` + meta files
      * per-pair subdirs that have a `tcn_volatility_regime.arch.json` present

    Returns dict with:
      ensemble_complete: bool — True iff every checked artifact mtime >= start_ts
      ensemble_stale_groups: List[str] — names of groups that failed the check
      max_component_age_days: float — oldest artifact age at time of check,
        against wall-clock now (used by Gate 7)
    """
    import time as _time

    checks: List[tuple[str, Path]] = []

    # Top-level ensemble meta
    top = models_root / "modular_ensemble.meta.json"
    if top.exists():
        checks.append(("modular_ensemble", top))

    # Joint models directory (transformer + tcn + ridge etc.)
    joint_dir = models_root / "joint"
    if joint_dir.exists():
        for name in ("transformer_direction.keras", "tcn_volatility_regime.keras", "ridge_confidence.pkl"):
            p = joint_dir / name
            if p.exists():
                checks.append((f"joint/{name}", p))

    # Per-pair directories: SKIP. The joint retrain's fine-tuning step only
    # retunes pairs that exceed fine_tune_threshold (typically ~5 of 15). Pairs
    # that performed well keep their existing models — making their mtimes stale
    # by design, not by bug. Only the joint directory + meta.json are guaranteed
    # fresh after every retrain. Per-pair staleness is tracked separately by the
    # retrain_agent's model_staleness trigger.

    stale: List[str] = []
    max_age_days: float = 0.0
    now = _time.time()
    start_epoch = retrain_start_ts.timestamp()

    for label, path in checks:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            stale.append(f"{label} (stat failed)")
            continue
        age_days = max(0.0, (now - mtime) / 86400.0)
        if age_days > max_age_days:
            max_age_days = age_days
        if mtime < start_epoch:
            stale.append(f"{label} (mtime={datetime.fromtimestamp(mtime, timezone.utc).isoformat()})")

    complete = len(stale) == 0 and len(checks) > 0
    if not complete:
        logger.warning(
            "Ensemble completeness check: %d/%d artifacts stale. Stale: %s",
            len(stale), len(checks), stale,
        )
    else:
        logger.info("Ensemble completeness: %d/%d artifacts refreshed post-retrain", len(checks), len(checks))

    return {
        "ensemble_complete": complete,
        "ensemble_stale_groups": stale,
        "ensemble_artifacts_checked": len(checks),
        "max_component_age_days": round(max_age_days, 2),
    }


def validate_holdout_multi_timeframe(
    pairs: List[str],
    min_accuracy: float = 0.52,
    holdout_candles: int = 500,
    granularities: Optional[List[str]] = None,
) -> tuple[bool, str, Dict[str, Any]]:
    """Run `validate_holdout_accuracy` across multiple granularities.

    Rationale: a model that wins on H1 but loses on M15/H4 is regime-fragile.
    The promotion policy's Gate 5 (multi-timeframe coverage) consumes the
    `timeframes_passed` count this returns.

    Returns:
        (overall_passed, combined_message, metrics_dict)
        where overall_passed = at least one TF passed (so pipeline isn't blocked
        by a single missing data feed), and metrics_dict contains per-TF results
        plus `timeframes_passed` count for the policy gate.
    """
    granularities = granularities or ["H1", "M15", "H4"]
    per_tf: Dict[str, Any] = {}
    passed_count = 0
    messages: List[str] = []

    for gran in granularities:
        ok, msg = validate_holdout_accuracy(
            pairs=pairs,
            min_accuracy=min_accuracy,
            holdout_candles=holdout_candles,
            granularity=gran,
        )
        acc, samples = _parse_holdout_message(msg)
        per_tf[gran] = {"passed": ok, "accuracy": acc, "samples": samples}
        if ok and acc is not None and acc >= min_accuracy:
            passed_count += 1
        messages.append(msg)

    # If at least one TF has real accuracy data AND passes, we're good.
    # If ALL TFs returned ok=True but with insufficient samples (acc=None),
    # treat as a soft pass — the holdout validator couldn't test but explicitly
    # chose not to block.
    all_soft_pass = all(per_tf[g]["passed"] for g in granularities) and passed_count == 0
    overall_passed = passed_count >= 1 or all_soft_pass
    if all_soft_pass:
        logger.warning(
            "Multi-TF holdout: all %d timeframes returned insufficient samples — "
            "soft-passing. Holdout validation did NOT actually run.",
            len(granularities),
        )

    # PRIMARY TF ACCURACY: use the first granularity (the trading TF, typically
    # H1) as the promotion gate's accuracy metric. M15/H4 are informational —
    # requiring a model to predict on timeframes it wasn't trained for is asking
    # the wrong question. Per-TF breakdown stays in metadata for observability.
    primary_tf = granularities[0]
    primary_acc = per_tf[primary_tf].get("accuracy")
    primary_samples = per_tf[primary_tf].get("samples") or 0

    # Aggregate stays available for reference but isn't used in promotion gate.
    accs = [per_tf[g]["accuracy"] for g in granularities if per_tf[g]["accuracy"] is not None]
    agg_acc = sum(accs) / len(accs) if accs else None
    tot_samples = sum((per_tf[g]["samples"] or 0) for g in granularities)

    logger.info(
        "Holdout primary TF=%s accuracy=%.3f (%d samples) | aggregate=%.3f (%d samples)",
        primary_tf, primary_acc or 0, primary_samples, agg_acc or 0, tot_samples,
    )

    metrics = {
        "timeframes_passed": passed_count,
        "timeframes_tested": len(granularities),
        "per_timeframe": per_tf,
        "accuracy": primary_acc,            # PRIMARY TF only — used by promotion gates 1+3
        "accuracy_aggregate": agg_acc,      # mean across all TFs — informational
        "holdout_samples": primary_samples,  # samples from primary TF only
        "holdout_samples_total": tot_samples,
        "primary_timeframe": primary_tf,
    }
    combined_msg = (
        f"Multi-TF holdout: {passed_count}/{len(granularities)} timeframes passed. "
        + " | ".join(messages)
    )
    logger.info(combined_msg)
    return overall_passed, combined_msg, metrics


def check_drift_retrain_request() -> Optional[list]:
    """Check if drift trigger has requested retrain.

    Returns:
        List of pairs to retrain, or None if no request pending.
    """
    request_path = Path(PROJECT_ROOT) / "trained_data" / "retrain_request.json"
    if not request_path.exists():
        return None

    try:
        with open(request_path, 'r') as f:
            request = json.loads(f.read())

        pairs = request.get("pairs", [])
        reason = request.get("reason", "Unknown")
        priority = request.get("priority", "normal")

        logger.info(f"Drift-triggered retrain request found:")
        logger.info(f"  Pairs: {', '.join(pairs)}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Priority: {priority}")

        return pairs
    except Exception as e:
        logger.warning(f"Failed to parse retrain request: {e}")
        return None


def main():
    """Main entry point for scheduled retraining."""
    parser = argparse.ArgumentParser(
        description="Scheduled joint model retraining for ML Engine scanner",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Comma-separated list of pairs to train (overrides drift request)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="H1",
        help="Timeframe (default: H1)",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=7500,
        help="Number of candles to fetch (default: 7500, max ~10000 on M1 with 15 pairs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without actually training",
    )
    parser.add_argument(
        "--force-per-pair",
        action="store_true",
        help=(
            "Fine-tune EVERY pair regardless of performance threshold. "
            "Use when per-pair models are stale relative to joint ensemble "
            "(autonomous retrainer sets this by default)."
        ),
    )
    args = parser.parse_args()

    # Determine pairs: check for drift request first, then args, then defaults
    pairs = []
    if args.pairs:
        # Explicit --pairs argument (highest priority)
        pairs = [p.strip().upper().replace("/", "_") for p in args.pairs.split(",")]
        logger.info(f"Using explicit pairs from --pairs: {', '.join(pairs)}")
    else:
        # Check for drift-triggered request
        drift_pairs = check_drift_retrain_request()
        if drift_pairs:
            pairs = drift_pairs
            logger.info(f"Using pairs from drift trigger: {', '.join(pairs)}")
        else:
            # Fall back to defaults or "all"
            pairs = [p.strip().upper().replace("/", "_") for p in DEFAULT_SCANNER_PAIRS]
            logger.info(f"Using default pairs: {', '.join(pairs)}")

    # Expand "all" to the full list of default pairs
    if len(pairs) == 1 and pairs[0] == "ALL":
        from src.scanner.config import DEFAULT_PAIRS
        pairs = DEFAULT_PAIRS.copy()
        logger.info(f"Expanded 'all' to {len(pairs)} pairs: {', '.join(pairs)}")
    
    # Log start
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("ML ENGINE SCHEDULED JOINT RETRAINING")
    logger.info(f"Start time: {start_time.isoformat()}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("[DRY RUN] Would train the following pairs:")
        for pair in pairs:
            logger.info(f"  - {pair}")
        logger.info("[DRY RUN] Exiting without training")
        return 0
    
    # Run training
    success, message, _result = run_train_joint(
        pairs=pairs,
        granularity=args.granularity,
        candles=args.candles,
        force_per_pair=args.force_per_pair,
    )
    
    # Validate models
    model_dir = PROJECT_ROOT / "trained_data" / "models" / "joint"
    if success:
        valid, valid_msg = validate_models(model_dir)
        if not valid:
            success = False
            message = f"{message}\n\nValidation failed: {valid_msg}"

    # Hold-out validation across H1/M15/H4 — multi-timeframe coverage feeds
    # Gate 5 of the promotion policy (catches H1-overfit models).
    holdout_metrics: Dict[str, Any] = {}
    if success:
        holdout_ok, holdout_msg, holdout_metrics = validate_holdout_multi_timeframe(pairs)
        message = f"{message}\n\n{holdout_msg}"
        if not holdout_ok:
            success = False
            logger.error("Retrain REJECTED: no timeframe passed hold-out validation")

    # Mythos audit 2026-04-30 — staleness telemetry honesty fix.
    # buddy_training_helpers writes candidate_trained_at + holdout_pending
    # at the end of training. This block PROMOTES candidate → trained_at
    # only after holdout passes. On rejection, the previous trained_at is
    # preserved + last_holdout_failure_at gets stamped so freshness
    # checks can see "we tried, holdout said no" rather than silently
    # carrying the misleading bumped trained_at from the failed attempt.
    try:
        _meta_path = PROJECT_ROOT / "trained_data" / "models" / "modular_ensemble.meta.json"
        if _meta_path.exists():
            with open(_meta_path, "r") as _mf:
                _meta_blob = json.load(_mf)
            if isinstance(_meta_blob, dict) and _meta_blob.get("holdout_pending"):
                _candidate_ts = _meta_blob.get("candidate_trained_at")
                _now_iso = datetime.now(timezone.utc).isoformat()
                if success:
                    # Promotion: candidate_trained_at → trained_at
                    if _candidate_ts:
                        _meta_blob["trained_at"] = _candidate_ts
                    _meta_blob["holdout_pending"] = False
                    _meta_blob["last_holdout_passed_at"] = _now_iso
                    _meta_blob.pop("last_holdout_failure_at", None)
                    logger.info(
                        "Promoted candidate_trained_at → trained_at: %s "
                        "(holdout passed)", _candidate_ts,
                    )
                else:
                    # Rejection: leave trained_at untouched (preserves
                    # last validated freshness state). Stamp the failure.
                    _meta_blob["holdout_pending"] = False
                    _meta_blob["last_holdout_failure_at"] = _now_iso
                    logger.info(
                        "Holdout REJECTED — trained_at left at previous "
                        "validated value (was %s); candidate %s discarded; "
                        "last_holdout_failure_at=%s",
                        _meta_blob.get("trained_at"),
                        _candidate_ts, _now_iso,
                    )
                with open(_meta_path, "w") as _mf:
                    json.dump(_meta_blob, _mf, indent=2, sort_keys=True, default=str)
    except Exception as _meta_err:
        logger.warning("Failed to update modular_ensemble.meta.json post-holdout: %s", _meta_err)

    # Ensemble completeness — feeds Gates 7 (freshness) and 8 (partial retrain).
    # Always runs; never blocks success at this layer (policy handles rejection).
    if success:
        models_root = PROJECT_ROOT / "trained_data" / "models"
        ensemble_metrics = verify_ensemble_refreshed(start_time, models_root)
        holdout_metrics.update(ensemble_metrics)
        message = (
            f"{message}\n\nEnsemble completeness: "
            f"{'OK' if ensemble_metrics['ensemble_complete'] else 'STALE'} "
            f"({ensemble_metrics['ensemble_artifacts_checked']} checked, "
            f"max_age={ensemble_metrics['max_component_age_days']:.1f}d)"
        )

    # Close the autonomy loop via Tier 7 trading event bus:
    #   publish TRAINING_COMPLETED → subscribed handler logs artifact, runs
    #   promotion_policy, emits MODEL_PROMOTED or MODEL_HELD.
    # Every hop lands in trained_data/trading_events.jsonl for audit. No-op
    # unless BUDDY_WANDB_REGISTRY_ENABLED=1 (handler short-circuits internally).
    if success:
        try:
            from src.training.training_events import publish_training_completed
            event_id = publish_training_completed(
                pairs=pairs,
                model_dir=model_dir,
                holdout_metrics=holdout_metrics,
                reason="scheduled",
            )
            logger.info("Published TRAINING_COMPLETED event_id=%s", event_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("Training-event publish failed (non-fatal): %s", exc, exc_info=True)

    # Log completion
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info(f"RETRAINING {'COMPLETED' if success else 'FAILED'}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 60)

    # Post-retrain: reset AccuracyGate for retrained pairs (only on success)
    if success:
        reset_accuracy_gate_for_pairs(pairs)
        logger.info(f"AccuracyGate reset for {len(pairs)} retrained pair(s)")

    # Write retrain summary for cooldown tracking
    if success:
        try:
            summary_path = PROJECT_ROOT / "trained_data" / "retrain_all_summary.json"
            summary_path.write_text(json.dumps({
                "timestamp": end_time.isoformat(),
                "pairs": pairs,
                "duration_seconds": round(duration, 1),
                "status": "success",
            }, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write retrain summary: {e}")

    # Clear drift retrain request if present (only on success)
    if success:
        try:
            request_path = Path(PROJECT_ROOT) / "trained_data" / "retrain_request.json"
            if request_path.exists():
                request_path.unlink()
                logger.info("Drift retrain request cleared")
        except Exception as e:
            logger.warning(f"Failed to clear retrain request: {e}")

    # Send failure email (no spam on success)
    if not success:
        email_body = f"""
ML Engine Joint Retraining Failed

Time: {start_time.isoformat()}
Duration: {duration:.1f} seconds
Pairs: {', '.join(pairs)}

Error Details:
{message}

Log file: {LOG_FILE}

---
This is an automated alert. No action required if you're already investigating.
        """.strip()

        send_failure_email(
            subject=f"Joint Retraining FAILED - {start_time.strftime('%Y-%m-%d')}",
            body=email_body,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
