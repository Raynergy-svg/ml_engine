#!/usr/bin/env python3
"""
Final Validation Report Generator — M1 Mac.

Scans all trained models, loads their metrics, validates direction accuracy
gaps, and produces a PASS/FAIL report per model per pair.

Usage:
    python scripts/generate_validation_report.py
    python scripts/generate_validation_report.py --output trained_data/validation_report.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("validation")

MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
MAX_GAP = 0.06  # 6% max acceptable gap

# All expected models per instrument
EXPECTED_CORE_MODELS = [
    "transformer_direction.keras",
    "tcn_volatility.keras",
    "lgbm_momentum.pkl",
    "lgbm_risk.pkl",
    "ridge_confidence.pkl",
    "histgb_direction.pkl",
]

EXPECTED_REGIME_MODELS = [
    "transformer_regime.keras",
    "tcn_volatility_regime.keras",
]

# Direction-critical models that must pass gap check
GAP_CHECKED = {"transformer_direction.keras", "tcn_volatility.keras", "histgb_direction.pkl"}

MASTER_PAIRS = ["GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD"]

ALL_PAIRS = [
    "GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD",
    "EUR_JPY", "AUD_JPY", "NZD_JPY", "USD_JPY",
    "EUR_USD", "EUR_AUD",
    "GBP_USD", "GBP_AUD",
    "NZD_USD",
    "USD_CHF",
]


def check_model_exists(pair: str, model_file: str) -> dict:
    """Check if a model file exists and get its size."""
    model_path = MODELS_DIR / pair / model_file
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return {"exists": True, "size_mb": round(size_mb, 2), "path": str(model_path)}
    return {"exists": False}


def load_training_summary(pair: str) -> dict:
    """Load training summary JSON if available."""
    summary_path = MODELS_DIR / pair / "training_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return {}


def validate_keras_model(pair: str, model_file: str) -> dict:
    """Validate a Keras model by loading it and checking its structure."""
    model_path = MODELS_DIR / pair / model_file
    if not model_path.exists():
        return {"valid": False, "error": "File not found"}

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(model_path), compile=False)
        input_shape = model.input_shape
        output_shape = model.output_shape
        n_params = model.count_params()
        tf.keras.backend.clear_session()
        return {
            "valid": True,
            "input_shape": str(input_shape),
            "output_shape": str(output_shape),
            "n_params": n_params,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_sklearn_model(pair: str, model_file: str) -> dict:
    """Validate a sklearn/lgbm model by loading it."""
    model_path = MODELS_DIR / pair / model_file
    if not model_path.exists():
        return {"valid": False, "error": "File not found"}

    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_type = type(model).__name__
        return {"valid": True, "model_type": model_type}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def generate_report(output_path: str) -> dict:
    """Generate the full validation report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_gap_threshold": MAX_GAP,
        "master_pairs": {},
        "target_pairs": {},
        "rl_suite": {},
        "meta_calibrator": {},
        "summary": {},
    }

    total_models = 0
    total_pass = 0
    total_fail = 0
    total_missing = 0

    # ── Master Pairs ──────────────────────────────────────────────────────────
    for pair in MASTER_PAIRS:
        pair_report = {"models": {}, "summary": {}}
        summary = load_training_summary(pair)

        for model_file in EXPECTED_CORE_MODELS + EXPECTED_REGIME_MODELS:
            total_models += 1
            status = check_model_exists(pair, model_file)

            if not status["exists"]:
                pair_report["models"][model_file] = {"status": "MISSING"}
                total_missing += 1
                continue

            # Validate model
            if model_file.endswith(".keras"):
                validation = validate_keras_model(pair, model_file)
            else:
                validation = validate_sklearn_model(pair, model_file)

            # Check gap from training summary
            gap_info = {}
            if summary and "results" in summary:
                for r in summary["results"]:
                    if r.get("model") and model_file.startswith(r["model"].replace("_direction", "")):
                        gap_info = {
                            "train_acc": r.get("train_acc"),
                            "val_acc": r.get("val_acc"),
                            "gap": r.get("gap"),
                        }
                        break

            # Determine pass/fail
            if model_file in GAP_CHECKED and gap_info.get("gap") is not None:
                passed = gap_info["gap"] <= MAX_GAP
            else:
                passed = validation.get("valid", False)

            if passed:
                total_pass += 1
            else:
                total_fail += 1

            pair_report["models"][model_file] = {
                "status": "PASS" if passed else "FAIL",
                **status,
                **validation,
                **gap_info,
            }

        pair_report["summary"] = {
            "models_found": sum(1 for m in pair_report["models"].values() if m.get("status") != "MISSING"),
            "models_pass": sum(1 for m in pair_report["models"].values() if m.get("status") == "PASS"),
            "models_fail": sum(1 for m in pair_report["models"].values() if m.get("status") == "FAIL"),
            "models_missing": sum(1 for m in pair_report["models"].values() if m.get("status") == "MISSING"),
        }
        report["master_pairs"][pair] = pair_report

    # ── Target Pairs ──────────────────────────────────────────────────────────
    target_pairs = [p for p in ALL_PAIRS if p not in MASTER_PAIRS]
    for pair in target_pairs:
        pair_report = {"models": {}}
        for model_file in EXPECTED_CORE_MODELS:
            total_models += 1
            status = check_model_exists(pair, model_file)
            if status["exists"]:
                total_pass += 1
                pair_report["models"][model_file] = {"status": "PASS", **status}
            else:
                total_missing += 1
                pair_report["models"][model_file] = {"status": "MISSING"}

        report["target_pairs"][pair] = pair_report

    # ── RL Suite ──────────────────────────────────────────────────────────────
    rl_models = {
        "position_sizer": MODELS_DIR / "rl_position_sizer.zip",
        "gate_thresholds": MODELS_DIR / "sac_gate_thresholds.zip",
        "optimal_exits": MODELS_DIR / "ppo_optimal_exit.zip",
        "reward_model": MODELS_DIR / "reward_model.pkl",
    }
    for name, path in rl_models.items():
        total_models += 1
        if path.exists():
            total_pass += 1
            report["rl_suite"][name] = {
                "status": "PASS",
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            }
        else:
            total_missing += 1
            report["rl_suite"][name] = {"status": "MISSING"}

    # ── Meta + Calibrator ─────────────────────────────────────────────────────
    meta_models = {
        "meta_labeler": MODELS_DIR / "joint" / "meta_labeler.pkl",
        "confidence_calibration": MODELS_DIR / "confidence_calibration.json",
    }
    for name, path in meta_models.items():
        total_models += 1
        if path.exists():
            total_pass += 1
            report["meta_calibrator"][name] = {"status": "PASS"}
        else:
            total_missing += 1
            report["meta_calibrator"][name] = {"status": "MISSING"}

    # ── Load sub-reports ──────────────────────────────────────────────────────
    for sub_report_name in ["transfer_learning_report.json", "rl_suite_report.json", "meta_calibrator_report.json"]:
        sub_path = MODELS_DIR / sub_report_name
        if sub_path.exists():
            try:
                with open(sub_path) as f:
                    report[f"sub_report_{sub_report_name}"] = json.load(f)
            except Exception:
                pass

    # ── Summary ───────────────────────────────────────────────────────────────
    report["summary"] = {
        "total_models": total_models,
        "passed": total_pass,
        "failed": total_fail,
        "missing": total_missing,
        "pass_rate": f"{(total_pass / total_models * 100):.1f}%" if total_models > 0 else "N/A",
        "overall_status": "PASS" if total_fail == 0 and total_missing == 0 else "FAIL",
    }

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)

    return report


def print_report(report: dict):
    """Print human-readable report to console."""
    print("\n" + "="*70)
    print("BUDDY ML ENGINE — TRAINING VALIDATION REPORT")
    print("="*70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Max Gap Threshold: {report['max_gap_threshold']}")

    # Master pairs
    print(f"\n{'─'*70}")
    print("MASTER PAIRS (Core Ensemble + Regime)")
    print(f"{'─'*70}")
    for pair, data in report.get("master_pairs", {}).items():
        s = data.get("summary", {})
        print(f"\n  {pair}:  {s.get('models_pass', 0)} PASS / {s.get('models_fail', 0)} FAIL / {s.get('models_missing', 0)} MISSING")
        for model, info in data.get("models", {}).items():
            status = info.get("status", "?")
            gap_str = f"  gap={info.get('gap', 'N/A')}" if info.get("gap") is not None else ""
            val_str = f"  val_acc={info.get('val_acc', 'N/A')}" if info.get("val_acc") is not None else ""
            icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "○")
            print(f"    {icon} {model}: {status}{gap_str}{val_str}")

    # Target pairs
    print(f"\n{'─'*70}")
    print("TARGET PAIRS (Transfer Learning)")
    print(f"{'─'*70}")
    for pair, data in report.get("target_pairs", {}).items():
        found = sum(1 for m in data.get("models", {}).values() if m.get("status") == "PASS")
        total = len(data.get("models", {}))
        print(f"  {pair}: {found}/{total} models")

    # RL Suite
    print(f"\n{'─'*70}")
    print("RL SUITE")
    print(f"{'─'*70}")
    for name, info in report.get("rl_suite", {}).items():
        icon = "✓" if info["status"] == "PASS" else "○"
        print(f"  {icon} {name}: {info['status']}")

    # Meta + Calibrator
    print(f"\n{'─'*70}")
    print("META-LABELER + CALIBRATOR")
    print(f"{'─'*70}")
    for name, info in report.get("meta_calibrator", {}).items():
        icon = "✓" if info["status"] == "PASS" else "○"
        print(f"  {icon} {name}: {info['status']}")

    # Summary
    s = report.get("summary", {})
    print(f"\n{'='*70}")
    print(f"OVERALL: {s.get('overall_status', '?')}  —  "
          f"{s.get('passed', 0)} passed, {s.get('failed', 0)} failed, "
          f"{s.get('missing', 0)} missing  ({s.get('pass_rate', 'N/A')})")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate training validation report")
    parser.add_argument("--output", default="trained_data/validation_report.json")
    args = parser.parse_args()

    report = generate_report(args.output)
    print_report(report)
    logger.info(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
