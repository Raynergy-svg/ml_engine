"""Gated retrain entrypoint for the risk-target model family (forward
volatility + forward drawdown-state).

Offline/research only. Writes to `trained_data/risk_targets/models/` — a
namespace distinct from the FX per-pair `trained_data/models/{PAIR}/`
ensemble dirs `src/scanner/gates.py` scans, so this can never collide with
or be picked up by the live FX gate/inference path. Never imports
`execution.py`/`gates.py`/`state_engine`/any broker client (see
`tests/test_risk_target_no_hotpath_coupling.py`).

Reuses the SAME gate primitives already enforced in
`cli/training_ops.py::retrain_gates()` (`_cli_gate_verdict`,
`_cli_gate_degenerate`, `_log_cli_gate`) rather than duplicating gate logic
— "keep it gated" per operator directive. A candidate may only overwrite
the incumbent artifact if it does not regress (beyond a 5% relative
tolerance, same as every other CLI-gated head) on EITHER head's OOS/val
metric (QLIKE for volatility, Brier for drawdown-state — both natively
lower-is-better, so they plug directly into the shared MAE-style gate
primitive). Missing/unscorable incumbent = first deploy = PASSED, same
convention as `_apply_rf_gate`/`_apply_xgb_gate`/`_apply_ridge_gate`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from cli.training_ops import (
    CLI_GATE_PASSED,
    CLI_GATE_REFUSED,
    _cli_gate_verdict,
    _log_cli_gate,
)
from src.training.trainers.config import TrainerConfig
from src.training.trainers.risk_target_trainer import RiskTargetTrainer

logger = logging.getLogger(__name__)

MODEL_DIR = Path("trained_data/risk_targets/models")
MODEL_BASENAME = "risk_target_model"


def _score_existing_metrics(model_path: Path) -> Optional[Dict[str, Any]]:
    """Read the incumbent's saved metrics from its meta.json, if any exists."""
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("metrics")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("risk_target gate: could not read existing meta %s: %s", meta_path, exc)
        return None


def train_risk_targets(
    X_train,
    y_train_vol,
    y_train_dd,
    x_val,
    y_val_vol,
    y_val_dd,
    feature_names,
    *,
    model_dir: Path = MODEL_DIR,
    model_basename: str = MODEL_BASENAME,
    **trainer_kwargs: Any,
) -> Dict[str, Any]:
    """Train a fresh `RiskTargetTrainer` candidate and apply the gate.

    Both heads are saved together in one artifact (`RiskTargetTrainer.save`
    writes a single `.pkl`/`.meta.json` pair), so the gate requires BOTH
    the volatility head (QLIKE) and drawdown head (Brier) to not regress
    vs. the incumbent before writing — refusing if either regresses,
    mirroring `_apply_rf_gate`'s "refuse if EITHER sub-target regresses"
    pattern for a multi-head artifact.

    Returns a dict with `vol_gate`, `drawdown_gate`, `verdict`, `written`,
    `metrics` (the candidate's just-computed val metrics).
    """
    trainer = RiskTargetTrainer(TrainerConfig())
    metrics = trainer.train(
        X_train, y_train_vol, x_val, y_val_vol,
        feature_names=feature_names,
        instrument="pooled_fx_pairs",
        y_train_drawdown=y_train_dd,
        y_val_drawdown=y_val_dd,
        **trainer_kwargs,
    )

    model_path = Path(model_dir) / model_basename
    existing_metrics = _score_existing_metrics(model_path)
    n_val = int(metrics["n_val"])

    vol_gate = _cli_gate_verdict(
        candidate_mae=metrics["val_qlike"],
        existing_mae=(existing_metrics or {}).get("val_qlike"),
        n_test=n_val,
    )
    _log_cli_gate("risk_target_volatility", vol_gate)

    drawdown_gate = _cli_gate_verdict(
        candidate_mae=metrics["val_brier_drawdown"],
        existing_mae=(existing_metrics or {}).get("val_brier_drawdown"),
        n_test=n_val,
    )
    _log_cli_gate("risk_target_drawdown", drawdown_gate)

    both_pass = (
        vol_gate["verdict"] == CLI_GATE_PASSED
        and drawdown_gate["verdict"] == CLI_GATE_PASSED
    )
    result: Dict[str, Any] = {
        "vol_gate": vol_gate,
        "drawdown_gate": drawdown_gate,
        "verdict": CLI_GATE_PASSED if both_pass else CLI_GATE_REFUSED,
        "written": False,
        "metrics": metrics,
    }

    if both_pass:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        trainer.save(str(model_path))
        result["written"] = True
        logger.info(
            "risk_target retrain: gate PASSED both heads — artifact written to %s", model_path,
        )
    else:
        logger.info(
            "risk_target retrain: gate REFUSED (volatility=%s, drawdown=%s) — incumbent untouched",
            vol_gate["verdict"], drawdown_gate["verdict"],
        )

    return result


__all__ = ["train_risk_targets", "MODEL_DIR", "MODEL_BASENAME"]
