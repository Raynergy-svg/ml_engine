#!/usr/bin/env python3
"""Quick experiment: test hypotheses to beat 51% direction accuracy."""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import logging
import json
from pathlib import Path
import numpy as np

from sota_core.raw_sequence_model import RawSequenceModel, ModelConfig
from sota_core.trainer import SOTATrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATHS = [str(p) for p in Path("trained_data/harvest").glob("*_M15.parquet")]
if not DATA_PATHS:
    raise RuntimeError("No harvest data found")

logger.info(f"Found {len(DATA_PATHS)} data files")

EXPERIMENTS = [
    {
        "name": "baseline_random_2class",
        "model_cfg": {},
        "trainer_cfg": {"use_temporal_split": False, "direction_threshold": 0.0},
        "epochs": 15,
    },
    {
        "name": "temporal_2class",
        "model_cfg": {},
        "trainer_cfg": {"use_temporal_split": True, "direction_threshold": 0.0},
        "epochs": 15,
    },
    {
        "name": "temporal_3class_thresh0005",
        "model_cfg": {"direction_classes": 3},
        "trainer_cfg": {"use_temporal_split": True, "direction_threshold": 0.0005, "finetune_epochs": 15},
        "epochs": 15,
    },
    {
        "name": "temporal_3class_thresh0010",
        "model_cfg": {"direction_classes": 3},
        "trainer_cfg": {"use_temporal_split": True, "direction_threshold": 0.0010, "finetune_epochs": 15},
        "epochs": 15,
    },
    {
        "name": "itransformer_temporal_3class",
        "model_cfg": {"direction_classes": 3, "encoder_type": "itransformer"},
        "trainer_cfg": {"use_temporal_split": True, "direction_threshold": 0.0010, "finetune_epochs": 15},
        "epochs": 15,
    },
]

results = []
for exp in EXPERIMENTS:
    logger.info(f"\n{'='*60}\nRunning: {exp['name']}\n{'='*60}")
    try:
        model_cfg = ModelConfig(**exp["model_cfg"])
        model = RawSequenceModel(model_cfg)
        trainer_cfg = TrainerConfig(**exp["trainer_cfg"])
        trainer = SOTATrainer(model=model, config=trainer_cfg)

        # Skip pretrain — load existing weights if available
        metrics = trainer.finetune(
            data_paths=DATA_PATHS,
            pretrain_weights_dir="trained_data/models/sota_pretrain",
            save_dir=f"trained_data/models/experiments/{exp['name']}",
        )
        results.append({
            "name": exp["name"],
            "status": "OK",
            "val_direction_loss": metrics.get("val_direction_loss"),
            "val_direction_accuracy": metrics.get("val_direction_accuracy"),
            "val_regime_loss": metrics.get("val_regime_loss"),
            "val_regime_accuracy": metrics.get("val_regime_accuracy"),
        })
        logger.info(f"Results: {metrics}")
    except Exception as e:
        logger.error(f"Experiment {exp['name']} failed: {e}")
        results.append({"name": exp["name"], "status": "FAILED", "error": str(e)})

# Summary
logger.info(f"\n{'='*60}\nEXPERIMENT SUMMARY\n{'='*60}")
for r in results:
    logger.info(json.dumps(r, indent=2))

# Write report
Path("trained_data/models/experiments").mkdir(parents=True, exist_ok=True)
with open("trained_data/models/experiments/report.json", "w") as f:
    json.dump(results, f, indent=2)
