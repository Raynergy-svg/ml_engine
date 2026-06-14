#!/usr/bin/env python3
"""
Training script for SOTA Core + Neural Agents (Goals 1 & 2).

Usage:
    python scripts/train_sota_and_neural.py --phase pretrain --data data/raw/*.csv
    python scripts/train_sota_and_neural.py --phase finetune --data data/labeled/*.csv
    python scripts/train_sota_and_neural.py --phase neural --journal trained_data/journal.jsonl

This script is the entry point for taking the scanner from rule-based
heuristics to learned end-to-end models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sota_core import SOTATrainer, TrainerConfig as SOTATrainerConfig
from src.scanner.agents.neural import NeuralAgentTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_sota_and_neural")


def load_trade_history(journal_path: str) -> List[Dict[str, Any]]:
    """Load trade history from journal JSONL."""
    records = []
    with open(journal_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    logger.info(f"Loaded {len(records)} journal records from {journal_path}")
    return records


def phase_pretrain(data_paths: List[str], save_dir: str) -> None:
    """Phase 1: Self-supervised pre-training on raw OHLCV."""
    logger.info("=== Phase 1: Self-supervised pre-training ===")
    cfg = SOTATrainerConfig(batch_size=32, pretrain_epochs=50)
    trainer = SOTATrainer(config=cfg)
    metrics = trainer.pretrain(data_paths, save_dir=save_dir)
    logger.info(f"Pre-training complete. Metrics: {metrics}")


def phase_finetune(data_paths: List[str], save_dir: str) -> None:
    """Phase 2: Supervised fine-tuning on labeled outcomes."""
    logger.info("=== Phase 2: Supervised fine-tuning ===")
    cfg = SOTATrainerConfig(batch_size=32, finetune_epochs=30)
    trainer = SOTATrainer(config=cfg)
    metrics = trainer.finetune(
        data_paths,
        pretrain_weights_dir="trained_data/models/sota_pretrain",
        save_dir=save_dir,
    )
    logger.info(f"Fine-tuning complete. Metrics: {metrics}")


def phase_neural(journal_path: str, save_dir: str) -> None:
    """Phase 3: Train neural agent policies from trade outcomes."""
    logger.info("=== Phase 3: Neural agent policy training ===")
    history = load_trade_history(journal_path)
    if len(history) < 100:
        logger.warning(f"Insufficient trade history ({len(history)} records). Need >= 100.")
        return

    trainer = NeuralAgentTrainer()
    losses = trainer.batch_train(history, epochs=10)
    trainer.save_policies(save_dir)
    logger.info(f"Neural agent training complete. Losses: {losses}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SOTA + Neural Agents")
    parser.add_argument("--phase", choices=["pretrain", "finetune", "neural", "all"], required=True)
    parser.add_argument("--data", nargs="+", help="CSV paths for SOTA training")
    parser.add_argument("--journal", help="Journal JSONL path for neural agent training")
    parser.add_argument("--save-dir", default="trained_data/models")
    args = parser.parse_args()

    # MED-6 FIX: Pre-flight validation for data files
    if args.phase in ("pretrain", "finetune", "all"):
        if not args.data:
            logger.error("--data required for pretrain/finetune phase")
            return 1
        missing = [p for p in args.data if not Path(p).exists()]
        if missing:
            logger.error(f"Missing data files: {missing}")
            return 1
        logger.info(f"Validated {len(args.data)} data file(s)")

    if args.phase in ("neural", "all"):
        if not args.journal:
            logger.error("--journal required for neural phase")
            return 1
        if not Path(args.journal).exists():
            logger.error(f"Journal file not found: {args.journal}")
            return 1
        logger.info(f"Validated journal file: {args.journal}")

    if args.phase in ("pretrain", "all"):
        phase_pretrain(args.data, save_dir=f"{args.save_dir}/sota_pretrain")

    if args.phase in ("finetune", "all"):
        phase_finetune(args.data, save_dir=f"{args.save_dir}/sota_finetuned")

    if args.phase in ("neural", "all"):
        phase_neural(args.journal, save_dir=f"{args.save_dir}/neural_agents")

    logger.info("Training pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
