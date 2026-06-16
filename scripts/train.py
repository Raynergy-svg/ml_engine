#!/usr/bin/env python3
"""Unified training CLI — one entry point for all model types.

Replaces the fragmented training scripts:
  scripts/train_sota.py
  scripts/train_neural_agents.py
  scripts/train_single_model.py
  scripts/train_full_ensemble.py
  scripts/train_meta_calibrator.py

Usage:
    python scripts/train.py --model-type sota --data trained_data/harvest/*.parquet
    python scripts/train.py --model-type neural --data trained_data/harvest/EUR_USD_M15.parquet --dry-run
    python scripts/train.py --model-type sota --pretrain-only
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.harness import list_trainers, train

logger = logging.getLogger(__name__)


def resolve_data_paths(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for p in patterns:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p))
        elif Path(p).exists():
            paths.append(p)
        else:
            logger.warning("Data path not found: %s", p)
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ML Engine Training Harness")
    parser.add_argument("--model-type", required=True, choices=list_trainers(), help="Registered trainer name")
    parser.add_argument("--data", nargs="+", required=True, help="Glob patterns or explicit paths to OHLCV data")
    parser.add_argument("--save-dir", default=None, help="Checkpoint output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without writing artifacts")
    parser.add_argument("--pretrain-only", action="store_true", help="SOTA: run only phase 1 pretraining")
    parser.add_argument("--finetune-only", action="store_true", help="SOTA: run only phase 2 finetuning")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    data_paths = resolve_data_paths(args.data)
    if not data_paths:
        parser.error("No valid data paths resolved.")

    kwargs: Dict[str, Any] = {}
    if args.pretrain_only:
        kwargs["pretrain_only"] = True
    if args.finetune_only:
        kwargs["finetune_only"] = True
    if args.epochs is not None:
        kwargs["pretrain_epochs"] = args.epochs
        kwargs["finetune_epochs"] = args.epochs
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr

    logger.info("Training model_type=%s on %d path(s)", args.model_type, len(data_paths))
    result = train(
        model_type=args.model_type,
        data_paths=data_paths,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
        **kwargs,
    )

    if result.get("status") == "ok":
        logger.info("Training succeeded: %s", result)
    else:
        logger.error("Training failed: %s", result.get("error"))
        sys.exit(1)


if __name__ == "__main__":
    main()
