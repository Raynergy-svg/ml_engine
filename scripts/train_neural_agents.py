#!/usr/bin/env python3
"""Neural Agent Training Script — bridge trade outcomes to online policy updates.

Collects verdicts + outcomes from the scanner, pushes them into each neural
agent's replay buffer, and triggers lightweight gradient updates.

Usage:
    python scripts/train_neural_agents.py --collect-only
    python scripts/train_neural_agents.py --update-all
    python scripts/train_neural_agents.py --shadow-mode EUR_USD
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scanner.agents.neural import NeuralAgentTrainer, NeuralAgentConfig
from scanner.agents.neural.policies import (
    TrendPolicy,
    MomentumPolicy,
    MeanReversionPolicy,
    VolatilityPolicy,
)

logger = logging.getLogger(__name__)

SAVE_DIR = _PROJECT_ROOT / "trained_data" / "models" / "neural_agents"
MIN_SAMPLES = 50


def load_outcome_log(log_path: Path) -> List[Dict[str, Any]]:
    """Read trade outcome records produced by the scanner."""
    records: List[Dict[str, Any]] = []
    if not log_path.exists():
        logger.warning(f"No outcome log at {log_path}")
        return records
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning(f"Could not read {log_path}: {exc}")
    return records


def push_outcomes_to_replay(
    trainer: NeuralAgentTrainer,
    records: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Push each trade outcome into the appropriate agent replay buffers.

    Each record is expected to contain:
      {
        "pair": str,
        "direction": "LONG" | "SHORT",
        "outcome": "WIN" | "LOSS",
        "pnl_pips": float,
        "regime": "LOW" | "NORMAL" | "HIGH" | "EXTREME",
        "agent_verdicts": [
          {"name": "neural_trend", "score": 0.72, "passed": true, ...},
          ...
        ],
        "features": {  # optional pre-computed features
          "neural_trend": [...],
          "neural_momentum": [...],
        }
      }
    """
    pushed: Dict[str, int] = {}
    for rec in records:
        pair = rec.get("pair", "UNKNOWN")
        outcome = rec.get("outcome", "BREAKEVEN")
        won = outcome == "WIN"
        for vd in rec.get("agent_verdicts", []):
            name = vd.get("name", "")
            if name not in trainer.agents:
                continue
            agent = trainer.agents[name]
            # Reward shaping: +1 for correct vote, -1 for incorrect vote
            # (voted FOR and won, or voted AGAINST and lost)
            voted_for = vd.get("passed", False)
            target_score = 1.0 if (voted_for == won) else 0.0
            # Extract or synthesize feature vector
            features = rec.get("features", {}).get(name)
            if features is None:
                # Fallback: simple feature vector derived from verdict metadata
                features = np.array([
                    vd.get("score", 0.5),
                    1.0 if voted_for else 0.0,
                    1.0 if won else 0.0,
                    rec.get("pnl_pips", 0.0) / 100.0,
                ], dtype=np.float32)
            agent.push_experience(features, target_score)
            pushed[name] = pushed.get(name, 0) + 1
    for name, count in pushed.items():
        logger.info(f"Pushed {count} experiences to {name}")
    return pushed


def update_policies(trainer: NeuralAgentTrainer, min_samples: int = MIN_SAMPLES) -> Dict[str, Any]:
    """Run online gradient updates for every agent with enough data."""
    results: Dict[str, Any] = {}
    for name, agent in trainer.agents.items():
        n_ready = agent.replay_size()
        if n_ready < min_samples:
            logger.info(f"{name}: {n_ready} samples (need {min_samples}); skipping update")
            results[name] = {"status": "insufficient_data", "samples": n_ready}
            continue
        try:
            loss_before, loss_after = agent.update_from_replay()
            results[name] = {
                "status": "updated",
                "samples": n_ready,
                "loss_before": round(loss_before, 6),
                "loss_after": round(loss_after, 6),
                "improvement": round(loss_before - loss_after, 6),
            }
            logger.info(
                f"{name} updated: loss {loss_before:.4f} -> {loss_after:.4f} "
                f"(n={n_ready})"
            )
        except Exception as exc:
            logger.warning(f"{name} update failed: {exc}")
            results[name] = {"status": "error", "error": str(exc)}
    return results


def evaluate_policies(
    trainer: NeuralAgentTrainer,
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute per-agent validation AUC on held-out records."""
    from sklearn.metrics import roc_auc_score
    aucs: Dict[str, float] = {}
    for name, agent in trainer.agents.items():
        y_true = []
        y_score = []
        for rec in records:
            for vd in rec.get("agent_verdicts", []):
                if vd.get("name") == name:
                    won = rec.get("outcome") == "WIN"
                    y_true.append(1.0 if won else 0.0)
                    y_score.append(vd.get("score", 0.5))
        if len(y_true) >= 10 and len(set(y_true)) > 1:
            try:
                aucs[name] = roc_auc_score(y_true, y_score)
            except Exception:
                aucs[name] = 0.5
        else:
            aucs[name] = 0.5
    return aucs


def bootstrap_shadow(
    trainer: NeuralAgentTrainer,
    pair: str,
    n_episodes: int = 100,
) -> None:
    """Generate synthetic outcomes to bootstrap a cold-start agent.

    Shadow mode: produce verdicts but don't vote until enough samples.
    This is a helper for unit testing / bootstrap only.
    """
    import random
    logger.info(f"Bootstrapping shadow mode for {pair} with {n_episodes} synthetic episodes")
    for i in range(n_episodes):
        features = np.random.randn(20).astype(np.float32)
        target = random.choice([0.0, 1.0])
        for agent in trainer.agents.values():
            agent.push_experience(features, target)
    update_policies(trainer, min_samples=10)
    trainer.save_policies()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural scanner agents")
    parser.add_argument("--collect-only", action="store_true", help="Collect outcomes, skip updates")
    parser.add_argument("--update-all", action="store_true", help="Update all agents from replay")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--outcome-log", type=Path, default=_PROJECT_ROOT / "trained_data" / "logs" / "trade_outcomes.jsonl")
    parser.add_argument("--shadow-mode", metavar="PAIR", default=None, help="Bootstrap synthetic data for a pair")
    parser.add_argument("--save-dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--evaluate", action="store_true", help="Compute per-agent AUC on outcome log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = NeuralAgentConfig()
    trainer = NeuralAgentTrainer()
    trainer.cfg.save_dir = str(args.save_dir)
    trainer.load_policies()

    if args.shadow_mode:
        bootstrap_shadow(trainer, args.shadow_mode)
        sys.exit(0)

    # Load outcomes
    records = load_outcome_log(args.outcome_log)
    if not records:
        logger.error(f"No outcome records at {args.outcome_log}")
        logger.error("Neural agents need trade outcomes to learn. Run scanner trades first.")
        sys.exit(1)
    logger.info(f"Loaded {len(records)} outcome records")

    # Push into replay buffers
    pushed = push_outcomes_to_replay(trainer, records)
    if not pushed:
        logger.warning("No neural agent verdicts found in outcome log")

    if args.evaluate:
        aucs = evaluate_policies(trainer, records)
        for name, auc in aucs.items():
            print(f"{name}: AUC = {auc:.3f}")
        sys.exit(0)

    if args.collect_only:
        logger.info("Collect-only mode; not running gradient updates")
        sys.exit(0)

    # Gradient updates
    if args.update_all:
        results = update_policies(trainer, min_samples=args.min_samples)
        for name, info in results.items():
            print(f"{name}: {info}")
        trainer.save_policies()
    else:
        logger.info("Pass --update-all to run gradient updates")


if __name__ == "__main__":
    main()
