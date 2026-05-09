"""Launch W&B sweeps for the 4 master pairs (Phase 5.D).

Path 4 OOS validated 3/3 masters generalize (USD_JPY 60.2%, EUR_USD 54.9%,
EUR_JPY 55.8%); USD_CAD pending. This script kicks off Bayesian
hyperparameter sweeps for the master pairs to push accuracy higher.

Per-pair budget:
  pilot mode   : 10 trials × ~5min = ~50 min  (cheap proof-of-lift on USD_JPY)
  full mode    : 30 trials × ~5min = ~2.5h    (production tuning per pair)

Usage:
  # Pilot on USD_JPY only (recommended first):
  python scripts/launch_master_pair_sweeps.py --pairs USD_JPY --trials 10

  # Full sweep on all 4 masters:
  python scripts/launch_master_pair_sweeps.py \\
      --pairs EUR_USD,EUR_JPY,USD_JPY,USD_CAD --trials 30

The script:
  1. Calls wandb.sweep() per pair → returns sweep_id
  2. Runs wandb.agent() with the trial budget
  3. Each trial executes scripts/run_transformer_direction_sweep.py
  4. Reports the best trial's hyperparameters + val_balanced_accuracy

Halt remains True throughout. Sweeps don't unhalt the bot.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("master_pair_sweeps")

DEFAULT_MASTERS = ("EUR_USD", "EUR_JPY", "USD_JPY", "USD_CAD")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs",
        default=",".join(DEFAULT_MASTERS),
        help="Comma-separated master pair list",
    )
    ap.add_argument("--granularity", default="H1")
    ap.add_argument("--candles", type=int, default=25000)
    ap.add_argument("--lookahead", type=int, default=24)
    ap.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Trials per pair (pilot=10, full=30)",
    )
    ap.add_argument(
        "--project",
        default="buddy-master-tuning",
        help="W&B project name",
    )
    ap.add_argument(
        "--launch-only",
        action="store_true",
        help="Create sweep + print sweep_id; don't run agent (so you can run "
        "the agent in a separate terminal or via wandb agent CLI)",
    )
    args = ap.parse_args()

    import wandb
    from src.training.wandb_sweep_config import get_transformer_direction_sweep_config

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    log.info("Launching sweeps for %d pair(s): %s", len(pairs), pairs)
    log.info("Per-pair: %d trials × ~5min = ~%d min", args.trials, args.trials * 5)

    sweep_summary = []
    for pair in pairs:
        log.info("=" * 60)
        log.info("PAIR: %s", pair)
        log.info("=" * 60)

        cfg = get_transformer_direction_sweep_config(
            pair=pair,
            granularity=args.granularity,
            candles=args.candles,
            lookahead=args.lookahead,
        )
        spec = cfg.to_wandb_spec()
        # Override project from CLI (sweep config defaults set per-spec)
        spec["project"] = args.project

        log.info("Creating W&B sweep for %s...", pair)
        sweep_id = wandb.sweep(spec, project=args.project)
        full_id = f"{args.project}/{sweep_id}"
        log.info("Sweep created: %s", full_id)

        if args.launch_only:
            log.info(
                "[--launch-only] Run agent manually:\n"
                "  wandb agent --count %d %s",
                args.trials, full_id,
            )
            sweep_summary.append({"pair": pair, "sweep_id": full_id, "status": "created"})
            continue

        log.info("Running %d trials...", args.trials)
        try:
            wandb.agent(sweep_id, count=args.trials, project=args.project)
        except Exception as e:
            log.error("Agent failed for %s: %s", pair, e, exc_info=True)
            sweep_summary.append({"pair": pair, "sweep_id": full_id, "status": "agent_error", "error": str(e)})
            continue

        # Fetch the best trial
        api = wandb.Api()
        sweep = api.sweep(full_id)
        best = sweep.best_run(order="-val_balanced_accuracy") if sweep else None
        if best is None:
            sweep_summary.append({"pair": pair, "sweep_id": full_id, "status": "no_best"})
            continue
        best_metric = best.summary.get("val_balanced_accuracy", 0.0)
        best_cfg = {k: v for k, v in best.config.items() if not k.startswith("_")}
        log.info("BEST TRIAL for %s: val_balanced_accuracy=%.4f", pair, best_metric)
        log.info("  Best hyperparameters: %s", best_cfg)
        sweep_summary.append({
            "pair": pair,
            "sweep_id": full_id,
            "status": "done",
            "best_val_balanced_accuracy": float(best_metric),
            "best_config": best_cfg,
        })

    print("\n" + "=" * 60)
    print("MASTER PAIR SWEEP SUMMARY")
    print("=" * 60)
    for entry in sweep_summary:
        if entry.get("status") == "done":
            print(f"  {entry['pair']:10}  best val_bal={entry['best_val_balanced_accuracy']*100:.2f}%  sweep={entry['sweep_id']}")
        else:
            print(f"  {entry['pair']:10}  status={entry.get('status')}  sweep={entry.get('sweep_id', 'N/A')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
