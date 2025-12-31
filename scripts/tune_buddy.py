#!/usr/bin/env python3
"""Buddy sweet-spot tuner.

Runs a small hyperparameter sweep over (lr, batch_size, patience) using the
existing `main.train_buddy()` training routine and records the best validation
epoch/metric from each run.

Outputs:
- Per-run artifacts in `trained_data/models/` as `buddy_tf_<run_tag>.keras` and
  `buddy_tf_<run_tag>.meta.json`
- A summary JSON at `trained_data/models/buddy_tuning_summary.json`

Example:
  python3 scripts/tune_buddy.py --epochs 200 \
    --lrs 0.001 0.0003 0.0001 \
    --batch-sizes 32 64 \
    --patiences 10 20 \
    --topk 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    lr: float
    batch_size: int
    patience: int
    seed: int


def _safe_tag(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_summary(model_dir: Path, results: list[dict[str, Any]]) -> None:
    def score(m: dict[str, Any]) -> tuple[float, float]:
        # Primary: best validation directional accuracy.
        # Tie-break: post-fit average confidence.
        b = m.get("best_val_direction_accuracy")
        if b is None:
            b = m.get("val_direction_accuracy")
        c = m.get("val_avg_confidence") or 0.0
        return (float(b or 0.0), float(c))

    ranked = sorted(results, key=score, reverse=True)
    summary_path = model_dir / "buddy_tuning_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "runs": results,
                "ranked_run_tags": [m.get("run_tag") for m in ranked],
            },
            indent=2,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Tune Buddy training sweet spot")
    p.add_argument("--config", default="./config_tuned.yaml")
    p.add_argument("--csv", default=None, help="Optional CSV override")
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)

    p.add_argument("--lrs", type=float, nargs="+", default=[0.001, 0.0003, 0.0001])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    p.add_argument("--patiences", type=int, nargs="+", default=[10, 20])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--run-prefix", default="tune")
    p.add_argument("--topk", type=int, default=5)

    args = p.parse_args()

    # Ensure repo root is importable (so `import main` works when executed as a script).
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import lazily so argparse help is fast.
    from main import train_buddy  # type: ignore

    model_dir = Path("trained_data") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    run_prefix = _safe_tag(str(args.run_prefix))

    run_cfgs: list[RunConfig] = []
    for lr in args.lrs:
        for bs in args.batch_sizes:
            for pat in args.patiences:
                run_cfgs.append(RunConfig(lr=float(lr), batch_size=int(bs), patience=int(pat), seed=int(args.seed)))

    results: list[dict[str, Any]] = []

    try:
        for i, rc in enumerate(run_cfgs, start=1):
            tag = _safe_tag(f"{run_prefix}_lr{rc.lr:g}_bs{rc.batch_size}_pat{rc.patience}_seed{rc.seed}")
            print(f"\n=== [{i}/{len(run_cfgs)}] {tag} ===")

            meta_path = model_dir / f"buddy_tf_{tag}.meta.json"
            if meta_path.exists():
                meta = _read_json(meta_path)
                results.append(meta)
                print("(skip) Found existing meta, not retraining")
            else:
                train_buddy(
                    args.config,
                    args.csv,
                    seq_len=int(args.seq_len),
                    epochs=int(args.epochs),
                    batch_size=int(rc.batch_size),
                    lr=float(rc.lr),
                    patience=int(rc.patience),
                    seed=int(rc.seed),
                    run_tag=tag,
                    all_features=True,
                )
                meta = _read_json(meta_path)
                results.append(meta)

            best = meta.get("best_val_direction_accuracy")
            best_epoch = meta.get("best_epoch")
            post = meta.get("val_direction_accuracy")
            print(
                "Result: "
                f"best_val_dir_acc={None if best is None else round(float(best) * 100.0, 2)}% "
                f"@epoch={best_epoch} | "
                f"postfit_val_dir_acc={round(float(post) * 100.0, 2)}%"
            )

            _write_summary(model_dir, results)

    except KeyboardInterrupt:
        print("\nInterrupted; writing partial summary.")

    finally:
        _write_summary(model_dir, results)

    ranked_run_tags = _read_json(model_dir / "buddy_tuning_summary.json").get("ranked_run_tags", [])

    print("\n=== Top configs ===")
    for j, tag in enumerate(ranked_run_tags[: int(args.topk)], start=1):
        m = next((r for r in results if r.get("run_tag") == tag), {})
        print(
            f"#{j} tag={m.get('run_tag')} "
            f"lr={m.get('lr')} bs={m.get('batch_size')} pat={m.get('early_stopping_patience')} "
            f"best={m.get('best_val_direction_accuracy')}@{m.get('best_epoch')} "
            f"post={m.get('val_direction_accuracy')} conf={m.get('val_avg_confidence')}"
        )

    print(f"\nWrote: {model_dir / 'buddy_tuning_summary.json'}")


if __name__ == "__main__":
    main()
