"""Apply FX confidence tuning to a config file and write tuned config.

Usage:
    PYTHONPATH=. python3 scripts/apply_confidence_tuning.py --in config_tuned.yaml --out config_tuned.yaml
  or
  PYTHONPATH=. python3 scripts/apply_confidence_tuning.py --set "base_confidence=0.75" "spread_penalty_max=0.0"

This is a small convenience to persist and test calibration results produced
by `scripts/calibrate_fx_confidence.py` without editing `config_tuned.yaml` directly.
"""

import argparse
import yaml
from pathlib import Path

DEFAULTS = {
    "base_confidence": 0.75,
    "spread_soft_start_ratio": 0.5,
    "spread_penalty_max": 0.0,
    "atr_soft_start_pct": 0.0005,
    "atr_penalty_max": 0.0,
}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def parse_set_items(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Invalid set argument: {it}. Expect key=value")
        k, v = it.split("=", 1)
        try:
            # try numeric
            if "." in v:
                val = float(v)
            else:
                val = int(v)
        except Exception:
            val = v
        out[k.strip()] = val
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default="config_tuned.yaml")
    p.add_argument("--out", dest="outfile", default="config_tuned.yaml")
    p.add_argument("--set", dest="set", nargs="*", default=[], help="Override specific tuning params key=value")
    args = p.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile)

    if not in_path.exists():
        raise SystemExit(f"Input config not found: {in_path}")

    cfg = load_yaml(in_path)

    tuning = DEFAULTS.copy()
    if args.set:
        overrides = parse_set_items(args.set)
        tuning.update(overrides)

    cfg.setdefault("fx", {})
    cfg["fx"].setdefault("confidence_model", {})
    cfg["fx"]["confidence_model"].update(tuning)

    write_yaml(cfg, out_path)
    print(f"Wrote tuned config to: {out_path}")
    print("Applied tuning:")
    for k, v in tuning.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
# — Raynergy-svg —
