#!/usr/bin/env python3
"""Is the ~52% wall FX-specific? Run the SAME daily-direction method on other assets.

The escapability test for "get me out of the coin flip": take the identical
HistGradientBoosting daily walk-forward that scored 50.3% balanced on FX majors and
point it at instruments with documented inefficiency (crypto: strong momentum /
autocorrelation, retail-dominated) plus an equity index, with EUR/USD as control.

If crypto lands materially >52% balanced across many years -> the wall is FX-specific
and the escape is real (change the instrument, keep the method). If crypto also
coin-flips -> directional prediction is broadly dead.

Free FRED daily data: CBBTCUSD (BTC 2014+), CBETHUSD (ETH 2016+), NASDAQCOM (1971+),
SP500 (2016+), DEXUSEU (EUR/USD control). Same leakage-safe causal features + label.

Usage: python scripts/experiment_cross_asset_direction.py
"""
from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiment_daily_direction_oos import make_features  # noqa: E402

INSTRUMENTS = {  # FRED id -> (name, ann_days, test_start)
    "CBBTCUSD": ("BTC_USD", 365, 2018),
    "CBETHUSD": ("ETH_USD", 365, 2019),
    "NASDAQCOM": ("NASDAQ", 252, 2005),
    "SP500": ("SP500", 252, 2019),
    "DEXUSEU": ("EUR_USD_ctrl", 252, 2005),
}


def fetch(fid: str) -> pd.Series:
    import requests
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}", timeout=20)
    df = pd.read_csv(StringIO(r.text)); df.columns = ["date", "px"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    return df.dropna().set_index("date")["px"].sort_index()


def run() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score

    results = {}
    for fid, (name, ann, t0) in INSTRUMENTS.items():
        px = fetch(fid)
        close = pd.DataFrame({name: px})
        feat = make_features(close)
        feat = feat[(feat["pair"] == name) & feat["fwd_ret"].notna()]
        fcols = [c for c in feat.columns if c not in ("y", "fwd_ret", "pair", "year")]
        per_year = []
        for ytest in range(t0, 2026):
            tr = feat[feat["year"] < ytest]; te = feat[feat["year"] == ytest]
            if len(te) < 120 or len(tr) < 500:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_depth=3,
                l2_regularization=1.0, min_samples_leaf=60, random_state=42)
            clf.fit(tr[fcols].to_numpy(), tr["y"].to_numpy())
            proba = clf.predict_proba(te[fcols].to_numpy())[:, 1]
            pred = (proba >= 0.5).astype(int)
            y = te["y"].to_numpy(); fwd = te["fwd_ret"].to_numpy()
            pos = np.where(pred == 1, 1.0, -1.0)
            cost = 0.0010 if name in ("BTC_USD", "ETH_USD") else 0.0002  # per-flip ~10bp crypto/2bp eq
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            ls_net = pos * fwd - flips * cost
            longonly = np.where(pred == 1, 1.0, 0.0) * fwd
            def sh(x):
                return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann))
            per_year.append({
                "year": ytest, "n": int(len(te)),
                "bal_acc": round(float(balanced_accuracy_score(y, pred)), 4),
                "model_ls_net": round(sh(ls_net), 2),       # long/short, after cost
                "model_long_net": round(sh(longonly), 2),    # long-when-up only
                "buy_hold": round(sh(fwd), 2)})              # always long (beta)
        if per_year:
            bals = np.array([r["bal_acc"] for r in per_year])
            results[name] = {
                "n_years": len(per_year),
                "mean_bal_acc": round(float(bals.mean()), 4),
                "median_bal_acc": round(float(np.median(bals)), 4),
                "share_above_52": round(float((bals > 0.52).mean()), 3),
                "share_above_54": round(float((bals > 0.54).mean()), 3),
                "mean_model_ls_net": round(float(np.mean([r["model_ls_net"] for r in per_year])), 2),
                "mean_model_long_net": round(float(np.mean([r["model_long_net"] for r in per_year])), 2),
                "mean_buy_hold": round(float(np.mean([r["buy_hold"] for r in per_year])), 2),
                "per_year": per_year,
            }
    return results


def main() -> int:
    t0 = time.time()
    res = run()
    print("=== CROSS-ASSET DAILY DIRECTION — same method as FX (50.3% balanced) ===")
    print(f"{'instrument':14s} {'yrs':>4s} {'bal_acc':>8s} {'model_LS':>9s} "
          f"{'model_Lo':>9s} {'buy_hold':>9s}  alpha_vs_B&H")
    for name, r in res.items():
        alpha = r["mean_model_long_net"] - r["mean_buy_hold"]
        print(f"{name:14s} {r['n_years']:>4d} {r['mean_bal_acc']:>8.4f} "
              f"{r['mean_model_ls_net']:>9.2f} {r['mean_model_long_net']:>9.2f} "
              f"{r['mean_buy_hold']:>9.2f}  {alpha:>+6.2f}")
    print("\nREAD: model_LS = long/short after cost; model_Lo = long-only (long when model says up); "
          "buy_hold = always long (pure beta). alpha_vs_B&H = does the model's timing BEAT just "
          "holding? If alpha <= 0, the positive Sharpe is drift/beta capture, NOT directional skill "
          "— so the 'escape' is a risk-premium (buy-and-hold) play, not a prediction edge.")
    out = {"results": res, "elapsed_s": round(time.time() - t0, 1)}
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"cross_asset_direction_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
