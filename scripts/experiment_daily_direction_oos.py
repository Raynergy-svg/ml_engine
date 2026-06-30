#!/usr/bin/env python3
"""Daily one-day-ahead FX direction — does the ~58% literature claim survive as SKILL?

Recent peer-reviewed-ish papers claim ~55-60% OOS daily directional accuracy on majors
(arXiv 2409.04471: 58.52% EUR/USD one-day, headline return for 2022; MDPI Forecasting
BFSA: 55-60% at H=1, negative by H=2/3, authors concede it's "exaggerated by feature
selection, causing structural directional imbalance"). This is the one piece of evidence
that challenges this project's intraday ~52% ceiling.

Decisive test: is a daily one-day-ahead direction edge real SKILL, or trend-year/class-
imbalance capture? The discriminator is BALANCED accuracy (mean of up-day and down-day
hit-rates). If raw acc ~58% but balanced acc ~50%, the "edge" is the market drifting one
way that year, not prediction.

Honest, leakage-safe by construction:
  - Features at day t use only data <= t (causal lookbacks).
  - Label = sign(close[t+1]/close[t]-1); the t->t+1 return is never an input.
  - Walk-forward by CALENDAR YEAR: predict year Y using a model trained ONLY on years < Y.
  - Pools all 7 majors (pair-relative features) for sample size; reports per-year raw +
    balanced accuracy, class balance, and a cost-aware long/short directional Sharpe.

Data: FRED daily closes (cached by experiment_pre2014_factor.py). Model: HistGradientBoosting.
Usage: python scripts/experiment_daily_direction_oos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reuse the FRED daily loader from the pre-2014 experiment (cached CSVs).
from scripts.experiment_pre2014_factor import load_fred_close  # noqa: E402

SPREAD_PIPS = {"EUR_USD": 0.8, "USD_JPY": 0.9, "GBP_USD": 1.2, "AUD_USD": 1.0,
               "NZD_USD": 1.4, "USD_CAD": 1.5, "USD_CHF": 1.5}


def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def make_features(close: pd.DataFrame) -> pd.DataFrame:
    """Causal per-(pair,day) features + next-day label. Long format, poolable across pairs."""
    rows = []
    for pair in close.columns:
        px = close[pair].dropna()
        r = np.log(px / px.shift(1))
        f = pd.DataFrame(index=px.index)
        for k in (1, 2, 3, 5, 10, 21):
            f[f"ret_{k}"] = np.log(px / px.shift(k))
        f["mom_sign"] = (np.sign(r.rolling(5).sum()) + np.sign(r.rolling(21).sum())) / 2.0
        f["vol_10"] = r.rolling(10).std()
        f["vol_21"] = r.rolling(21).std()
        f["z_20"] = (px - px.rolling(20).mean()) / px.rolling(20).std()
        f["z_50"] = (px - px.rolling(50).mean()) / px.rolling(50).std()
        # RSI(14)
        up = r.clip(lower=0).rolling(14).mean()
        dn = (-r.clip(upper=0)).rolling(14).mean()
        f["rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-12))
        f["dow"] = px.index.dayofweek
        fwd = np.log(px.shift(-1) / px)          # t -> t+1 return (LABEL ONLY, never a feature)
        f["y"] = (fwd > 0).astype(int)
        f["fwd_ret"] = fwd
        f["pair"] = pair
        f["year"] = px.index.year
        rows.append(f.dropna(subset=[c for c in f.columns if c not in ("fwd_ret",)]))
    return pd.concat(rows).sort_index()


def run() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score

    close = load_fred_close(refresh=False)
    feat = make_features(close)
    feat = feat[feat["fwd_ret"].notna()]
    feat_cols = [c for c in feat.columns if c not in ("y", "fwd_ret", "pair", "year")]

    per_year = []
    for ytest in range(2005, 2027):
        tr = feat[feat["year"] < ytest]
        te = feat[feat["year"] == ytest]
        if len(te) < 200 or len(tr) < 1000:
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_depth=3,
            l2_regularization=1.0, min_samples_leaf=200, random_state=42)
        clf.fit(tr[feat_cols].to_numpy(), tr["y"].to_numpy())
        proba = clf.predict_proba(te[feat_cols].to_numpy())[:, 1]
        pred = (proba >= 0.5).astype(int)
        y = te["y"].to_numpy()
        raw = float((pred == y).mean())
        bal = float(balanced_accuracy_score(y, pred))
        up_rate = float(y.mean())
        # cost-aware: long if pred up else short, earn fwd_ret minus round-trip half-spread
        fwd = te["fwd_ret"].to_numpy()
        pos = np.where(pred == 1, 1.0, -1.0)
        strat = pos * fwd
        net = strat - 0.00007  # ~0.7bp round-trip spread approx on majors (flat, conservative)
        sharpe = float(net.mean() / (net.std() + 1e-12) * np.sqrt(252))
        per_year.append({"year": ytest, "n": int(len(te)), "up_rate": round(up_rate, 3),
                         "raw_acc": round(raw, 4), "bal_acc": round(bal, 4),
                         "net_sharpe": round(sharpe, 2)})

    raws = np.array([r["raw_acc"] for r in per_year])
    bals = np.array([r["bal_acc"] for r in per_year])
    summary = {
        "n_years": len(per_year),
        "mean_raw_acc": round(float(raws.mean()), 4),
        "mean_bal_acc": round(float(bals.mean()), 4),
        "median_bal_acc": round(float(np.median(bals)), 4),
        "bal_acc_above_52_share": round(float((bals > 0.52).mean()), 3),
        "best_year_raw": max(per_year, key=lambda r: r["raw_acc"]),
        "best_year_bal": max(per_year, key=lambda r: r["bal_acc"]),
    }
    return {"per_year": per_year, "summary": summary}


def main() -> int:
    t0 = time.time()
    out = run()
    print("=== DAILY 1-DAY-AHEAD FX DIRECTION — walk-forward OOS, 7 majors pooled ===")
    print(f"{'year':>5s} {'n':>5s} {'up_rate':>7s} {'raw_acc':>7s} {'bal_acc':>7s} {'net_Sh':>7s}")
    for r in out["per_year"]:
        flag = "  <- raw>>bal (imbalance/trend capture)" if (
            r["raw_acc"] - r["bal_acc"] > 0.03) else ""
        print(f"{r['year']:>5d} {r['n']:>5d} {r['up_rate']:>7.3f} {r['raw_acc']:>7.4f} "
              f"{r['bal_acc']:>7.4f} {r['net_sharpe']:>7.2f}{flag}")
    s = out["summary"]
    print(f"\nmean raw acc = {s['mean_raw_acc']:.4f} | mean BALANCED acc = {s['mean_bal_acc']:.4f} "
          f"| median balanced = {s['median_bal_acc']:.4f}")
    print(f"share of years with balanced acc > 0.52: {s['bal_acc_above_52_share']:.0%}")
    print(f"best raw-acc year: {s['best_year_raw']}")
    print(f"best balanced-acc year: {s['best_year_bal']}")
    print("\nREAD: if mean RAW acc is high but mean BALANCED acc ~0.50-0.52, the ~58% daily "
          "literature claim is class-imbalance/trend-year capture, NOT day-by-day skill — "
          "confirming the coin flip. If balanced acc is consistently >0.53 across many years, "
          "there IS a real daily directional edge this project under-tested.")
    out["elapsed_s"] = round(time.time() - t0, 1)
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"daily_direction_oos_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
