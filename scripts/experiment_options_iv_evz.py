#!/usr/bin/env python3
"""Options-implied vol lever — reduced test on the ONLY free FX IV series (EVZ).

The lone peer-reviewed survivor of the edge search (IJF, options-IV term-structure
factor) is CROSS-SECTIONAL across currencies+tenors. Free FX implied-vol data is
exactly ONE series — EVZCLS (CBOE EuroCurrency Volatility Index = EUR/USD 1-month
ATM vol, 2007-2025, now discontinued). Yen/Pound vol indices are discontinued (404).
So the published factor is UNBUILDABLE at retail data scale — itself a confirmation
of the access-wall thesis.

This runs the reduced test the data DOES support: does EVZ (level, change, and the
variance-risk-premium proxy IV - realized-vol) add predictive lift for EUR/USD
next-day direction beyond price-only features? If balanced accuracy with EVZ ~=
price-only ~= 50%, the accessible slice of the options lever is dead too.

Usage: python scripts/experiment_options_iv_evz.py
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

from scripts.experiment_pre2014_factor import load_fred_close, fetch_fred_daily  # noqa: E402
from scripts.experiment_daily_direction_oos import make_features  # noqa: E402


def fetch_evz() -> pd.Series:
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=EVZCLS", timeout=20)
    df = pd.read_csv(StringIO(r.text)); df.columns = ["date", "evz"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["evz"] = pd.to_numeric(df["evz"], errors="coerce")
    return df.dropna().set_index("date")["evz"].sort_index()


def run() -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score

    close = load_fred_close(refresh=False)[["EUR_USD"]]
    feat = make_features(close)
    feat = feat[(feat["pair"] == "EUR_USD") & feat["fwd_ret"].notna()].copy()

    evz = fetch_evz()
    px = close["EUR_USD"]
    r = np.log(px / px.shift(1))
    rv = r.rolling(21).std() * np.sqrt(252) * 100.0   # annualized realized vol, %
    evz_d = evz.reindex(px.index).ffill()
    evz_df = pd.DataFrame(index=px.index)
    evz_df["evz_z60"] = (evz_d - evz_d.rolling(60).mean()) / evz_d.rolling(60).std()
    evz_df["evz_chg5"] = evz_d - evz_d.shift(5)
    evz_df["vrp"] = evz_d - rv                          # variance-risk-premium proxy (IV - RV)
    feat = feat.join(evz_df, how="left")
    feat = feat[feat["evz_z60"].notna()]                # restrict to EVZ-covered span

    price_cols = [c for c in feat.columns
                  if c not in ("y", "fwd_ret", "pair", "year", "evz_z60", "evz_chg5", "vrp")]
    evz_cols = price_cols + ["evz_z60", "evz_chg5", "vrp"]

    def walk(cols):
        rows = []
        for ytest in range(2009, 2025):
            tr = feat[feat["year"] < ytest]; te = feat[feat["year"] == ytest]
            if len(te) < 150 or len(tr) < 500:
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.03, max_depth=3,
                l2_regularization=1.0, min_samples_leaf=80, random_state=42)
            clf.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
            pred = (clf.predict_proba(te[cols].to_numpy())[:, 1] >= 0.5).astype(int)
            rows.append({"year": ytest, "n": int(len(te)),
                         "bal_acc": round(float(balanced_accuracy_score(te["y"], pred)), 4)})
        return rows

    price_only = walk(price_cols)
    with_evz = walk(evz_cols)
    pa = np.array([x["bal_acc"] for x in price_only])
    ea = np.array([x["bal_acc"] for x in with_evz])
    return {
        "evz_span": f"{evz.index[0].date()}->{evz.index[-1].date()}",
        "n_years": len(price_only),
        "price_only_mean_bal_acc": round(float(pa.mean()), 4),
        "with_evz_mean_bal_acc": round(float(ea.mean()), 4),
        "evz_lift_pp": round(float((ea.mean() - pa.mean()) * 100), 2),
        "per_year": [{"year": p["year"], "price_only": p["bal_acc"],
                      "with_evz": e["bal_acc"]} for p, e in zip(price_only, with_evz)],
    }


def main() -> int:
    t0 = time.time()
    out = run()
    print("=== OPTIONS-IV (EVZ) REDUCED TEST — EUR/USD next-day direction ===")
    print(f"EVZ span: {out['evz_span']} | years tested: {out['n_years']}")
    print(f"{'year':>5s} {'price_only':>11s} {'with_EVZ':>9s}")
    for r in out["per_year"]:
        print(f"{r['year']:>5d} {r['price_only']:>11.4f} {r['with_evz']:>9.4f}")
    print(f"\nprice-only mean balanced acc = {out['price_only_mean_bal_acc']:.4f}")
    print(f"with-EVZ   mean balanced acc = {out['with_evz_mean_bal_acc']:.4f}")
    print(f"EVZ lift = {out['evz_lift_pp']:+.2f} pp")
    print("\nREAD: EVZ is the ONLY free FX implied-vol series (EUR/USD 1m, discontinued 2025). "
          "The published cross-sectional options-IV term-structure factor is UNBUILDABLE free "
          "(needs multi-currency multi-tenor surfaces). If EVZ lift ~0, even the accessible "
          "slice of the options lever adds no directional signal -> lever closed end-to-end.")
    out["elapsed_s"] = round(time.time() - t0, 1)
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"options_iv_evz_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
