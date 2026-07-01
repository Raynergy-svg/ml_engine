#!/usr/bin/env python3
"""SMART Gate G1 — is the ETH timing alpha robust, or one lucky config?

Re-runs the daily-direction walk-forward on ETH + BTC across multiple model configs
(shallow GBM, deep GBM, logistic) x 2 feature sets, counting how often the long-only
timed model beats buy-and-hold per year. G1 PASSES if ETH long-only beats buy-and-hold
in >=5/7 years across >=3 of the config x feature-set combinations.

See docs/crypto-edge-smart-plan-2026-06-18.md. Free FRED data (CBBTCUSD, CBETHUSD).
Usage: python scripts/experiment_crypto_robustness.py
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

COINS = {"CBBTCUSD": ("BTC_USD", 2018), "CBETHUSD": ("ETH_USD", 2019)}
ANN = 365
COST = 0.0010  # ~10bp per flip (crypto)
REDUCED = ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_21", "mom_sign", "vol_10", "vol_21"]


def fetch(fid: str) -> pd.Series:
    import requests
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}", timeout=20)
    df = pd.read_csv(StringIO(r.text)); df.columns = ["date", "px"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["px"] = pd.to_numeric(df["px"], errors="coerce")
    return df.dropna().set_index("date")["px"].sort_index()


def model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "gbm_shallow":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                              l2_regularization=1.0, min_samples_leaf=60, random_state=42)
    if kind == "gbm_deep":
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6,
                                              l2_regularization=0.5, min_samples_leaf=30, random_state=7)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5))


def sh(x):
    return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN))


def run() -> dict:
    out = {}
    for fid, (name, t0) in COINS.items():
        feat = make_features(pd.DataFrame({name: fetch(fid)}))
        feat = feat[(feat["pair"] == name) & feat["fwd_ret"].notna()]
        full = [c for c in feat.columns if c not in ("y", "fwd_ret", "pair", "year")]
        combos = {}
        for mk in ("gbm_shallow", "gbm_deep", "logistic"):
            for fs_name, cols in (("full", full), ("reduced", REDUCED)):
                wins = tot = 0; alphas = []
                for ytest in range(t0, 2026):
                    tr = feat[feat["year"] < ytest]; te = feat[feat["year"] == ytest]
                    if len(te) < 120 or len(tr) < 500:
                        continue
                    clf = model(mk); clf.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
                    pred = (clf.predict_proba(te[cols].to_numpy())[:, 1] >= 0.5).astype(int)
                    fwd = te["fwd_ret"].to_numpy()
                    a = sh(np.where(pred == 1, 1.0, 0.0) * fwd) - sh(fwd)
                    alphas.append(a); wins += a > 0; tot += 1
                combos[f"{mk}/{fs_name}"] = {"years_win": wins, "years": tot,
                                             "mean_alpha": round(float(np.mean(alphas)), 3)}
        out[name] = combos
    return out


def main() -> int:
    t0 = time.time()
    res = run()
    print("=== SMART G1 — crypto timing-alpha robustness (long-only vs buy-&-hold) ===")
    eth_pass = 0
    for name, combos in res.items():
        print(f"\n{name}:")
        print(f"  {'config/features':22s} {'years_beat_B&H':>15s} {'mean_alpha':>11s}")
        for k, v in combos.items():
            mark = " PASS" if (name == "ETH_USD" and v["years_win"] * 7 >= 5 * v["years"]) else ""
            if name == "ETH_USD" and v["years_win"] * 7 >= 5 * v["years"]:
                eth_pass += 1
            print(f"  {k:22s} {str(v['years_win'])+'/'+str(v['years']):>15s} "
                  f"{v['mean_alpha']:>+11.3f}{mark}")
    verdict = ("G1 PASS — ETH timing alpha is robust across configs" if eth_pass >= 3
               else "G1 FAIL — ETH alpha does not survive config variation (likely a fluke)")
    print(f"\n  ETH combos meeting >=5/7-years bar: {eth_pass}/6")
    print(f"  VERDICT: {verdict}")
    out = {"results": res, "eth_combos_passing": eth_pass, "g1_pass": eth_pass >= 3,
           "elapsed_s": round(time.time() - t0, 1)}
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"crypto_robustness_g1_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
