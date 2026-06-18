#!/usr/bin/env python3
"""SMART Gate G3 — does the ETH defensive-timing book pass the ship gate?

The only signal that survived G1 (models) + G2 (close conventions) is ETH long-only
timing. G3 asks the deployability question with the project's MECHANICAL gate:
net Sharpe >= 0.40 AND maxDD <= 25%, after realistic crypto costs, on a continuous
walk-forward out-of-sample equity curve. Buy-and-hold ETH is the benchmark.

Tested under BOTH close conventions (FRED CBETHUSD, Coinbase) and 2 cost levels.
See docs/crypto-edge-smart-plan-2026-06-18.md. Usage: python scripts/experiment_eth_shipgate_g3.py
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

ANN = 365
GATE_SHARPE, GATE_MAXDD = 0.40, 0.25


def fred_eth() -> pd.Series:
    import requests
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CBETHUSD", timeout=20)
    d = pd.read_csv(StringIO(r.text)); d.columns = ["t", "px"]
    d["t"] = pd.to_datetime(d["t"], utc=True); d["px"] = pd.to_numeric(d["px"], errors="coerce")
    return d.dropna().set_index("t")["px"].sort_index()


def cbase_eth() -> pd.Series:
    s = pd.read_csv(REPO_ROOT / "market_data/crypto/ETH-USD_1d.csv", index_col=0, parse_dates=True)["close"]
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s.sort_index()


def maxdd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def shipgate(px: pd.Series, cost: float) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    feat = make_features(pd.DataFrame({"ETH": px}))
    feat = feat[(feat["pair"] == "ETH") & feat["fwd_ret"].notna()].sort_index()
    cols = [c for c in feat.columns if c not in ("y", "fwd_ret", "pair", "year")]
    strat_ret, bh_ret = [], []
    prev = 0.0
    for ytest in sorted(feat["year"].unique()):
        tr = feat[feat["year"] < ytest]; te = feat[feat["year"] == ytest]
        if len(te) < 120 or len(tr) < 500:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                             l2_regularization=1.0, min_samples_leaf=60, random_state=42)
        clf.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        pred = (clf.predict_proba(te[cols].to_numpy())[:, 1] >= 0.5).astype(float)
        fwd = te["fwd_ret"].to_numpy()
        for i in range(len(pred)):
            flip = abs(pred[i] - prev)
            strat_ret.append(pred[i] * fwd[i] - flip * cost)   # long-only timed (cash when flat)
            bh_ret.append(fwd[i])
            prev = pred[i]
    s = np.array(strat_ret); b = np.array(bh_ret)
    def stats(x):
        eq = np.cumprod(1 + x)
        sharpe = float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN))
        cagr = float(eq[-1] ** (ANN / len(x)) - 1)
        return {"net_sharpe": round(sharpe, 3), "max_dd": round(maxdd(eq), 3),
                "cagr": round(cagr, 4), "n_days": len(x)}
    return {"timed_long": stats(s), "buy_hold": stats(b)}


def main() -> int:
    t0 = time.time()
    out = {}
    for conv, px in (("FRED-close", fred_eth()), ("Coinbase-close", cbase_eth())):
        for cost_bps, cost in ((10, 0.0010), (30, 0.0030)):
            r = shipgate(px, cost)
            t = r["timed_long"]
            passes = bool(t["net_sharpe"] >= GATE_SHARPE and t["max_dd"] <= GATE_MAXDD)
            out[f"{conv}/{cost_bps}bps"] = {**r, "gate_pass": passes}
    print("=== SMART G3 — ETH defensive-timing ship gate (net Sharpe>=0.40 AND maxDD<=25%) ===")
    print(f"{'scenario':22s} {'timed_Sharpe':>12s} {'timed_maxDD':>11s} {'B&H_Sharpe':>10s} "
          f"{'B&H_maxDD':>9s}  GATE")
    for k, v in out.items():
        t, b = v["timed_long"], v["buy_hold"]
        print(f"{k:22s} {t['net_sharpe']:>12.3f} {t['max_dd']:>11.1%} {b['net_sharpe']:>10.3f} "
              f"{b['max_dd']:>9.1%}  {'PASS' if v['gate_pass'] else 'FAIL'}")
    any_pass = any(v["gate_pass"] for v in out.values())
    print(f"\nG3 {'PASS (some scenario clears the gate)' if any_pass else 'FAIL — no scenario clears net Sharpe>=0.40 AND maxDD<=25%'}")
    print("READ: timed-long cushions ETH drawdowns but crypto maxDD is structurally huge; "
          "the gate's 25% maxDD rail is the binding constraint. If G3 fails, the surviving "
          "signal is real but not deployable under the current risk mandate (same as carry).")
    payload = {"scenarios": out, "g3_any_pass": any_pass,
               "gate": {"min_net_sharpe": GATE_SHARPE, "max_dd": GATE_MAXDD},
               "elapsed_s": round(time.time() - t0, 1)}
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"eth_shipgate_g3_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
