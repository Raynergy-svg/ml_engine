#!/usr/bin/env python3
"""SMART Gate G2 — does the ETH timing alpha generalize across a crypto universe?

Fetches multi-year daily closes for a >=8-coin universe via the Coinbase Exchange
public API (paginated, 300 candles/request, no key) and runs the same long-only-vs-
buy-and-hold timing test (canonical config: shallow GBM, full features).

G2 PASS = mean timing alpha across coins-with-sufficient-history >= +0.20 AND positive
on a MAJORITY of coins. Disambiguates "ETH has real timing structure" from "ETH's
7-year path was lucky." See docs/crypto-edge-smart-plan-2026-06-18.md.

Usage: python scripts/experiment_crypto_universe_g2.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.experiment_daily_direction_oos import make_features  # noqa: E402

PRODUCTS = ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD", "XRP-USD", "ADA-USD",
            "SOL-USD", "DOGE-USD", "LINK-USD", "XLM-USD"]
FLOOR = datetime(2017, 1, 1, tzinfo=timezone.utc)
ANN = 365
COST = 0.0010
CACHE = REPO_ROOT / "market_data" / "crypto"


def coinbase_daily(product: str) -> pd.Series:
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{product}_1d.csv"
    if cf.exists():
        s = pd.read_csv(cf, index_col=0, parse_dates=True)["close"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s.rename(product)
    rows = {}
    end = datetime.now(timezone.utc)
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    while end > FLOOR:
        start = end - timedelta(days=290)
        try:
            r = requests.get(url, params={"granularity": 86400,
                             "start": start.isoformat(), "end": end.isoformat()},
                             timeout=25, headers={"User-Agent": "research"})
            if r.status_code != 200:
                time.sleep(1.0)
                end = start - timedelta(days=1)
                continue
            data = r.json()
            if not data:
                break
            for c in data:  # [time, low, high, open, close, volume]
                rows[int(c[0])] = float(c[4])
        except Exception:
            time.sleep(1.0)
        end = start - timedelta(days=1)
        time.sleep(0.34)
    if not rows:
        return pd.Series(dtype=float, name=product)
    idx = pd.to_datetime(sorted(rows), unit="s", utc=True)
    s = pd.Series([rows[int(t.timestamp())] for t in idx], index=idx, name=product)
    s = s[s > 0].sort_index()
    s.to_frame("close").to_csv(cf)
    return s


def coin_alpha(px: pd.Series, name: str) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    feat = make_features(pd.DataFrame({name: px}))
    feat = feat[(feat["pair"] == name) & feat["fwd_ret"].notna()]
    cols = [c for c in feat.columns if c not in ("y", "fwd_ret", "pair", "year")]
    years = sorted(feat["year"].unique())
    py = []
    for ytest in years:
        tr = feat[feat["year"] < ytest]; te = feat[feat["year"] == ytest]
        if len(te) < 120 or len(tr) < 500:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                             l2_regularization=1.0, min_samples_leaf=60, random_state=42)
        clf.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        pred = (clf.predict_proba(te[cols].to_numpy())[:, 1] >= 0.5).astype(int)
        fwd = te["fwd_ret"].to_numpy()
        def sh(x):
            return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN))
        flips = np.abs(np.diff(np.concatenate([[0.0], np.where(pred == 1, 1.0, -1.0)])))
        py.append({"year": int(ytest),
                   "bal_acc": round(float(balanced_accuracy_score(te["y"], pred)), 4),
                   "alpha": round(sh(np.where(pred == 1, 1.0, 0.0) * fwd) - sh(fwd), 3),
                   "ls_net": round(sh(np.where(pred == 1, 1.0, -1.0) * fwd - flips * COST), 2)})
    if len(py) < 3:
        return None
    al = np.array([p["alpha"] for p in py])
    return {"test_years": len(py), "mean_alpha": round(float(al.mean()), 3),
            "years_pos_alpha": int((al > 0).sum()),
            "mean_ls_net": round(float(np.mean([p["ls_net"] for p in py])), 2),
            "mean_bal_acc": round(float(np.mean([p["bal_acc"] for p in py])), 4),
            "span": f"{px.index[0].date()}->{px.index[-1].date()}", "per_year": py}


def main() -> int:
    t0 = time.time()
    res = {}
    for p in PRODUCTS:
        s = coinbase_daily(p)
        if len(s) < 800:
            print(f"  skip {p}: only {len(s)} days"); continue
        r = coin_alpha(s, p.replace("-", "_"))
        if r:
            res[p] = r
            print(f"  {p:9s} span {r['span']}  yrs={r['test_years']}  "
                  f"mean_alpha={r['mean_alpha']:+.3f}  pos={r['years_pos_alpha']}/{r['test_years']}  "
                  f"bal={r['mean_bal_acc']:.4f}  ls_net={r['mean_ls_net']:+.2f}")
    alphas = np.array([r["mean_alpha"] for r in res.values()])
    pos_share = float((alphas > 0).mean()) if len(alphas) else 0.0
    g2 = bool(len(alphas) >= 8 and alphas.mean() >= 0.20 and pos_share > 0.5)
    print("\n=== SMART G2 — crypto universe timing-alpha generalization ===")
    print(f"coins with usable history: {len(res)} | mean alpha across coins = {alphas.mean():+.3f} "
          f"| coins positive = {int((alphas>0).sum())}/{len(alphas)} ({pos_share:.0%})")
    print(f"G2 bar: >=8 coins AND mean alpha >=+0.20 AND majority positive  ->  "
          f"{'PASS' if g2 else 'FAIL'}")
    out = {"results": res, "n_coins": len(res), "mean_alpha": round(float(alphas.mean()), 3) if len(alphas) else None,
           "coins_positive": int((alphas > 0).sum()) if len(alphas) else 0, "g2_pass": g2,
           "elapsed_s": round(time.time() - t0, 1)}
    pth = REPO_ROOT / "trained_data" / "backtests" / (
        f"crypto_universe_g2_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    pth.parent.mkdir(parents=True, exist_ok=True)
    tmp = pth.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(pth)
    print(f"\nwrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
