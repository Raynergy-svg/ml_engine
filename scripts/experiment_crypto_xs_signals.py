#!/usr/bin/env python3
"""H2/H3 — cross-sectional crypto signals on the SAME verified harness as H1.

Pre-registered: docs/experiment-crypto-edge-hunt-2026-06-29.md (frozen; NO sweep).
Research/backtest only — no live execution, no real money, no order path.

Backtest/metrics/gate logic is COPIED VERBATIM from the verifier-confirmed H1 script
(scripts/experiment_crypto_funding_carry.py); only the SIGNAL swaps. Funding P&L is
always modeled (perp positions accrue funding incidentally) so net is realistic.

  H2 momentum  : signal = trailing-14d return, direction=+1 (long winners / short
                 losers), quintiles, $-neutral, 1d lag, daily.
  H3 orderflow : signal = trailing-1d taker-buy ratio (taker_base/volume) - 0.5,
                 direction=-1 (contrarian: long faded-selling / short faded-buying).

Usage: python scripts/experiment_crypto_xs_signals.py {momentum|orderflow} [--refresh-klines]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crypto import data_layer as dl  # noqa: E402

ANN = 365.0
QUINTILE = 0.20
MIN_HISTORY_D = 90
MIN_ADV_USD = 10_000_000
ADV_WINDOW_D = 30
COST_BPS = 10.0
STRESS_MULT = 2.0
OOS_START = "2024-01-01"
N_TRIALS = 3
EULER = 0.5772156649
WORKERS = 24

SIGNALS = {
    "momentum": {"lookback": 14, "direction": +1, "needs_taker": False,
                 "h": "H2"},
    "orderflow": {"lookback": 1, "direction": -1, "needs_taker": True,
                  "h": "H3"},
}


# --------------------------------------------------------------------------- #
def _concurrent(fn, symbols, label):
    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fn, s): s for s in symbols}
        done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            done += 1
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    out[s] = df
            except Exception as exc:
                print(f"  WARN {label} {s}: {exc}", file=sys.stderr)
            if done % 100 == 0:
                print(f"  {label}: {done}/{len(symbols)} ({time.time()-t0:.0f}s)")
    print(f"  {label}: {len(out)}/{len(symbols)} non-empty in {time.time()-t0:.0f}s")
    return out


def build_panels(needs_taker: bool, refresh_klines: bool):
    symbols = dl.list_binance_perp_symbols("USDT")
    print(f"universe enumerated (incl delisted): {len(symbols)} USDT perps")
    klines = _concurrent(
        lambda s: dl.fetch_binance_klines(s, "1d", refresh=refresh_klines),
        symbols, "klines")
    close = pd.DataFrame({s: k["close"] for s, k in klines.items()})
    qvol = pd.DataFrame({s: k["quote_volume"] for s, k in klines.items()})
    for df in (close, qvol):
        df.index = pd.to_datetime(df.index).tz_convert("UTC").normalize()
    close = close.sort_index()
    qvol = qvol.reindex(close.index)

    taker_ratio = None
    if needs_taker:
        tr = {}
        for s, k in klines.items():
            if "taker_base" in k.columns and "volume" in k.columns:
                v = k["volume"].replace(0, np.nan)
                r = (k["taker_base"] / v)
                r.index = pd.to_datetime(r.index).tz_convert("UTC").normalize()
                tr[s] = r
        taker_ratio = pd.DataFrame(tr).reindex(close.index)

    adv = qvol.rolling(ADV_WINDOW_D, min_periods=10).median().shift(1)
    hist_ok = close.notna().cumsum().shift(1) >= MIN_HISTORY_D
    ever_liquid = [s for s in close.columns if (adv[s] >= MIN_ADV_USD).any()]
    print(f"symbols ever >= ${MIN_ADV_USD/1e6:.0f}M ADV: {len(ever_liquid)}")

    fund = _concurrent(lambda s: dl.fetch_binance_funding(s), ever_liquid, "funding")
    fd = {}
    for s, f in fund.items():
        d = f["funding_rate"].copy()
        d.index = pd.to_datetime(d.index).tz_convert("UTC").normalize()
        fd[s] = d.groupby(level=0).sum()
    funding_daily = pd.DataFrame(fd).reindex(close.index)

    eligible = (hist_ok & (adv >= MIN_ADV_USD)
                & funding_daily.notna() & close.notna())[ever_liquid].fillna(False)
    btc_ret = close["BTCUSDT"].pct_change()
    return close, funding_daily, eligible, btc_ret, ever_liquid, taker_ratio


def make_signal(name, close, taker_ratio, cols):
    spec = SIGNALS[name]
    lb = spec["lookback"]
    if name == "momentum":
        raw = close[cols] / close[cols].shift(lb) - 1.0
    elif name == "orderflow":
        raw = taker_ratio[cols] - 0.5
    else:
        raise ValueError(name)
    return raw.shift(1)  # known at t-1 (causal)


# --------------------------------------------------------------------------- #
def backtest(close, funding_daily, eligible, cols, signal_df, direction, *,
             cost_bps, restrict_alive_at_end=None):
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols].copy()
    if restrict_alive_at_end is not None:
        for s in cols:
            if s not in restrict_alive_at_end:
                elig[s] = False

    price_ret = px.pct_change()
    sig = signal_df.where(elig.shift(1).fillna(False))

    prev_w = pd.Series(0.0, index=cols)
    rows = []
    for d in px.index:
        s = sig.loc[d].dropna()
        w = pd.Series(0.0, index=cols)
        if len(s) >= 10:
            k = max(1, int(len(s) * QUINTILE))
            ranked = s.sort_values()
            low, high = ranked.index[:k], ranked.index[-k:]
            if direction > 0:           # long high signal, short low
                longs, shorts = high, low
            else:                       # long low signal, short high
                longs, shorts = low, high
            w[longs] = 0.5 / k
            w[shorts] = -0.5 / k
        pr = price_ret.loc[d].reindex(cols).fillna(0.0)
        fr = fund.loc[d].reindex(cols).fillna(0.0)
        price_pnl = float((prev_w * pr).sum())
        carry_pnl = float((-prev_w * fr).sum())
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * (cost_bps / 1e4)
        rows.append({"date": d, "price": price_pnl, "carry": carry_pnl,
                     "turnover": turnover, "cost": cost,
                     "gross": price_pnl + carry_pnl,
                     "net": price_pnl + carry_pnl - cost})
        prev_w = w
    return pd.DataFrame(rows).set_index("date")


# --------------------------------------------------------------------------- #
def sharpe(x):
    x = x.dropna()
    return float(np.sqrt(ANN) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0


def max_drawdown(x):
    eq = (1 + x.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def psr(x, bench=0.0):
    x = x.dropna()
    t = len(x)
    if t < 30 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    sk = float(stats.skew(x))
    ku = float(stats.kurtosis(x, fisher=False))
    denom = np.sqrt((1 - sk * sr + (ku - 1) / 4 * sr ** 2) / (t - 1))
    return float(stats.norm.cdf((sr - bench) / denom))


def deflated_sr(x, n_trials=N_TRIALS):
    x = x.dropna()
    t = len(x)
    if t < 30:
        return float("nan"), float("nan")
    e_max = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
             + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = e_max / np.sqrt(t - 1)
    return float(psr(x, sr0)), float(sr0 * np.sqrt(ANN))


def block_bootstrap_p(x, n=5000, block=5, seed=12345):
    x = x.dropna().to_numpy()
    t = len(x)
    if t < 30:
        return float("nan")
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(t / block))
    means = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, t, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % t
        means[i] = x[idx[:t]].mean()
    return float((means <= 0).mean())


def btc_beta(net, btc_ret):
    df = pd.concat([net, btc_ret], axis=1).dropna()
    df.columns = ["s", "b"]
    if len(df) < 30 or df["b"].var() == 0:
        return float("nan")
    return float(np.cov(df["s"], df["b"])[0, 1] / df["b"].var())


def block_report(r, btc_ret, label):
    net = r["net"]
    return {"label": label, "n_days": int(len(net)),
            "net_sharpe": round(sharpe(net), 3),
            "gross_sharpe": round(sharpe(r["gross"]), 3),
            "carry_sharpe": round(sharpe(r["carry"]), 3),
            "price_sharpe": round(sharpe(r["price"]), 3),
            "net_ann_ret": round(float(net.mean() * ANN), 4),
            "carry_ann_ret": round(float(r["carry"].mean() * ANN), 4),
            "price_ann_ret": round(float(r["price"].mean() * ANN), 4),
            "cost_ann": round(float(r["cost"].mean() * ANN), 4),
            "avg_turnover": round(float(r["turnover"].mean()), 3),
            "max_drawdown": round(max_drawdown(net), 3),
            "btc_beta": round(btc_beta(net, btc_ret), 3)}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SIGNALS:
        print("usage: experiment_crypto_xs_signals.py {momentum|orderflow} "
              "[--refresh-klines]")
        return 2
    name = sys.argv[1]
    spec = SIGNALS[name]
    refresh = "--refresh-klines" in sys.argv
    t0 = time.time()
    print(f"=== {spec['h']} CRYPTO XS {name.upper()} (pre-registered, frozen) ===")

    close, funding_daily, eligible, btc_ret, cols, taker = build_panels(
        spec["needs_taker"], refresh)
    sig = make_signal(name, close, taker, cols)
    print(f"universe: {len(cols)} | span {close.index.min().date()}->"
          f"{close.index.max().date()}")

    gend = close.index.max()
    alive = {s for s in cols
             if close[s].last_valid_index() is not None
             and close[s].last_valid_index() >= gend - pd.Timedelta(days=45)}

    d = spec["direction"]
    r_full = backtest(close, funding_daily, eligible, cols, sig, d, cost_bps=COST_BPS)
    r_2x = backtest(close, funding_daily, eligible, cols, sig, d,
                    cost_bps=COST_BPS * STRESS_MULT)
    r_surv = backtest(close, funding_daily, eligible, cols, sig, d, cost_bps=COST_BPS,
                      restrict_alive_at_end=alive)

    oos = lambda r: r.loc[r.index >= OOS_START]   # noqa: E731
    is_ = lambda r: r.loc[r.index < OOS_START]    # noqa: E731
    full_all = block_report(r_full, btc_ret, "full_all")
    full_is = block_report(is_(r_full), btc_ret, "full_IS_pre2024")
    full_oos = block_report(oos(r_full), btc_ret, "full_OOS_2024+")
    stress_oos = block_report(oos(r_2x), btc_ret, "stress2x_OOS")
    surv_oos = block_report(oos(r_surv), btc_ret, "survivoronly_OOS")

    oos_net = oos(r_full)["net"]
    dsr, dsr_bench = deflated_sr(oos_net)
    boot_p = block_bootstrap_p(oos_net)
    psr0 = psr(oos_net, 0.0)

    g_sharpe = full_oos["net_sharpe"] >= 0.40
    g_dd = abs(full_all["max_drawdown"]) <= 0.25
    g_oos = (full_oos["net_sharpe"] >= 0.40
             and np.sign(full_oos["net_sharpe"]) == np.sign(full_is["net_sharpe"]))
    g_mt = (not np.isnan(dsr)) and dsr >= 0.95 and (not np.isnan(boot_p)) \
        and boot_p < 0.05 / N_TRIALS
    g_beta = abs(full_oos["btc_beta"]) <= 0.15

    verdict = {
        "experiment": f"{spec['h']}_crypto_xs_{name}",
        "pre_registered": "docs/experiment-crypto-edge-hunt-2026-06-29.md",
        "frozen_params": {"signal": name, "lookback": spec["lookback"],
                          "direction": d, "quintile": QUINTILE,
                          "min_adv_usd": MIN_ADV_USD, "cost_bps_per_side": COST_BPS,
                          "oos_start": OOS_START, "n_trials": N_TRIALS},
        "universe": {"ever_liquid": len(cols), "alive_at_end": len(alive),
                     "delisted_during_sample": len(cols) - len(alive),
                     "span": [str(close.index.min().date()),
                              str(close.index.max().date())]},
        "blocks": {b["label"]: b for b in
                   [full_all, full_is, full_oos, stress_oos, surv_oos]},
        "significance_oos": {"psr_gt0": round(psr0, 4),
                             "deflated_sr": round(dsr, 4),
                             "dsr_bench_ann_sharpe": round(dsr_bench, 3),
                             "bootstrap_p_mean_le0": round(boot_p, 4),
                             "bonferroni_alpha": round(0.05 / N_TRIALS, 4)},
        "gate": {"oos_net_sharpe>=0.40": bool(g_sharpe),
                 "max_drawdown<=0.25": bool(g_dd),
                 "oos_confirmed": bool(g_oos),
                 "multiple_testing(DSR>=.95 & p<.0167)": bool(g_mt),
                 "btc_beta_neutral(|b|<=0.15)": bool(g_beta),
                 "history_depth>=10y": False},
        "clears_ex_history": bool(g_sharpe and g_dd and g_oos and g_mt and g_beta),
        "elapsed_s": round(time.time() - t0, 1),
    }
    outdir = REPO_ROOT / "trained_data" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / (f"crypto_xs_{name}_"
                  f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    tmp.replace(p)

    print("\n--- OOS (2024+) ---")
    print(json.dumps(full_oos, indent=2))
    print("--- IS (pre-2024) ---")
    print(json.dumps(full_is, indent=2))
    print("--- significance (OOS) ---")
    print(json.dumps(verdict["significance_oos"], indent=2))
    print("--- gate ---")
    print(json.dumps(verdict["gate"], indent=2))
    print(f"\nclears_ex_history={verdict['clears_ex_history']} "
          f"(history>=10y FALSE by construction)\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
