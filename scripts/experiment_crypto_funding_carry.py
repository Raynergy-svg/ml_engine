#!/usr/bin/env python3
"""H1 — Perp funding carry (cross-sectional, dollar-neutral, market-neutral).

Pre-registered: docs/experiment-crypto-edge-hunt-2026-06-29.md  (frozen; NO sweep).
Research/backtest only — no live execution, no real money, no order path.

Rule (frozen):
  signal s_i(t) = trailing 3-day mean funding rate (last 9 x 8h intervals), known
                  at end of day t-1 (causal, 1-day execution lag).
  short top-quintile by s (receive funding), long bottom-quintile (receive funding),
  equal-weight within leg, scaled dollar-neutral (gross=1.0), daily rebalance.
Universe (frozen, point-in-time, survivorship-aware): Binance USDT perps with
  >=90d history and trailing-30d median quote-volume ADV >= $10M as of t; coins that
  later delist remain eligible while alive (closed at last price -> tail UNDER-counted).
Costs (frozen): 10 bps/side (5 taker + 5 slippage*1%ADV); +2x stress. Funding modeled
  exactly from data. OOS = 2024-01-01+ (untouched by construction; no sweep).
Gate (frozen): OOS net Sharpe>=0.40, maxDD<=0.25, walk-forward, OOS-confirmed,
  DSR>=0.95 & Bonferroni bootstrap p<0.05/3, |BTC-beta|<=0.15. MIN_TOTAL_YEARS=10
  CANNOT be met -> permanent "clears ex-history" caveat (never relaxed).
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

ANN = 365.0  # crypto trades 365 d/yr
QUINTILE = 0.20
LOOKBACK_D = 3
MIN_HISTORY_D = 90
MIN_ADV_USD = 10_000_000
ADV_WINDOW_D = 30
COST_BPS = 10.0          # per side, applied to turnover
STRESS_MULT = 2.0
OOS_START = "2024-01-01"
N_TRIALS = 3             # multiple-testing budget (H1,H2,H3)
EULER = 0.5772156649
WORKERS = 24


# --------------------------------------------------------------------------- #
# Data assembly
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
            except Exception as exc:  # surface, don't swallow (per repo gate)
                print(f"  WARN {label} {s}: {exc}", file=sys.stderr)
            if done % 100 == 0:
                print(f"  {label}: {done}/{len(symbols)} ({time.time()-t0:.0f}s)")
    print(f"  {label}: {len(out)}/{len(symbols)} non-empty in {time.time()-t0:.0f}s")
    return out


def build_panels(refresh: bool = False):
    """Return (close, adv, funding_daily, btc_ret) daily panels (date x symbol)."""
    symbols = dl.list_binance_perp_symbols("USDT")
    print(f"universe enumerated (incl delisted): {len(symbols)} USDT perps")

    klines = _concurrent(lambda s: dl.fetch_binance_klines(s, "1d", refresh=refresh),
                         symbols, "klines")
    close = pd.DataFrame({s: k["close"] for s, k in klines.items()})
    qvol = pd.DataFrame({s: k["quote_volume"] for s, k in klines.items()})
    close.index = pd.to_datetime(close.index).tz_convert("UTC").normalize()
    qvol.index = pd.to_datetime(qvol.index).tz_convert("UTC").normalize()
    close = close.sort_index()
    qvol = qvol.reindex(close.index)

    # trailing-30d median ADV (causal: shifted so decision at t uses <=t-1)
    adv = qvol.rolling(ADV_WINDOW_D, min_periods=10).median().shift(1)
    hist_ok = close.notna().cumsum().shift(1) >= MIN_HISTORY_D  # >=90 prior obs

    # funding only for symbols that ever clear the ADV floor (saves downloads)
    ever_liquid = [s for s in close.columns if (adv[s] >= MIN_ADV_USD).any()]
    print(f"symbols ever >= ${MIN_ADV_USD/1e6:.0f}M ADV: {len(ever_liquid)}")
    fund = _concurrent(lambda s: dl.fetch_binance_funding(s, refresh=refresh),
                       ever_liquid, "funding")
    # aggregate 8h funding -> daily sum (total funding paid that day)
    fd = {}
    for s, f in fund.items():
        d = f["funding_rate"].copy()
        d.index = pd.to_datetime(d.index).tz_convert("UTC").normalize()
        fd[s] = d.groupby(level=0).sum()
    funding_daily = pd.DataFrame(fd).reindex(close.index)

    eligible = hist_ok & (adv >= MIN_ADV_USD) & funding_daily.notna() & close.notna()
    eligible = eligible[ever_liquid].fillna(False)

    btc_ret = close["BTCUSDT"].pct_change()
    return close, adv, funding_daily, eligible, btc_ret, ever_liquid


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(close, funding_daily, eligible, cols, *, cost_bps,
             restrict_alive_at_end=None):
    """Run H1. Returns dict of daily series: net, gross, carry, price, turnover.

    restrict_alive_at_end: if a set, only trade symbols whose last close is within
    45d of global end (survivor-only run, for dead-coin decomposition).
    """
    px = close[cols]
    fund = funding_daily[cols]
    elig = eligible[cols].copy()

    if restrict_alive_at_end is not None:
        for s in cols:
            if s not in restrict_alive_at_end:
                elig[s] = False

    price_ret = px.pct_change()
    # signal: trailing LOOKBACK_D-day mean of daily funding, as of day d itself.
    # The 1-day signal-to-execution lag is already provided by the loop below
    # (w decided at iteration d is applied via prev_w starting at day d+1) — an
    # additional .shift(1) here double-lagged the signal, so day-d P&L was
    # actually using funding through d-2, not the intended d-1.
    sig = fund.rolling(LOOKBACK_D, min_periods=LOOKBACK_D).mean()
    sig = sig.where(elig.shift(1).fillna(False))

    dates = px.index
    prev_w = pd.Series(0.0, index=cols)
    rows = []
    for d in dates:
        s = sig.loc[d].dropna()
        w = pd.Series(0.0, index=cols)
        if len(s) >= 10:  # need a meaningful cross-section
            k = max(1, int(len(s) * QUINTILE))
            ranked = s.sort_values()
            longs = ranked.index[:k]    # lowest funding -> long (receive when <0)
            shorts = ranked.index[-k:]  # highest funding -> short (receive when >0)
            w[longs] = 0.5 / k
            w[shorts] = -0.5 / k
        # P&L for day d from positions decided at d-1 (prev_w), held over day d
        pr = price_ret.loc[d].reindex(cols).fillna(0.0)
        fr = fund.loc[d].reindex(cols).fillna(0.0)
        price_pnl = float((prev_w * pr).sum())
        carry_pnl = float((-prev_w * fr).sum())   # short(+w<0) receives +funding
        turnover = float((w - prev_w).abs().sum())
        cost = turnover * (cost_bps / 1e4)
        rows.append({
            "date": d, "price": price_pnl, "carry": carry_pnl,
            "turnover": turnover, "cost": cost,
            "gross": price_pnl + carry_pnl,
            "net": price_pnl + carry_pnl - cost,
        })
        prev_w = w
    r = pd.DataFrame(rows).set_index("date")
    return r


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def sharpe(x):
    x = x.dropna()
    return float(np.sqrt(ANN) * x.mean() / x.std(ddof=1)) if x.std(ddof=1) > 0 else 0.0


def max_drawdown(x):
    eq = (1 + x.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def psr(x, sr_bench_perperiod=0.0):
    """Probabilistic Sharpe Ratio: Prob(true SR > bench), non-normal adjusted."""
    x = x.dropna()
    t = len(x)
    if t < 30 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)  # per-period
    sk = float(stats.skew(x))
    ku = float(stats.kurtosis(x, fisher=False))  # Pearson (normal=3)
    denom = np.sqrt((1 - sk * sr + (ku - 1) / 4 * sr ** 2) / (t - 1))
    return float(stats.norm.cdf((sr - sr_bench_perperiod) / denom))


def deflated_sr(x, n_trials=N_TRIALS):
    """DSR = PSR at the deflated benchmark = E[max of n_trials SRs]."""
    x = x.dropna()
    t = len(x)
    if t < 30:
        return float("nan"), float("nan")
    # expected max of n iid standard-normal SR estimates (Bailey & Lopez de Prado)
    e_max = ((1 - EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
             + EULER * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = e_max / np.sqrt(t - 1)  # per-period deflated benchmark
    return float(psr(x, sr0)), float(sr0 * np.sqrt(ANN))


def block_bootstrap_p(x, n=5000, block=5, seed=12345):
    """One-sided p that mean<=0, stationary block bootstrap."""
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
    df.columns = ["strat", "btc"]
    if len(df) < 30 or df["btc"].var() == 0:
        return float("nan")
    return float(np.cov(df["strat"], df["btc"])[0, 1] / df["btc"].var())


def report_block(r, btc_ret, label):
    net = r["net"]
    return {
        "label": label, "n_days": int(len(net)),
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
        "btc_beta": round(btc_beta(net, btc_ret), 3),
    }


def main():
    t0 = time.time()
    print("=== H1 CRYPTO FUNDING CARRY (pre-registered, frozen) ===")
    close, adv, funding_daily, eligible, btc_ret, cols = build_panels()
    print(f"tradeable universe size (ever-liquid): {len(cols)}")
    print(f"backtest span: {close.index.min().date()} -> {close.index.max().date()}")

    # global "alive at end" set for survivor-only decomposition
    gend = close.index.max()
    alive = {s for s in cols
             if close[s].last_valid_index() is not None
             and close[s].last_valid_index() >= gend - pd.Timedelta(days=45)}
    n_dead = len(cols) - len(alive)
    print(f"alive-at-end: {len(alive)} | delisted-during-sample: {n_dead}")

    # full (survivorship-aware) run
    r_full = backtest(close, funding_daily, eligible, cols, cost_bps=COST_BPS)
    r_2x = backtest(close, funding_daily, eligible, cols,
                    cost_bps=COST_BPS * STRESS_MULT)
    r_surv = backtest(close, funding_daily, eligible, cols, cost_bps=COST_BPS,
                      restrict_alive_at_end=alive)

    oos = lambda r: r.loc[r.index >= OOS_START]  # noqa: E731
    is_ = lambda r: r.loc[r.index < OOS_START]   # noqa: E731

    full_all = report_block(r_full, btc_ret, "full_all")
    full_is = report_block(is_(r_full), btc_ret, "full_IS_pre2024")
    full_oos = report_block(oos(r_full), btc_ret, "full_OOS_2024+")
    stress_oos = report_block(oos(r_2x), btc_ret, "stress2x_OOS")
    surv_oos = report_block(oos(r_surv), btc_ret, "survivoronly_OOS")

    # significance on OOS net
    oos_net = oos(r_full)["net"]
    dsr, dsr_bench = deflated_sr(oos_net)
    boot_p = block_bootstrap_p(oos_net)
    psr0 = psr(oos_net, 0.0)

    # gate
    g_sharpe = full_oos["net_sharpe"] >= 0.40
    g_dd = abs(full_all["max_drawdown"]) <= 0.25
    g_oos_conf = (full_oos["net_sharpe"] >= 0.40
                  and np.sign(full_oos["net_sharpe"]) == np.sign(full_is["net_sharpe"]))
    g_mt = (not np.isnan(dsr)) and dsr >= 0.95 and (not np.isnan(boot_p)) \
        and boot_p < 0.05 / N_TRIALS
    g_beta = abs(full_oos["btc_beta"]) <= 0.15
    g_history = False  # MIN_TOTAL_YEARS=10 cannot be met (pre-registered caveat)

    dead_coin_inflation = round(full_oos["carry_ann_ret"] - surv_oos["carry_ann_ret"], 4)

    verdict = {
        "experiment": "H1_crypto_funding_carry",
        "pre_registered": "docs/experiment-crypto-edge-hunt-2026-06-29.md",
        "frozen_params": {
            "lookback_d": LOOKBACK_D, "quintile": QUINTILE,
            "min_adv_usd": MIN_ADV_USD, "min_history_d": MIN_HISTORY_D,
            "cost_bps_per_side": COST_BPS, "oos_start": OOS_START,
            "n_trials": N_TRIALS,
        },
        "universe": {"ever_liquid": len(cols), "alive_at_end": len(alive),
                     "delisted_during_sample": n_dead,
                     "span": [str(close.index.min().date()),
                              str(close.index.max().date())]},
        "blocks": {b["label"]: b for b in
                   [full_all, full_is, full_oos, stress_oos, surv_oos]},
        "significance_oos": {
            "psr_gt0": round(psr0, 4), "deflated_sr": round(dsr, 4),
            "dsr_bench_ann_sharpe": round(dsr_bench, 3),
            "bootstrap_p_mean_le0": round(boot_p, 4),
            "bonferroni_alpha": round(0.05 / N_TRIALS, 4),
        },
        "decomposition_oos": {
            "carry_ann_ret": full_oos["carry_ann_ret"],
            "price_ann_ret": full_oos["price_ann_ret"],
            "btc_beta": full_oos["btc_beta"],
            "survivoronly_carry_ann_ret": surv_oos["carry_ann_ret"],
            "dead_coin_carry_inflation": dead_coin_inflation,
        },
        "gate": {
            "oos_net_sharpe>=0.40": bool(g_sharpe),
            "max_drawdown<=0.25": bool(g_dd),
            "oos_confirmed": bool(g_oos_conf),
            "multiple_testing(DSR>=.95 & p<.0167)": bool(g_mt),
            "btc_beta_neutral(|b|<=0.15)": bool(g_beta),
            "history_depth>=10y": bool(g_history),
        },
        "clears_ex_history": bool(g_sharpe and g_dd and g_oos_conf and g_mt and g_beta),
        "elapsed_s": round(time.time() - t0, 1),
    }

    outdir = REPO_ROOT / "trained_data" / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"crypto_funding_carry_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    tmp.replace(p)

    print("\n--- OOS (2024+) headline ---")
    print(json.dumps(full_oos, indent=2))
    print("\n--- significance + decomposition (OOS) ---")
    print(json.dumps(verdict["significance_oos"], indent=2))
    print(json.dumps(verdict["decomposition_oos"], indent=2))
    print("\n--- gate ---")
    print(json.dumps(verdict["gate"], indent=2))
    print(f"\nclears_ex_history={verdict['clears_ex_history']}  "
          f"(history_depth>=10y is FALSE by construction -> binding caveat)")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
