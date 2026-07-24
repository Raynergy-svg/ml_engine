"""Kronos-small vs Axiom trend lane — pre-registered trade-level shootout.

Pre-registration: docs/prereg-kronos-vs-axiom-shootout-2026-07-24.md (frozen BEFORE
any out-of-sample outcome was computed). Research-only: imports no broker, execution,
or state module; writes only under trained_data/research/kronos_shootout/.

Arms
----
AXIOM : the live trend-lane construction, reused verbatim from
        src.equity.trend_sleeve.trend_sleeve_weights (long iff close[t-1] > SMA100,
        else flat; long-or-flat is the lane's real construction).
KRONOS: NeoQuasar/Kronos-small zero-shot, 400-bar context, pred_len=5, repo-default
        sampling (T=1.0, top_p=0.9, top_k=0), sample_count=3, seeded per week.
        Long/short/abstain around a cost-sized deadband.

Implementation choice documented before OOS read (prereg S6.4 left it open): the
weekly net-return series used for Sharpe/DSR/bootstrap charges the round-trip cost
in full on the week a trade OPENS (direction change from flat/opposite); held weeks
carry gross return only. Trade-level metrics use segment-merged trades per prereg S3.

Usage
-----
  python scripts/experiment_kronos_vs_axiom_shootout.py --mode smoke --skip-kronos
  python scripts/experiment_kronos_vs_axiom_shootout.py --mode full
Environment: KRONOS_REPO_PATH must point at a checkout of
https://github.com/shiyu-coder/Kronos @ 67b630e67f6a18c9e9be918d9b4337c960db1e9a.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.equity.trend_sleeve import trend_sleeve_weights  # noqa: E402
from src.research.gated_harness.significance import (  # noqa: E402
    circular_block_bootstrap_sharpe_pvalue,
    deflated_sharpe_ratio,
)

# ---- frozen protocol constants (prereg S2-S8; changing any = new experiment) ----
EXPERIMENT_ID = "kronos-vs-axiom-trend-shootout-2026-07-24"
PREREG_DOC = _REPO_ROOT / "docs" / "prereg-kronos-vs-axiom-shootout-2026-07-24.md"
PANEL_DIR_DEFAULT = _REPO_ROOT / "market_data" / "factor"
OUT_DIR_DEFAULT = _REPO_ROOT / "trained_data" / "research" / "kronos_shootout"

SMA_WINDOW = 100
KRONOS_CONTEXT = 400
PRED_LEN = 5
KRONOS_T = 1.0
KRONOS_TOP_P = 0.9
KRONOS_TOP_K = 0
KRONOS_SAMPLE_COUNT = 3
SEED_BASE = 42
HOLD_BARS = 5  # enter open[t+1], exit open[t+6]

COST_PRIMARY_BPS = 2.5
COST_SCENARIOS_BPS = (1.5, 2.5, 4.0)
DEADBAND = COST_PRIMARY_BPS / 1e4

PRIMARY_WINDOW = ("2025-09-01", "2026-06-11")
SECONDARY_WINDOW = ("2024-01-01", "2025-08-31")
SMOKE_WINDOW = ("2018-01-01", "2018-06-30")  # in-sample; OOS stays sealed

BOOT_BLOCK = 4
BOOT_REPS = 2000
BOOT_SEED = 7

KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_REPO_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


@dataclass
class WeekDecision:
    """One arm's decision + outcome for one pair-week."""

    pair: str
    date: pd.Timestamp          # decision bar t (first trading day of ISO week)
    direction: int              # +1 long / -1 short / 0 abstain
    gross: Optional[float]      # direction * (open[t+6]/open[t+1] - 1); None if abstain


@dataclass
class Trade:
    """A merged run of same-direction consecutive weekly decisions (prereg S3)."""

    pair: str
    direction: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    weeks: int
    gross: float                # compounded over the segment

    def net(self, cost_bps: float) -> float:
        return self.gross - cost_bps / 1e4


@dataclass
class ArmResult:
    weekly: List[WeekDecision] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Panel + calendar
# --------------------------------------------------------------------------- #

def load_panel(panel_dir: Path, pairs: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load per-pair daily OHLCV CSVs, date-indexed ascending, deduped."""
    out: Dict[str, pd.DataFrame] = {}
    for csv in sorted(panel_dir.glob("*_D.csv")):
        pair = csv.name[: -len("_D.csv")]
        if pairs and pair not in pairs:
            continue
        df = pd.read_csv(csv, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        need = ["open", "high", "low", "close", "volume"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"{csv.name}: missing columns {missing}")
        out[pair] = df[need].astype(float)
    if not out:
        raise ValueError(f"no *_D.csv panels found under {panel_dir}")
    return out


def decision_dates(index: pd.DatetimeIndex, start: str, end: str) -> List[pd.Timestamp]:
    """First trading day of each ISO week inside [start, end] (prereg S3)."""
    lo, hi = pd.Timestamp(start, tz=index.tz), pd.Timestamp(end, tz=index.tz)
    within = index[(index >= lo) & (index <= hi)]
    seen: set = set()
    picks: List[pd.Timestamp] = []
    for ts in within:
        key = ts.isocalendar()[:2]
        if key not in seen:
            seen.add(key)
            picks.append(ts)
    return picks


def week_outcome(df: pd.DataFrame, t: pd.Timestamp) -> Optional[float]:
    """(open[t+6] / open[t+1]) - 1, or None when forward bars don't exist yet."""
    pos = df.index.get_loc(t)
    if not isinstance(pos, (int, np.integer)):
        return None
    entry_i, exit_i = pos + 1, pos + 1 + HOLD_BARS
    if exit_i >= len(df.index):
        return None
    entry, exit_ = float(df["open"].iloc[entry_i]), float(df["open"].iloc[exit_i])
    if not (math.isfinite(entry) and math.isfinite(exit_)) or entry <= 0:
        return None
    return exit_ / entry - 1.0


# --------------------------------------------------------------------------- #
# Arm A — AXIOM trend lane (reused, not forked)
# --------------------------------------------------------------------------- #

def axiom_directions(df: pd.DataFrame, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
    """Long/flat per decision date via the live lane's trend_sleeve_weights."""
    px = df[["close"]].rename(columns={"close": "PAIR"})
    w = trend_sleeve_weights(px, sma_window=SMA_WINDOW, step=1)
    return {t: (1 if float(w.loc[t, "PAIR"]) > 0.0 else 0) for t in dates if t in w.index}


# --------------------------------------------------------------------------- #
# Arm B — Kronos zero-shot
# --------------------------------------------------------------------------- #

def kronos_direction_from_pred(pred_close_5: float, last_close: float) -> int:
    """Long/short/abstain around the cost-sized deadband (prereg S2 arm B)."""
    if not (math.isfinite(pred_close_5) and math.isfinite(last_close)) or last_close <= 0:
        return 0
    r = pred_close_5 / last_close - 1.0
    if r > DEADBAND:
        return 1
    if r < -DEADBAND:
        return -1
    return 0


def _load_kronos():
    repo = os.environ.get("KRONOS_REPO_PATH", "")
    if not repo or not Path(repo).exists():
        raise RuntimeError(
            "KRONOS_REPO_PATH must point at a checkout of shiyu-coder/Kronos "
            f"@ {KRONOS_REPO_COMMIT} (weights come from huggingface.co — "
            "blocked hosts must be reported, not routed around)"
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch  # noqa: F401  (fail here, loudly, if absent)
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)
    model = Kronos.from_pretrained(KRONOS_MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    return predictor


def kronos_directions(
    panel: Dict[str, pd.DataFrame],
    dates_by_pair: Dict[str, List[pd.Timestamp]],
) -> Dict[str, Dict[pd.Timestamp, int]]:
    """Batched per-decision-date zero-shot forecasts -> directions."""
    import torch

    predictor = _load_kronos()
    all_dates = sorted({t for ds in dates_by_pair.values() for t in ds})
    out: Dict[str, Dict[pd.Timestamp, int]] = {p: {} for p in dates_by_pair}
    for week_i, t in enumerate(all_dates):
        batch_pairs: List[str] = []
        dfs, x_ts, y_ts = [], [], []
        for pair, ds in dates_by_pair.items():
            if t not in ds:
                continue
            df = panel[pair]
            pos = df.index.get_loc(t)
            if pos + 1 < KRONOS_CONTEXT:
                continue  # not enough context; abstain implicitly
            ctx = df.iloc[pos + 1 - KRONOS_CONTEXT: pos + 1]
            future = pd.date_range(t, periods=PRED_LEN, freq="B") + pd.Timedelta(days=1)
            batch_pairs.append(pair)
            dfs.append(ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
            x_ts.append(pd.Series(ctx.index))
            y_ts.append(pd.Series(future))
        if not batch_pairs:
            continue
        torch.manual_seed(SEED_BASE + week_i)
        preds = predictor.predict_batch(
            dfs, x_ts, y_ts,
            pred_len=PRED_LEN, T=KRONOS_T, top_k=KRONOS_TOP_K,
            top_p=KRONOS_TOP_P, sample_count=KRONOS_SAMPLE_COUNT, verbose=False,
        )
        for pair, ctx_df, pred in zip(batch_pairs, dfs, preds):
            last_close = float(ctx_df["close"].iloc[-1])
            pred_close = float(pred["close"].iloc[-1])
            out[pair][t] = kronos_direction_from_pred(pred_close, last_close)
    return out


# --------------------------------------------------------------------------- #
# Ledger: weekly decisions -> merged trades -> metrics
# --------------------------------------------------------------------------- #

def build_weekly(
    pair: str,
    df: pd.DataFrame,
    dates: List[pd.Timestamp],
    directions: Dict[pd.Timestamp, int],
) -> List[WeekDecision]:
    weekly: List[WeekDecision] = []
    for t in dates:
        d = int(directions.get(t, 0))
        base = week_outcome(df, t)
        if base is None:
            continue  # forward bars missing: week unusable for BOTH arms
        weekly.append(WeekDecision(pair, t, d, d * base if d != 0 else None))
    return weekly


def segment_trades(weekly: List[WeekDecision]) -> List[Trade]:
    """Merge consecutive same-direction weeks into trades (prereg S3)."""
    trades: List[Trade] = []
    run: List[WeekDecision] = []

    def flush() -> None:
        if not run:
            return
        gross = 1.0
        for w in run:
            gross *= 1.0 + (w.gross or 0.0)
        trades.append(Trade(
            pair=run[0].pair, direction=run[0].direction,
            entry_date=run[0].date, exit_date=run[-1].date,
            weeks=len(run), gross=gross - 1.0,
        ))
        run.clear()

    for w in sorted(weekly, key=lambda x: x.date):
        if w.direction == 0 or w.gross is None:
            flush()
            continue
        if run and run[-1].direction != w.direction:
            flush()
        run.append(w)
    flush()
    return trades


def weekly_net_series(weekly: List[WeekDecision], cost_bps: float) -> pd.Series:
    """Weekly net returns; full round-trip cost charged on trade-opening weeks."""
    rows: Dict[pd.Timestamp, float] = {}
    prev_dir = 0
    for w in sorted(weekly, key=lambda x: x.date):
        r = w.gross if w.gross is not None else 0.0
        if w.direction != 0 and w.direction != prev_dir:
            r -= cost_bps / 1e4
        rows[w.date] = rows.get(w.date, 0.0) + r
        prev_dir = w.direction
    return pd.Series(rows).sort_index()


def arm_metrics(result: ArmResult, cost_bps: float) -> dict:
    nets = [tr.net(cost_bps) for tr in result.trades]
    wins = sum(1 for n in nets if n > 0)
    taken_weeks = [w for w in result.weekly if w.direction != 0]
    hits = sum(1 for w in taken_weeks if (w.gross or 0.0) > 0)
    return {
        "trades": len(nets),
        "winning_trades": wins,
        "trade_win_rate": wins / len(nets) if nets else None,
        "net_expectancy_per_trade": float(np.mean(nets)) if nets else None,
        "total_net_return": float(np.sum(nets)) if nets else 0.0,
        "weeks_taken": len(taken_weeks),
        "weeks_available": len(result.weekly),
        "participation": len(taken_weeks) / len(result.weekly) if result.weekly else None,
        "weekly_hit_rate": hits / len(taken_weeks) if taken_weeks else None,
    }


def matched_week_test(ax: List[WeekDecision], kr: List[WeekDecision]) -> dict:
    """Two-proportion z on weeks where BOTH arms took a position (prereg S6.2)."""
    ax_by_key = {(w.pair, w.date): w for w in ax}
    both = [
        (a, k) for k in kr
        if (a := ax_by_key.get((k.pair, k.date))) is not None
        and a.direction != 0 and k.direction != 0
    ]
    n = len(both)
    if n == 0:
        return {"n_matched_weeks": 0}
    pa = sum(1 for a, _ in both if (a.gross or 0.0) > 0) / n
    pk = sum(1 for _, k in both if (k.gross or 0.0) > 0) / n
    pool = (pa + pk) / 2.0
    se = math.sqrt(max(2.0 * pool * (1.0 - pool) / n, 1e-18))
    z = (pk - pa) / se
    return {"n_matched_weeks": n, "axiom_hit": pa, "kronos_hit": pk, "z_kronos_minus_axiom": z}


def evaluate_window(
    panel: Dict[str, pd.DataFrame],
    window: Tuple[str, str],
    kronos_dirs: Optional[Dict[str, Dict[pd.Timestamp, int]]],
) -> dict:
    axiom = ArmResult()
    kron = ArmResult()
    for pair, df in panel.items():
        dates = decision_dates(df.index, *window)
        ax_dirs = axiom_directions(df, dates)
        axiom.weekly += build_weekly(pair, df, dates, ax_dirs)
        if kronos_dirs is not None:
            kron.weekly += build_weekly(pair, df, dates, kronos_dirs.get(pair, {}))
    for arm in (axiom, kron):
        by_pair: Dict[str, List[WeekDecision]] = {}
        for w in arm.weekly:
            by_pair.setdefault(w.pair, []).append(w)
        for ws in by_pair.values():
            arm.trades += segment_trades(ws)

    out: dict = {"window": list(window), "cost_scenarios": {}}
    for bps in COST_SCENARIOS_BPS:
        out["cost_scenarios"][str(bps)] = {
            "axiom": arm_metrics(axiom, bps),
            "kronos": arm_metrics(kron, bps) if kronos_dirs is not None else None,
        }
    if kronos_dirs is not None:
        out["matched_weeks"] = matched_week_test(axiom.weekly, kron.weekly)
        ax_net = weekly_net_series(axiom.weekly, COST_PRIMARY_BPS)
        kr_net = weekly_net_series(kron.weekly, COST_PRIMARY_BPS)
        diff = (kr_net - ax_net).dropna()
        out["significance"] = {
            "axiom_dsr": deflated_sharpe_ratio(ax_net, n_trials=1),
            "kronos_dsr": deflated_sharpe_ratio(kr_net, n_trials=1),
            "diff_bootstrap": circular_block_bootstrap_sharpe_pvalue(
                diff, block_size=BOOT_BLOCK, n_reps=BOOT_REPS, seed=BOOT_SEED),
        }
        prim = out["cost_scenarios"][str(COST_PRIMARY_BPS)]
        aw, kw = prim["axiom"]["winning_trades"], prim["kronos"]["winning_trades"]
        headline = "KRONOS" if kw > aw else ("AXIOM" if aw > kw else "TIE")
        boot = out["significance"]["diff_bootstrap"]
        p_diff = boot["p_oos_sharpe_le_zero"] if boot else None
        exp_a = prim["axiom"]["net_expectancy_per_trade"] or float("-inf")
        exp_k = prim["kronos"]["net_expectancy_per_trade"] or float("-inf")
        decisive = False
        if headline == "KRONOS" and p_diff is not None:
            decisive = exp_k > exp_a and p_diff < 0.05
        elif headline == "AXIOM" and p_diff is not None:
            decisive = exp_a > exp_k and (1.0 - p_diff) < 0.05
        out["verdict"] = {
            "headline_more_winning_trades": headline,
            "statistical": "DECISIVE" if decisive else "INCONCLUSIVE_STATISTICALLY",
        }
    return out


# --------------------------------------------------------------------------- #
# Provenance + main
# --------------------------------------------------------------------------- #

def _provenance() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    prereg_digest = "missing"
    if PREREG_DOC.exists():
        prereg_digest = hashlib.sha256(PREREG_DOC.read_bytes()).hexdigest()
    return {
        "experiment_id": EXPERIMENT_ID,
        "repo_commit": _git("rev-parse", "HEAD"),
        "prereg_doc_sha256": prereg_digest,
        "kronos_repo_commit_pinned": KRONOS_REPO_COMMIT,
        "kronos_model_id": KRONOS_MODEL_ID,
        "kronos_tokenizer_id": KRONOS_TOKENIZER_ID,
        "seed_base": SEED_BASE,
        "sample_count": KRONOS_SAMPLE_COUNT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--skip-kronos", action="store_true",
                    help="pipeline validation without HF access (smoke only)")
    ap.add_argument("--panel-dir", type=Path, default=PANEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()
    if args.skip_kronos and args.mode != "smoke":
        ap.error("--skip-kronos is smoke-only: a full run without the challenger is not the experiment")

    pairs = ["EUR_USD", "USD_JPY"] if args.mode == "smoke" else None
    windows = [SMOKE_WINDOW] if args.mode == "smoke" else [PRIMARY_WINDOW, SECONDARY_WINDOW]
    panel = load_panel(args.panel_dir, pairs)
    print(f"panel: {len(panel)} pairs | mode={args.mode}")

    result = {"provenance": _provenance(), "mode": args.mode, "windows": []}
    for window in windows:
        kdirs = None
        if not args.skip_kronos:
            dates_by_pair = {p: decision_dates(df.index, *window) for p, df in panel.items()}
            kdirs = kronos_directions(panel, dates_by_pair)
        result["windows"].append(evaluate_window(panel, window, kdirs))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"result_{args.mode}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    tmp.rename(out_path)
    print(f"RESULT:{json.dumps({'out': str(out_path), 'mode': args.mode})}")
    for w in result["windows"]:
        prim = w["cost_scenarios"][str(COST_PRIMARY_BPS)]
        line = {"window": w["window"], "axiom": prim["axiom"]}
        if prim["kronos"] is not None:
            line["kronos"] = prim["kronos"]
            line["verdict"] = w.get("verdict")
        print(f"RESULT:{json.dumps(line, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
