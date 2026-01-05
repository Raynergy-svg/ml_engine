#!/usr/bin/env python3
"""Replay-evaluate a saved Buddy checkpoint.

Purpose:
- Verify whether historical high validation metrics are reproducible.
- Evaluate the same checkpoint on a different CSV (out-of-sample) using the
  checkpoint's saved feature_columns and standardization stats.

This script intentionally mirrors the windowing semantics used in `main.py`
(`make_sequence_dataset`): drop the final timestep and use the label at the end
of each window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure repository root is on sys.path when running from `scripts/`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_engineering import FeatureEngineering  # noqa: E402
from ml_head_engine import MLEngineHead  # noqa: E402
from mr_engine import MREngineHead  # noqa: E402
from ms_head_engine import MSEngineHead  # noqa: E402
from mt_engine import MTEngineHead  # noqa: E402
from mx_head_engine import MXEngineHead  # noqa: E402


def _load_meta(meta_path: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError("Invalid meta JSON")
    return meta


def _load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "MLEngineHead": MLEngineHead,
            "MREngineHead": MREngineHead,
            "MSEngineHead": MSEngineHead,
            "MTEngineHead": MTEngineHead,
            "MXEngineHead": MXEngineHead,
        },
    )


def _build_features(
    df_raw: pd.DataFrame,
    *,
    train_smoothing: bool,
    median_window: int | None,
    feature_columns: list[str],
    feature_engineering_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    # Keep ordering stable.
    df = df_raw.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    fe = FeatureEngineering(feature_engineering_cfg or {})
    df_fe = fe.create_features(
        df,
        include_all=True,
        apply_candle_smoothing=bool(train_smoothing),
        median_window=median_window,
    )

    numeric_df = df_fe.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Labels must align to the feature rows we keep.
    if "close" not in df_fe.columns:
        raise ValueError("Expected `close` column after feature engineering")
    closes = pd.to_numeric(df_fe["close"], errors="coerce").reindex(numeric_df.index)
    closes_np = closes.to_numpy(dtype=np.float32)

    # Force exact same feature order/shape.
    numeric_df = numeric_df.reindex(columns=feature_columns)
    numeric_df = numeric_df.fillna(0.0)

    x2d = numeric_df.to_numpy(dtype=np.float32)
    return x2d, closes_np


def _compute_direction_and_windows(
    x2d: np.ndarray,
    closes: np.ndarray,
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    seq_len: int,
    mask_top_k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    next_close = closes[1:]
    cur_close = closes[:-1]
    direction = (next_close > cur_close).astype(np.float32)

    # Align features to t->t+1 targets.
    x2d = x2d[:-1]

    n = int(x2d.shape[0])
    if n <= int(seq_len) + 10:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    split = int(n * 0.8)

    feature_mask: np.ndarray | None = None
    if mask_top_k is not None and int(mask_top_k) > 0 and int(mask_top_k) < int(x2d.shape[1]):
        # Mirror `main.py` ranking (train-only): Pearson corr with direction + abs return.
        # Ranking is computed on raw (unstandardized) features.
        x_rank_all = x2d.astype(np.float32, copy=False)
        n_rank = int(len(x_rank_all))
        split_rank = int(n_rank * 0.8)
        if split_rank > 10:
            x_rank = x_rank_all[:split_rank]
            y_dir_rank = direction[:split_rank]
            r0 = (closes[1 : split_rank + 1] - closes[:split_rank]) / np.maximum(closes[:split_rank], 1e-8)
            abs_ret_rank = np.abs(r0).astype(np.float32)

            # Correlations column-wise.
            # Use pandas for convenience/stability on constant columns.
            try:
                x_rank_df = pd.DataFrame(x_rank)
                s_dir = x_rank_df.corrwith(pd.Series(y_dir_rank), method="pearson").abs().fillna(0.0)
                s_conf = x_rank_df.corrwith(pd.Series(abs_ret_rank), method="pearson").abs().fillna(0.0)
                scores = (0.7 * s_dir) + (0.3 * s_conf)
                order = scores.sort_values(ascending=False).index.to_numpy(dtype=np.int32)
                k = int(mask_top_k)
                feature_mask = np.zeros((int(x2d.shape[1]),), dtype=np.float32)
                feature_mask[order[:k]] = 1.0
            except Exception:
                feature_mask = None

    # Standardize using meta's train-only stats.
    x_std = (x2d - mu) / np.where(sigma < 1e-6, 1.0, sigma)
    val_x = x_std[split:]
    val_y = direction[split:]

    if feature_mask is not None:
        val_x = val_x * feature_mask.reshape(1, -1)

    # Match `make_sequence_dataset`: drop the final timestep.
    val_x = val_x[:-1]
    val_y = val_y[:-1]

    n_windows = int(len(val_x) - int(seq_len) + 1)
    if n_windows <= 0:
        raise ValueError("Not enough validation windows")

    y_end = val_y[int(seq_len) - 1 :]
    if int(len(y_end)) != int(n_windows):
        # Defensive: keep consistent with window construction below.
        n_windows = min(int(n_windows), int(len(y_end)))
        y_end = y_end[:n_windows]

    # Materialize windows in batches for memory safety.
    return val_x, y_end


def evaluate_checkpoint(
    *,
    meta_path: Path,
    csv_path: Path,
    batch_size: int,
    config_path: Path | None,
    mask_top_k: int | None,
) -> dict[str, float]:
    meta = _load_meta(meta_path)

    model_path = Path(str(meta.get("model_path", "")))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    feature_columns = [str(c) for c in (meta.get("feature_columns") or [])]
    if not feature_columns:
        raise ValueError("Meta missing feature_columns")

    std = meta.get("standardize") or {}
    mu = np.asarray(std.get("mean"), dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std"), dtype=np.float32).reshape(1, -1)

    seq_len = int(meta.get("seq_len", 50))
    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window", None)
    median_window = int(median_window) if median_window is not None else None

    feature_engineering_cfg: dict[str, Any] | None = None
    if config_path is not None and config_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(config_path.read_text())
            if isinstance(cfg, dict):
                fe_cfg = cfg.get("feature_engineering")
                if isinstance(fe_cfg, dict):
                    feature_engineering_cfg = fe_cfg
        except Exception:
            feature_engineering_cfg = None

    df_raw = pd.read_csv(csv_path)
    x2d, closes = _build_features(
        df_raw,
        train_smoothing=train_smoothing,
        median_window=median_window,
        feature_columns=feature_columns,
        feature_engineering_cfg=feature_engineering_cfg,
    )

    val_x2d, y_end = _compute_direction_and_windows(
        x2d,
        closes,
        mu=mu,
        sigma=sigma,
        seq_len=seq_len,
        mask_top_k=mask_top_k,
    )

    model = _load_model(model_path)

    n_windows = int(len(y_end))
    correct = 0
    conf_sum = 0.0

    for i in range(0, n_windows, int(batch_size)):
        j = min(n_windows, i + int(batch_size))
        xb = np.stack([val_x2d[k : k + seq_len] for k in range(i, j)], axis=0).astype(np.float32, copy=False)
        yb = y_end[i:j]

        pred = model.predict(xb, verbose=0)
        p_dir = np.asarray(pred["direction"]).reshape(-1)
        p_conf = np.asarray(pred["confidence"]).reshape(-1)
        p_conf = np.clip(p_conf, 0.0, 1.0)

        y_hat = (p_dir >= 0.5).astype(np.float32)
        correct += int((y_hat == yb).sum())
        conf_sum += float(p_conf.sum())

    val_acc = float(correct / max(1, n_windows))
    avg_conf = float(conf_sum / max(1, n_windows))

    w_dir = float(meta.get("combined_w_dir", 0.7))
    w_conf = float(meta.get("combined_w_conf", 0.3))
    combined = float((w_dir * val_acc) + (w_conf * avg_conf))

    return {
        "val_direction_accuracy": val_acc,
        "val_avg_confidence": avg_conf,
        "val_combined_score": combined,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay-evaluate a Buddy checkpoint on a CSV.")
    ap.add_argument("--meta", required=True, help="Path to buddy_tf*.meta.json")
    ap.add_argument("--csv", default=None, help="CSV path to evaluate on (defaults to meta['csv_path'])")
    ap.add_argument(
        "--config",
        default=None,
        help="Optional config YAML (used for feature_engineering settings). Defaults to config_tuned.yaml when present.",
    )
    ap.add_argument(
        "--mask-top-k",
        type=int,
        default=None,
        help="Optional: apply a train-only correlation feature mask (top-K) before evaluation (approximates feature curriculum).",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    return ap.parse_args()


def _resolve_config_path(config_arg: str | None) -> Path | None:
    if config_arg is None:
        default_cfg = REPO_ROOT / "config_tuned.yaml"
        return default_cfg if default_cfg.exists() else None
    return Path(config_arg)


def _maybe_print_meta_deltas(meta: dict[str, Any], *, args_csv: str | None) -> None:
    if args_csv is not None:
        return
    for k in ["val_direction_accuracy", "val_avg_confidence", "val_combined_score"]:
        mv = meta.get(k)
        if mv is not None:
            print(f"meta_saved_{k}: {float(mv):.6f}")


def _print_report(
    *,
    meta_path: Path,
    csv_path: Path,
    config_path: Path | None,
    out: dict[str, float],
) -> None:
    print(f"meta: {meta_path}")
    print(f"csv:  {csv_path}")
    if config_path is not None:
        print(f"config: {config_path}")
    for k, v in out.items():
        print(f"{k}: {v:.6f}")


def main() -> int:
    args = _parse_args()

    meta_path = Path(args.meta)
    meta = _load_meta(meta_path)

    csv_path = Path(args.csv) if args.csv else Path(str(meta.get("csv_path", "")))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    config_path = _resolve_config_path(args.config)

    out = evaluate_checkpoint(
        meta_path=meta_path,
        csv_path=csv_path,
        batch_size=int(args.batch_size),
        config_path=config_path,
        mask_top_k=int(args.mask_top_k) if args.mask_top_k is not None else None,
    )

    _print_report(meta_path=meta_path, csv_path=csv_path, config_path=config_path, out=out)

    # If evaluating the same CSV as stored in meta, show delta vs saved values.
    try:
        _maybe_print_meta_deltas(meta, args_csv=args.csv)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# — Raynergy-svg —
