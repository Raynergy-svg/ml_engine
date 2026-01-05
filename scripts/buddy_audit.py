"""Audit Buddy training artifacts.

Recomputes validation metrics from the saved `buddy_tf.keras` + `buddy_tf.meta.json`,
checks weight/gradient sanity, and summarizes calibration signals.

Intended to be run locally:
  conda run -n ml_engine python scripts/buddy_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Ensure repo root is on sys.path when running via `conda run`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _rankdata(a: np.ndarray) -> np.ndarray:
    idx = a.argsort(kind="mergesort")
    ranks = np.empty_like(idx, dtype=np.float64)
    ranks[idx] = np.arange(len(a), dtype=np.float64)
    return ranks


def main() -> int:
    meta_path = Path("trained_data/models/buddy_tf.meta.json")
    meta = json.loads(meta_path.read_text())

    import tensorflow as tf

    # Buddy models can include custom layers (e.g., MLEngineHead). Import and pass
    # them explicitly so Keras can deserialize the .keras artifact.
    try:
        from ml_head_engine import MLEngineHead  # type: ignore
        from mr_engine import MREngineHead  # type: ignore
        from ms_head_engine import MSEngineHead  # type: ignore
        from mt_engine import MTEngineHead  # type: ignore
        from mx_head_engine import MXEngineHead  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import Buddy custom head classes required to load buddy_tf.keras"
        ) from e

    model_path = Path(str(meta["model_path"]))
    model = tf.keras.models.load_model(
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

    seq_len = int(meta["seq_len"])
    feature_columns = list(meta["feature_columns"])
    mu = np.asarray(meta["standardize"]["mean"], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(meta["standardize"]["std"], dtype=np.float32).reshape(1, -1)

    csv_path = Path(str(meta["csv_path"]))
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    from utils import load_config

    cfg = load_config("config_tuned.yaml")
    from feature_engineering import FeatureEngineering

    all_features = bool(meta.get("all_features", True))
    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window")
    median_window = int(median_window) if median_window is not None else None

    if all_features:
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=train_smoothing,
            median_window=median_window,
        )

    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    missing = [c for c in feature_columns if c not in numeric_df.columns]
    extra = [c for c in numeric_df.columns if c not in set(feature_columns)]
    if missing:
        print(f"Reindex: {len(missing)} missing features filled with 0.0")
    if extra:
        print(f"Reindex: {len(extra)} extra features dropped")

    numeric_df = numeric_df.reindex(columns=feature_columns, fill_value=0.0)

    closes = df.loc[numeric_df.index, "close"].to_numpy(dtype=np.float32)
    direction_y_all = (closes[1:] > closes[:-1]).astype(np.float32)

    feats = numeric_df.to_numpy(dtype=np.float32)[:-1]
    n = int(feats.shape[0])
    if n != int(direction_y_all.shape[0]):
        raise RuntimeError(f"label/feature mismatch: feats={n} dir={direction_y_all.shape[0]}")

    if mu.shape[1] != feats.shape[1]:
        raise RuntimeError(f"Feature dim mismatch: meta mu={mu.shape[1]} vs feats={feats.shape[1]}")

    split = int(n * 0.8)
    val_feats = ((feats - mu) / sigma)[split:].astype(np.float32, copy=False)
    val_dir = direction_y_all[split:].astype(np.float32, copy=False)

    # Match make_sequence_dataset semantics (drop last row)
    x2d_eff = val_feats[:-1]
    y_dir_eff = val_dir[:-1]
    n_windows = int(len(x2d_eff) - seq_len + 1)
    if n_windows <= 0:
        raise RuntimeError("Not enough validation rows")

    y_dir_end = y_dir_eff[seq_len - 1 :]

    from numpy.lib.stride_tricks import sliding_window_view

    batch = 256
    p_dir_all = np.empty((n_windows,), dtype=np.float32)
    p_conf_all = np.empty((n_windows,), dtype=np.float32)

    for start in range(0, n_windows, batch):
        end = min(n_windows, start + batch)
        chunk = x2d_eff[start : end + seq_len - 1]
        win = sliding_window_view(chunk, window_shape=(seq_len, chunk.shape[1]))
        win = win.reshape(-1, seq_len, chunk.shape[1]).astype(np.float32, copy=False)

        pred = model.predict(win, verbose=0)
        p_dir = np.asarray(pred["direction"]).reshape(-1)
        p_conf = np.asarray(pred["confidence"]).reshape(-1)

        p_dir_all[start:end] = p_dir[: end - start]
        p_conf_all[start:end] = np.clip(p_conf[: end - start], 0.0, 1.0)

    pred_dir_bin = (p_dir_all >= 0.5).astype(np.float32)
    val_acc = float(np.mean(pred_dir_bin == y_dir_end))
    avg_conf = float(np.mean(p_conf_all))

    w_dir = float(meta.get("combined_w_dir", 0.7))
    w_conf = float(meta.get("combined_w_conf", 0.3))
    combined = (w_dir * val_acc) + (w_conf * avg_conf)

    print("\n=== Recomputed validation metrics (from saved model) ===")
    print(f"val_direction_accuracy: {val_acc*100:.2f}%")
    print(f"val_avg_confidence:     {avg_conf*100:.2f}%")
    print(f"val_combined_score:     {combined*100:.2f}%")

    print("\n=== Meta-recorded validation metrics ===")
    print(f"meta val_direction_accuracy: {float(meta['val_direction_accuracy'])*100:.2f}%")
    print(f"meta val_avg_confidence:     {float(meta['val_avg_confidence'])*100:.2f}%")
    print(f"meta val_combined_score:     {float(meta['val_combined_score'])*100:.2f}%")
    print(f"meta best_epoch:             {int(meta.get('best_epoch') or -1)}")

    correct = (pred_dir_bin == y_dir_end).astype(np.float32)
    qs = np.quantile(p_conf_all, np.linspace(0, 1, 11))
    qs = np.unique(qs)
    rows = []
    for i in range(1, len(qs)):
        lo, hi = float(qs[i - 1]), float(qs[i])
        if hi <= lo + 1e-12:
            continue
        if i == 1:
            m = (p_conf_all >= lo) & (p_conf_all <= hi)
        else:
            m = (p_conf_all > lo) & (p_conf_all <= hi)
        n_i = int(m.sum())
        if n_i < 20:
            continue
        rows.append((n_i, float(np.mean(p_conf_all[m])), float(np.mean(correct[m])), lo, hi))

    rows.sort(key=lambda r: r[1])
    print("\n=== Confidence bins vs direction accuracy ===")
    print(" n    mean_conf   dir_acc   [lo,hi]")
    for n_i, mc, acc, lo, hi in rows:
        print(f"{n_i:4d}  {mc:9.4f}  {acc:7.4f}  [{lo:.4f},{hi:.4f}]")
    if len(rows) >= 3:
        mean_confs = np.asarray([r[1] for r in rows], dtype=np.float64)
        accs = np.asarray([r[2] for r in rows], dtype=np.float64)
        corr = float(np.corrcoef(mean_confs, accs)[0, 1])
        print(f"bin-level corr(mean_conf, dir_acc): {corr:+.3f}")

    cal = (meta.get("tier2") or {}).get("calibration")
    if cal and "bins" in cal:
        ts = cal.get("temperature_scaling") if isinstance(cal, dict) else None
        if isinstance(ts, dict):
            print("\n=== Tier-2 temperature scaling (meta) ===")
            print(f"enabled: {bool(ts.get('enabled', False))}")
            if ts.get("T") is not None:
                try:
                    print(f"T: {float(ts['T']):.6g}")
                except Exception:
                    print(f"T: {ts.get('T')}")
            if ts.get("nll_pre") is not None or ts.get("nll_post") is not None:
                print(f"NLL: {ts.get('nll_pre')} -> {ts.get('nll_post')}")
            if ts.get("spearman_pre") is not None or ts.get("spearman_post") is not None:
                print(f"Spearman(score,y): {ts.get('spearman_pre')} -> {ts.get('spearman_post')}")

        bins = cal["bins"]
        scores = np.asarray([b["score"] for b in bins], dtype=np.float64)
        pwin = np.asarray([b["p_win"] for b in bins], dtype=np.float64)
        rs = _rankdata(scores)
        rp = _rankdata(pwin)
        spearman = float(np.corrcoef(rs, rp)[0, 1])
        weights = np.asarray([b["n"] for b in bins], dtype=np.float64)
        ece_like = float(np.average(np.abs(pwin - scores), weights=weights))
        print("\n=== Tier-2 calibration sanity ===")
        print(f"bin-level Spearman(score, p_win): {spearman:+.3f}")
        print(f"ECE-like (|p_win-score|, weighted): {ece_like:.3f}  (score=dir_conf*p_conf)")
        print(f"overall win_rate: {float(cal.get('win_rate', float('nan')))*100:.2f}% (n={int(cal.get('n',0))})")

    print("\n=== Weight sanity (trained model) ===")
    nonfinite = 0
    max_abs = 0.0
    for w in model.weights:
        arr = w.numpy()
        if not np.isfinite(arr).all():
            nonfinite += int(np.size(arr) - np.isfinite(arr).sum())
        max_abs = max(max_abs, float(np.max(np.abs(arr))))
    print(f"total weights: {len(model.weights)} | non-finite elements: {nonfinite} | max |w|: {max_abs:.4g}")

    stats = []
    for w in model.weights:
        arr = w.numpy().astype(np.float64, copy=False)
        stats.append((float(np.max(np.abs(arr))), float(np.std(arr)), w.name, int(arr.size)))
    stats.sort(reverse=True, key=lambda t: t[0])
    print("\nTop tensors by max |w|:")
    for mx, sd, name, size in stats[:10]:
        print(f"{mx:10.4g}  std={sd:10.4g}  n={size:8d}  {name}")

    print("\n=== Gradient norm check (single batch, direction loss) ===")
    x_batch = tf.convert_to_tensor(
        sliding_window_view(x2d_eff[: (batch + seq_len - 1)], window_shape=(seq_len, x2d_eff.shape[1]))
        .reshape(-1, seq_len, x2d_eff.shape[1])
        .astype(np.float32, copy=False)
    )
    y_batch = tf.convert_to_tensor(y_dir_end[: x_batch.shape[0]].reshape(-1, 1).astype(np.float32))

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    with tf.GradientTape() as tape:
        out = model(x_batch, training=True)
        loss = bce(y_batch, tf.cast(out["direction"], tf.float32))
    grads = tape.gradient(loss, model.trainable_variables)
    finite_grads = [g for g in grads if g is not None]
    norms = []
    for g in finite_grads:
        gnp = g.numpy()
        norms.append(float(np.linalg.norm(gnp.reshape(-1), ord=2)) if np.isfinite(gnp).all() else float("nan"))
    if norms:
        norms_np = np.asarray(norms, dtype=np.float64)
        print(f"trainable vars: {len(model.trainable_variables)} | grads computed: {len(finite_grads)}")
        print(
            f"grad L2 norm: min={np.nanmin(norms_np):.4g} median={np.nanmedian(norms_np):.4g} max={np.nanmax(norms_np):.4g}"
        )

    print("\n=== Baseline vs trained (random init clone, subset) ===")
    clone = tf.keras.models.clone_model(model)
    _ = clone(tf.zeros((1, seq_len, int(mu.shape[1])), dtype=tf.float32))

    sub_n = min(n_windows, 2000)
    sub_x = tf.convert_to_tensor(
        sliding_window_view(x2d_eff[: (sub_n + seq_len - 1)], window_shape=(seq_len, x2d_eff.shape[1]))
        .reshape(-1, seq_len, x2d_eff.shape[1])
        .astype(np.float32, copy=False)
    )
    sub_y = y_dir_end[:sub_n]

    pred_t = model.predict(sub_x, verbose=0)
    pred_r = clone.predict(sub_x, verbose=0)
    acc_t = float(np.mean((np.asarray(pred_t["direction"]).reshape(-1) >= 0.5) == sub_y))
    acc_r = float(np.mean((np.asarray(pred_r["direction"]).reshape(-1) >= 0.5) == sub_y))
    print(f"subset windows: {sub_n}")
    print(f"trained dir_acc: {acc_t*100:.2f}%")
    print(f"random  dir_acc: {acc_r*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# — Raynergy-svg —
