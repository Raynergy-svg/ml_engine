"""One-shot migration: apply EMA weights to per-pair transformer .keras files.

Phase 5.D's transformer_trainer.py:save() saves MAIN model weights to .keras
but EMA weights only to .ema.pkl. Runtime loads .keras → small-magnitude
outputs in 0.48-0.52 range, never crossing 0.55 confidence gate.

Diagnostic 2026-05-10: EMA-applied EUR_USD lifts accuracy 52.82% → 54.49%.

This script applies ema.pkl positionally to each per-pair .keras file
(matching EMACallback.apply() in callbacks.py:167-172).

Note: pickle.load is used on our OWN trainer artifacts (.ema.pkl is
produced by transformer_trainer.py); these are trusted local files.
"""
from __future__ import annotations
import argparse, logging, os, pickle, shutil, sys, time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("apply_ema")

MODELS_DIR = REPO_ROOT / "trained_data" / "models"
EXCLUDE = {
    "EUR_JPY_oos_train", "EUR_JPY_tuned",
    "EUR_USD_news_first_run", "EUR_USD_news_k32", "EUR_USD_news_lb4",
    "EUR_USD_oos_train", "EUR_USD_pre_tuned_phase5d", "EUR_USD_tuned",
    "USD_CAD_investigation", "USD_CAD_oos_train", "USD_CAD_tuned",
    "USD_JPY_oos_train", "USD_JPY_pre_tuned_phase5d", "USD_JPY_tuned",
    "AUD_NZD", "weight_snapshots", "joint", "shadow",
}


def list_pair_dirs() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDE)


def apply_ema(pair: str, dry_run: bool = False) -> Dict:
    import numpy as np
    from src.utils.keras_model_loader import load_keras_model

    pdir = MODELS_DIR / pair
    keras_p = pdir / "transformer_direction.keras"
    ema_p = pdir / "transformer_direction.ema.pkl"
    bak_p = pdir / "transformer_direction.keras.pre-ema-backup"

    out: Dict = {"pair": pair, "status": "skipped"}
    if not keras_p.exists():
        out["reason"] = "no .keras"; return out
    if not ema_p.exists():
        out["reason"] = "no .ema.pkl"; return out

    log.info(f"[{pair}] loading")
    km, _ = load_keras_model(str(keras_p))
    with open(ema_p, "rb") as f:
        ema = pickle.load(f)
    ema_w = ema.get("ema_weights", [])
    tw = km.trainable_weights

    if len(ema_w) != len(tw):
        out["reason"] = f"count mismatch ({len(ema_w)} ema vs {len(tw)} model)"; return out

    applied = 0
    for w, e in zip(tw, ema_w):
        if w.shape == e.shape:
            w.assign(e.astype(np.float32))
            applied += 1
    out["applied"] = applied
    out["total"] = len(ema_w)

    if dry_run:
        out["status"] = "dry_ok"
        log.info(f"[{pair}] dry-run ok ({applied}/{len(ema_w)})")
        return out

    shutil.copy2(keras_p, bak_p)
    km.save(str(keras_p))
    out["status"] = "applied"
    log.info(f"[{pair}] saved EMA-applied .keras (backup at {bak_p.name})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()] if args.pairs else list_pair_dirs()
    log.info(f"Mode: {'DRY' if args.dry_run else 'WRITE'} | pairs: {pairs}")

    results = []
    t0 = time.time()
    for p in pairs:
        try:
            r = apply_ema(p, dry_run=args.dry_run)
        except Exception as ex:
            log.exception(f"[{p}] failed")
            r = {"pair": p, "status": "error", "reason": str(ex)}
        results.append(r)

    log.info(f"\n=== SUMMARY ({time.time()-t0:.1f}s) ===")
    for r in results:
        log.info(f"  {r['pair']:12} {r.get('status','?'):10} weights={r.get('applied','-')}/{r.get('total','-')} {r.get('reason','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
