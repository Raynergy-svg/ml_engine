"""Extend the 2026-04-30 rollback backup directory before the follow-up PR.

Adds three new backup classes alongside the existing pre_leak_fix backups:

1. ``confidence_calibration.json.v2``       — current Platt calibrator (v2)
   that the follow-up PR will overwrite with v3 (archive+live calibration).
2. ``trade_journal_rl.json.pre_backfill``   — current live journal (with
   None / 14-feature ``ridge_features``) that the follow-up PR will rewrite
   with backfilled 24-feature vectors.
3. ``per_pair/{INSTRUMENT}_ridge_confidence.pkl.pre_per_pair_retrain``
   — current per-pair confidence models (still the OLD leaky ones because
   the previous PR skipped per-pair confidence). These will be replaced
   by the follow-up PR's per-pair fine-tunes.

Updates ``trained_data/_rollback_2026-04-30/MANIFEST.json`` with sha256
entries for every new backup. Atomic write of the manifest.

Run once at the START of the follow-up PR. Idempotent: if a backup file
already exists with a matching sha256, it is left alone but its manifest
entry is refreshed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAINED = _PROJECT_ROOT / "trained_data"
_ROLLBACK = _TRAINED / "_rollback_2026-04-30"
_PER_PAIR_DIR = _ROLLBACK / "per_pair"
_MANIFEST = _ROLLBACK / "MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_one(src: Path, dst: Path) -> Optional[Dict[str, Any]]:
    """Copy ``src`` to ``dst`` (idempotent), return manifest entry."""
    if not src.exists():
        return None
    src_sha = _sha256(src)
    if dst.exists() and _sha256(dst) == src_sha:
        # Already backed up identically — no-op.
        return {
            "source": str(src.relative_to(_PROJECT_ROOT)),
            "backup": str(dst.relative_to(_ROLLBACK)),
            "size_bytes": src.stat().st_size,
            "sha256_source": src_sha,
            "sha256_backup": src_sha,
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src.relative_to(_PROJECT_ROOT)),
        "backup": str(dst.relative_to(_ROLLBACK)),
        "size_bytes": src.stat().st_size,
        "sha256_source": src_sha,
        "sha256_backup": _sha256(dst),
    }


def _load_manifest() -> Dict[str, Any]:
    if not _MANIFEST.exists():
        return {"created_at": datetime.now(timezone.utc).isoformat(), "files": []}
    try:
        return json.loads(_MANIFEST.read_text())
    except (OSError, ValueError) as exc:
        print(f"WARN: manifest unreadable ({exc}); starting fresh", file=sys.stderr)
        return {"created_at": datetime.now(timezone.utc).isoformat(), "files": []}


def _write_manifest_atomic(payload: Dict[str, Any]) -> None:
    tmp = _MANIFEST.with_suffix(_MANIFEST.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, _MANIFEST)


def _merge_entry(files: List[Dict[str, Any]], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replace any existing entry for the same backup path; preserve order."""
    out = [f for f in files if f.get("backup") != entry["backup"]]
    out.append(entry)
    return out


def main() -> int:
    if not _ROLLBACK.exists():
        print(f"ERROR: rollback dir missing: {_ROLLBACK}", file=sys.stderr)
        return 1

    new_entries: List[Dict[str, Any]] = []

    # 1) Calibration v2 — current state before v3 overwrite
    cal_src = _TRAINED / "confidence_calibration.json"
    cal_dst = _ROLLBACK / "confidence_calibration.json.v2"
    e = _backup_one(cal_src, cal_dst)
    if e:
        new_entries.append(e)
        print(f"backed up: {cal_src} -> {cal_dst} ({e['sha256_source'][:12]})")
    else:
        print(f"WARN: no source for {cal_src}", file=sys.stderr)

    # 2) Live trade journal — pre-backfill snapshot
    j_src = _TRAINED / "trade_journal_rl.json"
    j_dst = _ROLLBACK / "trade_journal_rl.json.pre_backfill"
    e = _backup_one(j_src, j_dst)
    if e:
        new_entries.append(e)
        print(f"backed up: {j_src} -> {j_dst} ({e['sha256_source'][:12]})")
    else:
        print(f"WARN: no source for {j_src}", file=sys.stderr)

    # 3) Per-pair ridge_confidence — pre per-pair retrain snapshot
    _PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    pair_models = sorted((_TRAINED / "models").glob("*/ridge_confidence.pkl"))
    pair_models = [p for p in pair_models if "joint" not in p.parts]
    if not pair_models:
        print("WARN: no per-pair ridge_confidence.pkl found", file=sys.stderr)
    for src in pair_models:
        pair = src.parent.name
        dst = _PER_PAIR_DIR / f"{pair}_ridge_confidence.pkl.pre_per_pair_retrain"
        e = _backup_one(src, dst)
        if e:
            new_entries.append(e)
            print(f"backed up: {pair} pkl ({e['sha256_source'][:12]})")

    # Merge into manifest
    manifest = _load_manifest()
    files = manifest.get("files", [])
    for e in new_entries:
        files = _merge_entry(files, e)
    manifest["files"] = files
    manifest["follow_up_extended_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest_atomic(manifest)

    print(f"\nManifest updated: {_MANIFEST}")
    print(f"Total tracked files: {len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
