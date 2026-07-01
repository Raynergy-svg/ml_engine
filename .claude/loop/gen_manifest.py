#!/usr/bin/env python3
"""gen_manifest.py — regenerate .claude/loop/gate_manifest.json (SHA-256 of every gate script).

Run this AFTER a legitimate change to any enforcement script, and commit the updated manifest WITH
that change (so the diff shows both). verify_gate.py + loop_gate.py check current hashes against this
manifest and fail closed on drift. Usage: python3 gen_manifest.py [--repo PATH]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _integrity import GATE_SCRIPTS  # noqa: E402  (single source of truth for the pinned set)

DEFAULT_REPO = Path("/Users/buddy/Documents/ml_engine")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    a = ap.parse_args()
    repo = Path(a.repo)
    man = {}
    for rel in GATE_SCRIPTS:
        p = repo / rel
        if p.exists():
            man[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    out = repo / ".claude/loop/gate_manifest.json"
    out.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out} with {len(man)} gate hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
