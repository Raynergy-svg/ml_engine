#!/usr/bin/env python3
"""Generate or verify .claude/skills-lock.json.

Pins each skill directory under .claude/skills/ by a content hash so that
silent edits (which can change agent behavior) surface as a lockfile diff
in code review or CI.

Lockfile format mirrors buddy-autonomous-loop/skills-lock.json:
    {
      "version": 1,
      "skills": {
        "<name>": {
          "source": ".claude/skills/<name>",
          "sourceType": "local",        # "github" if pulled from a repo
          "computedHash": "<sha256>"
        }
      }
    }

Hash protocol: sha256 over the sorted list of (relative_path, file_sha256)
pairs inside the skill directory. Stable across platforms and re-clones.

Usage:
    python scripts/lock_skills.py            # write/refresh lockfile
    python scripts/lock_skills.py --check    # exit 1 if any skill drifts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
LOCK_PATH = REPO_ROOT / ".claude" / "skills-lock.json"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_skill_hash(skill_dir: Path) -> str:
    """Hash all files in a skill directory deterministically."""
    entries = []
    for p in sorted(skill_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(skill_dir).as_posix()
            entries.append(f"{rel}:{_file_sha256(p)}")
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_lockfile() -> dict:
    skills = {}
    for d in sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir()):
        skills[d.name] = {
            "source": f".claude/skills/{d.name}",
            "sourceType": "local",
            "computedHash": compute_skill_hash(d),
        }
    return {"version": 1, "skills": skills}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify the lockfile matches current skills; exit 1 on drift.",
    )
    args = ap.parse_args()

    current = build_lockfile()

    if args.check:
        if not LOCK_PATH.exists():
            print(f"ERROR: {LOCK_PATH} missing. Run scripts/lock_skills.py to create it.", file=sys.stderr)
            return 1
        try:
            locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"ERROR: {LOCK_PATH} is malformed: {e}", file=sys.stderr)
            return 1
        if locked != current:
            print("ERROR: skill drift detected vs .claude/skills-lock.json", file=sys.stderr)
            locked_skills = locked.get("skills", {})
            for name, meta in current["skills"].items():
                old = locked_skills.get(name, {}).get("computedHash")
                if old != meta["computedHash"]:
                    print(f"  drift: {name}  locked={old}  current={meta['computedHash']}", file=sys.stderr)
            for name in locked_skills:
                if name not in current["skills"]:
                    print(f"  removed: {name}", file=sys.stderr)
            print("Run scripts/lock_skills.py to refresh the lockfile after intentional edits.", file=sys.stderr)
            return 1
        print(f"OK: {len(current['skills'])} skills match lockfile.")
        return 0

    LOCK_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {LOCK_PATH} ({len(current['skills'])} skills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
