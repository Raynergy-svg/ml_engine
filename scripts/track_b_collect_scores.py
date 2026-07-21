"""Collect batch_*_result.json files (written by scoring subagents) into a
single raw_scores.json keyed by task_id. A task_id absent from every batch
result (or present but malformed) is an ABSTENTION -- dropped, never
zero-filled -- consistent with the inference-contract refusal rule used
throughout this repo. Prints a summary so abstentions are visible, not silent.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("trained_data/research/track_b_postcutoff_2026_07_02")
BATCH_DIR = OUT_DIR / "score_batches"
RAW_SCORES_PATH = OUT_DIR / "raw_scores.json"

REQUIRED_FIELDS = ("fundamental_quality", "accounting_red_flags", "forward_outlook", "conviction")


def _valid(fields: dict) -> bool:
    if not isinstance(fields, dict):
        return False
    for k in REQUIRED_FIELDS:
        if k not in fields:
            return False
        v = fields[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
    lo_hi = {
        "fundamental_quality": (-1.0, 1.0),
        "accounting_red_flags": (0.0, 1.0),
        "forward_outlook": (-1.0, 1.0),
        "conviction": (0.0, 1.0),
    }
    for k, (lo, hi) in lo_hi.items():
        if not (lo <= float(fields[k]) <= hi):
            return False
    return True


def main() -> None:
    manifest = json.loads((BATCH_DIR / "_manifest.json").read_text())
    expected_task_ids = set()
    for m in manifest:
        expected_task_ids.update(m["task_ids"])

    raw_scores: dict = {}
    malformed = []
    missing_result_file = []
    for m in manifest:
        result_path = Path(m["result_file"])
        if not result_path.exists():
            missing_result_file.append(result_path.name)
            continue
        try:
            payload = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            malformed.append(result_path.name)
            continue
        for tid, fields in payload.items():
            if tid not in expected_task_ids:
                continue
            if _valid(fields):
                raw_scores[tid] = {k: float(fields[k]) for k in REQUIRED_FIELDS}
                raw_scores[tid]["rationale"] = str(fields.get("rationale", ""))
                raw_scores[tid]["spans_used"] = [str(s) for s in fields.get("spans_used", [])]
            else:
                malformed.append(f"{tid} (in {result_path.name})")

    abstained = sorted(expected_task_ids - set(raw_scores.keys()))
    RAW_SCORES_PATH.write_text(json.dumps(raw_scores, indent=2, sort_keys=True))
    print(f"expected={len(expected_task_ids)} scored={len(raw_scores)} "
          f"abstained_or_malformed={len(abstained)} missing_result_files={len(missing_result_file)}")
    if malformed:
        print("malformed:", malformed[:20])
    if missing_result_file:
        print("missing result files:", missing_result_file)
    if abstained:
        print("abstained task_ids:", abstained[:30])


if __name__ == "__main__":
    main()
