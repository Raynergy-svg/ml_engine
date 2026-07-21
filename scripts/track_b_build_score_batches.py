"""Split blinded_tasks/*.txt into fixed-size batch files for parallel subagent
scoring, using DETERMINISTIC modulo sampling (not first-N, to avoid an
alphabetical-ticker bias) when the fetched set exceeds the session's practical
scoring ceiling. The sampling rule is fixed BEFORE any score is seen — no
result-dependent selection.

Usage: python3 scripts/track_b_build_score_batches.py [target_n] [batch_size]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

OUT_DIR = Path("trained_data/research/track_b_postcutoff_2026_07_02")
TASKS_DIR = OUT_DIR / "blinded_tasks"
BATCH_DIR = OUT_DIR / "score_batches"
INDEX_PATH = OUT_DIR / "_index.json"
SELECTION_LOG_PATH = OUT_DIR / "selection_log.json"


def main() -> None:
    target_n = int(sys.argv[1]) if len(sys.argv) > 1 else 999999
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    index = json.loads(INDEX_PATH.read_text())
    all_task_ids = sorted(index.keys())
    n_total = len(all_task_ids)

    if n_total > target_n:
        stride = n_total / target_n
        picked_idx = sorted({int(round(i * stride)) for i in range(target_n)})
        picked_idx = [i for i in picked_idx if i < n_total]
        selected = [all_task_ids[i] for i in picked_idx]
    else:
        selected = all_task_ids

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.txt"):
        old.unlink()

    n_batches = math.ceil(len(selected) / batch_size)
    batch_manifest = []
    for b in range(n_batches):
        batch_tasks = selected[b * batch_size:(b + 1) * batch_size]
        parts = []
        for tid in batch_tasks:
            text = (TASKS_DIR / f"{tid}.txt").read_text(encoding="utf-8")
            parts.append(f"##### TASK_ID: {tid} #####\n{text}\n##### END {tid} #####\n")
        batch_path = BATCH_DIR / f"batch_{b:03d}.txt"
        batch_path.write_text("\n\n".join(parts), encoding="utf-8")
        batch_manifest.append({
            "batch_file": str(batch_path),
            "task_ids": batch_tasks,
            "result_file": str(BATCH_DIR / f"batch_{b:03d}_result.json"),
        })

    SELECTION_LOG_PATH.write_text(json.dumps({
        "n_total_fetched": n_total,
        "n_selected": len(selected),
        "selection_rule": (
            "deterministic modulo sampling over sorted task_id order (task_id "
            "assigned in Wikipedia-alphabetical ticker order), fixed BEFORE any "
            "score was seen -- not an outcome-based selection"
            if n_total > target_n else "all fetched tasks used (no reduction needed)"
        ),
        "target_n": target_n,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "selected_task_ids": selected,
    }, indent=2), encoding="utf-8")

    (BATCH_DIR / "_manifest.json").write_text(json.dumps(batch_manifest, indent=2), encoding="utf-8")
    print(f"n_total_fetched={n_total} n_selected={len(selected)} n_batches={n_batches} "
          f"batch_size={batch_size}")
    for m in batch_manifest:
        print(m["batch_file"], "->", len(m["task_ids"]), "tasks")


if __name__ == "__main__":
    main()
