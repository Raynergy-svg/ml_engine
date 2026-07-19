#!/usr/bin/env python
"""One-time AUDITED supersession repair for the hedge twin-lane ledger.

Context (2026-07-19 re-review item 2): the canonical evidence view now FAILS
CLOSED on two rows claiming one (strategy, asof_date) evidence period unless
the later row carries an explicit ``supersedes`` pointer. The committed
ledger contains one such pair written BEFORE the supersession mechanism
existed: equity_harvester @ 2026-07-01, captured once on the operator's
machine (2026-07-08 cycle, weights not recorded) and once by the step-1
reconcile cycle (2026-07-18, weights recorded). Their MATERIAL evidence —
realized returns, costs, hedge decision, exposure, notional, review flags —
is identical; only provenance (cycle_ts, meta.source_path) and the presence
of the weights book differ.

This script performs the EXPLICIT supersession the fail-closed rule demands:

* For each period with multiple rows it verifies the material evidence
  matches EXACTLY (via ``_material_evidence``). Any disagreement ABORTS with
  a diff — a genuine conflict is a human decision, never a script's.
* It stamps the LATER row ``supersedes`` = the prior row's book identity
  (or the explicit ``pre_identity_row`` sentinel when the prior recorded no
  weights), plus a ``supersession_reason`` documenting why.
* The rewrite is atomic (tmp + fsync + rename) under the ledger's
  cross-process lock, and the ledger is git-tracked — the repair is fully
  reviewable in the commit diff.

Running it again is a no-op (already-stamped pairs are skipped). It never
deletes a row: superseded revisions stay in the ledger as history; only the
canonical ACTIVE view excludes them.

Usage:
    python scripts/repair_hedge_ledger_supersession.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evidence.forward_ledger import ledger_lock, read_rows  # noqa: E402
from src.hedge.hedged_shadow_lane import (  # noqa: E402
    RAW_VS_HEDGED_LEDGER_PATH,
    _material_evidence,
    _row_identity,
)


def _is_degraded_recapture(prior: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    """True when the LATER row is a strict information-subset of the prior:
    every material field either matches or is null where the prior resolved
    a value. Such a row asserts no evidence of its own — it may be VOIDED
    (never supersede the resolved revision)."""
    p = json.loads(_material_evidence(prior))
    c = json.loads(_material_evidence(cur))

    def _subset(pv: Any, cv: Any) -> bool:
        if cv is None and pv is not None:
            return True
        if isinstance(pv, dict) and isinstance(cv, dict):
            return set(cv) <= set(pv) and all(_subset(pv[k], cv[k]) for k in cv)
        return pv == cv
    return _subset(p, c) and p != c


def repair(path: Path) -> Dict[str, Any]:
    """Stamp explicit supersession on same-period rows whose material
    evidence is identical. Returns a summary; raises SystemExit(2) on any
    genuine (material) conflict."""
    with ledger_lock(path):
        rows = read_rows(path)                 # strict: corruption fails closed
        by_period: Dict[Tuple[str, str], List[int]] = {}
        for i, row in enumerate(rows):
            key = (str(row.get("strategy", "")), str(row.get("asof_date", "")))
            by_period.setdefault(key, []).append(i)

        stamped: List[Dict[str, Any]] = []
        for key, idxs in by_period.items():
            if len(idxs) < 2:
                continue
            for prev_i, cur_i in zip(idxs, idxs[1:]):
                prior, cur = rows[prev_i], rows[cur_i]
                if cur.get("supersedes") or cur.get("voided") is True:
                    continue                   # already explicit — no-op
                now = datetime.now(timezone.utc).isoformat()
                if _material_evidence(prior) == _material_evidence(cur):
                    cur["supersedes"] = (_row_identity(prior)[2]
                                         or "pre_identity_row")
                    cur["supersession_reason"] = (
                        "2026-07-19 audited repair: same evidence period "
                        "recorded twice before the supersession mechanism "
                        f"existed (prior cycle_ts={prior.get('cycle_ts')!r}); "
                        "material evidence verified identical — later capture "
                        "is the active revision, prior kept as history")
                    cur["supersession_stamped_at"] = now
                    stamped.append({"period": list(key), "action": "supersedes",
                                    "supersedes": cur["supersedes"],
                                    "prior_cycle_ts": prior.get("cycle_ts"),
                                    "active_cycle_ts": cur.get("cycle_ts")})
                elif _is_degraded_recapture(prior, cur):
                    cur["voided"] = True
                    cur["voided_reason"] = (
                        "2026-07-19 audited repair: unresolved re-capture "
                        "(null marks) of an evidence period already resolved "
                        f"by the prior row (cycle_ts={prior.get('cycle_ts')!r}) "
                        "— asserts no evidence and must never overwrite "
                        "resolved marks; excluded from the active view, kept "
                        "as history")
                    cur["voided_stamped_at"] = now
                    stamped.append({"period": list(key), "action": "voided",
                                    "prior_cycle_ts": prior.get("cycle_ts"),
                                    "voided_cycle_ts": cur.get("cycle_ts")})
                else:
                    print("GENUINE CONFLICT — material evidence differs for "
                          f"{key!r}; refusing to stamp. Human decision "
                          "required.\n"
                          f"  prior: {_material_evidence(prior)}\n"
                          f"  later: {_material_evidence(cur)}")
                    raise SystemExit(2)

        if stamped:
            fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                                       prefix=".ledger_", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, str(path))
        return {"n_rows": len(rows), "n_stamped": len(stamped), "stamped": stamped}


def main() -> int:
    summary = repair(RAW_VS_HEDGED_LEDGER_PATH)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["n_stamped"] == 0:
        print("no-op: no unstamped same-period pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
