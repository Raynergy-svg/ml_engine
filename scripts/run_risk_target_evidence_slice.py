#!/usr/bin/env python3
"""Run the Phase I risk-target evidence vertical slice end-to-end.

Realizes the roadmap §12 first-success criterion against the real cached daily
FX panel (``market_data/factor/*_D.csv``):

    one exact dataset -> one signed job -> one isolated run ->
    one immutable package -> one independently reproduced verdict ->
    one locally governed disposition

Offline/research only. Reads cached CSVs, writes signed evidence to
``trained_data/evidence/`` (or ``--out``). Never calls a broker, never touches
``.claude/state.json``, never promotes a champion — the slice stops at
QUARANTINED / REJECTED.

Usage:
    python scripts/run_risk_target_evidence_slice.py
    python scripts/run_risk_target_evidence_slice.py --pairs EUR_USD USD_JPY --out /tmp/ev
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evidence.risk_target.evaluation import EvaluationParams  # noqa: E402
from src.evidence.risk_target.slice import (  # noqa: E402
    build_evidence_store,
    build_slice_identities,
    risk_target_evidence_view,
    run_risk_target_evidence_slice,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_DIR = REPO_ROOT / "market_data" / "factor"
DEFAULT_PAIRS = ["AUD_USD", "EUR_USD", "GBP_USD", "USD_CAD", "USD_CHF", "USD_JPY"]
DEFAULT_OUT = REPO_ROOT / "trained_data" / "evidence"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        if len(out) >= 40 and all(c in "0123456789abcdef" for c in out.lower()):
            return out.lower()
    except (subprocess.SubprocessError, OSError):
        pass
    return "0" * 40


def _load_partitions(pairs: list[str]) -> dict[str, bytes]:
    partitions: dict[str, bytes] = {}
    for pair in pairs:
        path = FACTOR_DIR / f"{pair}_D.csv"
        if not path.exists():
            logger.warning("skipping %s — no cached CSV at %s", pair, path)
            continue
        partitions[pair] = path.read_bytes()
    if not partitions:
        raise SystemExit(f"no cached FX daily CSVs found under {FACTOR_DIR}")
    return partitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args()

    partitions = _load_partitions(args.pairs)
    now = datetime.now(timezone.utc)
    params = EvaluationParams(oos_start=args.oos_start)
    dataset_id = args.dataset_id or f"fx-daily-{now.strftime('%Y%m%dT%H%M%SZ')}"

    identities = build_slice_identities(now)
    store = build_evidence_store(args.out, identities, clock_now=now)

    logger.info("Running risk-target evidence slice over %d pairs -> %s", len(partitions), args.out)
    result = run_risk_target_evidence_slice(
        store, identities, partitions,
        dataset_id=dataset_id,
        coverage_start=date(2010, 1, 1), coverage_end=now.date(),
        retrieved_at=now, created_at=now, git_commit=_git_commit(), params=params,
    )

    summary = {
        "dataset_id": dataset_id,
        "git_commit": _git_commit(),
        "pairs": sorted(partitions),
        "outcomes": {
            lane: {
                "final_state": outcome.final_state.value,
                "decision": outcome.verdict.decision.value,
                "reason": outcome.reason,
                "package_digest": outcome.package_digest,
                "failed_checks": [c.check_id for c in outcome.verdict.checks if not c.passed],
            }
            for lane, outcome in result.outcomes.items()
        },
    }
    print(json.dumps(summary, indent=2))
    logger.info("Evidence cockpit view:\n%s", json.dumps(risk_target_evidence_view(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
