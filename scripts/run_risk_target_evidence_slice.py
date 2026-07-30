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
from src.evidence.risk_target.persistent_identity import (  # noqa: E402
    load_or_create_slice_identities,
)
from src.evidence.risk_target.slice import (  # noqa: E402
    build_evidence_store,
    risk_target_evidence_view,
    run_risk_target_evidence_slice,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTOR_DIR = REPO_ROOT / "market_data" / "factor"
DATASET_DESCRIPTOR = FACTOR_DIR / "axiom_risk_target_dataset.json"
DEFAULT_OUT = REPO_ROOT / "trained_data" / "evidence"
# Private signing keys live under trained_data/axiom/ (gitignored — never
# commit a private key). Persistent identities keep the durable store
# verifiable across processes; see persistent_identity.py.
DEFAULT_KEY_DIR = REPO_ROOT / "trained_data" / "axiom" / "signing"


def _load_descriptor() -> dict:
    try:
        return json.loads(DATASET_DESCRIPTOR.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("dataset descriptor unreadable (%s); falling back to *_D.csv glob", exc)
        return {}


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
    descriptor = _load_descriptor()
    descriptor_pairs = sorted(descriptor.get("partitions", {}))
    default_pairs = descriptor_pairs or sorted(
        p.name[:-len("_D.csv")] for p in FACTOR_DIR.glob("*_D.csv")
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=default_pairs)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--key-dir", type=Path, default=DEFAULT_KEY_DIR)
    parser.add_argument("--oos-start", default="2024-01-01")
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args()

    partitions = _load_partitions(args.pairs)
    now = datetime.now(timezone.utc)
    params = EvaluationParams(oos_start=args.oos_start)
    dataset_id = args.dataset_id or descriptor.get(
        "dataset_id", f"fx-daily-{now.strftime('%Y%m%dT%H%M%SZ')}"
    )
    coverage_start = date.fromisoformat(descriptor.get("coverage_start", "2014-01-01"))
    coverage_end = date.fromisoformat(descriptor.get("coverage_end", now.date().isoformat()))
    git_commit = _git_commit()

    identities = load_or_create_slice_identities(args.key_dir, now=now)
    store = build_evidence_store(args.out, identities, clock_now=now)

    logger.info("Running risk-target evidence slice over %d pairs -> %s", len(partitions), args.out)
    result = run_risk_target_evidence_slice(
        store, identities, partitions,
        dataset_id=dataset_id,
        coverage_start=coverage_start, coverage_end=coverage_end,
        retrieved_at=now, created_at=now, git_commit=git_commit, params=params,
    )

    reports_by_lane = {
        head.lane_id: head.evaluation_report_envelope.payload
        for head in result.worker_output.heads
    }
    summary = {
        "dataset_id": dataset_id,
        "dataset_manifest_digest": result.dataset_manifest_envelope.payload_digest,
        "job_manifest_digest": result.job_envelope.payload_digest,
        "git_commit": git_commit,
        "pairs": sorted(partitions),
        "store_root": str(args.out),
        "outcomes": {
            lane: {
                "final_state": outcome.final_state.value,
                "decision": outcome.verdict.decision.value,
                "reason": outcome.reason,
                "package_digest": outcome.package_digest,
                "disposition_head_digest": outcome.disposition_head_digest,
                "package_dir": str(args.out / "packages" / outcome.package_digest),
                "ledger_dir": str(args.out / "dispositions" / outcome.package_digest),
                "failed_checks": [c.check_id for c in outcome.verdict.checks if not c.passed],
                "gates": [
                    {
                        "gate_id": g["gate_id"],
                        "status": g["status"],
                        "observed": g["observed"],
                        "threshold": g["threshold"],
                    }
                    for g in reports_by_lane.get(lane, {}).get("gates", [])
                ],
                "metrics": {
                    name: metric["value"]
                    for name, metric in reports_by_lane.get(lane, {}).get("metrics", {}).items()
                },
                "head_passed": reports_by_lane.get(lane, {}).get("passed"),
            }
            for lane, outcome in result.outcomes.items()
        },
    }
    print(json.dumps(summary, indent=2))
    logger.info("Evidence cockpit view:\n%s", json.dumps(risk_target_evidence_view(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
