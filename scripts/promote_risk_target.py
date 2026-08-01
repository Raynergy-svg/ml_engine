#!/usr/bin/env python3
"""Promote one QUARANTINED risk-target evidence package to lane champion.

Thin CLI over :class:`src.evidence.promotion.PromotionService` (Phase L).
Requires EXPLICIT operator approval — no ``--operator-approve`` reason, no
promotion. Exits nonzero on any refusal; prints the promotion record on
success.

The ship gate is enforced inside the service (CLAUDE.md Hard NO #3): a head
whose pre-registered EvaluationReport gate result is FAIL is refused — for
risk-target that means the forward-volatility head may promote only if its
gate passed, and the drawdown head (which fails its bar) is always refused.

``--dry-run`` runs the identical verification pipeline READ-ONLY: it resolves
the package, re-verifies every digest, enforces the ship gate, checks the
feature-pipeline contract, and runs the load + dry-inference smoke — but
writes NOTHING (no ledger event, no pointer, no artifact copy, no key
minting, no JSONL entry) and needs no operator approval because it cannot
promote.

Usage:
    python scripts/promote_risk_target.py <package_digest> --dry-run
    python scripts/promote_risk_target.py <package_digest> \
        --operator-approve "validated on 2026-07 holdout, forward-vol gate PASS"
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evidence.promotion import (  # noqa: E402
    InferenceProbe,
    OperatorApproval,
    PromotionError,
    PromotionService,
    load_or_create_promotion_identities,
    preflight,
)
from src.evidence.risk_target.persistent_identity import (  # noqa: E402
    load_or_create_slice_identities,
)
from src.evidence.risk_target.slice import build_evidence_store  # noqa: E402
from src.evidence.store import EvidenceStoreError  # noqa: E402

DEFAULT_STORE_ROOT = REPO_ROOT / "trained_data" / "evidence"
DEFAULT_KEY_DIR = REPO_ROOT / "trained_data" / "axiom" / "signing"


def _risk_target_probe() -> InferenceProbe:
    """Real load + dry-inference for pickled risk-target LightGBM heads.

    The bytes are unpickled only AFTER the store has re-verified their digest
    against the producer-signed package (see evaluation.py security note).
    """

    def load(data: bytes) -> object:
        return pickle.loads(data)

    def predict(model: object) -> object:
        import numpy as np

        n_features = getattr(model, "n_features_", None)
        if n_features is None and hasattr(model, "booster_"):
            n_features = model.booster_.num_feature()
        if n_features is None and hasattr(model, "n_features_in_"):
            n_features = model.n_features_in_
        if n_features is None:
            raise ValueError("cannot determine model feature count for dry inference")
        sample = np.zeros((1, int(n_features)), dtype=float)
        return model.predict(sample)  # type: ignore[attr-defined]

    return InferenceProbe(load=load, predict=predict)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_digest", help="sha256 digest of the evidence package to promote")
    parser.add_argument(
        "--operator-approve",
        metavar="REASON",
        default=None,
        help=(
            "explicit operator approval reason, recorded in the promotion event. "
            "REQUIRED to promote; ignored with --dry-run"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="READ-ONLY preflight: verify everything, write nothing, promote nothing",
    )
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--key-dir", type=Path, default=DEFAULT_KEY_DIR)
    parser.add_argument("--lane", default=None, help="expected lane_id (refuses on mismatch)")
    parser.add_argument("--artifact-id", default=None, help="artifact to promote (default: the sole artifact)")
    parser.add_argument(
        "--expected-feature-pipeline-version",
        default=None,
        help="consumer's expected feature pipeline version (default: runtime risk-target constant)",
    )
    args = parser.parse_args()

    reason = (args.operator_approve or "").strip()
    if not args.dry_run and not reason:
        print(
            "REFUSED: promotion requires --operator-approve with a non-empty reason "
            "(use --dry-run for a read-only check)",
            file=sys.stderr,
        )
        return 2

    expected_version = args.expected_feature_pipeline_version
    if expected_version is None:
        from src.training.risk_target_features import RISK_TARGET_FEATURE_PIPELINE_VERSION

        expected_version = RISK_TARGET_FEATURE_PIPELINE_VERSION

    now = datetime.now(timezone.utc)

    if args.dry_run:
        # Read-only path: refuse to mint anything — the identities and the
        # store must already exist from a real evidence-slice run.
        from src.evidence.risk_target.persistent_identity import IDENTITY_METADATA_FILENAME

        if not (Path(args.key_dir) / IDENTITY_METADATA_FILENAME).exists():
            print(
                f"REFUSED: no signing identities at {args.key_dir}; "
                "run the evidence slice first (dry-run never creates keys)",
                file=sys.stderr,
            )
            return 2
        if not Path(args.store_root).is_dir():
            print(f"REFUSED: no evidence store at {args.store_root}", file=sys.stderr)
            return 2
        identities = load_or_create_slice_identities(args.key_dir, now=now)
        store = build_evidence_store(args.store_root, identities, clock_now=now)
        try:
            report = preflight(
                store,
                args.package_digest,
                probe=_risk_target_probe(),
                expected_feature_pipeline_version=expected_version,
                expected_lane_id=args.lane,
                artifact_id=args.artifact_id,
            )
        except (PromotionError, EvidenceStoreError) as exc:
            print(f"REFUSED (dry-run): {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0

    identities = load_or_create_slice_identities(args.key_dir, now=now)
    promotion_identities = load_or_create_promotion_identities(args.key_dir, now=now)
    promotion_identities.register(identities.trust_store, identities.authorities)
    store = build_evidence_store(args.store_root, identities, clock_now=now)

    from src.evidence.contracts import AuthorityRole

    service = PromotionService(
        store,
        operator_signer=promotion_identities.operator,
        operator_actor_id=promotion_identities.actors[AuthorityRole.OPERATOR],
        promotion_signer=promotion_identities.promotion,
        promotion_actor_id=promotion_identities.actors[AuthorityRole.PROMOTION_SERVICE],
    )
    try:
        record = service.promote(
            args.package_digest,
            approval=OperatorApproval(reason=reason),
            probe=_risk_target_probe(),
            expected_feature_pipeline_version=expected_version,
            expected_lane_id=args.lane,
            artifact_id=args.artifact_id,
            now=now,
        )
    except (PromotionError, EvidenceStoreError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(record.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
