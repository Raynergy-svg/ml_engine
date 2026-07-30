#!/usr/bin/env python3
"""Compute and persist the 60-forward-weekday Risk-target P2 readiness state.

The readiness COMPUTATION already existed and is correct
(``ForwardCaptureService.evaluate_p2_readiness``); what did not exist was a
WRITER. Without one, ``trained_data/forward_capture/p2_readiness.json`` was
never produced, so ``dashboard/server/training_cockpit.py`` permanently rendered
``readiness_unavailable`` and ``scripts/run_forward_capture_daily.py`` hard-failed
this job every night (audit §3.2, §6, gap G2).

This script closes the producer side only. It persists the report through
``ControlStateStore`` (compare-and-swap + ``flock`` + ``fsync`` + atomic rename)
at exactly the path the dashboard already reads:

    <control-root>/p2_readiness.json      (default: trained_data/forward_capture/)

ANALYSIS ONLY. Nothing here trades, unhalts, promotes, or touches the scan loop.

Exit codes (must stay inside ``run_forward_capture_daily.py``'s accepted set
``{0, 3}`` for this job):

    0  ready      — every required pair AND exposure have >= the minimum number
                    of forward-eligible weekdays; report written.
    3  not ready  — still accumulating forward evidence; report written with
                    per-pair ``blocking_reasons``. This is the normal nightly
                    outcome today and is NOT a job failure.
    1  failure    — readiness could not be computed or could not be persisted.
                    Deliberately outside the accepted set so the nightly job
                    surfaces it instead of silently writing nothing.

Examples:
    python scripts/check_risk_target_p2_readiness.py
    python scripts/check_risk_target_p2_readiness.py --pairs EUR_USD,USD_JPY
    python scripts/check_risk_target_p2_readiness.py --minimum-days 60 --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_platform.forward_capture import (  # noqa: E402
    DEFAULT_CONTROL_ROOT,
    DEFAULT_P2_MIN_TRADING_DAYS,
    ForwardCaptureService,
    P2ReadinessReport,
)
from src.data_platform.platform import ControlStateConflict  # noqa: E402
from src.evidence.hashing import sha256_bytes  # noqa: E402

logger = logging.getLogger("check_risk_target_p2_readiness")

CONTROL_STATE_NAME = "p2_readiness"
READINESS_FILENAME = f"{CONTROL_STATE_NAME}.json"

EXIT_READY = 0
EXIT_FAILURE = 1
EXIT_ACCUMULATING = 3

# The P2 tick contract must cover exactly the pairs tick capture streams;
# a readiness gate over a narrower list would report "ready" for a universe the
# training job does not have. Kept in lockstep with ``ALL_MAJOR_PAIRS`` in
# ``scripts/run_tick_capture.py`` (the ``--pairs ALL_FX`` list used by
# ``com.buddy.tick_capture``); ``tests/test_forward_capture_defect_fixes.py``
# fails if the two ever drift apart.
P2_REQUIRED_PAIRS: tuple[str, ...] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "EUR_CHF", "GBP_CHF", "AUD_JPY", "EUR_AUD", "GBP_AUD",
    "AUD_NZD", "CAD_JPY", "CHF_JPY", "EUR_CAD", "NZD_JPY",
)


def resolve_required_pairs(raw: str | None) -> tuple[str, ...]:
    """Parse a ``--pairs`` / env override, falling back to the P2 contract list."""
    source = raw if raw is not None else os.getenv("AXIOM_P2_REQUIRED_PAIRS")
    if not source or source.strip().upper() == "ALL_FX":
        return P2_REQUIRED_PAIRS
    pairs = tuple(item.strip().upper() for item in source.split(",") if item.strip())
    if not pairs:
        raise ValueError("--pairs was provided but resolved to an empty list")
    return pairs


def persist_readiness(
    service: ForwardCaptureService,
    report: P2ReadinessReport,
    control_root: Path,
    *,
    attempts: int = 3,
) -> Path:
    """Atomically persist ``report`` to ``<control_root>/p2_readiness.json``.

    Uses ``ControlStateStore`` so the write is compare-and-swap guarded under an
    exclusive lock and lands via ``fsync`` + ``os.replace``. A concurrent writer
    (two nightly runs overlapping) loses the CAS and is retried against the fresh
    digest rather than clobbering the other run's bytes.
    """
    store = service.platform.control_state_store(control_root)
    path = control_root / READINESS_FILENAME
    payload: dict[str, Any] = report.model_dump(mode="json", round_trip=True)
    last_error: ControlStateConflict | None = None
    for _attempt in range(max(1, attempts)):
        expected_digest = sha256_bytes(path.read_bytes()) if path.exists() else None
        try:
            store.write(CONTROL_STATE_NAME, payload, expected_digest=expected_digest)
        except ControlStateConflict as exc:
            last_error = exc
            continue
        return path
    raise ControlStateConflict(
        f"p2 readiness control state remained contended at {path}: {last_error}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Comma-separated pairs or 'ALL_FX' (default: the 20-pair P2 contract list)",
    )
    parser.add_argument(
        "--minimum-days",
        type=int,
        default=int(os.getenv("AXIOM_P2_MIN_TRADING_DAYS", DEFAULT_P2_MIN_TRADING_DAYS)),
        help=f"Minimum forward-eligible weekdays required (default: {DEFAULT_P2_MIN_TRADING_DAYS})",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=None,
        help="Directory holding p2_readiness.json (default: trained_data/forward_capture/)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the JSON summary on stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    control_root = Path(
        args.control_root
        or os.getenv("AXIOM_FORWARD_CAPTURE_ROOT", str(DEFAULT_CONTROL_ROOT))
    )

    try:
        required_pairs = resolve_required_pairs(args.pairs)
        if args.minimum_days < 1:
            raise ValueError("--minimum-days must be >= 1")
        service = ForwardCaptureService.default()
        report = service.evaluate_p2_readiness(
            required_pairs, minimum_trading_days=args.minimum_days
        )
    except Exception as exc:  # surfaced as exit 1, never as a silent "not ready"
        logger.exception("P2 readiness evaluation failed: %s", exc)
        return EXIT_FAILURE

    try:
        path = persist_readiness(service, report, control_root)
    except Exception as exc:
        logger.exception("P2 readiness report could not be persisted: %s", exc)
        return EXIT_FAILURE

    summary = {
        "path": str(path),
        "ready": report.ready,
        "minimum_trading_days": report.minimum_trading_days,
        "exposure_trading_days": report.exposure_trading_days,
        "required_pairs": list(report.required_pairs),
        "tick_trading_days_by_pair": dict(report.tick_trading_days_by_pair),
        "blocking_reasons": list(report.blocking_reasons),
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if report.ready:
        logger.info("P2 readiness: READY — report written to %s", path)
        return EXIT_READY
    logger.info(
        "P2 readiness: accumulating forward evidence (%d blocking reason(s)) — report written to %s",
        len(report.blocking_reasons), path,
    )
    return EXIT_ACCUMULATING


if __name__ == "__main__":
    raise SystemExit(main())
