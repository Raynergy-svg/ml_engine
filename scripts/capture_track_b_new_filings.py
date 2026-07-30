#!/usr/bin/env python3
"""Capture newly-filed Track B SEC documents through the canonical capture plane.

``scripts/run_forward_capture_daily.py`` has referenced this script since the
``com.buddy.forward_daily`` LaunchAgent was written, but the file never existed —
so that job exited 2 every night and no filing ever reached the canonical
contract on a schedule (audit §3.2, gap G8).

WHAT THIS ACTUALLY DOES (no fake capture — the ingestion is real and pre-existing):

  1. Universe: current S&P 500 constituents via ``src.equity.sp500_membership``
     (the same source ``scripts/track_b_fetch_and_blind_postcutoff.py`` used at
     503-name scale), or an explicit ``--tickers`` / ``--universe-file`` list.
  2. Ticker -> CIK via SEC's official ``company_tickers.json``
     (``src.equity.edgar_fundamentals.load_cik_map``).
  3. Cheap pre-filter: read each issuer's submissions index and pick the most
     recent qualifying filing with ``filingDate <= today`` using the loader's own
     PIT gate (``select_pit_filing``). Only filings NEWER than the per-ticker
     watermark AND filed within ``--max-age-days`` are followed up — this is what
     makes it a "new filings" job instead of a full re-download every night.
  4. Canonical write: ``src.equity.research.pit_text_loader.load_pit_filing``,
     which already hooks ``ForwardCaptureService.capture_track_b_filing`` (raw
     filing bytes into the immutable ``filings`` domain + a forward-capture
     history record). ``capture_track_b_filing`` marks a filing FORWARD only when
     it is <= 7 days old, so the ``--max-age-days`` default of 7 matches the
     contract that decides training eligibility.
  5. Watermark: last captured accession per ticker, persisted atomically through
     ``ControlStateStore`` (CAS + flock + fsync + rename).

ANALYSIS ONLY: network reads from SEC/Wikipedia and disk writes under
``axiom-data/`` + ``trained_data/``. Nothing trades, unhalts, or promotes.

Exit codes (``run_forward_capture_daily.py`` accepts only ``{0}`` for this job,
so anything non-zero is deliberately a visible nightly failure):

    0  run completed; every followed-up filing reached the canonical plane
       (zero new filings is a success, not a failure)
    1  hard failure: universe or CIK map unavailable, or the per-ticker failure
       ratio exceeded ``--max-failure-ratio``
    4  at least one canonical capture FAILED after its document was fetched
       (the filing exists only in the legacy text cache — needs a re-run).
       Deliberately not 2, which is what the interpreter itself returns when the
       script file is missing — the exact failure this script was written to end.

Examples:
    python scripts/capture_track_b_new_filings.py
    python scripts/capture_track_b_new_filings.py --tickers AAPL,MSFT --max-age-days 30
    python scripts/capture_track_b_new_filings.py --universe-file my_universe.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests  # noqa: E402

from src.data_platform.forward_capture import (  # noqa: E402
    DEFAULT_CONTROL_ROOT,
    ForwardCaptureService,
)
from src.data_platform.platform import ControlStateConflict, DataIntegrityError  # noqa: E402
from src.equity.edgar_fundamentals import load_cik_map  # noqa: E402
from src.equity.research import pit_text_loader  # noqa: E402
from src.evidence.hashing import sha256_bytes  # noqa: E402

logger = logging.getLogger("capture_track_b_new_filings")

WATERMARK_STATE_NAME = "track_b_filing_watermark"
DEFAULT_CACHE_DIR = REPO_ROOT / "trained_data" / "research" / "track_b_edgar_cache"
DEFAULT_MAX_AGE_DAYS = 7  # matches capture_track_b_filing's FORWARD window

EXIT_OK = 0
EXIT_HARD_FAILURE = 1
EXIT_CAPTURE_FAILED = 4


def load_universe(
    tickers: Optional[str], universe_file: Optional[Path]
) -> Tuple[str, ...]:
    """Resolve the issuer universe. Raises on failure — never silently narrows."""
    if tickers:
        parsed = tuple(item.strip().upper() for item in tickers.split(",") if item.strip())
        if not parsed:
            raise ValueError("--tickers was provided but resolved to an empty list")
        return parsed
    if universe_file is not None:
        payload = json.loads(Path(universe_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("tickers") or payload.get("members") or []
        parsed = tuple(
            str(item.get("ticker") if isinstance(item, dict) else item).strip().upper()
            for item in payload
        )
        parsed = tuple(sorted({item for item in parsed if item}))
        if not parsed:
            raise ValueError(f"universe file {universe_file} contained no tickers")
        return parsed
    from src.equity.sp500_membership import _fetch_wiki_tables, _parse_current

    constituents, _changes = _fetch_wiki_tables()
    parsed = tuple(_parse_current(constituents))
    if not parsed:
        raise RuntimeError("S&P 500 constituent table parsed to an empty universe")
    return parsed


def latest_filing_row(
    session: requests.Session, cik10: str, as_of: str, forms: Sequence[str]
) -> Optional[Dict[str, str]]:
    """Most recent qualifying filing from the issuer's ``recent`` submissions block.

    Index-only: no primary document is fetched here. Returns ``None`` when EDGAR
    is unreachable for this issuer or nothing qualifies. Deliberately does NOT
    page into ``filings.files[]`` — a *new* filing is always in the recent block,
    and paging deep history would turn a nightly delta job into a full backfill.
    """
    url = pit_text_loader.SUBMISSIONS_URL.format(name=f"CIK{cik10}.json")
    submissions = pit_text_loader._get_json(session, url)
    if not isinstance(submissions, dict):
        return None
    recent = (submissions.get("filings") or {}).get("recent") or {}
    row = pit_text_loader.select_pit_filing(
        pit_text_loader._iter_recent_rows(recent), as_of, forms
    )
    return None if row is None else {str(k): str(v) for k, v in row.items()}


def read_watermarks(store: Any, path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return store.read(WATERMARK_STATE_NAME)
    except (DataIntegrityError, OSError, ValueError) as exc:
        logger.error(
            "track_b watermark at %s is unreadable (%s) — treating every ticker as "
            "unseen for this run; the file is left in place for inspection",
            path, exc,
        )
        return {}


def write_watermarks(
    store: Any, path: Path, payload: Dict[str, Any], *, attempts: int = 3
) -> None:
    last_error: Optional[ControlStateConflict] = None
    for _attempt in range(max(1, attempts)):
        expected_digest = sha256_bytes(path.read_bytes()) if path.exists() else None
        try:
            store.write(WATERMARK_STATE_NAME, payload, expected_digest=expected_digest)
        except ControlStateConflict as exc:
            last_error = exc
            continue
        return
    raise ControlStateConflict(f"track_b watermark remained contended at {path}: {last_error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: S&P 500)")
    parser.add_argument("--universe-file", type=Path, default=None, help="JSON list/obj of tickers")
    parser.add_argument(
        "--forms", type=str, default="10-K,10-Q,8-K", help="Comma-separated EDGAR forms"
    )
    parser.add_argument(
        "--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
        help=f"Only follow up filings filed within N days (default: {DEFAULT_MAX_AGE_DAYS})",
    )
    parser.add_argument("--max-tickers", type=int, default=0, help="Bound the universe (0 = no bound)")
    parser.add_argument(
        "--max-failure-ratio", type=float, default=0.25,
        help="Hard-fail if this fraction of tickers errored (default: 0.25)",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--control-root", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report which filings are new without fetching documents or capturing",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the JSON summary on stdout")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    )

    if args.max_age_days < 0:
        logger.error("--max-age-days must be >= 0")
        return EXIT_HARD_FAILURE
    forms = tuple(item.strip() for item in args.forms.split(",") if item.strip())
    if not forms:
        logger.error("--forms resolved to an empty list")
        return EXIT_HARD_FAILURE

    control_root = Path(
        args.control_root
        or os.getenv("AXIOM_FORWARD_CAPTURE_ROOT", str(DEFAULT_CONTROL_ROOT))
    )
    as_of = datetime.now(timezone.utc).date()
    cutoff = (as_of - timedelta(days=args.max_age_days)).isoformat()
    as_of_iso = as_of.isoformat()

    session = requests.Session()
    try:
        universe = load_universe(args.tickers, args.universe_file)
    except Exception as exc:
        logger.exception(
            "Track B universe unavailable (%s) — refusing to run against a partial "
            "universe; pass --universe-file or --tickers to proceed offline", exc,
        )
        return EXIT_HARD_FAILURE
    if args.max_tickers > 0:
        universe = universe[: args.max_tickers]

    try:
        cik_map = load_cik_map(session=session)
    except Exception as exc:
        logger.exception("SEC company_tickers.json unavailable: %s", exc)
        return EXIT_HARD_FAILURE
    if not cik_map:
        logger.error("SEC company_tickers.json returned an empty CIK map — hard failure")
        return EXIT_HARD_FAILURE

    service = ForwardCaptureService.default()
    store = service.platform.control_state_store(control_root)
    watermark_path = control_root / f"{WATERMARK_STATE_NAME}.json"
    watermarks = read_watermarks(store, watermark_path)

    pit_text_loader.reset_track_b_capture_failures()
    outcomes: Dict[str, str] = {}
    captured: List[Dict[str, str]] = []
    errors: Dict[str, str] = {}

    for ticker in universe:
        cik = cik_map.get(ticker) or cik_map.get(ticker.replace("-", "."))
        if not cik:
            outcomes[ticker] = "no_cik"
            continue
        try:
            time.sleep(pit_text_loader._REQUEST_SPACING_S)
            row = latest_filing_row(session, cik, as_of_iso, forms)
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
            outcomes[ticker] = "error"
            logger.warning("track_b: submissions lookup failed for %s: %s", ticker, exc)
            continue
        if row is None:
            outcomes[ticker] = "no_filing"
            continue
        accession = row.get("accessionNumber", "")
        filed = row.get("filingDate", "")
        previous = watermarks.get(ticker)
        if isinstance(previous, dict) and previous.get("accession") == accession:
            outcomes[ticker] = "already_captured"
            continue
        if filed < cutoff:
            outcomes[ticker] = "not_new"
            continue
        if args.dry_run:
            outcomes[ticker] = "would_capture"
            captured.append({"ticker": ticker, "accession": accession, "filed": filed, "form": row.get("form", "")})
            continue
        try:
            filing = pit_text_loader.load_pit_filing(
                ticker, cik, as_of_iso, forms=forms, session=session, cache_dir=args.cache_dir,
            )
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
            outcomes[ticker] = "error"
            logger.warning("track_b: filing fetch failed for %s: %s", ticker, exc)
            continue
        if filing is None:
            outcomes[ticker] = "no_filing"
            continue
        outcomes[ticker] = "captured"
        captured.append({"ticker": ticker, "accession": accession, "filed": filing.filed, "form": filing.form})
        watermarks[ticker] = {"accession": accession, "filed": filing.filed, "form": filing.form}

    capture_failures = pit_text_loader.track_b_capture_failures()

    if not args.dry_run:
        try:
            write_watermarks(store, watermark_path, watermarks)
        except Exception as exc:
            logger.exception("track_b watermark could not be persisted: %s", exc)
            return EXIT_HARD_FAILURE

    error_ratio = (len(errors) / len(universe)) if universe else 0.0
    summary: Dict[str, Any] = {
        "as_of": as_of_iso,
        "cutoff": cutoff,
        "universe_size": len(universe),
        "dry_run": bool(args.dry_run),
        "captured_count": len(captured),
        "captured": captured[:200],
        "outcome_counts": {
            value: sum(1 for item in outcomes.values() if item == value)
            for value in sorted(set(outcomes.values()))
        },
        "error_count": len(errors),
        "error_ratio": round(error_ratio, 4),
        "errors": dict(list(errors.items())[:50]),
        "canonical_capture_failures": list(capture_failures),
        "watermark_path": str(watermark_path),
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))

    if capture_failures:
        logger.error(
            "%d filing(s) were fetched but FAILED canonical capture — they exist only "
            "in the legacy text cache", len(capture_failures),
        )
        return EXIT_CAPTURE_FAILED
    if error_ratio > args.max_failure_ratio:
        logger.error(
            "track_b: %d/%d tickers errored (%.1f%% > %.1f%% ceiling)",
            len(errors), len(universe), error_ratio * 100, args.max_failure_ratio * 100,
        )
        return EXIT_HARD_FAILURE
    logger.info(
        "track_b: %d new filing(s) captured from %d issuers (%d errors)",
        len(captured), len(universe), len(errors),
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
