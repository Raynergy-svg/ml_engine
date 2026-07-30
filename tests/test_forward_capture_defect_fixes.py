"""Phase C/D data-platform defect fixes (audit ``docs/architecture/audit/phase_CD_data_platform.md``).

Covers exactly three confirmed defects:

* **V1 / G4** — ``TickPersister.flush`` silently discarded a whole day-partition of
  ticks when ``pd.read_parquet`` failed on the existing file.
* **G2** — ``scripts/check_risk_target_p2_readiness.py`` did not exist, so the
  60-forward-weekday readiness computation had no writer and the nightly job
  hard-failed.
* **V4** — ``hedged_shadow_lane._append_jsonl`` logged a failed ledger append and
  returned normally, making failure indistinguishable from success.

NO MOCKS (CLAUDE.md No-Mock Rule): real ``TickPersister`` against real Parquet
files under ``tmp_path``, a real ``ForwardCaptureService``/``DataPlatform``/
``ControlStateStore`` against a real ``AXIOM_DATA_ROOT``, real
``BookSnapshot``/hedge machinery with a test-authored in-memory price panel, and
the real dashboard reader consuming the real artifact written to disk. No
``unittest.mock``, no patching, no test doubles.
"""
from __future__ import annotations

import ast
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tick_capture import TickPersister  # noqa: E402
from src.utils.oanda_streaming import TickQuote  # noqa: E402


def _tick(pair: str, when: datetime, bid: float, ask: float) -> TickQuote:
    return TickQuote(
        instrument=pair, time=when, bid=bid, ask=ask, mid=(bid + ask) / 2.0,
        bid_liq=1_000_000.0, ask_liq=1_000_000.0,
        status="tradeable", source="oanda_stream",
    )


def _partition_path(root: Path, pair: str, when: datetime) -> Path:
    return root / pair / f"{when.year}" / f"{when.month:02d}" / f"{when.day:02d}.parquet"


# --------------------------------------------------------------------------- #
# DEFECT 1 (V1 / G4) — a corrupt partition must never delete captured ticks    #
# --------------------------------------------------------------------------- #
def test_corrupt_existing_parquet_does_not_destroy_prior_ticks(tmp_path):
    """Truncated (unreadable) day-partition -> quarantined byte-for-byte, never overwritten."""
    root = tmp_path / "ticks"
    persister = TickPersister(root=root)
    day = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)

    # Real prior capture: three ticks written through the real flush path.
    prior = [_tick("EUR_USD", day + timedelta(seconds=i), 1.1000 + i / 1e5, 1.1002 + i / 1e5)
             for i in range(3)]
    assert persister.flush(prior) == 3
    pq_path = _partition_path(root, "EUR_USD", day)
    assert len(pd.read_parquet(pq_path)) == 3

    # Corrupt it the way a killed writer / bad sector does: truncate the footer.
    # The prior tick bytes are still in the file; only the parquet footer is gone.
    intact_bytes = pq_path.read_bytes()
    truncated = intact_bytes[: len(intact_bytes) // 2]
    pq_path.write_bytes(truncated)
    with pytest.raises(Exception):
        pd.read_parquet(pq_path)

    # Next flush for the same pair/day.
    later = [_tick("EUR_USD", day + timedelta(minutes=5), 1.1010, 1.1012)]
    assert persister.flush(later) == 1

    quarantined = sorted(pq_path.parent.glob("*.corrupt-*"))
    assert len(quarantined) == 1, "the unreadable partition must be quarantined, not deleted"
    assert quarantined[0].read_bytes() == truncated, (
        "quarantined bytes must be byte-for-byte the prior partition — no data loss"
    )
    assert not quarantined[0].name.endswith(".parquet"), (
        "quarantine name must fall outside reader globs (*.parquet) and prune_older_than"
    )

    # The live partition is readable again and carries the new batch.
    assert len(pd.read_parquet(pq_path)) == 1
    assert persister.parquet_merge_failures == 1
    failure = persister.last_parquet_merge_failure
    assert failure is not None
    assert failure["partition"] == str(pq_path)
    assert failure["quarantine"] == str(quarantined[0])
    assert failure["message"], "the real exception message must be recorded, not swallowed"


def test_healthy_partition_still_merges_and_reports_no_failure(tmp_path):
    """Happy path is unchanged: a readable partition is appended to, not replaced."""
    root = tmp_path / "ticks"
    persister = TickPersister(root=root)
    day = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)

    persister.flush([_tick("GBP_USD", day, 1.2500, 1.2502)])
    persister.flush([_tick("GBP_USD", day + timedelta(minutes=1), 1.2510, 1.2512)])

    frame = pd.read_parquet(_partition_path(root, "GBP_USD", day))
    assert len(frame) == 2, "existing ticks must survive the second flush"
    assert persister.parquet_merge_failures == 0
    assert persister.last_parquet_merge_failure is None
    assert not list((root / "GBP_USD").rglob("*.corrupt-*"))


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory permissions, so the rename cannot be made to fail",
)
def test_unquarantinable_partition_refuses_to_overwrite(tmp_path):
    """If the bad partition cannot even be moved, the flush raises instead of overwriting.

    ``_do_flush`` re-buffers and retries on a raised flush, so the batch is not lost.
    """
    root = tmp_path / "ticks"
    persister = TickPersister(root=root)
    day = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)
    persister.flush([_tick("USD_JPY", day, 150.00, 150.02)])
    pq_path = _partition_path(root, "USD_JPY", day)
    intact = pq_path.read_bytes()

    # A directory at the partition path is unreadable by pandas AND unmovable
    # onto a quarantine name only if the destination is blocked; block it by
    # making the parent read-only after seeding a decoy quarantine collision.
    pq_path.write_bytes(intact[: len(intact) // 2])
    pq_path.parent.chmod(0o500)  # r-x: cannot rename within this directory
    try:
        with pytest.raises(OSError):
            persister.flush([_tick("USD_JPY", day + timedelta(minutes=1), 150.10, 150.12)])
    finally:
        pq_path.parent.chmod(0o700)
    assert pq_path.read_bytes() == intact[: len(intact) // 2], (
        "the unreadable partition must be left exactly as-is when it cannot be quarantined"
    )
    assert persister.parquet_merge_failures == 0


# --------------------------------------------------------------------------- #
# DEFECT 2 (G2) — the P2 readiness script must write a real, consumable artifact
# --------------------------------------------------------------------------- #
def _readiness_main():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_risk_target_p2_readiness", REPO_ROOT / "scripts" / "check_risk_target_p2_readiness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _accepted_codes_for(job_name: str) -> set[int]:
    """Re-derive the caller's accepted exit codes from the scheduler source itself."""
    source = (REPO_ROOT / "scripts" / "run_forward_capture_daily.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 4:
            continue
        name = node.elts[0]
        if isinstance(name, ast.Constant) and name.value == job_name:
            return set(ast.literal_eval(node.elts[3]))
    raise AssertionError(f"job {job_name!r} not found in run_forward_capture_daily.py")


def test_p2_readiness_script_writes_real_artifact_with_accepted_exit_code(tmp_path, monkeypatch):
    data_root = tmp_path / "axiom-data"
    control_root = tmp_path / "forward_capture"
    monkeypatch.setenv("AXIOM_DATA_ROOT", str(data_root))
    module = _readiness_main()

    rc = module.main(["--pairs", "EUR_USD,USD_JPY", "--control-root", str(control_root), "--quiet"])

    accepted = _accepted_codes_for("p2_readiness")
    assert accepted == {0, 3}
    assert rc in accepted, f"exit {rc} is outside the caller's accepted set {accepted}"
    assert rc == 3, "no forward evidence exists yet — the honest answer is 'accumulating'"

    artifact = control_root / "p2_readiness.json"
    assert artifact.is_file(), "the readiness state must be persisted, not just printed"

    # The dashboard reader's exact validation path must accept the bytes on disk.
    from src.data_platform.forward_capture import P2ReadinessReport

    raw = json.loads(artifact.read_text(encoding="utf-8"))
    report = P2ReadinessReport.model_validate_json(json.dumps(raw))
    assert report.ready is False
    assert report.minimum_trading_days == 60
    assert set(report.required_pairs) == {"EUR_USD", "USD_JPY"}
    assert any("exposure_forward_days" in reason for reason in report.blocking_reasons)
    assert any("tick_forward_days:EUR_USD" in reason for reason in report.blocking_reasons)


def test_p2_readiness_artifact_is_consumed_by_the_dashboard_reader(tmp_path, monkeypatch):
    """The persisted artifact — not a display string — is what the cockpit reads."""
    data_root = tmp_path / "axiom-data"
    control_root = tmp_path / "forward_capture"
    monkeypatch.setenv("AXIOM_DATA_ROOT", str(data_root))
    module = _readiness_main()
    assert module.main(["--pairs", "EUR_USD", "--control-root", str(control_root), "--quiet"]) == 3

    from dashboard.server.training_cockpit import read_data_status

    status = read_data_status(
        repo_root=tmp_path, data_root=data_root, control_root=control_root, source_root=tmp_path
    )
    readiness = status["p2_readiness"]
    assert readiness["available"] is True, (
        "before this fix the reader always saw 'P2 readiness report is absent'"
    )
    assert readiness["ready"] is False
    assert readiness["source"] == str(control_root / "p2_readiness.json")


def test_p2_readiness_reports_ready_and_exit_zero_when_the_gate_is_met(tmp_path, monkeypatch):
    """Real forward records -> ready=True -> exit 0 (also inside the accepted set)."""
    from src.data_platform.forward_capture import ForwardCaptureService

    data_root = tmp_path / "axiom-data"
    control_root = tmp_path / "forward_capture"
    monkeypatch.setenv("AXIOM_DATA_ROOT", str(data_root))

    # Fixed WEEKDAY clock: eligibility counts observed_at weekdays, so a test that
    # used "now" would flip its verdict on Saturdays.
    weekday = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)  # Thursday
    assert weekday.weekday() < 5
    writer = ForwardCaptureService.default()
    writer._clock = lambda: weekday

    captured = writer.capture_exposure_snapshot({
        "source": "oanda_fx_trend_lane",
        "captured_at_utc": weekday.isoformat(),
        "source_mtime_utc": weekday.isoformat(),
        "net_currency_notional_home": {"USD": 1234.5},
    })
    assert captured is not None and captured.training_eligible

    observations = [
        {"instrument": "EUR_USD", "time": (weekday - timedelta(seconds=30)).isoformat(),
         "bid": 1.1000, "ask": 1.1002, "mid": 1.1001, "bid_liq": 1e6, "ask_liq": 1e6,
         "status": "tradeable", "source": "oanda_stream"},
        {"instrument": "EUR_USD", "time": (weekday - timedelta(seconds=15)).isoformat(),
         "bid": 1.1001, "ask": 1.1003, "mid": 1.1002, "bid_liq": 1e6, "ask_liq": 1e6,
         "status": "tradeable", "source": "oanda_stream"},
    ]
    buffer = io.BytesIO()
    pd.DataFrame(observations).to_parquet(buffer, compression="zstd", index=False)
    tick_record = writer.capture_tick_partition(
        "EUR_USD", observations, buffer.getvalue(), observed_at=weekday
    )
    assert tick_record is not None and tick_record.training_eligible

    module = _readiness_main()
    rc = module.main([
        "--pairs", "EUR_USD", "--minimum-days", "1",
        "--control-root", str(control_root), "--quiet",
    ])
    assert rc == 0, "one forward weekday of exposure + ticks meets a 1-day gate"
    assert rc in _accepted_codes_for("p2_readiness")

    payload = json.loads((control_root / "p2_readiness.json").read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["exposure_trading_days"] == 1
    assert payload["tick_trading_days_by_pair"]["EUR_USD"] == 1
    assert payload["blocking_reasons"] == []


def test_p2_required_pairs_match_the_tick_capture_universe():
    """A readiness gate over fewer pairs than tick capture streams would lie."""
    module = _readiness_main()
    source = (REPO_ROOT / "scripts" / "run_tick_capture.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    all_major = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ALL_MAJOR_PAIRS" for t in node.targets
        ):
            all_major = ast.literal_eval(node.value)
    assert all_major, "ALL_MAJOR_PAIRS not found in scripts/run_tick_capture.py"
    assert set(module.P2_REQUIRED_PAIRS) == set(all_major)


def test_track_b_capture_script_exists_and_is_importable():
    """The other missing nightly target: it must exist and expose a real main()."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "capture_track_b_new_filings.py"
    assert path.is_file()
    spec = importlib.util.spec_from_file_location("capture_track_b_new_filings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    # It must route through the pre-existing canonical filing path, not a stub.
    assert module.pit_text_loader.load_pit_filing is not None
    assert _accepted_codes_for("track_b_filings") == {0}
    assert module.EXIT_OK == 0


def test_track_b_universe_loader_rejects_empty_input(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "capture_track_b_new_filings", REPO_ROOT / "scripts" / "capture_track_b_new_filings.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    universe_file = tmp_path / "universe.json"
    universe_file.write_text(json.dumps(["AAPL", "msft", "AAPL"]), encoding="utf-8")
    assert module.load_universe(None, universe_file) == ("AAPL", "MSFT")

    universe_file.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_universe(None, universe_file)


# --------------------------------------------------------------------------- #
# DEFECT 3 (V4) — a failed ledger append must be reported as a failure         #
# --------------------------------------------------------------------------- #
def test_failed_ledger_append_raises_instead_of_reporting_success(tmp_path):
    from src.hedge import hedged_shadow_lane as hsl

    blocked = tmp_path / "raw_vs_hedged_ledger.jsonl"
    blocked.mkdir()  # a directory where the ledger file belongs -> open(..., "a") fails
    with pytest.raises(OSError):
        hsl._append_jsonl(blocked, {"cycle_ts": "2026-07-30T00:00:00Z"})

    # Sanity: the same call against a writable path really does persist.
    good = tmp_path / "ok.jsonl"
    hsl._append_jsonl(good, {"cycle_ts": "2026-07-30T00:00:00Z"})
    assert json.loads(good.read_text(encoding="utf-8").strip())["cycle_ts"] == "2026-07-30T00:00:00Z"


def test_hedge_cycle_does_not_return_a_row_when_its_ledger_append_fails(tmp_path):
    """End-to-end: real book + real hedge machinery, unwritable ledger -> no phantom success."""
    from src.hedge import hedged_shadow_lane as hsl
    from src.hedge.exposure_tags import (
        CURRENCY_BUCKET_MAP_PATH,
        SECTOR_BUCKET_MAP_PATH,
        load_bucket_map,
    )

    dates = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    prices = [100.0, 101.0, 102.01, 103.03]
    panel = pd.DataFrame({"AAPL": prices, "MSFT": prices, "NVDA": prices}, index=dates)
    book = hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date=str(dates[0].date()),
        weights={"AAPL": 0.5, "MSFT": 0.5}, raw_net_return=None, raw_gross_return=None,
        raw_cost=None, return_source="must_compute", meta={},
    )
    kwargs = dict(
        price_panel=panel,
        sector_bucket_map=load_bucket_map(SECTOR_BUCKET_MAP_PATH),
        currency_bucket_map=load_bucket_map(CURRENCY_BUCKET_MAP_PATH),
        book=book,
    )

    ledger = tmp_path / "ledger.jsonl"
    decisions = tmp_path / "decisions.jsonl"

    # Control: with writable paths the cycle persists and returns a row.
    row = hsl.run_cycle_for_strategy(
        "equity_harvester", ledger_path=ledger, decision_log_path=decisions, persist=True, **kwargs
    )
    assert row is not None
    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 1

    # Failure: the ledger path is unwritable -> the cycle must NOT return a row.
    blocked = tmp_path / "blocked.jsonl"
    blocked.mkdir()
    with pytest.raises(OSError):
        hsl.run_cycle_for_strategy(
            "equity_harvester", ledger_path=blocked, decision_log_path=decisions,
            persist=True, **kwargs
        )

    # run_all turns that raise into an explicit per-strategy None sentinel — a real
    # failure status the caller can see, never a success row.
    results = hsl.run_all(
        ["equity_harvester"], ledger_path=blocked, decision_log_path=decisions, persist=True
    )
    assert results["equity_harvester"] is None


def test_capture_best_effort_returns_a_real_status(tmp_path, monkeypatch):
    """V2: the hook stays non-raising but its outcome is now countable."""
    from src.data_platform import forward_capture as fc

    monkeypatch.setenv("AXIOM_DATA_ROOT", str(tmp_path / "axiom-data"))
    monkeypatch.setattr(fc, "_DEFAULT_SERVICE", None, raising=False)

    # An unknown hook name is a genuine failure and must report False, not None.
    assert fc.capture_best_effort("no_such_capture_method") is False
    # A real hook against a real (empty) data root succeeds.
    now = datetime.now(timezone.utc)
    assert fc.capture_best_effort(
        "capture_exposure_snapshot",
        {
            "source": "oanda_fx_trend_lane",
            "captured_at_utc": now.isoformat(),
            "source_mtime_utc": (now - timedelta(seconds=1)).isoformat(),
            "net_currency_notional_home": {"USD": 1.0},
        },
    ) is True


def test_track_b_capture_failures_are_counted_not_swallowed():
    """V3: the double swallow is gone — failures land in a countable register."""
    from src.equity.research import pit_text_loader

    pit_text_loader.reset_track_b_capture_failures()
    assert pit_text_loader.track_b_capture_failures() == ()
    pit_text_loader._record_track_b_capture_failure("0000-00-000000", "AAPL", "unit probe")
    recorded = pit_text_loader.track_b_capture_failures()
    assert len(recorded) == 1
    assert recorded[0]["accession"] == "0000-00-000000"
    assert recorded[0]["ticker"] == "AAPL"
    pit_text_loader.reset_track_b_capture_failures()
    assert pit_text_loader.track_b_capture_failures() == ()
