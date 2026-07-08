"""No-mock tests for the P2 tick-capture activation (docs/ENGINEERING_BRAIN.md).

Covers: (1) TickPersister writes valid, deduplicated, partitioned Parquet from
real TickQuote objects on real disk (tmp_path); (2) retention/TTL pruning
removes only partitions older than the cutoff, keyed off partition date, not
mtime; (3) TickCaptureDaemon's gap detector honestly records idle windows to
an append-only sidecar instead of silently absorbing them; (4) the health
sidecar is written atomically and reflects real daemon state; (5) structural
guard -- no broker/order/execution code exists anywhere in the capture path,
and the practice-only client factory refuses to build anything but a
practice-pinned stream client.

Real classes, real disk (tmp_path), no `unittest.mock` -- per repo policy.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.data.tick_capture as tick_capture  # noqa: E402
from src.data.tick_capture import (  # noqa: E402
    TickCaptureDaemon,
    TickPersister,
    build_practice_stream_client,
)
from src.utils.oanda_streaming import (  # noqa: E402
    PRACTICE_STREAM_URL,
    OandaStreamClient,
    TickQuote,
)


def _tick(instrument: str, when: datetime, bid: float = 1.1000, ask: float = 1.1002) -> TickQuote:
    return TickQuote(
        instrument=instrument,
        time=when,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2.0,
        bid_liq=500_000.0,
        ask_liq=500_000.0,
        status="tradeable",
        source="oanda_stream",
    )


class TestTickPersister:
    def test_flush_writes_valid_partitioned_parquet(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        when = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)
        ticks = [_tick("EUR_USD", when), _tick("EUR_USD", when + timedelta(seconds=1))]

        written = persister.flush(ticks)

        assert written == 2
        pq_path = tmp_path / "EUR_USD" / "2026" / "07" / "07.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)
        assert len(df) == 2
        assert set(df.columns) == {"instrument", "bid", "ask", "mid", "bid_liq", "ask_liq", "status", "source"}
        assert df["mid"].iloc[0] == pytest.approx(1.1001)

    def test_flush_dedupes_on_reflush_same_timestamp(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        when = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)

        persister.flush([_tick("EUR_USD", when, bid=1.1000, ask=1.1002)])
        persister.flush([_tick("EUR_USD", when, bid=1.1010, ask=1.1012)])  # same time, restart-replay

        df = pd.read_parquet(tmp_path / "EUR_USD" / "2026" / "07" / "07.parquet")
        assert len(df) == 1  # deduped by (instrument, time)
        assert df["bid"].iloc[0] == pytest.approx(1.1010)  # keep="last" wins

    def test_flush_partitions_multiple_instruments_and_days(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        d1 = datetime(2026, 7, 6, 23, 59, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 7, 7, 0, 1, 0, tzinfo=timezone.utc)
        written = persister.flush([_tick("EUR_USD", d1), _tick("GBP_USD", d2)])

        assert written == 2
        assert (tmp_path / "EUR_USD" / "2026" / "07" / "06.parquet").exists()
        assert (tmp_path / "GBP_USD" / "2026" / "07" / "07.parquet").exists()

    def test_flush_empty_list_is_a_noop(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        assert persister.flush([]) == 0
        assert list(tmp_path.glob("**/*.parquet")) == []


class TestRetentionPruning:
    def test_prune_removes_only_partitions_older_than_cutoff(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        today = datetime.now(timezone.utc)
        old = today - timedelta(days=400)
        recent = today - timedelta(days=5)

        persister.flush([_tick("EUR_USD", old)])
        persister.flush([_tick("EUR_USD", recent)])

        old_path = tmp_path / "EUR_USD" / f"{old.year}" / f"{old.month:02d}" / f"{old.day:02d}.parquet"
        recent_path = tmp_path / "EUR_USD" / f"{recent.year}" / f"{recent.month:02d}" / f"{recent.day:02d}.parquet"
        assert old_path.exists() and recent_path.exists()

        removed = persister.prune_older_than(retention_days=180)

        assert removed == 1
        assert not old_path.exists()
        assert recent_path.exists()

    def test_prune_disabled_when_retention_days_is_none_or_zero(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        old = datetime.now(timezone.utc) - timedelta(days=1000)
        persister.flush([_tick("EUR_USD", old)])

        assert persister.prune_older_than(retention_days=None) == 0
        assert persister.prune_older_than(retention_days=0) == 0

    def test_prune_ignores_non_partition_files(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        stray = tmp_path / "EUR_USD" / "2026" / "07" / "_gaps.jsonl"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("{}\n")
        # Only *.parquet is globbed, so a .jsonl sidecar never even reaches
        # the int()-parse path -- this just documents that invariant.
        assert persister.prune_older_than(retention_days=1) == 0
        assert stray.exists()


class _FakeStreamClient:
    """Minimal stand-in exposing only what TickCaptureDaemon calls on a client.

    Not a mock: a real, hand-written object with real behavior (yields a
    fixed tick sequence with a deliberate gap), used because driving the
    daemon's gap-detection logic requires controlling tick arrival timing,
    and hitting the real OANDA stream from a unit test would be flaky/slow.
    The real-network path is exercised separately as an integration test.
    """

    def __init__(self, ticks):
        self._ticks = ticks
        self.base_url = PRACTICE_STREAM_URL
        self.shutdown_called = False

    def stream_ticks(self, instruments, snapshot=True):
        for t in self._ticks:
            yield t

    def shutdown(self):
        self.shutdown_called = True


class TestGapHonesty:
    def test_gap_detected_and_persisted_to_sidecar(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        client = _FakeStreamClient([_tick("EUR_USD", datetime.now(timezone.utc))])
        daemon = TickCaptureDaemon(
            client=client, persister=persister, gap_threshold_sec=5.0, retention_days=None,
        )

        daemon._last_tick_time = time.time() - 30.0  # idle 30s > 5s threshold
        daemon._stream_loop(["EUR_USD"])

        assert len(daemon._gaps) == 1
        assert daemon._gaps[0]["duration_sec"] >= 30.0
        gaps_path = tmp_path / "_gaps.jsonl"
        assert gaps_path.exists()
        lines = [json.loads(ln) for ln in gaps_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0]["duration_sec"] >= 30.0

    def test_no_gap_recorded_within_threshold(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        client = _FakeStreamClient([_tick("EUR_USD", datetime.now(timezone.utc))])
        daemon = TickCaptureDaemon(
            client=client, persister=persister, gap_threshold_sec=20.0, retention_days=None,
        )

        daemon._last_tick_time = time.time() - 1.0  # well within threshold
        daemon._stream_loop(["EUR_USD"])

        assert daemon._gaps == []
        assert not (tmp_path / "_gaps.jsonl").exists()

    def test_first_tick_never_counted_as_a_gap(self, tmp_path):
        """No prior tick time yet -- nothing to be idle relative to."""
        persister = TickPersister(root=tmp_path)
        client = _FakeStreamClient([_tick("EUR_USD", datetime.now(timezone.utc))])
        daemon = TickCaptureDaemon(client=client, persister=persister, retention_days=None)

        daemon._stream_loop(["EUR_USD"])

        assert daemon._gaps == []


class TestHealthSidecar:
    def test_do_flush_persists_health_json_atomically(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        client = _FakeStreamClient([])
        daemon = TickCaptureDaemon(client=client, persister=persister, retention_days=None)
        daemon._stats["started_at"] = "2026-07-07T00:00:00+00:00"

        daemon._do_flush()

        health_path = tmp_path / "_health.json"
        assert health_path.exists()
        health = json.loads(health_path.read_text())
        assert health["started_at"] == "2026-07-07T00:00:00+00:00"
        assert health["gaps_detected"] == 0
        assert health["retention_days"] is None
        assert list(tmp_path.glob("*.tmp")) == []  # no leftover tmp file

    def test_health_reflects_recorded_gaps(self, tmp_path):
        persister = TickPersister(root=tmp_path)
        client = _FakeStreamClient([])
        daemon = TickCaptureDaemon(client=client, persister=persister, retention_days=None)
        daemon._record_gap(1_000_000.0, 1_000_030.0)

        health = daemon.get_health()

        assert health["gaps_detected"] == 1
        assert health["last_gap"]["duration_sec"] == pytest.approx(30.0)


class TestPracticeOnlySafety:
    def test_build_practice_stream_client_pins_practice_url(self, monkeypatch):
        monkeypatch.setenv("OANDA_API_TOKEN", "test-token")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "101-000-00000000-001")
        monkeypatch.setenv("OANDA_ENVIRONMENT", "live")  # even if this leaks in, must be ignored

        client = build_practice_stream_client()

        assert isinstance(client, OandaStreamClient)
        assert client.environment == "practice"
        assert client.base_url == PRACTICE_STREAM_URL

    def test_build_practice_stream_client_refuses_without_credentials(self, monkeypatch):
        monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)

        with pytest.raises(RuntimeError):
            build_practice_stream_client()

    def test_build_practice_stream_client_refuses_non_practice_base_url(self, monkeypatch):
        """The practice-URL check is an explicit `if ... raise RuntimeError`, not
        an `assert` -- verify it actually fires (not just that it's absent under
        the normal, already-practice path covered above). Uses a real,
        hand-written stand-in class (same convention as _FakeStreamClient), not
        a Mock, per the repo's No-Mock rule."""
        monkeypatch.setenv("OANDA_API_TOKEN", "test-token")
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "101-000-00000000-001")

        class _BadUrlOandaStreamClient:
            def __init__(self, *, account_id, api_token, environment):
                self.account_id = account_id
                self.api_token = api_token
                self.environment = environment
                self.base_url = "https://stream-fxtrade.oanda.com"  # LIVE url, wrong on purpose

        monkeypatch.setattr(tick_capture, "OandaStreamClient", _BadUrlOandaStreamClient)

        with pytest.raises(RuntimeError, match="live stream URL"):
            build_practice_stream_client()


class TestNoExecutionPath:
    """Structural guard: tick capture never touches a broker/order path."""

    def test_no_broker_or_order_references(self):
        forbidden_patterns = (
            "place_equity_order(", "place_market_order(", "import src.brokers",
            "from src.brokers", "OandaPracticeClient(", "execute_trade(", "create_broker(",
            "/orders", "/trades",
        )
        for rel in ("src/data/tick_capture.py", "scripts/run_tick_capture.py", "src/utils/oanda_streaming.py"):
            lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
            code_lines = "\n".join(ln for ln in lines if not ln.strip().startswith("#"))
            for forbidden in forbidden_patterns:
                assert forbidden not in code_lines, f"{rel} references execution surface {forbidden!r}"

    def test_stream_client_only_hits_pricing_stream_endpoint(self):
        src = (REPO_ROOT / "src/utils/oanda_streaming.py").read_text(encoding="utf-8")
        assert "/pricing/stream" in src
        assert "/orders" not in src and "/trades" not in src
