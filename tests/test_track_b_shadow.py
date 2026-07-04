"""No-mock tests for the Track B (SEC filing-text research-alpha) SHADOW lane.

Covers: (1) the frozen construction's imported constants trace to the
verifier-confirmed research harness (never a locally redefined/tuned value);
(2) score loading strictly excludes pre-cutoff (contamination-arm) scores and
dedupes by (ticker, as_of); (3) the ledger's append/skip-duplicate/compounding
semantics on real disk (tmp_path); (4) the shadow lane script's fail-closed
per-lane halt gate never reaches the (expensive) book computation; (5) no
broker/execution code exists anywhere in this lane.

Real classes, real disk (tmp_path), no `unittest.mock` — per repo policy. The
tests that pull the real cached price panel / scored-filing artifacts are
marked `integration` (deselect with `-m "not integration"`), matching the
project's "skip / mark integration / sandbox — don't mock" rule for external
data sources. The lane-halt fixture tests build an ISOLATED tmp_path repo with
its own `.claude/state.json` (never the real one) — same precedent as
`tests/test_crypto_momentum_shadow.py::TestLaneHaltGating`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.equity import track_b_shadow as tbs  # noqa: E402
from src.equity.research import harness as H  # noqa: E402
from src.equity.research.contracts import PRIMARY_WEIGHTS  # noqa: E402
import track_b_run_harness_postcutoff as _tb_postcutoff  # noqa: E402


class TestFrozenConstructionRegression:
    def test_constants_trace_to_imported_harness_defaults(self):
        # Every value traces to the imported harness function's own default
        # signature, never a repeated local literal — a future change to
        # run_research_backtest's defaults would fail this test instead of
        # silently drifting.
        import inspect
        params = inspect.signature(H.run_research_backtest).parameters
        assert tbs.COST_BPS == pytest.approx(float(params["cost_bps"].default))
        assert tbs.VOL_TARGET == pytest.approx(float(params["vol_target"].default))
        assert tbs.REBALANCE_STEP_DAYS == int(params["rebalance_step_days"].default)
        assert tbs.COMPOSITE_WEIGHTS == dict(PRIMARY_WEIGHTS)
        assert tbs.QUINTILE == H.QUINTILE
        assert tbs.MIN_NAMES_FOR_QUINTILE == H.MIN_NAMES_FOR_QUINTILE
        assert tbs.OVERLAY_DD_SOFT == H.OVERLAY_DD_SOFT
        assert tbs.OVERLAY_DD_HARD == H.OVERLAY_DD_HARD
        assert tbs.OVERLAY_MAX_LEV == H.OVERLAY_MAX_LEV
        assert tbs.MODEL_CUTOFF == _tb_postcutoff.MODEL_CUTOFF
        assert tbs.PRICE_START == _tb_postcutoff.PRICE_START

    def test_construction_manifest_reports_honest_coverage(self):
        # Post the 2026-07-04 scale-up (N=420 scored, 419 priced) the coverage
        # note reports a well-powered NULL, not "underpowered": the +0.16 rank-IC
        # FADED to +0.09 (non-significant) as N grew past the ~405 target, and
        # the remaining gap is temporal (n_rebalances=1), not cross-sectional N.
        manifest = tbs.construction_manifest(n_scored=420)
        assert manifest["n_scored_filings"] == 420
        assert manifest["power_required_n_filings"] == 405
        note = manifest["coverage_note"]
        assert "420" in note
        # Honest self-labelling: a measured null, not a fabricated edge.
        assert "NULL" in note
        assert "INSUFFICIENT" in note
        # The binding constraint is now temporal (rebalance dates), surfaced.
        assert "rebalance" in note.lower()
        assert "INSUFFICIENT" in manifest["gate_verdict"]
        assert manifest["composite_weights"] == dict(PRIMARY_WEIGHTS)


class TestLoadFrozenScores:
    def _write_artifact(self, path: Path, scores: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "meta": {"model_cutoff": tbs.MODEL_CUTOFF}, "pipeline_version": "1.0.0",
            "schema": "research_scores", "score_count": len(scores), "scores": scores,
        }))

    def _raw(self, ticker, as_of, quality=0.5):
        return {
            "ticker": ticker, "as_of": as_of, "fundamental_quality": quality,
            "accounting_red_flags": 0.1, "forward_outlook": 0.2, "conviction": 0.7,
            "rationale": "test fixture", "spans_used": [],
        }

    def test_excludes_pre_cutoff_scores(self, tmp_path):
        path = tmp_path / "artifact.json"
        self._write_artifact(path, [
            self._raw("PRE", "2025-06-01"),   # before MODEL_CUTOFF (2026-02-01) -> excluded
            self._raw("POST", "2026-03-01"),  # after cutoff -> included
        ])
        scores = tbs.load_frozen_scores(paths=[path])
        tickers = {s.ticker for s in scores}
        assert tickers == {"POST"}

    def test_combines_multiple_batches_and_dedupes(self, tmp_path):
        p1 = tmp_path / "batch1.json"
        p2 = tmp_path / "batch2.json"
        self._write_artifact(p1, [self._raw("AAA", "2026-03-01", quality=0.5)])
        self._write_artifact(p2, [
            self._raw("BBB", "2026-04-01", quality=0.3),
            self._raw("AAA", "2026-03-01", quality=0.9),  # exact dupe key -> last-write-wins
        ])
        scores = tbs.load_frozen_scores(paths=[p1, p2])
        by_ticker = {s.ticker: s for s in scores}
        assert set(by_ticker) == {"AAA", "BBB"}
        assert by_ticker["AAA"].fundamental_quality == pytest.approx(0.9)

    def test_missing_artifact_file_is_skipped_not_fatal(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        present = tmp_path / "present.json"
        self._write_artifact(present, [self._raw("ONLY", "2026-03-01")])
        scores = tbs.load_frozen_scores(paths=[missing, present])
        assert [s.ticker for s in scores] == ["ONLY"]

    def test_corrupt_artifact_file_is_skipped_not_fatal(self, tmp_path):
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not valid json")
        present = tmp_path / "present.json"
        self._write_artifact(present, [self._raw("ONLY", "2026-03-01")])
        scores = tbs.load_frozen_scores(paths=[corrupt, present])
        assert [s.ticker for s in scores] == ["ONLY"]

    def test_out_of_range_score_field_raises(self, tmp_path):
        path = tmp_path / "artifact.json"
        bad = self._raw("BAD", "2026-03-01")
        bad["fundamental_quality"] = 5.0  # out of [-1, 1]
        self._write_artifact(path, [bad])
        with pytest.raises(ValueError):
            tbs.load_frozen_scores(paths=[path])


class TestLedger:
    def _result(self, asof="2026-01-01", net=0.01, n_scored=48):
        return tbs.ShadowCycleResult(
            asof_date=asof, n_scored_filings=n_scored, universe_size=48,
            longs={"AAA": 0.1, "BBB": 0.1}, gross_leverage=0.2,
            today_net_return=net, today_gross_return=net + 0.0005,
            today_cost=0.0005, today_turnover=0.1,
            construction=tbs.construction_manifest(n_scored),
        )

    def test_first_entry_schema_and_cumulative(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        row = tbs.record_shadow_cycle(self._result(net=0.02), ledger_path=ledger_path,
                                      cycle_ts_iso="2026-01-01T00:00:00+00:00")
        assert row is not None
        assert row["forward_cycle_seq"] == 1
        assert row["cumulative_shadow_return"] == pytest.approx(0.02)
        assert row["orders_placed"] == 0
        assert row["broker"] is None
        assert row["n_longs"] == 2 and row["n_shorts"] == 0
        assert row["book"]["longs"]["AAA"] == pytest.approx(0.1)
        assert row["book"]["shorts"] == {}
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["asof_date"] == "2026-01-01"

    def test_compounds_across_cycles(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        tbs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                                cycle_ts_iso="2026-01-01T00:00:00+00:00")
        row2 = tbs.record_shadow_cycle(self._result(asof="2026-02-01", net=-0.01), ledger_path=ledger_path,
                                       cycle_ts_iso="2026-02-01T00:00:00+00:00")
        expected = (1.02) * (0.99) - 1.0
        assert row2["cumulative_shadow_return"] == pytest.approx(expected)
        assert row2["forward_cycle_seq"] == 2

    def test_skips_duplicate_asof_date(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        tbs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                                cycle_ts_iso="2026-01-01T00:00:00+00:00")
        second = tbs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.05), ledger_path=ledger_path,
                                         cycle_ts_iso="2026-01-01T01:00:00+00:00")
        assert second is None
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_skips_corrupt_ledger_line_without_losing_valid_rows(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        tbs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                                cycle_ts_iso="2026-01-01T00:00:00+00:00")
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write('{"asof_date": "2026-01-15", "cumulative_shad\n')  # truncated mid-write
        row3 = tbs.record_shadow_cycle(self._result(asof="2026-02-01", net=-0.01), ledger_path=ledger_path,
                                       cycle_ts_iso="2026-02-01T00:00:00+00:00")
        expected = (1.02) * (0.99) - 1.0
        assert row3["cumulative_shadow_return"] == pytest.approx(expected)
        assert row3["forward_cycle_seq"] == 2
        summary = tbs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 2

    def test_forward_oos_summary_empty(self, tmp_path):
        summary = tbs.forward_oos_summary(ledger_path=tmp_path / "missing.jsonl")
        assert summary["n_cycles"] == 0
        assert summary["cumulative_return"] == 0.0
        assert summary["first_asof_date"] is None
        assert "forward_sharpe_annualized" not in summary  # never fabricated, key absent on n=0

    def test_forward_oos_summary_single_cycle_sharpe_undefined(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        tbs.record_shadow_cycle(self._result(net=0.01), ledger_path=ledger_path,
                                cycle_ts_iso="2026-01-01T00:00:00+00:00")
        summary = tbs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 1
        assert summary["forward_sharpe_annualized"] is None  # n<2 -> never fabricated

    def test_forward_oos_summary_multi_cycle_sharpe_computed(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        for i, net in enumerate([0.01, -0.005, 0.02]):
            tbs.record_shadow_cycle(self._result(asof=f"2026-0{i+1}-01", net=net), ledger_path=ledger_path,
                                    cycle_ts_iso=f"2026-0{i+1}-01T00:00:00+00:00")
        summary = tbs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 3
        assert isinstance(summary["forward_sharpe_annualized"], float)


class TestLaneHaltGating:
    """The shadow lane script must never reach compute_shadow_cycle (heavier
    price/backtest code path) while the track_b lane is halted."""

    def _build_state(self, tmp_path: Path, *, global_halted: bool, lane_halted: bool) -> Path:
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {
            "halted": global_halted,
            "mode": "live",
            "halted_lanes": {
                "oanda_fx": True, "equity": True, "brain": True,
                "crypto_momentum": True, "track_b": lane_halted,
            },
        }
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return root

    def test_tick_refuses_when_lane_halted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=False, lane_halted=True)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_track_b_shadow as runner  # noqa: E402

        result = runner._tick(root, refresh_prices=False)
        assert result["ran"] is False
        assert result["reason"].startswith("halted:")
        assert result["orders"] == 0
        # no trained_data directory was ever created — compute_shadow_cycle was never reached
        assert not (root / "trained_data").exists()

    def test_tick_refuses_when_globally_halted_even_if_lane_unhalted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=True, lane_halted=False)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_track_b_shadow as runner  # noqa: E402

        result = runner._tick(root, refresh_prices=False)
        assert result["ran"] is False
        assert result["orders"] == 0


class TestNoExecutionPath:
    """Structural guard: no broker/order-placement code anywhere in this lane."""

    def test_no_broker_or_order_references(self):
        forbidden_patterns = (
            "place_equity_order(", "place_market_order(", "import src.brokers",
            "from src.brokers", "OandaPracticeClient(", "execute_trade(", "create_broker(",
            "IBKRClient(",
        )
        for rel in ("src/equity/track_b_shadow.py", "scripts/run_track_b_shadow.py"):
            lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
            code_lines = "\n".join(ln for ln in lines if not ln.strip().startswith("#"))
            if code_lines.lstrip().startswith('"""'):
                first = code_lines.index('"""')
                second = code_lines.index('"""', first + 3)
                code_lines = code_lines[second + 3:]
            for forbidden in forbidden_patterns:
                assert forbidden not in code_lines, f"{rel} references execution surface {forbidden!r}"

    def test_live_gate_never_armed_anywhere_in_lane(self):
        for rel in ("src/equity/track_b_shadow.py", "scripts/run_track_b_shadow.py"):
            lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
            code_lines = "\n".join(ln for ln in lines if not ln.strip().startswith("#"))
            if code_lines.lstrip().startswith('"""'):
                first = code_lines.index('"""')
                second = code_lines.index('"""', first + 3)
                code_lines = code_lines[second + 3:]
            assert ".arm(" not in code_lines, f"{rel} calls LiveGate.arm() — arming must never happen in the shadow path"


@pytest.mark.integration
class TestLiveShadowCycleIntegration:
    """Pulls real (cached) scored-filing artifacts + cached price panel — skip
    with -m 'not integration' in fast runs."""

    def test_compute_shadow_cycle_cached_produces_valid_result(self, tmp_path):
        result = tbs.compute_shadow_cycle(refresh_prices=False)
        assert result.universe_size > 0
        assert result.n_scored_filings > 0
        assert result.asof_date
        assert isinstance(result.longs, dict)
        assert len(result.longs) > 0  # combined universe clears MIN_NAMES_FOR_QUINTILE

        ledger_path = tmp_path / "ledger.jsonl"
        row = tbs.record_shadow_cycle(result, ledger_path=ledger_path, cycle_ts_iso="2026-07-02T00:00:00+00:00")
        assert row is not None
        assert row["asof_date"] == result.asof_date
        assert row["orders_placed"] == 0
        assert row["broker"] is None

    def test_run_track_b_shadow_script_one_shot(self, tmp_path):
        """Real end-to-end script invocation against an ISOLATED fixture repo
        (own .claude/state.json under tmp_path — never the real one). Mirrors
        tests/test_crypto_momentum_shadow.py's own precedent for this pattern.
        `--project-root` only scopes the HALT check; the ledger path is the
        module-default under the real REPO_ROOT, so this also produces this
        task's "first real ledger entry" as a side effect. The duplicate-
        asof_date guard makes repeat runs safe no-ops."""
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {"halted": False, "mode": "live",
                 "halted_lanes": {"oanda_fx": True, "equity": True, "brain": True,
                                  "crypto_momentum": True, "track_b": False}}
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_track_b_shadow.py"),
             "--project-root", str(root)],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CYCLE_RESULT: ran=True" in proc.stdout
        assert "orders=0" in proc.stdout
        real_ledger = REPO_ROOT / "trained_data" / "research" / "track_b_shadow_ledger.jsonl"
        assert real_ledger.exists()
