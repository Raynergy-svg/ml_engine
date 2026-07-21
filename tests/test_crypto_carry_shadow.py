"""No-mock tests for the crypto cash-and-carry SHADOW lane.

Covers: (1) the book+P&L construction is byte-identical to the frozen,
verifier-confirmed harness (`_cc.backtest`) -- the regression guard that
proves this module never silently re-derives/tunes the pre-registered
construction; (2) the ledger's append/skip-duplicate/compounding semantics on
real disk (tmp_path); (3) the shadow lane script's fail-closed per-lane halt
gate never reaches the (expensive, network-touching) book computation; (4) no
broker/execution code exists anywhere in this lane.

Real classes, real disk (tmp_path), no `unittest.mock` -- per repo policy.
The one test that pulls real crypto data (`test_compute_shadow_cycle_live_...`)
is marked `integration` (deselect with `-m "not integration"`), matching the
project's "skip / mark integration / sandbox -- don't mock" rule for external
data sources.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.crypto import carry_shadow as cs  # noqa: E402
import experiment_crypto_cash_and_carry as _cc  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic panel (deterministic, no network) for fast regression checks.
# --------------------------------------------------------------------------- #
def _synthetic_panel():
    rng = np.random.default_rng(7)
    n_days, n_syms = 200, 20
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"SYM{i}USDT" for i in range(n_syms)]

    funding = pd.DataFrame(rng.normal(0.0002, 0.0006, size=(n_days, n_syms)), index=dates, columns=cols)
    eligible = pd.DataFrame(True, index=dates, columns=cols)
    return funding, eligible, cols


class TestFrozenConstructionRegression:
    def test_book_and_returns_matches_frozen_harness(self):
        funding, eligible, cols = _synthetic_panel()
        returns, book = cs._compute_book_and_returns(funding, eligible, cols)
        ref = _cc.backtest(funding, eligible, cols, cost_bps_roundtrip=cs.COST_BPS_ROUNDTRIP, vol_target=True)

        for col in ("carry", "cost", "gross", "net", "turnover"):
            diff = (returns[col] - ref[col]).abs().max()
            assert diff < 1e-9, f"{col} series diverged from frozen backtest by {diff}"
        assert list(book.columns) == cols
        assert book.shape[0] == len(funding.index)

    def test_construction_manifest_matches_imported_constants(self):
        manifest = cs.construction_manifest()
        assert manifest["signal"] == "cash_and_carry_positive_funding_only"
        assert manifest["lookback_d"] == _cc.LOOKBACK_D
        assert manifest["cost_bps_roundtrip"] == pytest.approx(_cc.COST_BPS_ROUNDTRIP)
        assert manifest["vol_target_ann"] == pytest.approx(_cc.TARGET_ANN_VOL)
        assert manifest["max_leverage"] == pytest.approx(_cc.MAX_LEV)
        # every value traces to the imported harness module, never a local literal
        assert cs.LOOKBACK_D == _cc.LOOKBACK_D
        assert cs.COST_BPS_ROUNDTRIP == _cc.COST_BPS_ROUNDTRIP
        assert cs.TARGET_ANN_VOL == _cc.TARGET_ANN_VOL
        assert cs.VOL_WINDOW == _cc.VOL_WINDOW
        assert cs.MAX_LEV == _cc.MAX_LEV

    def test_book_never_short_and_never_negative_weight(self):
        """The construction only ever harvests positive funding (long-cash /
        short-perp) -- there is no reverse-carry leg, so every weight is >= 0."""
        funding, eligible, cols = _synthetic_panel()
        _, book = cs._compute_book_and_returns(funding, eligible, cols)
        assert (book.to_numpy() >= -1e-12).all()

    def test_last_populated_date_trims_sparse_nan_tail(self):
        _funding, eligible, cols = _synthetic_panel()
        sparse = eligible.copy()
        well_populated_cutoff = sparse.index[-6]
        sparse.loc[sparse.index[-5]:, cols[3:]] = False  # only 3/20 symbols eligible past cutoff
        cutoff = cs._last_populated_date(sparse, cols, min_valid=10)
        assert cutoff == well_populated_cutoff

    def test_last_populated_date_falls_back_to_last_index_if_never_populated(self):
        _funding, eligible, cols = _synthetic_panel()
        sparse = eligible.copy()
        sparse.loc[:, :] = False
        cutoff = cs._last_populated_date(sparse, cols, min_valid=10)
        assert cutoff == sparse.index[-1]


class TestLedger:
    def _result(self, asof="2026-01-01", net=0.01):
        return cs.ShadowCycleResult(
            asof_date=asof, universe_size=20,
            longs={"SYM1USDT": 0.1, "SYM2USDT": 0.1},
            shorts={},
            gross_leverage=0.2, today_net_return=net,
            today_price_return=0.0, today_carry_return=net + 0.0005,
            today_cost=0.0005, today_turnover=0.1,
        )

    def test_first_entry_schema_and_cumulative(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        row = cs.record_shadow_cycle(self._result(net=0.02), ledger_path=ledger_path,
                                     cycle_ts_iso="2026-01-01T00:00:00+00:00")
        assert row is not None
        assert row["forward_cycle_seq"] == 1
        assert row["cumulative_shadow_return"] == pytest.approx(0.02)
        assert row["orders_placed"] == 0
        assert row["broker"] is None
        assert row["n_longs"] == 2 and row["n_shorts"] == 0
        assert row["book"]["longs"]["SYM1USDT"] == pytest.approx(0.1)
        assert row["book"]["shorts"] == {}
        assert row["today_price_return"] == pytest.approx(0.0)
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["asof_date"] == "2026-01-01"

    def test_compounds_across_cycles(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        cs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        row2 = cs.record_shadow_cycle(self._result(asof="2026-01-08", net=-0.01), ledger_path=ledger_path,
                                      cycle_ts_iso="2026-01-08T00:00:00+00:00")
        expected = (1.02) * (0.99) - 1.0
        assert row2["cumulative_shadow_return"] == pytest.approx(expected)
        assert row2["forward_cycle_seq"] == 2

    def test_skips_duplicate_asof_date(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        cs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        second = cs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.05), ledger_path=ledger_path,
                                        cycle_ts_iso="2026-01-01T01:00:00+00:00")
        assert second is None
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_skips_corrupt_ledger_line_without_losing_valid_rows(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        cs.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write('{"asof_date": "2026-01-05", "cumulative_shad\n')  # truncated mid-write
        row3 = cs.record_shadow_cycle(self._result(asof="2026-01-08", net=-0.01), ledger_path=ledger_path,
                                      cycle_ts_iso="2026-01-08T00:00:00+00:00")
        expected = (1.02) * (0.99) - 1.0
        assert row3["cumulative_shadow_return"] == pytest.approx(expected)
        assert row3["forward_cycle_seq"] == 2
        summary = cs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 2

    def test_forward_oos_summary_empty(self, tmp_path):
        summary = cs.forward_oos_summary(ledger_path=tmp_path / "missing.jsonl")
        assert summary["n_cycles"] == 0
        assert summary["cumulative_return"] == 0.0
        assert summary["first_asof_date"] is None
        assert "forward_sharpe_annualized" not in summary  # never fabricated on empty ledger

    def test_forward_oos_summary_single_cycle_sharpe_undefined(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        cs.record_shadow_cycle(self._result(net=0.01), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        summary = cs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 1
        assert summary["forward_sharpe_annualized"] is None

    def test_forward_oos_summary_multi_cycle_sharpe_computed(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        for i, net in enumerate([0.01, -0.005, 0.02]):
            cs.record_shadow_cycle(self._result(asof=f"2026-01-0{i+1}", net=net), ledger_path=ledger_path,
                                   cycle_ts_iso=f"2026-01-0{i+1}T00:00:00+00:00")
        summary = cs.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 3
        assert isinstance(summary["forward_sharpe_annualized"], float)


class TestLaneHaltGating:
    """The shadow lane script must never reach compute_shadow_cycle (network
    + broker-shaped code path) while the crypto_carry lane is halted."""

    def _build_state(self, tmp_path: Path, *, global_halted: bool, lane_halted: bool) -> Path:
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {
            "halted": global_halted,
            "mode": "live",
            "halted_lanes": {
                "oanda_fx": True, "equity": True, "brain": True,
                "crypto_carry": lane_halted,
            },
        }
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return root

    def test_tick_refuses_when_lane_halted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=False, lane_halted=True)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_crypto_carry_shadow as runner  # noqa: E402

        result = runner._tick(root, refresh=False)
        assert result["ran"] is False
        assert result["reason"].startswith("halted:")
        assert result["orders"] == 0
        # no ledger directory was ever created -- compute_shadow_cycle was never reached
        assert not (root / "trained_data").exists()

    def test_tick_refuses_when_globally_halted_even_if_lane_unhalted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=True, lane_halted=False)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_crypto_carry_shadow as runner  # noqa: E402

        result = runner._tick(root, refresh=False)
        assert result["ran"] is False
        assert result["orders"] == 0


class TestNoExecutionPath:
    """Structural guard: no broker/order-placement code anywhere in this lane."""

    def test_no_broker_or_order_references(self):
        forbidden_patterns = (
            "place_equity_order(", "place_market_order(", "import src.brokers",
            "from src.brokers", "OandaPracticeClient(", "execute_trade(", "create_broker(",
        )
        for rel in ("src/crypto/carry_shadow.py", "scripts/run_crypto_carry_shadow.py"):
            lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
            code_lines = "\n".join(ln for ln in lines if not ln.strip().startswith("#"))
            if code_lines.lstrip().startswith('"""'):
                first = code_lines.index('"""')
                second = code_lines.index('"""', first + 3)
                code_lines = code_lines[second + 3:]
            for forbidden in forbidden_patterns:
                assert forbidden not in code_lines, f"{rel} references execution surface {forbidden!r}"

    def test_live_gate_starts_disarmed(self, tmp_path):
        from src.crypto.crypto_carry_live_gate import make_crypto_carry_live_gate
        gate = make_crypto_carry_live_gate(
            state_path=tmp_path / "live_gate_state.json",
            audit_path=tmp_path / "live_gate_audit.jsonl",
        )
        assert gate.is_armed() is False


@pytest.mark.integration
class TestLiveShadowCycleIntegration:
    """Pulls real (cached) crypto data -- skip with -m 'not integration' in fast runs."""

    def test_compute_shadow_cycle_live_cache_produces_valid_result(self, tmp_path):
        result = cs.compute_shadow_cycle(refresh_klines=False)
        assert result.universe_size >= 0
        assert result.asof_date
        assert isinstance(result.longs, dict)
        assert result.shorts == {}
        assert result.today_price_return == 0.0

        ledger_path = tmp_path / "ledger.jsonl"
        row = cs.record_shadow_cycle(result, ledger_path=ledger_path, cycle_ts_iso="2026-07-07T00:00:00+00:00")
        assert row is not None
        assert row["asof_date"] == result.asof_date

    def test_run_crypto_carry_shadow_script_one_shot(self, tmp_path):
        """Real end-to-end script invocation against the real production
        ledger (same convention as the crypto_momentum lane's equivalent
        test) -- this doubles as this task's "produce a first real ledger
        entry" check. The duplicate-`asof_date` guard makes repeat runs safe
        no-ops."""
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {"halted": False, "mode": "live",
                 "halted_lanes": {"oanda_fx": True, "equity": True, "brain": True, "crypto_carry": False}}
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_crypto_carry_shadow.py"),
             "--project-root", str(root)],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CYCLE_RESULT: ran=True" in proc.stdout
        assert "orders=0" in proc.stdout
        real_ledger = REPO_ROOT / "trained_data" / "crypto_carry" / "shadow_carry_ledger.jsonl"
        assert real_ledger.exists()
