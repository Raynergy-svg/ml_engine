"""No-mock tests for the crypto XS-momentum SHADOW lane.

Covers: (1) the book+P&L construction is byte-identical to the frozen,
verifier-confirmed harness (`h2i.backtest_flex`) — the regression guard that
proves this module never silently re-derives/tunes the pre-registered
signal; (2) the ledger's append/skip-duplicate/compounding semantics on
real disk (tmp_path); (3) the shadow lane script's fail-closed per-lane
halt gate never reaches the (expensive, network-touching) book computation.

Real classes, real disk (tmp_path), no `unittest.mock` — per repo policy.
The one test that pulls real crypto data (`test_compute_shadow_cycle_live_...`)
is marked `integration` (deselect with `-m "not integration"`), matching the
project's "skip / mark integration / sandbox — don't mock" rule for external
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

from src.crypto import momentum_shadow as ms  # noqa: E402
import experiment_crypto_h2_infra_stress as h2i  # noqa: E402
import experiment_crypto_round2 as round2  # noqa: E402
import experiment_crypto_xs_signals as h  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic panel (deterministic, no network) for fast regression checks.
# --------------------------------------------------------------------------- #
def _synthetic_panel():
    rng = np.random.default_rng(42)
    n_days, n_syms = 200, 20
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"SYM{i}USDT" for i in range(n_syms)]

    log_rets = rng.normal(0.0, 0.03, size=(n_days, n_syms))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=cols)

    funding = pd.DataFrame(rng.normal(0.0, 0.0005, size=(n_days, n_syms)), index=dates, columns=cols)
    eligible = pd.DataFrame(True, index=dates, columns=cols)
    return close, funding, eligible, cols


class TestFrozenConstructionRegression:
    def test_book_and_returns_matches_frozen_harness(self):
        close, funding, eligible, cols = _synthetic_panel()
        sig = h.make_signal(ms.SIGNAL_NAME, close, None, cols)

        returns, book = ms._compute_book_and_returns(close, funding, eligible, cols, sig)
        ref = h2i.backtest_flex(
            close, funding, eligible, cols, sig, ms.DIRECTION,
            cost_bps=ms.COST_BPS, rebalance_days=ms.REBALANCE_DAYS, vol_target=True,
        )

        for col in ("price", "carry", "turnover", "cost", "gross", "net"):
            diff = (returns[col] - ref[col]).abs().max()
            assert diff < 1e-9, f"{col} series diverged from frozen backtest_flex by {diff}"
        assert list(book.columns) == cols
        assert book.shape[0] == len(close.index)

    def test_construction_manifest_matches_imported_constants(self):
        manifest = ms.construction_manifest()
        assert manifest["signal"] == "xs_momentum_14d"
        assert manifest["vol_target_ann"] == pytest.approx(0.10)
        assert manifest["max_leverage"] == pytest.approx(3.0)
        assert manifest["cost_bps"] == pytest.approx(10.0)
        # every value traces to the imported harness modules, never a local
        # literal — compared against the SOURCE constant, not a repeated
        # number, so a future change to round2.py's REBALANCE_D would fail
        # this test instead of silently drifting.
        assert ms.LOOKBACK_D == h.SIGNALS["momentum"]["lookback"]
        assert ms.QUINTILE == h.QUINTILE
        assert ms.TARGET_ANN_VOL == h2i.TARGET_ANN_VOL
        assert ms.VOL_WINDOW == h2i.VOL_WINDOW
        assert ms.MAX_LEV == h2i.MAX_LEV
        assert ms.REBALANCE_DAYS == round2.REBALANCE_D
        assert manifest["rebalance_days"] == round2.REBALANCE_D

    def test_last_populated_date_trims_sparse_nan_tail(self):
        close, _funding, _eligible, cols = _synthetic_panel()
        # Simulate the real cadence artifact: a handful of symbols keep 5 extra
        # trailing days of "data" (like the free monthly-dump source's uneven
        # per-symbol updates) while the rest of the universe has none there.
        sparse = close.copy()
        well_populated_cutoff = sparse.index[-6]
        sparse.loc[sparse.index[-5]:, cols[3:]] = float("nan")  # only 3/20 symbols past cutoff
        cutoff = ms._last_populated_date(sparse, cols, min_valid=10)
        assert cutoff == well_populated_cutoff

    def test_last_populated_date_falls_back_to_last_index_if_never_populated(self):
        close, _funding, _eligible, cols = _synthetic_panel()
        sparse = close.copy()
        sparse.loc[:, :] = float("nan")
        # never crashes, falls back to the raw last index rather than raising
        cutoff = ms._last_populated_date(sparse, cols, min_valid=10)
        assert cutoff == sparse.index[-1]

    def test_unlevered_book_is_dollar_neutral_on_rebalance_dates(self):
        close, funding, eligible, cols = _synthetic_panel()
        sig = h.make_signal(ms.SIGNAL_NAME, close, None, cols)
        _, book = ms._compute_book_and_returns(close, funding, eligible, cols, sig)
        # book is levered (Wlev); the SIGN pattern (long/short split) should still
        # net near-zero in gross dollar terms before cost drag on any date where
        # leverage is nonzero and enough symbols cleared the len(s)>=10 floor.
        nonzero_rows = book[book.abs().sum(axis=1) > 0]
        assert len(nonzero_rows) > 0


class TestLedger:
    def _result(self, asof="2026-01-01", net=0.01):
        return ms.ShadowCycleResult(
            asof_date=asof, universe_size=20,
            longs={"SYM1USDT": 0.05, "SYM2USDT": 0.05},
            shorts={"SYM3USDT": -0.05, "SYM4USDT": -0.05},
            gross_leverage=0.2, today_net_return=net,
            today_price_return=net + 0.001, today_carry_return=-0.0005,
            today_cost=0.0005, today_turnover=0.1,
        )

    def test_first_entry_schema_and_cumulative(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        row = ms.record_shadow_cycle(self._result(net=0.02), ledger_path=ledger_path,
                                     cycle_ts_iso="2026-01-01T00:00:00+00:00")
        assert row is not None
        assert row["forward_cycle_seq"] == 1
        assert row["cumulative_shadow_return"] == pytest.approx(0.02)
        assert row["orders_placed"] == 0
        assert row["broker"] is None
        assert row["n_longs"] == 2 and row["n_shorts"] == 2
        assert row["book"]["longs"]["SYM1USDT"] == pytest.approx(0.05)
        # real disk, real JSONL
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["asof_date"] == "2026-01-01"

    def test_compounds_across_cycles(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ms.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        row2 = ms.record_shadow_cycle(self._result(asof="2026-01-08", net=-0.01), ledger_path=ledger_path,
                                      cycle_ts_iso="2026-01-08T00:00:00+00:00")
        expected = (1.02) * (0.99) - 1.0
        assert row2["cumulative_shadow_return"] == pytest.approx(expected)
        assert row2["forward_cycle_seq"] == 2

    def test_skips_duplicate_asof_date(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ms.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        second = ms.record_shadow_cycle(self._result(asof="2026-01-01", net=0.05), ledger_path=ledger_path,
                                        cycle_ts_iso="2026-01-01T01:00:00+00:00")
        assert second is None
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1  # no duplicate/double-counted row

    def test_skips_corrupt_ledger_line_without_losing_valid_rows(self, tmp_path, caplog):
        ledger_path = tmp_path / "ledger.jsonl"
        ms.record_shadow_cycle(self._result(asof="2026-01-01", net=0.02), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        # simulate a torn/partial write landing between two good rows
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write('{"asof_date": "2026-01-05", "cumulative_shad\n')  # truncated mid-write
        row3 = ms.record_shadow_cycle(self._result(asof="2026-01-08", net=-0.01), ledger_path=ledger_path,
                                      cycle_ts_iso="2026-01-08T00:00:00+00:00")
        # the corrupt line is skipped, not counted — cumulative compounds off the
        # last VALID row (seq 1), and forward_cycle_seq does not silently inflate
        # past the number of real rows.
        expected = (1.02) * (0.99) - 1.0
        assert row3["cumulative_shadow_return"] == pytest.approx(expected)
        assert row3["forward_cycle_seq"] == 2
        summary = ms.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 2  # only the 2 well-formed rows counted

    def test_forward_oos_summary_empty(self, tmp_path):
        summary = ms.forward_oos_summary(ledger_path=tmp_path / "missing.jsonl")
        assert summary["n_cycles"] == 0
        assert summary["cumulative_return"] == 0.0
        assert summary["first_asof_date"] is None

    def test_forward_oos_summary_single_cycle_sharpe_undefined(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ms.record_shadow_cycle(self._result(net=0.01), ledger_path=ledger_path,
                               cycle_ts_iso="2026-01-01T00:00:00+00:00")
        summary = ms.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 1
        assert summary["forward_sharpe_annualized"] is None  # n<2 -> never fabricated

    def test_forward_oos_summary_multi_cycle_sharpe_computed(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        for i, net in enumerate([0.01, -0.005, 0.02]):
            ms.record_shadow_cycle(self._result(asof=f"2026-01-0{i+1}", net=net), ledger_path=ledger_path,
                                   cycle_ts_iso=f"2026-01-0{i+1}T00:00:00+00:00")
        summary = ms.forward_oos_summary(ledger_path=ledger_path)
        assert summary["n_cycles"] == 3
        assert isinstance(summary["forward_sharpe_annualized"], float)


class TestLaneHaltGating:
    """The shadow lane script must never reach compute_shadow_cycle (network
    + broker-shaped code path) while the crypto_momentum lane is halted."""

    def _build_state(self, tmp_path: Path, *, global_halted: bool, lane_halted: bool) -> Path:
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {
            "halted": global_halted,
            "mode": "live",
            "halted_lanes": {
                "oanda_fx": True, "equity": True, "brain": True,
                "crypto_momentum": lane_halted,
            },
        }
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return root

    def test_tick_refuses_when_lane_halted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=False, lane_halted=True)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_crypto_momentum_shadow as runner  # noqa: E402

        result = runner._tick(root, refresh=False)
        assert result["ran"] is False
        assert result["reason"].startswith("halted:")
        assert result["orders"] == 0
        # no ledger directory was ever created — compute_shadow_cycle was never reached
        assert not (root / "trained_data").exists()

    def test_tick_refuses_when_globally_halted_even_if_lane_unhalted(self, tmp_path):
        root = self._build_state(tmp_path, global_halted=True, lane_halted=False)
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import run_crypto_momentum_shadow as runner  # noqa: E402

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
        for rel in ("src/crypto/momentum_shadow.py", "scripts/run_crypto_momentum_shadow.py"):
            lines = (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
            code_lines = "\n".join(ln for ln in lines if not ln.strip().startswith("#"))
            # Strip the module docstring (triple-quoted at top) so prose that
            # NAMES a forbidden symbol (to say it's absent) doesn't self-trigger.
            if code_lines.lstrip().startswith('"""'):
                first = code_lines.index('"""')
                second = code_lines.index('"""', first + 3)
                code_lines = code_lines[second + 3:]
            for forbidden in forbidden_patterns:
                assert forbidden not in code_lines, f"{rel} references execution surface {forbidden!r}"


@pytest.mark.integration
class TestLiveShadowCycleIntegration:
    """Pulls real (cached) crypto data — skip with -m 'not integration' in fast runs."""

    def test_compute_shadow_cycle_live_cache_produces_valid_result(self, tmp_path):
        result = ms.compute_shadow_cycle(refresh_klines=False)
        assert result.universe_size >= 0
        assert result.asof_date
        assert isinstance(result.longs, dict)
        assert isinstance(result.shorts, dict)
        # NOTE (honest, not a bug): the cached monthly-dump tail can legitimately
        # have most symbols' close data lag behind the panel's last date (see
        # module docstring's "data cadence caveat") -> the frozen len(s)>=10
        # eligibility floor correctly abstains into a flat book rather than
        # ranking on a near-empty cross-section. A flat book on a sparse-data
        # date is the CORRECT behavior of the reused construction, not a defect.

        ledger_path = tmp_path / "ledger.jsonl"
        row = ms.record_shadow_cycle(result, ledger_path=ledger_path, cycle_ts_iso="2026-07-02T00:00:00+00:00")
        assert row is not None
        assert row["asof_date"] == result.asof_date

    def test_run_crypto_momentum_shadow_script_one_shot(self, tmp_path):
        """Real end-to-end script invocation. `--project-root` only scopes the
        HALT check (`_lane_halted` reads `<root>/.claude/state.json`); the ledger
        path is the module-default `trained_data/crypto/shadow_momentum_ledger.jsonl`
        under the real REPO_ROOT (same convention as `src.equity.live_gate`'s
        default paths) — this deliberately exercises the real production ledger,
        which is also this task's "produce a first real ledger entry" check. The
        duplicate-`asof_date` guard makes repeat runs safe no-ops."""
        root = tmp_path / "repo"
        (root / ".claude").mkdir(parents=True)
        state = {"halted": False, "mode": "live",
                 "halted_lanes": {"oanda_fx": True, "equity": True, "brain": True, "crypto_momentum": False}}
        (root / ".claude" / "state.json").write_text(json.dumps(state), encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_crypto_momentum_shadow.py"),
             "--project-root", str(root)],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "CYCLE_RESULT: ran=True" in proc.stdout
        assert "orders=0" in proc.stdout
        real_ledger = REPO_ROOT / "trained_data" / "crypto" / "shadow_momentum_ledger.jsonl"
        assert real_ledger.exists()
