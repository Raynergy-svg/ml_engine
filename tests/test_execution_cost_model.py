"""No-mock tests for the P2 offline execution-cost model (docs/ENGINEERING_BRAIN.md).

Covers: (1) fill-quality extraction from a hand-built ORDER_FILL transaction
matches hand-calculated spread/slippage (reproduce-then-resolve); (2)
malformed/partial transactions are skipped, not crashed on or fabricated;
(3) the model degrades honestly (source="insufficient_data", no fabricated
numbers) when fill/tick samples are sparse; (4) tick-derived spread stats are
None (not zero) when there is no tick coverage yet; (5) holdout validation
reports an honest MAE comparison, including "insufficient data" for
under-sampled instruments; (6) atomic JSON persistence round-trips; (7) a
real-data sanity smoke test against the repo's actual transactions.jsonl if
present (skipped otherwise -- no fabricated real-data assumptions in CI);
(8) structural guard -- no broker/execution import anywhere in this model.

Real classes, real disk (tmp_path), no `unittest.mock` -- per repo policy.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.execution_cost_model import (  # noqa: E402
    DEFAULT_TRANSACTIONS_PATH,
    build_cost_model,
    estimate_execution_cost,
    fill_quality_from_transaction,
    fill_stats_by_instrument,
    load_fill_qualities,
    save_cost_model,
    tick_spread_stats,
    validate_against_holdout,
)


def _make_fill(
    instrument="EUR_USD",
    bid=1.10000,
    ask=1.10020,
    price=1.10015,
    units=100_000,
    requested_units=None,
    half_spread_cost=5.0,
    time="2026-07-01T12:00:00.000000000Z",
):
    return {
        "type": "ORDER_FILL",
        "instrument": instrument,
        "time": time,
        "price": price,
        "units": str(units),
        "requestedUnits": str(requested_units if requested_units is not None else units),
        "halfSpreadCost": str(half_spread_cost),
        "fullPrice": {
            "bids": [{"price": str(bid), "liquidity": "500000"}],
            "asks": [{"price": str(ask), "liquidity": "500000"}],
        },
    }


class TestFillQualityExtraction:
    def test_buy_fill_slippage_matches_hand_calculation(self):
        tx = _make_fill(bid=1.10000, ask=1.10020, price=1.10015, units=100_000)
        fq = fill_quality_from_transaction(tx)

        assert fq is not None
        assert fq.side == "buy"
        assert fq.mid == pytest.approx(1.10010)
        assert fq.spread_raw == pytest.approx(0.00020)
        # buy fills above mid = adverse = positive slippage
        assert fq.slippage_raw == pytest.approx(1.10015 - 1.10010)
        assert fq.notional_units == pytest.approx(100_000)
        assert fq.half_spread_cost == pytest.approx(5.0)
        assert fq.partial_fill is False

    def test_sell_fill_slippage_sign_convention(self):
        # Sell filling BELOW mid is adverse -> positive slippage under our convention.
        tx = _make_fill(bid=1.10000, ask=1.10020, price=1.10002, units=-100_000)
        fq = fill_quality_from_transaction(tx)

        assert fq.side == "sell"
        assert fq.slippage_raw == pytest.approx(1.10010 - 1.10002)
        assert fq.slippage_raw > 0

    def test_favorable_fill_yields_negative_slippage(self):
        # Buy filling BELOW mid is favorable -> negative ("cost") slippage.
        tx = _make_fill(bid=1.10000, ask=1.10020, price=1.10005, units=100_000)
        fq = fill_quality_from_transaction(tx)
        assert fq.slippage_raw < 0

    def test_partial_fill_detected(self):
        tx = _make_fill(units=80_000, requested_units=100_000)
        fq = fill_quality_from_transaction(tx)
        assert fq.partial_fill is True

    def test_missing_book_ladder_returns_none_not_fabricated(self):
        tx = _make_fill()
        tx["fullPrice"] = {"bids": [], "asks": []}
        assert fill_quality_from_transaction(tx) is None

    def test_missing_price_field_returns_none(self):
        tx = _make_fill()
        del tx["price"]
        assert fill_quality_from_transaction(tx) is None

    def test_malformed_transaction_never_raises(self):
        assert fill_quality_from_transaction({"type": "ORDER_FILL"}) is None
        assert fill_quality_from_transaction({}) is None


class TestLoadFillQualities:
    def test_load_skips_corrupt_lines_and_non_fill_transactions(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        lines = [
            json.dumps(_make_fill(instrument="EUR_USD")),
            "not json at all {{{",
            json.dumps({"type": "ORDER_CANCEL", "instrument": "EUR_USD"}),
            json.dumps(_make_fill(instrument="GBP_USD")),
            "",
        ]
        path.write_text("\n".join(lines))

        qualities = load_fill_qualities(path)

        assert len(qualities) == 2
        assert {q.instrument for q in qualities} == {"EUR_USD", "GBP_USD"}

    def test_load_from_missing_path_returns_empty(self, tmp_path):
        assert load_fill_qualities(tmp_path / "nonexistent.jsonl") == []


class TestFillStatsByInstrument:
    def test_stats_computed_correctly_for_known_fills(self):
        qualities = [
            fill_quality_from_transaction(_make_fill(bid=1.1000, ask=1.1002, price=1.1001)),
            fill_quality_from_transaction(_make_fill(bid=1.1000, ask=1.1004, price=1.1003)),
        ]
        stats = fill_stats_by_instrument(qualities)

        assert stats["EUR_USD"]["n"] == 2
        assert stats["EUR_USD"]["spread_mean"] == pytest.approx((0.0002 + 0.0004) / 2)


class TestTickSpreadStats:
    def test_returns_none_when_no_tick_coverage(self, tmp_path):
        assert tick_spread_stats("EUR_USD", tick_root=tmp_path) is None

    def test_computes_spread_stats_from_real_parquet_partitions(self, tmp_path):
        inst_root = tmp_path / "EUR_USD" / "2026" / "07"
        inst_root.mkdir(parents=True)
        today = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "bid": [1.1000] * 250,
            "ask": [1.1002] * 250,
        })
        df.to_parquet(inst_root / f"{today.day:02d}.parquet")
        # Rewrite the partition under today's actual year/month so lookback matches.
        real_root = tmp_path / "EUR_USD" / f"{today.year}" / f"{today.month:02d}"
        real_root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(real_root / f"{today.day:02d}.parquet")

        stats = tick_spread_stats("EUR_USD", tick_root=tmp_path, lookback_days=30)

        assert stats is not None
        assert stats["n"] == 250
        assert stats["spread_mean"] == pytest.approx(0.0002)

    def test_stale_partitions_outside_lookback_are_excluded(self, tmp_path):
        old = datetime.now(timezone.utc) - timedelta(days=400)
        old_root = tmp_path / "EUR_USD" / f"{old.year}" / f"{old.month:02d}"
        old_root.mkdir(parents=True)
        pd.DataFrame({"bid": [1.10], "ask": [1.1002]}).to_parquet(old_root / f"{old.day:02d}.parquet")

        assert tick_spread_stats("EUR_USD", tick_root=tmp_path, lookback_days=30) is None


class TestEstimateExecutionCostHonestDegradation:
    def test_insufficient_data_reports_none_not_zero(self, tmp_path):
        empty_path = tmp_path / "transactions.jsonl"
        empty_path.write_text("")

        est = estimate_execution_cost("EUR_USD", transactions_path=empty_path, tick_root=tmp_path)

        assert est.source == "insufficient_data"
        assert est.spread_raw is None
        assert est.slippage_raw is None
        assert est.confidence == 0.0
        assert any("insufficient data" in n for n in est.notes)

    def test_sufficient_fills_yields_fills_sourced_estimate(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        fills = [_make_fill(bid=1.1000, ask=1.1002, price=1.1001 + i * 0.00001) for i in range(5)]
        path.write_text("\n".join(json.dumps(f) for f in fills))

        est = estimate_execution_cost("EUR_USD", transactions_path=path, tick_root=tmp_path)

        assert est.source == "fills"
        assert est.sample_count_fills == 5
        assert est.spread_raw == pytest.approx(0.0002)
        assert est.slippage_raw is not None
        assert est.confidence > 0.0

    def test_known_pip_instrument_converts_to_pips(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        fills = [_make_fill(bid=1.1000, ask=1.1002, price=1.1001) for _ in range(5)]
        path.write_text("\n".join(json.dumps(f) for f in fills))

        est = estimate_execution_cost("EUR_USD", transactions_path=path, tick_root=tmp_path)

        assert est.spread_pips == pytest.approx(est.spread_raw / 0.0001)

    def test_unknown_instrument_reports_raw_only_no_fabricated_pips(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        fills = [_make_fill(instrument="ZZZ_FAKE", bid=1.1, ask=1.1002, price=1.1001) for _ in range(5)]
        path.write_text("\n".join(json.dumps(f) for f in fills))

        est = estimate_execution_cost("ZZZ_FAKE", transactions_path=path, tick_root=tmp_path)

        assert est.spread_pips is None
        assert any("unknown pip_value" in n for n in est.notes)


class TestValidateAgainstHoldout:
    def test_insufficient_samples_reports_honestly(self):
        qualities = [fill_quality_from_transaction(_make_fill()) for _ in range(2)]
        result = validate_against_holdout(qualities)
        assert result["EUR_USD"]["holdout_n"] == 0
        assert "insufficient data" in result["EUR_USD"]["reason"]

    def test_model_mae_computed_against_naive_baseline(self):
        # Consistent slight positive slippage across all fills -> model (mean)
        # should beat the naive "assume zero" baseline on held-out data.
        fills = [
            _make_fill(bid=1.1000, ask=1.1002, price=1.10011 + i * 0.000001, time=f"2026-07-0{(i % 9) + 1}T00:00:00Z")
            for i in range(20)
        ]
        qualities = [fill_quality_from_transaction(f) for f in fills]

        result = validate_against_holdout(qualities, holdout_frac=0.25)

        assert result["EUR_USD"]["holdout_n"] == 5
        assert result["EUR_USD"]["model_beats_naive"] is True


class TestPersistence:
    def test_save_and_reload_round_trips(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        fills = [_make_fill(bid=1.1000, ask=1.1002, price=1.1001) for _ in range(5)]
        path.write_text("\n".join(json.dumps(f) for f in fills))

        model = build_cost_model(transactions_path=path, tick_root=tmp_path)
        out_path = save_cost_model(model, path=tmp_path / "cost_model.json")

        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert payload["advisory_only"] is True
        assert "EUR_USD" in payload["estimates"]
        assert list(tmp_path.glob("*.tmp")) == []  # no leftover tmp file

    def test_save_merges_into_existing_artifact_instead_of_clobbering(self, tmp_path):
        path = tmp_path / "transactions.jsonl"
        fills = (
            [_make_fill(instrument="EUR_USD", bid=1.1000, ask=1.1002, price=1.1001) for _ in range(5)]
            + [_make_fill(instrument="GBP_USD", bid=1.30, ask=1.3002, price=1.3001) for _ in range(5)]
        )
        path.write_text("\n".join(json.dumps(f) for f in fills))
        out_path = tmp_path / "cost_model.json"

        full_model = build_cost_model(transactions_path=path, tick_root=tmp_path)
        save_cost_model(full_model, path=out_path)
        assert set(json.loads(out_path.read_text())["estimates"]) == {"EUR_USD", "GBP_USD"}

        # A later partial run (e.g. --instrument EUR_USD) must not erase GBP_USD.
        partial_model = build_cost_model(["EUR_USD"], transactions_path=path, tick_root=tmp_path)
        save_cost_model(partial_model, path=out_path)

        estimates = json.loads(out_path.read_text())["estimates"]
        assert set(estimates) == {"EUR_USD", "GBP_USD"}


@pytest.mark.integration
class TestRealDataSmoke:
    """Sanity check against the repo's real ORDER_FILL history, if present."""

    def test_real_transactions_produce_sane_estimates(self):
        if not DEFAULT_TRANSACTIONS_PATH.exists():
            pytest.skip("no real trained_data/oanda/transactions.jsonl in this environment")

        qualities = load_fill_qualities()
        if not qualities:
            pytest.skip("transactions.jsonl exists but has no ORDER_FILL rows")

        model = build_cost_model()
        assert model  # at least one instrument estimated
        for inst, est in model.items():
            if est.source == "insufficient_data":
                continue
            assert est.spread_raw is not None and est.spread_raw >= 0
            assert est.confidence >= 0.0


class TestNoExecutionPath:
    """Structural guard: the cost model never touches a broker/order path."""

    def test_no_broker_or_order_references(self):
        forbidden_patterns = (
            "place_equity_order(", "place_market_order(", "import src.brokers.oanda",
            "OandaPracticeClient(", "execute_trade(", "create_broker(",
            "import src.scanner.execution", "from src.scanner.execution", "/orders", "/trades",
        )
        for rel in ("src/data/execution_cost_model.py", "scripts/build_execution_cost_model.py"):
            code = (REPO_ROOT / rel).read_text(encoding="utf-8")
            for forbidden in forbidden_patterns:
                assert forbidden not in code, f"{rel} references execution surface {forbidden!r}"
