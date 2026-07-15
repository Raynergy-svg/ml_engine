"""Phase F regression tests for profiling and compute-once RL matrices."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.crypto.research_store import (
    load_history,
    materialize_eligibility,
    materialize_strategy_dataset,
    partition_history,
)
from src.equity.research.contracts import ResearchScore
from src.equity.research.partitioned_store import EquityResearchStore
from src.equity.research.pit_text_loader import load_pit_filings
from src.equity.research.scorer import BaselineScorer, ScoringTask, score_prepared_batch
from src.training.performance.partition_store import ImmutableMonthlyParquetStore
from src.training.performance.metrics import (
    PerformanceBaseline,
    PerformanceProbe,
    write_baseline,
)
from src.training.performance.rl_dataset import (
    RLMatrixBundle,
    find_rl_matrices,
    fixed_window_view,
    load_rl_matrices,
    materialize_rl_matrices,
    predict_fixed_windows,
)
from src.training.risk_target_cache import RiskTargetFeatureCache, pool_final_valid_rows


class _MeanModel:
    def __init__(self) -> None:
        self.batch_shapes: list[tuple[int, ...]] = []

    def predict(self, values, *, verbose, batch_size):
        assert verbose == 0
        assert batch_size == 2
        self.batch_shapes.append(tuple(values.shape))
        return values.mean(axis=(1, 2), keepdims=True)


def test_fixed_window_view_matches_legacy_prefix_loop_without_copy():
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    expected = np.array([features[i - 4 : i] for i in range(4, len(features))])
    actual = fixed_window_view(features, sequence_length=4)
    assert np.array_equal(actual, expected)
    assert np.shares_memory(actual, features)
    assert actual.flags.writeable is False


def test_prediction_is_batched_over_fixed_windows():
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    model = _MeanModel()
    result = predict_fixed_windows(
        model,
        features,
        sequence_length=4,
        feature_dimension=3,
        batch_size=2,
    )
    expected = np.array(
        [features[i - 4 : i].mean() for i in range(4, len(features))],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)
    assert model.batch_shapes == [(2, 4, 3), (2, 4, 3), (2, 4, 3)]


def _bundle() -> RLMatrixBundle:
    return RLMatrixBundle(
        features=np.arange(40, dtype=np.float32).reshape(10, 4),
        ensemble_predictions=np.full((10, 2), 0.5, dtype=np.float32),
        prices=np.linspace(1.0, 2.0, 10, dtype=np.float32),
    )


def test_rl_matrices_are_content_addressed_reused_and_memory_mapped(tmp_path):
    first = materialize_rl_matrices(_bundle(), tmp_path, source_identity="source-a")
    second = materialize_rl_matrices(_bundle(), tmp_path, source_identity="source-a")
    assert first == second
    assert find_rl_matrices(tmp_path, source_identity="source-a") == first
    loaded = load_rl_matrices(first)
    assert isinstance(loaded.features, np.memmap)
    assert np.array_equal(loaded.features, _bundle().features)
    assert json.loads((first / "manifest.json").read_text())["rows"] == 10


def test_rl_matrix_tampering_fails_before_fit(tmp_path):
    artifact = materialize_rl_matrices(_bundle(), tmp_path, source_identity="source-a")
    with (artifact / "features.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_rl_matrices(artifact)


def test_performance_report_contains_every_phase_f_exit_metric(tmp_path):
    artifact = tmp_path / "artifact.bin"
    with PerformanceProbe(sample_interval_seconds=0.001) as probe:
        with probe.phase("feature_computation"):
            values = np.arange(10_000, dtype=np.float64)
        with probe.phase("fit"):
            values = values * 2
        with probe.phase("evaluation"):
            assert values.mean() > 0
        artifact.write_bytes(values.tobytes())
        time.sleep(0.003)
    report = probe.baseline(
        family="fixture",
        workload="phase-f-test",
        dataset={"id": "fixture-v1", "rows": len(values), "size_bytes": values.nbytes},
        artifact_path=artifact,
    )
    output = write_baseline(report, tmp_path / "baseline.json")
    payload = json.loads(output.read_text())
    assert payload["wall_time_seconds"] >= 0
    assert payload["peak_rss_bytes"] > 0
    assert payload["cpu_utilization_percent"] >= 0
    assert "gpu_utilization" in payload
    assert "io_volume" in payload
    assert payload["io_volume"]["logical_input_bytes"] == values.nbytes
    assert payload["dataset"]["size_bytes"] == values.nbytes
    assert set(payload["phases_seconds"]) >= {
        "feature_computation",
        "fit",
        "evaluation",
    }
    assert payload["artifact"]["size_bytes"] == artifact.stat().st_size


def test_monthly_store_pushes_entity_date_and_column_predicates(tmp_path):
    store = ImmutableMonthlyParquetStore(tmp_path)
    dates = pd.date_range("2024-01-20", periods=40, tz="UTC")
    for entity, offset in (("AAA", 0.0), ("BBB", 100.0)):
        frame = pd.DataFrame(
            {"value": np.arange(40) + offset, "unused": "large-payload"},
            index=dates,
        )
        store.write("prices", entity, frame)
    result = store.query(
        "prices",
        entities=["BBB"],
        start="2024-02-01",
        end="2024-02-03",
        columns=["value"],
    )
    assert list(result.columns) == ["event_time", "entity", "value"]
    assert result["entity"].unique().tolist() == ["BBB"]
    assert len(result) == 3
    assert "unused" not in result


def test_monthly_store_appends_only_new_rows_and_refuses_history_revision(tmp_path):
    store = ImmutableMonthlyParquetStore(tmp_path)
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    original = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=dates)
    store.write("prices", "AAA", original)
    extended = pd.concat(
        [
            original,
            pd.DataFrame(
                {"value": [4.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-01-04", tz="UTC")]),
            ),
        ]
    )
    store.write("prices", "AAA", extended)
    queried = store.query(
        "prices",
        entities=["AAA"],
        start="2024-01-01",
        end="2024-01-31",
        columns=["value"],
    )
    assert queried["value"].tolist() == [1.0, 2.0, 3.0, 4.0]
    revised = extended.copy()
    revised.iloc[0, 0] = 99.0
    with pytest.raises(ValueError, match="revision refused"):
        store.write("prices", "AAA", revised)


def test_equity_store_reuses_pit_snapshot_and_reads_only_requested_names(tmp_path):
    store = EquityResearchStore(tmp_path)
    dates = pd.date_range("2024-01-01", periods=10, tz="UTC")
    prices = pd.DataFrame({"AAA": np.arange(10.0), "BBB": np.arange(10.0) + 20}, index=dates)
    store.partition_prices(prices)
    selected = store.read_prices(tickers=["BBB"], start="2024-01-03", end="2024-01-05")
    assert selected.columns.tolist() == ["BBB"]
    assert len(selected) == 3

    universe = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["A", "B"]})
    first = store.write_pit_universe(universe, as_of="2024-01-01")
    second = store.write_pit_universe(universe, as_of="2024-01-01")
    assert first == second
    changed_universe = universe.copy()
    changed_universe.loc[0, "sector"] = "changed"
    with pytest.raises(ValueError, match="already frozen"):
        store.write_pit_universe(changed_universe, as_of="2024-01-01")
    loaded = store.read_pit_universe(as_of="2024-01-15", tickers=["AAA"])
    assert loaded["ticker"].tolist() == ["AAA"]
    rebalance = pd.DataFrame({"ticker": ["AAA", "BBB"], "weight": [0.6, 0.4]})
    store.write_rebalance_partition(rebalance, rebalance_date="2024-01-31")
    selected_rebalance = store.read_rebalance_partition(
        rebalance_date="2024-01-31", tickers=["BBB"]
    )
    assert selected_rebalance.to_dict(orient="records") == [
        {"ticker": "BBB", "weight": 0.4}
    ]


def test_filing_and_scoring_workers_refuse_unbounded_corpora():
    items = [(f"T{i}", str(i + 1), "2024-01-01") for i in range(65)]
    with pytest.raises(ValueError, match="maximum is 64"):
        load_pit_filings(items)

    tasks = [
        ScoringTask(
            ticker=f"T{i}",
            as_of="2024-01-01",
            form="10-K",
            text_to_score="strong demand and free cash flow",
            blinded=True,
        )
        for i in range(3)
    ]
    scores, failures = score_prepared_batch(tasks, BaselineScorer(), max_batch_size=3)
    assert len(scores) == 3
    assert all(isinstance(score, ResearchScore) for score in scores)
    assert failures == {}
    with pytest.raises(ValueError, match="maximum is 2"):
        score_prepared_batch(tasks, BaselineScorer(), max_batch_size=2)


def test_crypto_histories_are_month_partitioned_and_eligibility_is_separate(tmp_path):
    dates = pd.date_range("2024-01-20", periods=40, tz="UTC", name="open_time")
    frame = pd.DataFrame({"close": np.arange(40.0), "volume": 100.0}, index=dates)
    paths = partition_history("BTCUSDT", "klines_1d", frame, root=tmp_path)
    assert len(paths) == 2
    queried = load_history(
        ["BTCUSDT"],
        "klines_1d",
        start="2024-02-01",
        end="2024-02-02",
        columns=["close"],
        root=tmp_path,
    )
    assert len(queried) == 2
    eligible = pd.DataFrame(
        {"BTCUSDT": [True, False], "ETHUSDT": [False, True]},
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )
    momentum = materialize_eligibility(
        eligible, strategy_id="crypto_momentum", parameters={"lookback": 14}, root=tmp_path
    )
    carry = materialize_eligibility(
        eligible, strategy_id="crypto_carry", parameters={"lookback": 3}, root=tmp_path
    )
    assert momentum != carry
    assert momentum.exists() and carry.exists()
    changed = eligible.copy()
    changed.iloc[0, 0] = False
    assert materialize_eligibility(
        changed,
        strategy_id="crypto_momentum",
        parameters={"lookback": 14},
        root=tmp_path,
    ) != momentum
    momentum_data = materialize_strategy_dataset(
        frame[["close"]],
        strategy_id="crypto_momentum",
        dataset_role="lagged_signal_input",
        parameters={"lookback": 14},
        root=tmp_path,
    )
    carry_data = materialize_strategy_dataset(
        frame[["close"]],
        strategy_id="crypto_carry",
        dataset_role="funding_signal_input",
        parameters={"lookback": 3},
        root=tmp_path,
    )
    assert momentum_data != carry_data


def _ohlcv(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, tz="UTC")
    close = np.exp(np.linspace(0, 0.2, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.arange(n, dtype=float) + 100,
        }
    )


def test_risk_target_cache_is_per_pair_immutable_and_pools_only_valid_rows(tmp_path):
    cache = RiskTargetFeatureCache(tmp_path)
    raw = _ohlcv()
    first = cache.get_or_compute("EUR_USD", raw)
    paths_before = sorted(tmp_path.rglob("*.parquet"))
    second = cache.get_or_compute("EUR_USD", raw)
    assert len(paths_before) == 1
    assert sorted(tmp_path.rglob("*.parquet")) == paths_before
    pd.testing.assert_frame_equal(first, second)
    pooled = pool_final_valid_rows({"EUR_USD": first, "GBP_USD": second})
    assert set(pooled["pair"]) == {"EUR_USD", "GBP_USD"}
    feature_columns = [column for column in first if column.startswith("realized_vol_")]
    assert not pooled[feature_columns].isna().any().any()


@pytest.mark.parametrize(
    "family",
    [
        "legacy_rl",
        "equity_research",
        "track_b",
        "crypto_momentum",
        "crypto_carry",
        "risk_target",
    ],
)
def test_every_phase_f_family_has_a_complete_real_baseline(family):
    root = Path(__file__).resolve().parents[1]
    path = root / "trained_data/performance_baselines/phase_f" / f"{family}.json"
    report = PerformanceBaseline.model_validate_json(path.read_text(encoding="utf-8"))
    assert report.family == family
    assert report.dataset["rows"] > 0
    assert report.dataset["size_bytes"] > 0
    assert report.wall_time_seconds >= 0
    assert report.peak_rss_bytes > 0
    assert report.cpu_utilization_percent >= 0
    assert set(report.phases_seconds) >= {"feature_computation", "fit", "evaluation"}
    assert report.io_volume["logical_input_bytes"] == report.dataset["size_bytes"]
    assert report.artifact["size_bytes"] > 0
