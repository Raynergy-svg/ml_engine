#!/usr/bin/env python3
"""Generate Phase F performance baselines from bounded, real local workloads.

No network calls, broker access, model promotion, or full training runs occur.
Each family exercises its actual data/feature seam and records model-free fit
time as zero where fitting is not part of that strategy construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.equity.research.partitioned_store import EquityResearchStore  # noqa: E402
from src.equity.research.scorer import read_scores_artifact  # noqa: E402
from src.training.performance.metrics import (  # noqa: E402
    PerformanceBaseline,
    PerformanceProbe,
    write_baseline,
)
from src.training.performance.rl_dataset import (  # noqa: E402
    RLMatrixBundle,
    fixed_window_view,
    materialize_rl_matrices,
)
from src.training.risk_target_cache import RiskTargetFeatureCache, pool_final_valid_rows  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "trained_data" / "performance_baselines" / "phase_f"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "trained_data" / "performance_artifacts" / "phase_f"


def _source_identity(paths: list[Path]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
                total += len(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest(), total


def _write_summary(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def profile_legacy_rl(artifacts: Path, max_rows: int) -> tuple[PerformanceProbe, dict, Path]:
    sources = sorted((PROJECT_ROOT / "trained_data/cache/training_data").glob("*.csv"))
    if not sources:
        raise FileNotFoundError("legacy RL baseline requires a cached training CSV")
    source = sources[0]
    source_id, source_bytes = _source_identity([source])
    probe = PerformanceProbe()
    with probe:
        with probe.phase("feature_computation"):
            raw = pd.read_csv(source).tail(max_rows)
            numeric = raw.select_dtypes(include=[np.number]).fillna(0.0)
            if "close" not in numeric:
                raise ValueError(f"{source} has no numeric close column")
            features = numeric.drop(columns=["close"]).to_numpy(dtype=np.float32)
            if features.shape[1] == 0:
                features = numeric[["close"]].pct_change().fillna(0.0).to_numpy(dtype=np.float32)
            sequence_length = min(60, max(1, len(features) // 4))
            windows = fixed_window_view(features, sequence_length=sequence_length)
            predictions = np.full((len(features), 2), 0.5, dtype=np.float32)
            bundle = RLMatrixBundle(
                features=features,
                ensemble_predictions=predictions,
                prices=numeric["close"].to_numpy(dtype=np.float32),
            )
            artifact = materialize_rl_matrices(
                bundle, artifacts / "legacy_rl", source_identity=source_id
            )
        with probe.phase("fit"):
            pass  # preparation baseline: fitting is intentionally separated
        with probe.phase("evaluation"):
            _ = float(np.mean(windows[-1]))
    dataset = {
        "id": source_id,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "rows": len(features),
        "size_bytes": source_bytes,
        "feature_columns": features.shape[1],
        "fixed_windows": len(windows),
    }
    return probe, dataset, artifact


def profile_equity(artifacts: Path, max_rows: int) -> tuple[PerformanceProbe, dict, Path]:
    source = PROJECT_ROOT / "market_data/equity/sp500_prices.parquet"
    if not source.exists():
        raise FileNotFoundError(source)
    source_id, source_bytes = _source_identity([source])
    store = EquityResearchStore(artifacts / "equity_research")
    probe = PerformanceProbe()
    with probe:
        with probe.phase("feature_computation"):
            schema_names = pq.ParquetFile(source).schema_arrow.names
            value_columns = [name for name in schema_names if not name.startswith("__index_level_")]
            selected_columns = value_columns[: min(12, len(value_columns))]
            panel = pd.read_parquet(source, columns=selected_columns).tail(max_rows)
            store.partition_prices(panel)
            selected = store.read_prices(
                tickers=list(panel.columns[: min(6, panel.shape[1])]),
                start=str(panel.index.min()),
                end=str(panel.index.max()),
            )
            returns = selected.pct_change(fill_method=None)
        with probe.phase("fit"):
            pass  # rule-based equity construction has no fitted model
        with probe.phase("evaluation"):
            _ = returns.mean().sort_values(ascending=False)
    dataset = {
        "id": source_id,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "rows": len(panel),
        "entities": panel.shape[1],
        "size_bytes": source_bytes,
        "queried_rows": len(selected),
        "queried_entities": selected.shape[1],
    }
    return probe, dataset, artifacts / "equity_research"


def profile_track_b(artifacts: Path, max_rows: int) -> tuple[PerformanceProbe, dict, Path]:
    candidates = sorted((PROJECT_ROOT / "trained_data/research").glob("*/scores_artifact.json"))
    if not candidates:
        raise FileNotFoundError("Track B baseline requires a scores artifact")
    source = candidates[-1]
    source_id, source_bytes = _source_identity([source])
    probe = PerformanceProbe()
    with probe:
        with probe.phase("feature_computation"):
            scores, meta = read_scores_artifact(source)
            scores = scores[:max_rows]
            composites = np.asarray([score.composite() for score in scores], dtype=np.float64)
        with probe.phase("fit"):
            pass  # frozen Track B scoring is a model-free research construction here
        with probe.phase("evaluation"):
            ranked = np.argsort(composites) if len(composites) else np.empty(0, dtype=int)
            summary = {
                "scores": len(scores),
                "median_composite": float(np.median(composites)) if len(composites) else None,
                "ranked": len(ranked),
                "meta_keys": sorted(meta),
            }
            artifact = _write_summary(artifacts / "track_b" / "summary.json", summary)
    dataset = {
        "id": source_id,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "rows": len(scores),
        "size_bytes": source_bytes,
        "batch_limit": 32,
    }
    return probe, dataset, artifact


def _crypto_sources(kind: str, limit: int = 8) -> list[Path]:
    if kind == "momentum":
        base = PROJECT_ROOT / "crypto_cache/binance/klines/1d"
    else:
        base = PROJECT_ROOT / "crypto_cache/binance/funding"
    return sorted(base.glob("*.parquet"))[:limit]


def profile_crypto(
    family: str, artifacts: Path, max_rows: int
) -> tuple[PerformanceProbe, dict, Path]:
    kind = "momentum" if family == "crypto_momentum" else "carry"
    sources = _crypto_sources(kind)
    if not sources:
        raise FileNotFoundError(f"{family} baseline requires cached Binance parquet")
    source_id, source_bytes = _source_identity(sources)
    probe = PerformanceProbe()
    with probe:
        with probe.phase("feature_computation"):
            series: dict[str, pd.Series] = {}
            for path in sources:
                frame = pd.read_parquet(path).tail(max_rows)
                column = "close" if kind == "momentum" else "funding_rate"
                if column in frame:
                    values = frame[column].astype(float)
                    if kind == "carry":
                        values.index = pd.to_datetime(values.index, utc=True).normalize()
                        values = values.groupby(level=0).sum()
                    series[path.stem] = values
            panel = pd.DataFrame(series).sort_index()
            signal = (
                panel.pct_change(14, fill_method=None).shift(1)
                if kind == "momentum"
                else panel.rolling(3).mean().shift(1)
            )
        with probe.phase("fit"):
            pass  # momentum and carry are frozen rule constructions
        with probe.phase("evaluation"):
            last = signal.dropna(how="all").tail(1)
            ranked = last.rank(axis=1, pct=True) if not last.empty else last
            artifact = _write_summary(
                artifacts / family / "summary.json",
                {"rows": len(panel), "entities": panel.shape[1], "ranked_entities": int(ranked.notna().sum().sum())},
            )
    dataset = {
        "id": source_id,
        "sources": [str(path.relative_to(PROJECT_ROOT)) for path in sources],
        "rows": len(panel),
        "entities": panel.shape[1],
        "size_bytes": source_bytes,
        "dataset_role": kind,
    }
    return probe, dataset, artifact


def profile_risk_target(artifacts: Path, max_rows: int) -> tuple[PerformanceProbe, dict, Path]:
    source = PROJECT_ROOT / "market_data/factor/EUR_USD_D.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    source_id, source_bytes = _source_identity([source])
    cache = RiskTargetFeatureCache(artifacts / "risk_target")
    probe = PerformanceProbe()
    with probe:
        with probe.phase("feature_computation"):
            raw = pd.read_csv(source).tail(max_rows).reset_index(drop=True)
            raw["date"] = pd.to_datetime(raw["date"], utc=True)
            features = cache.get_or_compute("EUR_USD", raw)
            pooled = pool_final_valid_rows({"EUR_USD": features})
        with probe.phase("fit"):
            pass  # feature/cache baseline; model fitting remains a separate measured phase
        with probe.phase("evaluation"):
            _ = pooled.select_dtypes(include=[np.number]).mean()
    dataset = {
        "id": source_id,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "rows": len(raw),
        "valid_rows": len(pooled),
        "size_bytes": source_bytes,
    }
    return probe, dataset, artifacts / "risk_target"


PROFILERS: dict[str, Callable[[Path, int], tuple[PerformanceProbe, dict, Path]]] = {
    "legacy_rl": profile_legacy_rl,
    "equity_research": profile_equity,
    "track_b": profile_track_b,
    "crypto_momentum": lambda artifacts, rows: profile_crypto("crypto_momentum", artifacts, rows),
    "crypto_carry": lambda artifacts, rows: profile_crypto("crypto_carry", artifacts, rows),
    "risk_target": profile_risk_target,
}


def generate_baselines(
    *, output_dir: Path, artifacts_dir: Path, families: list[str], max_rows: int
) -> list[PerformanceBaseline]:
    reports: list[PerformanceBaseline] = []
    for family in families:
        probe, dataset, artifact = PROFILERS[family](artifacts_dir, max_rows)
        report = probe.baseline(
            family=family,
            workload=f"phase_f_{family}_bounded_baseline",
            dataset=dataset,
            artifact_path=artifact,
        )
        write_baseline(report, output_dir / f"{family}.json")
        reports.append(report)
        print(f"{family}: {report.baseline_digest} ({report.wall_time_seconds:.3f}s)")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--families", nargs="+", choices=sorted(PROFILERS), default=list(PROFILERS))
    args = parser.parse_args()
    if args.max_rows < 100:
        parser.error("--max-rows must be at least 100")
    generate_baselines(
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        families=args.families,
        max_rows=args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
