"""Deterministic, authority-free execution-cost model training and replay."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import GateResult, GateStatus

from .models import ModelResult

FEATURES = (
    "spread_bps",
    "volatility_1m",
    "participation_rate",
    "log_order_notional_usd",
    "book_imbalance_abs",
)
RAW_FEATURES = (
    "spread_bps", "volatility_1m", "participation_rate",
    "order_notional_usd", "book_imbalance",
)


@dataclass(frozen=True)
class EvaluationParams:
    oos_start: str = "2026-01-01T00:00:00+00:00"
    ridge_penalty: float = 1e-6
    min_train_rows: int = 20
    min_test_rows: int = 10
    max_p95_underprediction_bps: float = 2.0
    replay_tolerance: float = 1e-10


def _time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _rows(partitions: Mapping[str, bytes]) -> list[dict]:
    if not partitions:
        raise ValueError("at least one immutable fill partition is required")
    rows: list[dict] = []
    seen: set[str] = set()
    for partition_id, data in sorted(partitions.items()):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"partition {partition_id!r} is not UTF-8") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"partition {partition_id!r} line {line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError("each fill record must be a JSON object")
            required = {"order_id", "instrument", "feature_as_of", "decision_at", "fill_at",
                        "realized_cost_bps", *RAW_FEATURES}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"fill record missing fields {missing}")
            order_id = row["order_id"]
            if not isinstance(order_id, str) or not order_id:
                raise ValueError("order_id must be a non-empty string")
            if order_id in seen:
                raise ValueError(f"duplicate order_id {order_id!r}")
            seen.add(order_id)
            feature_at = _time(row["feature_as_of"], "feature_as_of")
            decision_at = _time(row["decision_at"], "decision_at")
            fill_at = _time(row["fill_at"], "fill_at")
            if feature_at > decision_at or decision_at >= fill_at:
                raise ValueError(f"order {order_id!r} violates causal feature/decision/fill ordering")
            numeric: dict[str, float] = {}
            for name in (*RAW_FEATURES, "realized_cost_bps"):
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"order {order_id!r} has non-numeric {name}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"order {order_id!r} has non-finite {name}")
                numeric[name] = value
            if numeric["spread_bps"] < 0 or numeric["volatility_1m"] < 0:
                raise ValueError("spread and volatility cannot be negative")
            if not 0 <= numeric["participation_rate"] <= 1:
                raise ValueError("participation_rate must be in [0, 1]")
            if numeric["order_notional_usd"] <= 0 or abs(numeric["book_imbalance"]) > 1:
                raise ValueError("notional must be positive and book_imbalance must be in [-1, 1]")
            rows.append({**row, **numeric, "_fill_at": fill_at})
    if not rows:
        raise ValueError("fill partitions contain no records")
    return sorted(rows, key=lambda row: (row["_fill_at"], row["order_id"]))


def _vector(row: dict) -> list[float]:
    return [
        row["spread_bps"], row["volatility_1m"], row["participation_rate"],
        math.log1p(row["order_notional_usd"]), abs(row["book_imbalance"]),
    ]


def _solve(matrix: list[list[float]], target: list[float]) -> list[float]:
    """Solve a small positive-definite system using pivoted elimination."""
    n = len(target)
    augmented = [list(matrix[i]) + [target[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise ValueError("execution-cost design matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * base
                for current, base in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[i][-1] for i in range(n)]


def _fit(train: list[dict], penalty: float) -> tuple[list[float], list[float], list[float]]:
    vectors = [_vector(row) for row in train]
    means = [sum(row[j] for row in vectors) / len(vectors) for j in range(len(FEATURES))]
    scales = []
    for j, mean in enumerate(means):
        variance = sum((row[j] - mean) ** 2 for row in vectors) / len(vectors)
        scales.append(max(math.sqrt(variance), 1e-12))
    design = [[1.0] + [(value - means[j]) / scales[j] for j, value in enumerate(row)] for row in vectors]
    width = len(design[0])
    gram = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    rhs = [sum(row[i] * sample["realized_cost_bps"] for row, sample in zip(design, train, strict=True)) for i in range(width)]
    for i in range(1, width):
        gram[i][i] += penalty
    return _solve(gram, rhs), means, scales


def _predict(row: dict, coefficients: list[float], means: list[float], scales: list[float]) -> float:
    vector = _vector(row)
    return coefficients[0] + sum(
        coefficients[j + 1] * ((vector[j] - means[j]) / scales[j])
        for j in range(len(FEATURES))
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def evaluate_partitions(partitions: Mapping[str, bytes], params: EvaluationParams | None = None) -> ModelResult:
    params = params or EvaluationParams()
    if params.ridge_penalty < 0:
        raise ValueError("ridge_penalty cannot be negative")
    cutoff = _time(params.oos_start, "oos_start")
    rows = _rows(partitions)
    train = [row for row in rows if row["_fill_at"] < cutoff]
    test = [row for row in rows if row["_fill_at"] >= cutoff]
    if len(train) < params.min_train_rows or len(test) < params.min_test_rows:
        raise ValueError(
            f"insufficient chronological rows: train={len(train)}, test={len(test)}; "
            f"required={params.min_train_rows}/{params.min_test_rows}"
        )
    coefficients, means, scales = _fit(train, params.ridge_penalty)
    actual = [row["realized_cost_bps"] for row in test]
    predicted = [_predict(row, coefficients, means, scales) for row in test]
    baseline_level = float(median(row["realized_cost_bps"] for row in train))
    baseline = [baseline_level] * len(test)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)
    baseline_mae = sum(abs(a - p) for a, p in zip(actual, baseline, strict=True)) / len(actual)
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual))
    under = [max(0.0, a - p) for a, p in zip(actual, predicted, strict=True)]
    baseline_under = [max(0.0, a - p) for a, p in zip(actual, baseline, strict=True)]
    p95_under = _percentile(under, 0.95)
    baseline_p95_under = _percentile(baseline_under, 0.95)
    metrics = {
        "oos_mae_bps": mae,
        "oos_baseline_mae_bps": baseline_mae,
        "oos_rmse_bps": rmse,
        "oos_p95_underprediction_bps": p95_under,
        "oos_baseline_p95_underprediction_bps": baseline_p95_under,
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
    }
    gates = (
        GateResult(gate_id="oos_mae_beats_train_median", status=GateStatus.PASS if mae < baseline_mae else GateStatus.FAIL,
                   observed=mae, threshold=baseline_mae, reason="chronological OOS MAE must beat a train-only median baseline"),
        GateResult(gate_id="tail_underprediction_bounded", status=GateStatus.PASS if p95_under <= params.max_p95_underprediction_bps and p95_under <= baseline_p95_under else GateStatus.FAIL,
                   observed=p95_under, threshold=min(params.max_p95_underprediction_bps, baseline_p95_under),
                   reason="OOS p95 cost underprediction must be absolutely bounded and no worse than baseline"),
    )
    artifact = {
        "schema_version": 1,
        "model_family": "ridge_linear",
        "target": "realized_execution_cost_bps",
        "features": list(FEATURES),
        "causal_lag": "feature_as_of <= decision_at < fill_at; label known only after fill",
        "oos_start": cutoff.isoformat(),
        "ridge_penalty": params.ridge_penalty,
        "coefficients": coefficients,
        "feature_means": means,
        "feature_scales": scales,
        "baseline_train_median_bps": baseline_level,
    }
    return ModelResult(
        metrics=metrics,
        gates=gates,
        passed=all(g.status == GateStatus.PASS for g in gates),
        artifact_bytes=canonical_bytes(artifact),
        effective_sample_size=float(len(test)),
        temporal_holdout=f"fills at or after {cutoff.isoformat()} (strict chronological holdout)",
        incumbent_comparison={"baseline": "train-only median realized cost", "baseline_mae_bps": baseline_mae},
    )


__all__ = ["FEATURES", "RAW_FEATURES", "EvaluationParams", "evaluate_partitions"]
