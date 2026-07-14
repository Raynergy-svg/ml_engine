"""Independent Track B score-artifact × rebalance-date × fold evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import sha256_bytes

from .lineage import validate_score_lineage

MIN_NAMES = 5
MIN_FORWARD_CELLS = 3
RANK_IC_BAR = 0.03
PLACEBO_ABS_LIMIT = 0.03


@dataclass(frozen=True)
class EvaluationParams:
    expected_cells: tuple[str, ...] = ()
    min_forward_cells: int = MIN_FORWARD_CELLS
    rank_ic_bar: float = RANK_IC_BAR
    placebo_abs_limit: float = PLACEBO_ABS_LIMIT
    replay_tolerance: float = 1e-12


def evaluation_cell_id(score_artifact_digest: str, rebalance_date: str, fold: str) -> str:
    if len(score_artifact_digest) != 64 or any(c not in "0123456789abcdef" for c in score_artifact_digest):
        raise ValueError("score_artifact_digest must be a lowercase SHA-256 digest")
    if any(not value or "::" in value for value in (rebalance_date, fold)):
        raise ValueError("rebalance_date and fold must be non-empty and cannot contain '::'")
    return f"{score_artifact_digest}::{rebalance_date}::{fold}"


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[ordered[offset]] = average
        position = end
    return ranks


def _corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def _parse_partition(cell_id: str, data: bytes) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"evaluation cell {cell_id!r} is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"evaluation cell {cell_id!r} line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"evaluation cell {cell_id!r} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"evaluation cell {cell_id!r} is empty")
    return rows


def score_cell(cell_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"evaluation cell {cell_id!r} has no rows")
    first = rows[0]
    artifact_digest = first.get("score_artifact_digest")
    rebalance_date = first.get("rebalance_date")
    fold = first.get("fold")
    if cell_id != evaluation_cell_id(str(artifact_digest), str(rebalance_date), str(fold)):
        raise ValueError(f"evaluation cell {cell_id!r} identity mismatch")
    tickers: set[str] = set()
    composites: list[float] = []
    returns: list[float] = []
    for row in rows:
        if any(row.get(key) != expected for key, expected in (
            ("score_artifact_digest", artifact_digest),
            ("rebalance_date", rebalance_date),
            ("fold", fold),
        )):
            raise ValueError(f"evaluation cell {cell_id!r} contains mixed identity")
        score = row.get("score_record")
        if not isinstance(score, dict):
            raise ValueError("evaluation row score_record must be an object")
        validate_score_lineage(score)
        ticker = score.get("ticker")
        if not isinstance(ticker, str) or not ticker or ticker in tickers:
            raise ValueError(f"evaluation cell {cell_id!r} has invalid/duplicate ticker")
        tickers.add(ticker)
        composite = score.get("composite")
        forward = row.get("forward_return")
        for value, name in ((composite, "composite"), (forward, "forward_return")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric")
        composites.append(float(composite))
        returns.append(float(forward))
    if len(rows) < MIN_NAMES:
        return {
            "cell_id": cell_id, "score_artifact_digest": artifact_digest,
            "rebalance_date": rebalance_date, "fold": fold,
            "n_names": len(rows), "rank_ic": None, "placebo_rank_ic": None,
        }
    rank_ic = _corr(_rank(composites), _rank(returns))
    # Deterministic derangement: rotate sorted-ticker score values one position.
    ordered = sorted(range(len(rows)), key=lambda index: str(rows[index]["score_record"]["ticker"]))
    permuted = list(composites)
    for index, source in zip(ordered, ordered[1:] + ordered[:1]):
        permuted[index] = composites[source]
    placebo = _corr(_rank(permuted), _rank(returns))
    return {
        "cell_id": cell_id, "score_artifact_digest": artifact_digest,
        "rebalance_date": rebalance_date, "fold": fold, "n_names": len(rows),
        "rank_ic": rank_ic, "placebo_rank_ic": placebo,
    }


def evaluate_partitions(
    partitions: Mapping[str, bytes], params: EvaluationParams | None = None
) -> dict[str, Any]:
    if not partitions:
        raise ValueError("no Track B evaluation partitions supplied")
    params = params or EvaluationParams()
    expected = list(params.expected_cells or tuple(sorted(partitions)))
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected evaluation cells")
    if set(partitions) != set(expected):
        raise ValueError(
            f"evaluation cell set mismatch: missing={sorted(set(expected)-set(partitions))}, "
            f"unexpected={sorted(set(partitions)-set(expected))}"
        )
    cards = [score_cell(cell_id, _parse_partition(cell_id, partitions[cell_id])) for cell_id in sorted(partitions)]
    valid = [card for card in cards if card["rank_ic"] is not None]
    mean_ic = sum(float(card["rank_ic"]) for card in valid) / len(valid) if valid else None
    placebo_values = [float(card["placebo_rank_ic"]) for card in valid if card["placebo_rank_ic"] is not None]
    mean_placebo = sum(placebo_values) / len(placebo_values) if placebo_values else None
    enough_time = len(valid) >= params.min_forward_cells
    signal = mean_ic is not None and mean_ic >= params.rank_ic_bar
    placebo_clean = mean_placebo is not None and abs(mean_placebo) <= params.placebo_abs_limit
    passed = enough_time and signal and placebo_clean
    result = {
        "n_cells": len(cards), "n_valid_forward_cells": len(valid),
        "mean_rank_ic": mean_ic, "mean_placebo_rank_ic": mean_placebo,
        "enough_forward_time": enough_time, "signal_gate_passed": signal,
        "placebo_gate_passed": placebo_clean, "passed": passed,
        "bars": {
            "min_forward_cells": params.min_forward_cells,
            "rank_ic_bar": params.rank_ic_bar,
            "placebo_abs_limit": params.placebo_abs_limit,
        },
        "cells": cards,
    }
    result["artifact_bytes"] = canonical_bytes(result)
    return result


def build_evaluation_partition(
    score_artifact: bytes,
    *,
    rebalance_date: str,
    fold: str,
    forward_returns: Mapping[str, float],
) -> tuple[str, bytes]:
    """Join a scored artifact to realized returns without mutating score lineage."""
    artifact_digest = sha256_bytes(score_artifact)
    cell_id = evaluation_cell_id(artifact_digest, rebalance_date, fold)
    rows: list[dict[str, Any]] = []
    try:
        text = score_artifact.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"evaluation cell {cell_id!r} score artifact is not valid UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            score = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"evaluation cell {cell_id!r} score artifact line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(score, dict):
            raise ValueError(
                f"evaluation cell {cell_id!r} score artifact line {line_number} is not an object"
            )
        validate_score_lineage(score)
        ticker = score.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("score record ticker must be a non-empty string")
        if ticker not in forward_returns:
            continue
        forward = forward_returns[ticker]
        if isinstance(forward, bool) or not isinstance(forward, (int, float)) or not math.isfinite(float(forward)):
            raise ValueError(f"forward return for {ticker!r} must be finite numeric")
        rows.append({
            "score_artifact_digest": artifact_digest,
            "rebalance_date": rebalance_date,
            "fold": fold,
            "score_record": score,
            "forward_return": float(forward),
        })
    if not rows:
        raise ValueError("no scored names matched forward returns")
    return cell_id, b"".join(canonical_bytes(row) + b"\n" for row in rows)


__all__ = [
    "EvaluationParams", "evaluation_cell_id", "score_cell", "evaluate_partitions",
    "build_evaluation_partition",
]
