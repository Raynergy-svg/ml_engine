"""US-004 — Strategy/engine interface contract for the equity-beta harvester.

This module defines the seam that US-006 (the live ``HarvesterStrategy``) and
US-017 (the rebalance control loop) must agree on, plus a stub implementation
the rebalance loop's tests can drive without dragging in the full harvester.

Contract
--------

Every strategy implements the typed :class:`EquityStrategy` protocol:

    compute_target_weights(asof: pd.Timestamp) -> dict[str, float]

with the following invariants (enforced by the helpers in this module):

* **Causal** — the strategy may only look at information observable at or
  before ``asof`` (close-of-session on ``asof``). The control loop passes a
  real wall-clock timestamp; downstream strategies that read price/score
  panels must slice ``loc[:asof]``, never ``shift(-1)``.
* **Bounds** — for each ticker ``|w_t| <= PER_PAIR_CAP * GROSS_CAP`` (i.e.
  ``0.30 * 4.0 = 1.20`` of NAV). This matches
  :func:`src.factor.portfolio.enforce_guards` exactly so the strategy step and
  the panel backtest cannot diverge.
* **Sums** — ``sum(|w_t|) <= GROSS_CAP`` (4:1 gross leverage). Long-only
  strategies additionally have ``w_t >= 0`` and ``sum(w_t) <= GROSS_CAP``.
* **Determinism** — same ``asof`` + same on-disk inputs => same dict. No
  network, no LLM, no random state in the hot path.

How US-017's rebalance loop consumes it
---------------------------------------

The control loop ticks once per scheduled rebalance (US-007 will add the
scheduler; US-017 is the surrounding loop). On every tick it calls::

    weights = strategy.compute_target_weights(asof=now)
    rebalance_loop_step(strategy, asof=now, state_path=state_path)

:func:`rebalance_loop_step` is the canonical adapter. It (a) invokes the
strategy, (b) validates the bounds the protocol promises, (c) persists the
target dict atomically (``.tmp`` then ``os.replace``) so a mid-rebalance
crash leaves either the old target or the new target on disk — never a
half-written one, and (d) returns the dict for the executor to diff against
current positions.

Helpers
-------

:func:`weights_from_factor_panel` is the canonical *reuse* glue: it takes a
score panel + returns panel + an ``asof`` and runs the existing
:func:`src.factor.portfolio.target_weights` + :func:`enforce_guards` panel
pipeline, then extracts the row for ``asof`` as a dict. US-006's harvester is
expected to call this helper rather than re-implementing leverage / cap math.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from src.factor.portfolio import (
    GROSS_CAP,
    PER_PAIR_CAP,
    VOL_TARGET,
    enforce_guards,
    target_weights,
)

logger = logging.getLogger(__name__)


PER_NAME_ABS_CAP = PER_PAIR_CAP * GROSS_CAP  # 0.30 * 4.0 = 1.20

__all__ = [
    "EquityStrategy",
    "StrategyError",
    "StubEqualWeightStrategy",
    "PER_NAME_ABS_CAP",
    "validate_weights",
    "weights_from_factor_panel",
    "rebalance_loop_step",
]


class StrategyError(RuntimeError):
    """Raised when a strategy returns weights that violate the protocol bounds.

    Strategies are allowed to be conservative (empty dict / all-zeros means
    "go flat"). They are NOT allowed to return per-name leverage above
    :data:`PER_NAME_ABS_CAP`, total gross above :data:`GROSS_CAP`, NaN/inf
    weights, or non-string keys. Surface the violation rather than letting a
    bad book reach the executor.
    """


@runtime_checkable
class EquityStrategy(Protocol):
    """Typed seam between the strategy and the control loop.

    Implementations:

    * **Stub** — :class:`StubEqualWeightStrategy` (this module). Used in
      tests + as the loop's smoke implementation before US-006 lands.
    * **Live** — ``HarvesterStrategy`` (US-006). Drives the vol-managed beta
      + drawdown-scaled book the ship gate (US-005) validated.

    The protocol intentionally returns a plain ``dict[str, float]`` rather
    than a ``pd.Series``: the executor wants explicit per-ticker entries and
    must treat absent tickers as "no position" (zero), not NaN.
    """

    def compute_target_weights(self, asof: pd.Timestamp) -> Dict[str, float]:
        """Return target portfolio weights as of ``asof`` (causal).

        Keys are tickers; values are target weights as a fraction of NAV
        (e.g. ``0.05`` = 5% long). The control loop interprets a missing
        key as zero weight; an explicit ``0.0`` carries the same meaning
        but is preferred for tickers the strategy actively decided to flat.
        """


def validate_weights(
    weights: Mapping[str, float],
    *,
    long_only: bool = True,
    gross_cap: float = GROSS_CAP,
    per_name_abs_cap: float = PER_NAME_ABS_CAP,
) -> Dict[str, float]:
    """Validate a target-weight dict against the protocol invariants.

    Returns a clean ``dict[str, float]`` (tickers as ``str``, weights as
    ``float``) with absent / zero entries dropped. Raises
    :class:`StrategyError` on any violation rather than silently clamping —
    a strategy that ships infeasible weights is a bug, not a paper cut.
    """

    cleaned: Dict[str, float] = {}
    for ticker, raw in weights.items():
        if not isinstance(ticker, str) or not ticker:
            raise StrategyError(
                f"weight key must be a non-empty str, got {ticker!r}"
            )
        try:
            w = float(raw)
        except (TypeError, ValueError) as exc:
            raise StrategyError(
                f"weight for {ticker!r} is not numeric: {raw!r}"
            ) from exc
        if not np.isfinite(w):
            raise StrategyError(
                f"weight for {ticker!r} is non-finite: {w!r}"
            )
        if abs(w) > per_name_abs_cap + 1e-9:
            raise StrategyError(
                f"per-name |w| for {ticker!r} = {w:.4f} exceeds cap "
                f"{per_name_abs_cap:.4f}"
            )
        if long_only and w < -1e-9:
            raise StrategyError(
                f"long-only protocol but {ticker!r} weight = {w:.4f} < 0"
            )
        if w != 0.0:
            cleaned[ticker] = w

    gross = float(sum(abs(w) for w in cleaned.values()))
    if gross > gross_cap + 1e-9:
        raise StrategyError(
            f"gross leverage {gross:.4f} exceeds cap {gross_cap:.4f}"
        )
    return cleaned


@dataclass(frozen=True)
class StubEqualWeightStrategy:
    """Trivial stub: equal-weight long-only book over a fixed ticker list.

    Used by the US-017 loop's own tests (and the test in this module) to
    confirm the seam wires up correctly before US-006 ships the live
    harvester. The stub is deterministic and Claude-free.

    ``target_gross`` is the total exposure (e.g. ``1.0`` = fully invested,
    ``0.5`` = half-invested). It is split equally across non-empty
    ``tickers``. ``target_gross`` must respect :data:`GROSS_CAP` and the
    per-name absolute cap, otherwise :func:`validate_weights` will reject
    the result.
    """

    tickers: Sequence[str] = field(default_factory=tuple)
    target_gross: float = 1.0

    def compute_target_weights(self, asof: pd.Timestamp) -> Dict[str, float]:
        names = [t for t in self.tickers if isinstance(t, str) and t]
        if not names or self.target_gross <= 0.0:
            return {}
        per_name = float(self.target_gross) / float(len(names))
        weights = {t: per_name for t in names}
        # validate_weights enforces the same bounds the protocol promises,
        # so a misconfigured stub fails loud at the seam rather than
        # surfacing as an executor surprise.
        return validate_weights(weights, long_only=True)


def weights_from_factor_panel(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    vol_target: float = VOL_TARGET,
    vol_lookback: int = 60,
    gross_cap: float = GROSS_CAP,
    per_pair_cap: float = PER_PAIR_CAP,
    long_only: bool = True,
) -> Dict[str, float]:
    """Reuse :mod:`src.factor.portfolio` to derive a per-``asof`` dict.

    This is the canonical glue the live harvester (US-006) is expected to
    use. We slice the panels to ``[..., asof]`` (so the leverage math
    never sees post-``asof`` rows), run the existing
    :func:`target_weights` + :func:`enforce_guards` pipeline, take the
    last row, optionally clip negatives for the long-only default, and
    return the dict.

    Raises :class:`StrategyError` if ``asof`` is not present in
    ``scores.index`` (the loop is expected to align on a real session
    timestamp; silent gaps are exactly the failure mode US-001 was
    written to prevent).
    """

    if asof not in scores.index:
        raise StrategyError(
            f"asof {asof!r} not present in score index "
            f"(first={scores.index.min()!r}, last={scores.index.max()!r})"
        )
    s_panel = scores.loc[:asof]
    r_panel = returns.reindex(index=s_panel.index, columns=s_panel.columns)

    raw = target_weights(
        s_panel,
        r_panel,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        gross_cap=gross_cap,
        per_pair_cap=per_pair_cap,
    )
    capped = enforce_guards(raw, gross_cap=gross_cap, per_pair_cap=per_pair_cap)
    row = capped.loc[asof]

    out: Dict[str, float] = {}
    for ticker, w in row.items():
        wf = float(w)
        if not np.isfinite(wf):
            continue
        if long_only and wf < 0.0:
            wf = 0.0
        if wf != 0.0:
            out[str(ticker)] = wf

    return validate_weights(out, long_only=long_only, gross_cap=gross_cap)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomic JSON write: ``tmp`` in the same dir then ``os.replace``.

    Same pattern as the data loader: a half-written file is never visible
    to a concurrent reader, and a crash mid-write leaves the previous
    target on disk rather than an empty file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.error(
            "atomic write failed for %s: %s", path, exc, exc_info=True
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise


def rebalance_loop_step(
    strategy: EquityStrategy,
    asof: pd.Timestamp,
    *,
    state_path: Path,
    long_only: bool = True,
    gross_cap: float = GROSS_CAP,
    per_name_abs_cap: float = PER_NAME_ABS_CAP,
    universe: Iterable[str] | None = None,
) -> Dict[str, float]:
    """One control-loop tick: call strategy, validate, persist atomically.

    This is the function US-017's loop calls every scheduled rebalance.
    It demonstrates exactly how the control loop consumes the strategy
    protocol so US-006 and US-017 can ship in parallel.

    Steps:

    1. Invoke ``strategy.compute_target_weights(asof)``.
    2. Run :func:`validate_weights` on the result so a bad book fails
       loud BEFORE it reaches the executor.
    3. Optionally intersect with ``universe`` (US-002's point-in-time
       roster) — tickers outside the universe are dropped with a
       warning rather than silently filled.
    4. Persist ``{asof, weights, gross, n_names}`` atomically to
       ``state_path`` (the executor reads from here on restart).
    5. Return the validated dict to the caller (executor diffs it
       against current positions to emit order deltas).

    Returns the validated target-weight dict.
    """

    if not isinstance(strategy, EquityStrategy):
        raise StrategyError(
            f"strategy {strategy!r} does not satisfy EquityStrategy protocol"
        )

    raw = strategy.compute_target_weights(asof)
    weights = validate_weights(
        raw,
        long_only=long_only,
        gross_cap=gross_cap,
        per_name_abs_cap=per_name_abs_cap,
    )

    if universe is not None:
        roster = {str(t) for t in universe if isinstance(t, str) and t}
        out_of_universe = [t for t in weights if t not in roster]
        for ticker in out_of_universe:
            logger.warning(
                "rebalance %s: dropping out-of-universe ticker %s (w=%.4f)",
                asof,
                ticker,
                weights[ticker],
            )
            del weights[ticker]

    payload = {
        "asof": pd.Timestamp(asof).isoformat(),
        "weights": dict(weights),
        "gross": float(sum(abs(w) for w in weights.values())),
        "n_names": int(len(weights)),
        "long_only": bool(long_only),
        "gross_cap": float(gross_cap),
        "per_name_abs_cap": float(per_name_abs_cap),
    }
    _atomic_write_json(Path(state_path), payload)
    return weights
