"""THE drawdown sign convention for every research gate in this repository.

Why this module exists
----------------------
Before 2026-07-30 the repo shipped two opposite drawdown sign conventions under
the same dict key ``max_drawdown``:

* NEGATIVE (one producer): ``src/research/gated_harness/backtest.py`` returned
  ``(equity/equity.cummax() - 1).min()``, and its gate tested
  ``metrics["max_drawdown"] >= -abs(limit)``.
* POSITIVE (every other producer): ``src/factor/backtest.py``,
  ``src/equity/backtest.py``, ``src/crypto/momentum_scorecard.py``,
  ``src/crypto/carry_scorecard.py``, ``src/hedge/hedge_scorecard.py``,
  ``src/training/walkforward_validation.py`` all emit a non-negative magnitude,
  and their gates test ``max_dd <= limit``.

Feeding a positive-convention metrics dict to the negative-convention gate made
an **86.3% drawdown pass a 25% drawdown gate** (``0.863 >= -0.25`` is ``True``).
Nothing cross-fed the two *today*, but Phase E's stated exit gate is to route
every lane through the shared harness — executing the roadmap was the trigger.

The fix, and why it is structural rather than documentary
---------------------------------------------------------
The canonical convention is the **positive fraction**: ``0.25`` means "a 25%
peak-to-trough decline", ``0.0`` means "no drawdown". Chosen because

1. it is what six of the seven producers already emit, so **no lane's reported
   business number changes** — a value that was ``0.863`` still means the same
   86.3% drawdown, and every on-disk artifact (``SHIP_GATE.json``, the
   ``trained_data/backtests/*.json`` reports, the evidence envelopes) keeps its
   existing schema and sign, so their readers need no migration;
2. under a positive convention **every** non-zero negative-convention value is
   strictly negative, so a single ``value < 0`` guard refuses 100% of
   wrong-convention input. The separation is total, not heuristic. (``0.0`` is
   the one shared value and it means the same thing in both conventions.)

Renaming the key was considered and rejected: ``max_drawdown`` /``max_dd`` is
read by ``src/factor/ship_gate.py:37``, ``src/equity/shadow_pipeline.py:204``,
``src/tui/widgets/state_strip.py:57``, ``src/tui/data_provider.py:856``, the
``EvaluationReport.metrics`` tuples in ``src/evidence`` and by every archived
``trained_data/backtests/*.json``. A rename would have to migrate on-disk
artifacts written by scripts this change is not permitted to touch. Enforcing
the convention at the seam achieves the same guarantee with no schema break.

Enforcement, not instruction (project rule; INTENT standing decision 7)
----------------------------------------------------------------------
:func:`drawdown_fraction` is a fail-closed validator: a wrong-signed,
non-finite, non-numeric or percent-scaled value **raises**, naming the key, the
offending value and the producing site. It never silently compares. Every
producer emits through it and every gate reads through it, so a future sign
flip fails loudly at the boundary instead of quietly satisfying a guard — the
same failure class as the documented "$3,527 dead-write" and "No-Mock
catastrophe" incidents.

Deliberately stdlib-only (``math`` alone) so the dependency-light contract
modules, the evidence workers and the pure-python scorecards can all import it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

#: The one convention. Emitted into artifacts so a reader can verify it.
DRAWDOWN_CONVENTION = "positive_fraction"

#: The canonical metrics key. ``max_dd`` remains a legal short alias in
#: existing artifact schemas and carries the same convention.
CANONICAL_DRAWDOWN_KEY = "max_drawdown"
DRAWDOWN_KEY_ALIASES = ("max_drawdown", "maximum_drawdown", "max_dd")

#: A fractional drawdown of a compounded equity curve cannot exceed 1.0.
#: Anything above it is almost always a percentage (86.3) mistaken for a
#: fraction (0.863), which is the second way this key goes wrong.
MAX_ADMISSIBLE_DRAWDOWN_FRACTION = 1.0


class DrawdownConventionError(ValueError):
    """A drawdown value violated the canonical positive-fraction convention."""


def drawdown_fraction(
    value: Any,
    *,
    key: str = CANONICAL_DRAWDOWN_KEY,
    source: str = "<unspecified>",
) -> float:
    """Validate ``value`` as a canonical drawdown and return it as a float.

    Fail-closed. Raises :class:`DrawdownConventionError` — never returns a
    coerced or silently corrected number — when ``value`` is:

    * not numeric, or not finite (``NaN`` / ``inf``);
    * **negative**, i.e. produced under the opposite sign convention;
    * greater than 1.0, i.e. a percentage where a fraction is required.

    ``key`` and ``source`` are echoed into the error so the offending producer
    is identifiable from the traceback alone.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrawdownConventionError(
            f"{source}: {key}={value!r} is not a real number; the canonical "
            f"drawdown convention is {DRAWDOWN_CONVENTION} (a float in [0, 1])"
        )
    number = float(value)
    if not math.isfinite(number):
        raise DrawdownConventionError(
            f"{source}: {key}={number!r} is not finite; refusing to gate on it"
        )
    if number < 0.0:
        raise DrawdownConventionError(
            f"{source}: {key}={number!r} is NEGATIVE. This repository's single "
            f"drawdown convention is {DRAWDOWN_CONVENTION!r}: a 25% peak-to-trough "
            f"decline is 0.25, never -0.25. A negative value means the producer "
            f"uses the opposite sign convention — the exact collision that let an "
            f"86.3% drawdown pass a 25% gate. Convert at the producer with "
            f"abs()/negation; this gate refuses rather than compares."
        )
    if number > MAX_ADMISSIBLE_DRAWDOWN_FRACTION:
        raise DrawdownConventionError(
            f"{source}: {key}={number!r} exceeds 1.0. The convention is a "
            f"FRACTION, not a percentage: 86.3% is 0.863, not 86.3. Refusing "
            f"rather than comparing against a fractional limit."
        )
    return number


def drawdown_limit(
    value: Any,
    *,
    key: str = "maximum_drawdown",
    source: str = "<unspecified>",
) -> float:
    """Validate a gate's drawdown *budget* under the same convention.

    A limit is the same kind of quantity as an observation, so it goes through
    the same guard. The pre-fix gate wrote ``-abs(maximum_drawdown)``, which
    accepted either sign for the budget and thereby hid the collision from the
    caller side too.
    """
    return drawdown_fraction(value, key=key, source=source)


def read_drawdown(
    metrics: Mapping[str, Any],
    *,
    key: str = CANONICAL_DRAWDOWN_KEY,
    source: str = "<unspecified>",
) -> float:
    """Read and validate ``metrics[key]``; raise a named error if absent.

    ``KeyError`` from a bare ``metrics["max_drawdown"]`` gives a gate no way to
    say *which* gate and *which* producer failed to supply the metric.
    """
    if key not in metrics:
        raise DrawdownConventionError(
            f"{source}: metrics dict has no {key!r} key "
            f"(present keys: {sorted(metrics)!r}); refusing to gate on a "
            f"missing drawdown rather than defaulting it"
        )
    return drawdown_fraction(metrics[key], key=key, source=source)


def drawdown_within_limit(
    observed: Any,
    limit: Any,
    *,
    key: str = CANONICAL_DRAWDOWN_KEY,
    source: str = "<unspecified>",
) -> bool:
    """The one drawdown comparison. Both sides validated before comparing.

    Returns ``observed <= limit`` in positive-fraction space.
    """
    return drawdown_fraction(observed, key=key, source=source) <= drawdown_limit(
        limit, source=source
    )


def drawdown_from_equity(equity: Sequence[float]) -> float:
    """Canonical max drawdown of an equity curve, as a non-negative fraction.

    Pure python so the dependency-light scorecards can share it. Returns 0.0
    for an empty or monotonically non-decreasing curve.
    """
    peak = None
    worst = 0.0
    for point in equity:
        value = float(point)
        if peak is None or value > peak:
            peak = value
        if peak and peak > 0.0:
            worst = max(worst, (peak - value) / peak)
    return worst


def drawdown_from_returns(returns: Sequence[float]) -> float:
    """Canonical max drawdown of a per-period fractional return series."""
    equity = 1.0
    curve = []
    for r in returns:
        equity *= 1.0 + float(r)
        curve.append(equity)
    return drawdown_from_equity(curve)
