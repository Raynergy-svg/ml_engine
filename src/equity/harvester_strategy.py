"""US-006 — ``HarvesterStrategy``: runtime, deterministic harvester book.

Implements the :class:`src.equity.strategy.EquityStrategy` protocol (US-004)
on top of the validated harvester pipeline (US-002 universe + US-003 backtest
helpers + US-005 ship gate).

Contract (mirrors the AC on this story)
---------------------------------------

* **Causal** — ``compute_target_weights(asof)`` slices every panel to
  ``loc[:asof]``; the overlay scalar is already shift(1) inside
  :func:`src.equity.backtest.overlay`. Same ``asof`` + same on-disk inputs
  always emits the same dict.
* **Claude-free** — no network calls, no LLM, no random state. Pure pandas.
* **Long-only by default** (FR-11). Flip ``long_only=False`` only when
  building a research book; the runtime path keeps it on.
* **Reuses** :mod:`src.factor.portfolio` constants (``GROSS_CAP``,
  ``PER_PAIR_CAP``) and :func:`enforce_guards` so the strategy step and the
  US-003 backtest cannot diverge on caps.
* **Ship-gate guard** — construction reads
  ``trained_data/backtests/SHIP_GATE.json`` and refuses to instantiate
  unless ``gate_pass==True`` AND the file's ``universe_hash`` equals the
  passed snapshot's ``universe_hash``. The guard is fail-closed: missing
  file, malformed JSON, ``False`` gate, hash mismatch all raise
  :class:`HarvesterStrategyError`.

Wiring to :class:`src.scanner.config.ScannerConfig`
---------------------------------------------------

The PRD acceptance criterion calls for the full three-layer pattern so a
profile override actually lands at the consumer:

1. **Dataclass field** — ``equity_harvester_target_vol`` (etc.) on
   :class:`ScannerConfig` with a sane default.
2. **Profile dict** — the ``balanced`` profile dict carries the same set;
   ``apply_profile`` setattrs them when the profile is activated.
3. **Consumer ``getattr``** — :func:`HarvesterStrategy.from_config` reads
   each field with ``getattr(config, ..., DEFAULTS[...])`` so an older
   config without the field still works.

Atomic writes
-------------

:func:`HarvesterStrategy.write_state` persists the most recently computed
target dict via ``.tmp`` + :func:`os.replace`. Crash mid-write leaves the
old state on disk, never a half-written one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from src.equity.backtest import overlay
from src.equity.ship_gate import build_equal_weight_panel
from src.equity.strategy import (
    EquityStrategy,
    PER_NAME_ABS_CAP,
    validate_weights,
)
from src.equity.universe import UniverseSnapshot
from src.factor.portfolio import GROSS_CAP, PER_PAIR_CAP, enforce_guards

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"


# Sane defaults that mirror src.equity.ship_gate's validated grid. Surface
# the strategy's own copy so consumers without a ScannerConfig still ship
# the harvester it was validated against (rather than silently picking up
# library-level constants that may drift).
DEFAULTS: Dict[str, float] = {
    "target_vol": 0.12,
    "dd_soft": 0.10,
    "dd_hard": 0.20,
    "max_lev": 1.0,
    "vol_lookback": 21,
    "long_only": True,
}


class HarvesterStrategyError(RuntimeError):
    """Raised when harvester inputs / ship-gate state make a run unsafe."""


@dataclass(frozen=True)
class HarvesterStrategy:
    """Runtime harvester strategy: equal-weight × causal vol/DD scalar.

    Construct via :func:`HarvesterStrategy.create` (does ship-gate guard)
    or :func:`HarvesterStrategy.from_config` (does ship-gate guard + reads
    overlay knobs from a :class:`ScannerConfig`).

    The frozen dataclass holds:

    * ``snapshot`` — point-in-time universe membership (US-002 artifact).
    * ``prices``  — date × ticker close-price panel (US-001 loader output).
    * ``ship_gate_path`` — verified at construction; held for telemetry.
    * Overlay/cap knobs — exposed so the dataclass repr is self-describing.

    The implementation is intentionally tiny: every gnarly bit
    (member-only weights, the causal overlay, the cap math) lives in
    already-tested modules. The strategy is just the seam that wires them
    together against the live ``asof`` timestamp.
    """

    snapshot: UniverseSnapshot
    prices: pd.DataFrame
    ship_gate_path: Path
    universe_hash: str
    target_vol: float = DEFAULTS["target_vol"]
    dd_soft: float = DEFAULTS["dd_soft"]
    dd_hard: float = DEFAULTS["dd_hard"]
    max_lev: float = DEFAULTS["max_lev"]
    vol_lookback: int = int(DEFAULTS["vol_lookback"])
    long_only: bool = bool(DEFAULTS["long_only"])
    gross_cap: float = GROSS_CAP
    per_pair_cap: float = PER_PAIR_CAP

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    @classmethod
    def create(
        cls,
        snapshot: UniverseSnapshot,
        prices: pd.DataFrame,
        ship_gate_path: Path,
        *,
        target_vol: float = DEFAULTS["target_vol"],
        dd_soft: float = DEFAULTS["dd_soft"],
        dd_hard: float = DEFAULTS["dd_hard"],
        max_lev: float = DEFAULTS["max_lev"],
        vol_lookback: int = int(DEFAULTS["vol_lookback"]),
        long_only: bool = bool(DEFAULTS["long_only"]),
        gross_cap: float = GROSS_CAP,
        per_pair_cap: float = PER_PAIR_CAP,
    ) -> "HarvesterStrategy":
        """Construct after running the ship-gate guard.

        Raises :class:`HarvesterStrategyError` if the gate refuses the
        run (missing file, bad JSON, ``gate_pass=False``, or universe-hash
        mismatch with the supplied snapshot).
        """
        _validate_inputs(
            snapshot,
            prices,
            target_vol=target_vol,
            dd_soft=dd_soft,
            dd_hard=dd_hard,
            max_lev=max_lev,
            vol_lookback=vol_lookback,
            gross_cap=gross_cap,
            per_pair_cap=per_pair_cap,
        )
        _enforce_ship_gate(Path(ship_gate_path), snapshot.universe_hash)
        return cls(
            snapshot=snapshot,
            prices=prices,
            ship_gate_path=Path(ship_gate_path),
            universe_hash=snapshot.universe_hash,
            target_vol=float(target_vol),
            dd_soft=float(dd_soft),
            dd_hard=float(dd_hard),
            max_lev=float(max_lev),
            vol_lookback=int(vol_lookback),
            long_only=bool(long_only),
            gross_cap=float(gross_cap),
            per_pair_cap=float(per_pair_cap),
        )

    @classmethod
    def from_config(
        cls,
        config: object,
        snapshot: UniverseSnapshot,
        prices: pd.DataFrame,
        *,
        ship_gate_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ) -> "HarvesterStrategy":
        """Build from a :class:`ScannerConfig`-shaped object via ``getattr``.

        Reads the seven overlay knobs and the ship-gate path from ``config``
        with :func:`getattr` so consumers without the harvester fields still
        get the validated defaults rather than an :class:`AttributeError`.
        This is the consumer-side of the 3-layer pattern.
        """
        target_vol = float(
            getattr(config, "equity_harvester_target_vol", DEFAULTS["target_vol"])
        )
        dd_soft = float(
            getattr(config, "equity_harvester_dd_soft", DEFAULTS["dd_soft"])
        )
        dd_hard = float(
            getattr(config, "equity_harvester_dd_hard", DEFAULTS["dd_hard"])
        )
        max_lev = float(
            getattr(config, "equity_harvester_max_lev", DEFAULTS["max_lev"])
        )
        vol_lookback = int(
            getattr(
                config,
                "equity_harvester_vol_lookback",
                DEFAULTS["vol_lookback"],
            )
        )
        long_only = bool(
            getattr(config, "equity_harvester_long_only", DEFAULTS["long_only"])
        )

        if ship_gate_path is None:
            cfg_path = getattr(
                config, "equity_harvester_ship_gate_path", SHIP_GATE_PATH_DEFAULT
            )
            ship_gate_path = Path(cfg_path)
            if not ship_gate_path.is_absolute() and project_root is not None:
                ship_gate_path = Path(project_root) / ship_gate_path

        return cls.create(
            snapshot=snapshot,
            prices=prices,
            ship_gate_path=Path(ship_gate_path),
            target_vol=target_vol,
            dd_soft=dd_soft,
            dd_hard=dd_hard,
            max_lev=max_lev,
            vol_lookback=vol_lookback,
            long_only=long_only,
        )

    # -----------------------------------------------------------------
    # Strategy protocol
    # -----------------------------------------------------------------
    def compute_target_weights(self, asof: pd.Timestamp) -> Dict[str, float]:
        """Return causal vol/DD-scaled member weights at ``asof``.

        Implementation:

        1. Build the EW membership weight panel from ``(snapshot, prices)``.
        2. Slice both the panel and the EW-base-return series to
           ``[:asof]`` so the overlay scalar can never peek past ``asof``.
        3. Compute the causal scalar via :func:`src.equity.backtest.overlay`
           (it does ``shift(1)`` internally).
        4. Take the row at ``asof``, multiply by the scalar at ``asof``,
           clip the result through :func:`enforce_guards` for caps, and
           drop negatives if ``long_only``.
        5. Hand the result to :func:`validate_weights` so a malformed book
           fails loud at the seam, not downstream.

        Raises :class:`HarvesterStrategyError` if ``asof`` is not in the
        derived weight-panel index (silent gaps are the failure mode the
        US-001 freshness guard exists to prevent).
        """
        asof_ts = _normalize_asof(asof)

        weight_panel = build_equal_weight_panel(self.snapshot, self.prices)
        if asof_ts not in weight_panel.index:
            raise HarvesterStrategyError(
                f"asof {asof_ts!r} not present in weight panel "
                f"(first={weight_panel.index.min()!r}, "
                f"last={weight_panel.index.max()!r})"
            )

        ew_panel = weight_panel.loc[:asof_ts]
        prices_panel = self.prices.reindex(index=ew_panel.index)
        if prices_panel.index.tz is None:
            prices_panel.index = prices_panel.index.tz_localize("UTC")

        # EW base returns drive the scalar — same series the ship gate
        # used so the runtime book is the validated book.
        base_returns = prices_panel.pct_change().mean(axis=1, skipna=True)
        scalar_series = overlay(
            base_returns.fillna(0.0),
            target_vol=self.target_vol,
            dd_soft=self.dd_soft,
            dd_hard=self.dd_hard,
            max_lev=self.max_lev,
        )
        scalar_series = scalar_series.reindex(ew_panel.index).fillna(0.0)

        scalar_today = float(scalar_series.loc[asof_ts])
        ew_row = ew_panel.loc[asof_ts].astype(float)
        managed = ew_row * scalar_today

        # Reuse the panel-level guards so the per-name cap and gross cap
        # are enforced identically to the backtest's enforce_guards.
        managed_frame = pd.DataFrame([managed.to_numpy()], columns=ew_panel.columns)
        capped = enforce_guards(
            managed_frame,
            gross_cap=self.gross_cap,
            per_pair_cap=self.per_pair_cap,
        )
        capped_row = capped.iloc[0]

        weights: Dict[str, float] = {}
        for ticker, weight in capped_row.items():
            wf = float(weight)
            if not np.isfinite(wf):
                continue
            if self.long_only and wf < 0.0:
                wf = 0.0
            if abs(wf) > 1e-12:
                weights[str(ticker)] = wf

        return validate_weights(
            weights,
            long_only=self.long_only,
            gross_cap=self.gross_cap,
            per_name_abs_cap=PER_NAME_ABS_CAP,
        )

    # -----------------------------------------------------------------
    # State persistence
    # -----------------------------------------------------------------
    def write_state(
        self,
        weights: Mapping[str, float],
        asof: pd.Timestamp,
        path: Path,
    ) -> Path:
        """Atomically persist ``weights`` for the executor to diff against.

        Uses the same ``tempfile.mkstemp`` + :func:`os.replace` pattern as
        :func:`src.equity.strategy._atomic_write_json`; a partial write is
        never visible.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "asof": pd.Timestamp(_normalize_asof(asof)).isoformat(),
            "weights": {str(k): float(v) for k, v in weights.items()},
            "gross": float(sum(abs(float(v)) for v in weights.values())),
            "n_names": int(len(weights)),
            "long_only": bool(self.long_only),
            "universe_hash": str(self.universe_hash),
            "ship_gate_path": str(self.ship_gate_path),
        }
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
                    "failed to remove temp file %s: %s",
                    tmp_path,
                    cleanup_exc,
                )
            raise
        return path


# ---------------------------------------------------------------------------
# Internals — ship gate + input validation
# ---------------------------------------------------------------------------


def _normalize_asof(asof: pd.Timestamp) -> pd.Timestamp:
    """Coerce ``asof`` to UTC midnight, matching the membership index."""
    ts = pd.Timestamp(asof)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to construct unless the ship gate clears for THIS universe.

    The guard is fail-closed: any failure (missing file, malformed JSON,
    missing keys, ``gate_pass`` not exactly ``True``, hash mismatch) raises
    :class:`HarvesterStrategyError`. The strategy should never be reachable
    without an explicit operator decision to unblock it.
    """
    if not path.exists():
        raise HarvesterStrategyError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvesterStrategyError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise HarvesterStrategyError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )

    gate_pass = payload.get("gate_pass")
    if gate_pass is not True:
        raise HarvesterStrategyError(
            f"ship gate refuses: gate_pass={gate_pass!r} at {path} "
            "(must be exactly True to unblock the strategy)"
        )

    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise HarvesterStrategyError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise HarvesterStrategyError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, snapshot={expected_universe_hash!r}) "
            f"at {path}"
        )


def _validate_inputs(
    snapshot: UniverseSnapshot,
    prices: pd.DataFrame,
    *,
    target_vol: float,
    dd_soft: float,
    dd_hard: float,
    max_lev: float,
    vol_lookback: int,
    gross_cap: float,
    per_pair_cap: float,
) -> None:
    """Reject obviously-broken inputs before the ship-gate check."""
    if not isinstance(snapshot, UniverseSnapshot):
        raise HarvesterStrategyError(
            f"snapshot must be UniverseSnapshot, got {type(snapshot)!r}"
        )
    if snapshot.membership.empty:
        raise HarvesterStrategyError(
            "snapshot has empty membership — nothing to harvest"
        )
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise HarvesterStrategyError(
            "prices must be a non-empty DataFrame"
        )
    if not isinstance(snapshot.universe_hash, str) or not snapshot.universe_hash:
        raise HarvesterStrategyError(
            "snapshot.universe_hash must be a non-empty string"
        )
    if target_vol <= 0:
        raise HarvesterStrategyError(
            f"target_vol must be > 0, got {target_vol!r}"
        )
    if dd_soft < 0 or dd_hard <= dd_soft:
        raise HarvesterStrategyError(
            f"require 0 <= dd_soft < dd_hard, got soft={dd_soft!r}, "
            f"hard={dd_hard!r}"
        )
    if max_lev <= 0:
        raise HarvesterStrategyError(
            f"max_lev must be > 0, got {max_lev!r}"
        )
    if vol_lookback < 2:
        raise HarvesterStrategyError(
            f"vol_lookback must be >= 2, got {vol_lookback!r}"
        )
    if gross_cap <= 0:
        raise HarvesterStrategyError(
            f"gross_cap must be > 0, got {gross_cap!r}"
        )
    if per_pair_cap <= 0:
        raise HarvesterStrategyError(
            f"per_pair_cap must be > 0, got {per_pair_cap!r}"
        )


# ---------------------------------------------------------------------------
# Module sanity
# ---------------------------------------------------------------------------

# A frozen dataclass instance satisfying the Protocol is the cleanest way to
# confirm the seam at import time without paying for an isinstance() at every
# call site. Mypy / runtime users get the failure surface here.
assert hasattr(HarvesterStrategy, "compute_target_weights")
_PROTOCOL_GUARD: type = EquityStrategy  # silence "unused import" linters


__all__ = [
    "DEFAULTS",
    "HarvesterStrategy",
    "HarvesterStrategyError",
    "SHIP_GATE_PATH_DEFAULT",
]


# Hummingbot Apache-2.0 attribution (PRD §notes):
# No source has been copied verbatim from Hummingbot into this module. The
# overlay/cap math here is original to this repo (US-003/US-005). If the
# runtime later adopts any Hummingbot strategy code, attribution lives in
# the file that uses it (Apache-2.0, NOTICE preserved). No freqtrade GPL.
