"""TrainingSignalApplicator — write homework training signal deltas to agent_weights.json.

Closes the v1 gap: HomeworkReviewer.approve()/edit() previously emitted TrainingSignal
payloads that the TUI just notified about and discarded. This applicator IS the
consumer — it reads agent_weights.json, applies the deltas, clips to a safe range,
and writes back atomically (tmp + rename) under fcntl.flock to serialize concurrent
operator actions.

See docs/superpowers/specs/2026-04-25-trade-homework-system-design.md §4.5 +
the Phase 96.5 close-the-loop fix.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.homework.types import TrainingSignal

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "trained_data" / "models" / "agent_weights.json"

DEFAULT_AGENT_BASELINE = 1.0  # weight assumed for an agent newly introduced into a regime row


class TrainingSignalApplicator:
    """Apply TrainingSignal.agent_weight_deltas to agent_weights.json.

    Args:
        weights_path: path to the regime-keyed agent weights file. Default: trained_data/models/agent_weights.json
        min_weight: floor for clipping. Default 0.1.
        max_weight: ceiling for clipping. Default 2.0.

    The agent_weights.json schema is dict[regime, dict[agent, weight]] with
    optional metadata keys (e.g. _meta). Metadata is preserved on rewrite.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    # ---------------- public API ----------------

    def apply(self, signal: TrainingSignal) -> Dict[str, Any]:
        """Apply signal's deltas to weights file. Returns a result dict.

        Result keys:
            regime          — the primary regime targeted (signal.regime)
            agents_updated  — list of agent names whose weight changed in signal.regime
            before          — dict of {agent: pre-clip weight} for primary-regime updates
            after           — dict of {agent: post-clip weight} for primary-regime updates
            clipped         — list of agent names whose post-delta weight hit a clip boundary
            other_regimes   — list of regimes touched via regime_prior_deltas
        """
        result: Dict[str, Any] = {
            "regime": signal.regime,
            "agents_updated": [],
            "before": {},
            "after": {},
            "clipped": [],
            "other_regimes": [],
        }

        # Early no-op: empty deltas (rejected signals, or edits that zeroed deltas)
        if not signal.agent_weight_deltas and not signal.regime_prior_deltas:
            logger.debug(
                "TrainingSignalApplicator: empty deltas for %s, no-op", signal.homework_id
            )
            return result

        if not self.weights_path.exists():
            logger.warning(
                "TrainingSignalApplicator: weights file does not exist at %s — skipping apply",
                self.weights_path,
            )
            return result

        # Acquire exclusive lock on the weights file for read-modify-write
        # We use a sidecar lockfile so the rename-replace doesn't blow away the lock holder.
        lock_path = self.weights_path.with_suffix(self.weights_path.suffix + ".lock")
        lock_fh = None
        try:
            lock_fh = open(lock_path, "a+", encoding="utf-8")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

            weights = self._read_weights()
            if weights is None:
                # corrupt or wrong-shape file; _read_weights logged the issue
                return result

            # Apply primary-regime deltas
            if signal.agent_weight_deltas:
                if signal.regime not in weights:
                    logger.warning(
                        "TrainingSignalApplicator: regime '%s' not in weights file (%s); "
                        "skipping primary deltas for %s",
                        signal.regime, list(weights.keys()), signal.homework_id,
                    )
                else:
                    self._apply_to_regime_row(
                        weights[signal.regime],
                        signal.agent_weight_deltas,
                        result,
                        is_primary=True,
                    )

            # Apply regime_prior_deltas to other regime rows
            for other_regime, other_deltas in (signal.regime_prior_deltas or {}).items():
                if other_regime not in weights:
                    logger.warning(
                        "TrainingSignalApplicator: regime_prior_deltas regime '%s' "
                        "not in weights file; skipping",
                        other_regime,
                    )
                    continue
                if not isinstance(other_deltas, dict):
                    logger.warning(
                        "TrainingSignalApplicator: regime_prior_deltas[%s] is not a dict; skipping",
                        other_regime,
                    )
                    continue
                self._apply_to_regime_row(
                    weights[other_regime],
                    other_deltas,
                    result,
                    is_primary=False,
                )
                if other_regime not in result["other_regimes"]:
                    result["other_regimes"].append(other_regime)

            # Skip rewrite if nothing changed (missing regime, etc.)
            if not result["agents_updated"] and not result["other_regimes"]:
                return result

            self._write_weights_atomic(weights)
            logger.info(
                "TrainingSignalApplicator: applied %d agent updates to regime=%s for %s "
                "(other regimes=%s, clipped=%s)",
                len(result["agents_updated"]),
                signal.regime,
                signal.homework_id,
                result["other_regimes"],
                result["clipped"],
            )
            return result

        finally:
            if lock_fh is not None:
                try:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_fh.close()

    # ---------------- internals ----------------

    def _read_weights(self) -> Optional[Dict[str, Any]]:
        """Read & schema-validate the weights file. Returns None on corrupt/wrong-shape."""
        try:
            with open(self.weights_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "TrainingSignalApplicator: failed to read %s: %s",
                self.weights_path, e,
            )
            return None

        if not isinstance(data, dict):
            logger.error(
                "TrainingSignalApplicator: weights file root is %s, expected dict",
                type(data).__name__,
            )
            return None

        # Validate that at least the regime rows we care about look right
        # (allow non-regime metadata like _meta to coexist).
        for key, val in data.items():
            if key.startswith("_"):
                continue  # metadata escape hatch
            if not isinstance(val, dict):
                logger.error(
                    "TrainingSignalApplicator: weights['%s'] is %s, expected dict[str, float]",
                    key, type(val).__name__,
                )
                return None
        return data

    def _apply_to_regime_row(
        self,
        regime_row: Dict[str, Any],
        deltas: Dict[str, float],
        result: Dict[str, Any],
        is_primary: bool,
    ) -> None:
        """Apply deltas to a single regime's agent->weight dict in-place; update result."""
        for agent, delta in deltas.items():
            try:
                delta_val = float(delta)
            except (TypeError, ValueError):
                logger.warning(
                    "TrainingSignalApplicator: non-numeric delta for %s: %r — skipping",
                    agent, delta,
                )
                continue

            current_raw = regime_row.get(agent, DEFAULT_AGENT_BASELINE)
            try:
                current = float(current_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "TrainingSignalApplicator: non-numeric existing weight %s[%s]=%r — using baseline",
                    is_primary, agent, current_raw,
                )
                current = DEFAULT_AGENT_BASELINE

            new_unclipped = current + delta_val
            new_clipped = max(self.min_weight, min(self.max_weight, new_unclipped))
            regime_row[agent] = new_clipped

            if is_primary:
                result["before"][agent] = current
                result["after"][agent] = new_clipped
                if agent not in result["agents_updated"]:
                    result["agents_updated"].append(agent)
            if new_unclipped != new_clipped and agent not in result["clipped"]:
                result["clipped"].append(agent)

    def _write_weights_atomic(self, weights: Dict[str, Any]) -> None:
        """Rewrite the weights file atomically via tmp + rename."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.weights_path.parent),
                prefix=self.weights_path.name,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(weights, tmp, indent=2, sort_keys=True)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.rename(tmp_name, str(self.weights_path))
            tmp_name = None  # successful rename consumed it
        finally:
            if tmp_name is not None and os.path.exists(tmp_name):
                # rename failed — clean up the tmp
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
