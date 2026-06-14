"""Real-time post-trade feedback loop.

Fires after every confirmed OANDA trade close. Responsible for:
    1. Idempotent logging of a shaped-reward record to logs/trade_feedback.jsonl.
    2. Append-only significant-outcome entries in .claude/learnings.md (>= 2R).
    3. Invoking PostTradeDiagnostics + SelfHeal (lazy imports; no-op if the
       sibling modules haven't shipped yet).

Hard scope:
    * DOES NOT call `update_weights_from_outcome()`. The batch path in
      `execution.sync_closed_trades_rl` owns RL weight updates — this module
      must never duplicate that work.
    * Stdlib only. No imports from `src.scanner.agents` or
      `src.scanner.execution` (circular dep risk — execution imports us).

The caller (execution.close_trade_oanda hook) wraps `run()` in its own
try/except. Even so, `run()` NEVER raises: every failure path returns a
status dict and logs the error with context.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PostTradeLoop:
    """Post-trade feedback loop: shaped-reward logging, diagnostics, self-heal.

    Usage:
        loop = PostTradeLoop()
        result = loop.run({"trade_id": "12345", "source": "close_trade_oanda"})

    `run()` returns a status dict and never raises. See the module docstring
    for scope rules.
    """

    # Shaped-reward coefficients — tunable in one place.
    SLIP_WEIGHT: float = 1.0
    MAE_WEIGHT: float = 0.5

    # File path constants (resolved relative to current working directory,
    # matching the pattern used by execution.py for trained_data/*).
    JOURNAL_PATH: Path = Path("trained_data/trade_journal_rl.json")
    FEEDBACK_LOG_PATH: Path = Path("logs/trade_feedback.jsonl")
    LEARNINGS_PATH: Path = Path(".claude/learnings.md")
    ERROR_LOG_PATH: Path = Path("logs/unhandled_errors.jsonl")

    def __init__(self):
        # US-011: RLHF reward shaper
        self._rlhf_shaper = None
        try:
            from src.rl.rlhf_reward import RLHFRewardShaper
            self._rlhf_shaper = RLHFRewardShaper()
        except Exception:
            pass
        # US-016: Self-healing config updater
        self._self_heal_config_updater = None
        try:
            from src.autonomy.self_heal_config import SelfHealConfigUpdater
            self._self_heal_config_updater = SelfHealConfigUpdater()
        except Exception:
            pass

    # How many recent lines of feedback log we scan for idempotency.
    IDEMPOTENCY_TAIL_LINES: int = 500

    def run(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single closed-trade event.

        Args:
            trade_result: Minimum contract is
                {"trade_id": "<oanda id>", "source": "<caller tag>"}.
                Extra keys are ignored here — the full record is loaded from
                the trade journal by trade_id.

        Returns:
            Status dict. Possible shapes:
                {"status": "skipped",   "reason": "already_processed",   ...}
                {"status": "skipped",   "reason": "journal_entry_missing", ...}
                {"status": "skipped",   "reason": "missing_trade_id",     ...}
                {"status": "processed", "shaped_reward": float, ...}
                {"status": "error",     "error": str, ...}
        """
        trade_id = str(trade_result.get("trade_id") or "").strip()
        source = trade_result.get("source", "unknown")

        if not trade_id:
            logger.warning("post_trade_loop: missing_trade_id source=%s", source)
            return {
                "status": "skipped",
                "reason": "missing_trade_id",
                "trade_id": None,
            }

        try:
            # 1) Idempotency: skip if we've already logged this trade_id.
            if self._already_processed(trade_id):
                logger.info(
                    "post_trade_loop: already_processed trade_id=%s source=%s",
                    trade_id, source,
                )
                return {
                    "status": "skipped",
                    "reason": "already_processed",
                    "trade_id": trade_id,
                }

            # 2) Load journal entry by trade_id.
            entry = self._load_journal_entry(trade_id)
            if entry is None:
                logger.warning(
                    "post_trade_loop: journal_entry_not_found trade_id=%s source=%s",
                    trade_id, source,
                )
                return {
                    "status": "skipped",
                    "reason": "journal_entry_missing",
                    "trade_id": trade_id,
                }

            # 3) Extract fields with safe defaults.
            fields = self._extract_fields(entry)

            # 4) Shaped reward.
            shaped_reward = self._compute_shaped_reward(
                pnl_pips=fields["pnl_pips"],
                slippage_pips=fields["slippage_pips"],
                mae_pips=fields["mae_pips"],
            )
            # 4b) US-011: RLHF reward shaping
            if self._rlhf_shaper is not None and getattr(self._rlhf_shaper, "_trained", False):
                try:
                    # Build a simple state vector from trade features
                    _state = np.array([
                        fields["pnl_pips"],
                        fields["slippage_pips"],
                        fields["mae_pips"],
                        float(fields["confidence"]) if fields.get("confidence") else 0.5,
                    ], dtype=np.float32)
                    shaped_reward = self._rlhf_shaper.shape_reward(_state, shaped_reward)
                except Exception as _rlhf_err:
                    logger.debug("RLHF shaping failed: %s", _rlhf_err)

            # 5) Append JSONL feedback record.
            self._append_feedback_record(
                trade_id=trade_id,
                pair=fields["pair"],
                shaped_reward=shaped_reward,
                pnl_pips=fields["pnl_pips"],
                slippage_pips=fields["slippage_pips"],
                mae_pips=fields["mae_pips"],
                regime=fields["regime"],
                source=source,
            )

            # 6) Significant-outcome learning (>= 2R move).
            sl_pips = fields["sl_pips"] if fields["sl_pips"] > 0 else 15.0
            if abs(shaped_reward) > 2.0 * sl_pips:
                self._append_learning(
                    trade_id=trade_id,
                    pair=fields["pair"],
                    regime=fields["regime"],
                    shaped_reward=shaped_reward,
                    pnl_pips=fields["pnl_pips"],
                    mae_pips=fields["mae_pips"],
                    slippage_pips=fields["slippage_pips"],
                    source=source,
                )

            logger.info(
                "post_trade_loop: processed #%s shaped_reward=%.1f pair=%s regime=%s",
                trade_id, shaped_reward, fields["pair"], fields["regime"],
            )

            # 7) Diagnostics (lazy import — graceful no-op if missing).
            diag_result = self._run_diagnostics(entry=entry, trade_id=trade_id)

            # 8) Self-heal if diagnostics came back unhealthy.
            heal_result: Optional[Dict[str, Any]] = None
            if diag_result is not None and diag_result.get("status") != "HEALTHY":
                heal_result = self._run_self_heal(
                    diag_result=diag_result, trade_id=trade_id,
                )

            return {
                "status": "processed",
                "trade_id": trade_id,
                "shaped_reward": shaped_reward,
                "diagnostics": diag_result,
                "self_heal": heal_result,
            }

        except Exception as exc:  # noqa: BLE001 — top-level safety net
            self._record_unhandled_error(trade_id=trade_id, exc=exc)
            logger.error(
                "post_trade_loop: unhandled error trade_id=%s error=%r",
                trade_id, exc,
            )
            return {
                "status": "error",
                "trade_id": trade_id,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Idempotency                                                        #
    # ------------------------------------------------------------------ #
    def _already_processed(self, trade_id: str) -> bool:
        """Return True if trade_id appears in the recent feedback log tail."""
        path = self.FEEDBACK_LOG_PATH
        if not path.exists():
            return False

        try:
            with path.open("r", encoding="utf-8") as fh:
                # Read all lines — the log is append-only and small in
                # practice. If it ever grows huge, swap for a tail read.
                lines = fh.readlines()
        except (OSError, UnicodeDecodeError) as err:
            logger.warning(
                "post_trade_loop: feedback log read failed in _already_processed "
                "trade_id=%s error=%r", trade_id, err,
            )
            return False

        tail = lines[-self.IDEMPOTENCY_TAIL_LINES:] if lines else []
        needle = '"trade_id": "{0}"'.format(trade_id)
        needle_sorted = '"trade_id":"{0}"'.format(trade_id)
        for line in tail:
            if needle in line or needle_sorted in line:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Journal loading + field extraction                                 #
    # ------------------------------------------------------------------ #
    def _load_journal_entry(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Load the journal entry matching trade_id. None on any failure."""
        path = self.JOURNAL_PATH
        if not path.exists():
            logger.debug(
                "post_trade_loop: journal file missing path=%s trade_id=%s",
                path, trade_id,
            )
            return None

        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "post_trade_loop: journal load failed trade_id=%s error=%r",
                trade_id, err,
            )
            return None

        if not isinstance(data, list):
            logger.warning(
                "post_trade_loop: journal root is not a list "
                "trade_id=%s type=%s", trade_id, type(data).__name__,
            )
            return None

        for entry in data:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("trade_id", "")) == trade_id:
                return entry
        return None

    def _extract_fields(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Pull shaped-reward inputs from a journal entry. Missing → 0.0."""
        outcome = entry.get("outcome") or {}
        regime_block = entry.get("regime") or {}

        pnl_pips = self._safe_float(
            outcome.get("pnl_pips", entry.get("pnl_pips", 0.0)),
            0.0,
        )
        slippage_pips = self._safe_float(entry.get("slippage_pips", 0.0), 0.0)
        mae_pips = self._safe_float(outcome.get("mae_pips", 0.0), 0.0)
        regime = regime_block.get("volatility_regime") or "UNKNOWN"
        pair = entry.get("pair") or "UNKNOWN"

        sl_pips = self._derive_sl_pips(entry)

        return {
            "pnl_pips": pnl_pips,
            "slippage_pips": slippage_pips,
            "mae_pips": mae_pips,
            "regime": regime,
            "pair": pair,
            "sl_pips": sl_pips,
        }

    def _derive_sl_pips(self, entry: Dict[str, Any]) -> float:
        """Best-effort SL distance in pips. Defaults to 15.0 if indeterminate."""
        stored = entry.get("sl_pips")
        if stored is not None:
            val = self._safe_float(stored, 0.0)
            if val > 0:
                return val

        entry_price = self._safe_float(entry.get("entry_price", 0.0), 0.0)
        sl_price = self._safe_float(entry.get("sl_price", 0.0), 0.0)
        if entry_price <= 0 or sl_price <= 0:
            return 15.0

        # Heuristic pip size: JPY pairs use 0.01, everything else 0.0001.
        pair = str(entry.get("pair") or "")
        pip_size = 0.01 if "JPY" in pair.upper() else 0.0001
        try:
            sl_pips = abs(entry_price - sl_price) / pip_size
        except ZeroDivisionError:
            return 15.0
        return sl_pips if sl_pips > 0 else 15.0

    # ------------------------------------------------------------------ #
    # Shaped reward                                                      #
    # ------------------------------------------------------------------ #
    def _compute_shaped_reward(
        self,
        pnl_pips: float,
        slippage_pips: float,
        mae_pips: float,
    ) -> float:
        """shaped = pnl - SLIP*slippage - MAE_WEIGHT*mae. MAE is drawdown proxy."""
        return (
            pnl_pips
            - (slippage_pips * self.SLIP_WEIGHT)
            - (mae_pips * self.MAE_WEIGHT)
        )

    # ------------------------------------------------------------------ #
    # Writers: feedback jsonl, learnings.md, error log                   #
    # ------------------------------------------------------------------ #
    def _append_feedback_record(
        self,
        trade_id: str,
        pair: str,
        shaped_reward: float,
        pnl_pips: float,
        slippage_pips: float,
        mae_pips: float,
        regime: str,
        source: str,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade_id,
            "pair": pair,
            "shaped_reward": round(float(shaped_reward), 2),
            "pnl_pips": round(float(pnl_pips), 2),
            "slippage_pips": round(float(slippage_pips), 2),
            "mae_pips": round(float(mae_pips), 2),
            "regime": regime,
            "source": source,
        }
        line = json.dumps(record, sort_keys=True) + "\n"

        try:
            self.FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            logger.error(
                "post_trade_loop: feedback log write failed "
                "trade_id=%s path=%s error=%r",
                trade_id, self.FEEDBACK_LOG_PATH, err,
            )

    def _append_learning(
        self,
        trade_id: str,
        pair: str,
        regime: str,
        shaped_reward: float,
        pnl_pips: float,
        mae_pips: float,
        slippage_pips: float,
        source: str,
    ) -> None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        outcome_tag = "win" if shaped_reward >= 0 else "loss"
        pair_safe = str(pair).replace(" ", "_")
        regime_safe = str(regime).replace(" ", "_")
        line = (
            "- [{date}] **OUTCOME/{pair}_{tag}_{regime}**: "
            "Trade {tid} closed at {shaped:.1f} pips "
            "({pnl:.1f} gross, MAE {mae:.1f}, slip {slip:.1f}). "
            "Source: {src}.\n"
        ).format(
            date=date_str,
            pair=pair_safe,
            tag=outcome_tag,
            regime=regime_safe,
            tid=trade_id,
            shaped=shaped_reward,
            pnl=pnl_pips,
            mae=mae_pips,
            slip=slippage_pips,
            src=source,
        )

        try:
            self.LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.LEARNINGS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            logger.warning(
                "post_trade_loop: learnings append failed "
                "trade_id=%s path=%s error=%r",
                trade_id, self.LEARNINGS_PATH, err,
            )

    def _record_unhandled_error(self, trade_id: str, exc: BaseException) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade_id,
            "module": "post_trade_loop",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            self.ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.ERROR_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            # Last-resort: we can't even write the error log. Log and move on.
            logger.error(
                "post_trade_loop: unhandled_errors.jsonl write failed "
                "trade_id=%s error=%r original=%r",
                trade_id, err, exc,
            )

    # ------------------------------------------------------------------ #
    # Lazy-loaded sibling modules                                        #
    # ------------------------------------------------------------------ #
    def _run_diagnostics(
        self,
        entry: Dict[str, Any],
        trade_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            from src.scanner.feedback.diagnostics import (  # type: ignore
                PostTradeDiagnostics,
            )
        except ImportError:
            logger.debug(
                "post_trade_loop: diagnostics module not yet available "
                "trade_id=%s", trade_id,
            )
            return None

        try:
            diag = PostTradeDiagnostics()
            result = diag.run(entry)
            if isinstance(result, dict):
                return result
            logger.warning(
                "post_trade_loop: diagnostics returned non-dict "
                "trade_id=%s type=%s",
                trade_id, type(result).__name__,
            )
            return None
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "post_trade_loop: diagnostics raised trade_id=%s error=%r",
                trade_id, err,
            )
            return None

    def _run_self_heal(
        self,
        diag_result: Dict[str, Any],
        trade_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            from src.scanner.feedback.self_heal import SelfHeal  # type: ignore
        except ImportError:
            logger.debug(
                "post_trade_loop: self_heal module not yet available "
                "trade_id=%s", trade_id,
            )
            return None

        try:
            heal = SelfHeal()
            result = heal.apply(diag_result)
            # US-016: Layer SelfHealConfigUpdater proposals
            if self._self_heal_config_updater is not None:
                try:
                    _updater_result = self._self_heal_config_updater.update(
                        current_config=dict(diag_result.get("config", {})),
                        recent_trades=diag_result.get("recent_trades", []),
                    )
                    if isinstance(result, dict):
                        result["config_updater"] = _updater_result
                except Exception as _updater_err:
                    logger.debug("SelfHealConfigUpdater error: %s", _updater_err)
            if isinstance(result, dict):
                return result
            return {"status": "applied", "detail": repr(result)}
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "post_trade_loop: self_heal raised trade_id=%s error=%r",
                trade_id, err,
            )
            return None

    # ------------------------------------------------------------------ #
    # Utilities                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


__all__ = ["PostTradeLoop"]
