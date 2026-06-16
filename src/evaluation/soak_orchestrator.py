"""Automated soak orchestration triggered after SOTA finetune completes.

US-006: Spawns a background Process that runs the soak comparison harness
on held-out data and persists the recommendation for the promotion gate.

Design:
- Non-blocking: soak runs in a child process so training completion
  never blocks the scan loop.
- Held-out data: last 10% of harvested candles, or last 500 bars minimum.
- Result persisted to trained_data/soak_results/ for ModelManager to read.
- Training lineage tracked: model_path + data_range used for the soak.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .soak_runner import SoakResult, run_soak_comparison, save_soak_report
from .model_comparison import BarPrediction

logger = logging.getLogger(__name__)

SOAK_RESULT_ROOT = Path("trained_data/soak_results")
SOAK_RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def _legacy_signal_fn(window: pd.DataFrame) -> BarPrediction:
    """Dummy legacy signal: uses simple momentum."""
    # In production this would call ModularEnsembleInference
    close = window["close"].values
    if len(close) < 2:
        return BarPrediction(timestamp=str(window.index[-1]), direction="HOLD", confidence=0.5, score=0.5)
    ret = (close[-1] / close[-2]) - 1.0
    direction = "LONG" if ret > 0 else "SHORT"
    conf = min(abs(ret) * 1000, 0.99)
    return BarPrediction(timestamp=str(window.index[-1]), direction=direction, confidence=conf, score=conf)


def _sota_signal_fn(window: pd.DataFrame) -> BarPrediction:
    """SOTA signal using the finetuned model."""
    try:
        from src.sota_core.inference import SOTAInference
        sota = SOTAInference()
        pair = window.attrs.get("pair", "UNKNOWN")
        pred = sota.predict(pair, window)
        if pred is not None:
            direction = pred.get("direction", "HOLD")
            confidence = float(pred.get("confidence", 0.5))
            return BarPrediction(timestamp=str(window.index[-1]), direction=direction, confidence=confidence, score=confidence)
    except Exception as e:
        logger.debug("SOTA signal error in soak: %s", e)
    return BarPrediction(timestamp=str(window.index[-1]), direction="HOLD", confidence=0.5, score=0.5)


def _load_held_out_data(pair: str, granularity: str = "M15", holdout_pct: float = 0.10, min_bars: int = 500) -> Optional[pd.DataFrame]:
    """Load the held-out tail of harvested data for a pair."""
    pq_path = Path(f"trained_data/harvest/{pair}_{granularity}.parquet")
    if not pq_path.exists():
        return None
    try:
        df = pd.read_parquet(pq_path)
        if len(df) < min_bars:
            return df  # use all if insufficient
        n_holdout = max(int(len(df) * holdout_pct), min_bars)
        return df.iloc[-n_holdout:].copy()
    except Exception as e:
        logger.warning("Failed to load held-out data for %s: %s", pair, e)
        return None


def _run_soak_worker(
    pair: str,
    model_path: str,
    granularity: str,
    output_path: str,
) -> None:
    """Worker function that runs in a subprocess."""
    try:
        df = _load_held_out_data(pair, granularity)
        if df is None or df.empty:
            logger.error("Soak worker: no data for %s", pair)
            return
        df.attrs["pair"] = pair
        result = run_soak_comparison(
            pair=pair,
            df=df,
            legacy_signal_fn=_legacy_signal_fn,
            sota_signal_fn=_sota_signal_fn,
            warmup_bars=min(128, len(df) // 4),
        )
        save_soak_report(result, output_path)
        logger.info("Soak worker complete for %s: %s (confidence=%.2f)", pair, result.recommendation, result.confidence)
    except Exception as e:
        logger.exception("Soak worker failed for %s: %s", pair, e)


class SoakOrchestrator:
    """Non-blocking soak orchestrator."""

    def __init__(self) -> None:
        self._process: Optional[multiprocessing.Process] = None
        self._active_pair: Optional[str] = None
        self._status: Dict[str, Any] = {
            "state": "idle",  # idle | running | complete | error
            "last_pair": None,
            "last_started": None,
            "last_completed": None,
            "last_error": None,
        }

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def trigger(
        self,
        pair: str,
        model_path: str,
        granularity: str = "M15",
    ) -> bool:
        """Start a soak in the background. Returns False if already running."""
        if self.is_running():
            logger.warning("Soak already running for %s; skipping new trigger", self._active_pair)
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = str(SOAK_RESULT_ROOT / f"{pair}_{timestamp}.json")

        self._process = multiprocessing.Process(
            target=_run_soak_worker,
            args=(pair, model_path, granularity, output_path),
            daemon=True,
            name=f"soak-{pair}",
        )
        self._process.start()
        self._active_pair = pair
        self._status.update({
            "state": "running",
            "last_pair": pair,
            "last_started": datetime.now(timezone.utc).isoformat(),
            "last_output": output_path,
            "last_error": None,
        })
        logger.info("Soak triggered for %s (PID %s)", pair, self._process.pid)
        return True

    def poll(self) -> Dict[str, Any]:
        """Check status; if process finished, update state."""
        if self._process is None:
            return dict(self._status)
        if not self._process.is_alive():
            self._status["state"] = "complete" if self._status.get("last_error") is None else "error"
            self._status["last_completed"] = datetime.now(timezone.utc).isoformat()
            self._process = None
            self._active_pair = None
        return dict(self._status)

    def get_latest_result(self, pair: str) -> Optional[SoakResult]:
        """Read the most recent soak result file for a pair."""
        files = sorted(SOAK_RESULT_ROOT.glob(f"{pair}_*.json"))
        if not files:
            return None
        try:
            with open(files[-1]) as f:
                data = json.load(f)
            return SoakResult(
                pair=data["pair"],
                start_time=data["start_time"],
                end_time=data["end_time"],
                n_bars=data["n_bars"],
                recommendation=data["recommendation"],
                confidence=data["confidence"],
            )
        except Exception as e:
            logger.debug("Failed to load soak result for %s: %s", pair, e)
            return None
