"""OANDA -> cache -> unified training session runner.

This is the main loop you described:
- Fetch new OANDA candles and append/dedupe into a local CSV cache.
- Train the unified multi-head model, resuming from the last/best checkpoint.
- Save the best checkpoint in `trained_data/models/` for Buddy to load.

This module is intentionally conservative and synchronous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from oanda_candle_cache import update_oanda_cache


@dataclass(frozen=True)
class OandaTrainSessionResult:
    instruments: List[str]
    csv_paths: Dict[str, str]
    model_path: str
    metrics_path: str


def _split_instruments(value: str) -> List[str]:
    parts: List[str] = []
    for chunk in str(value).replace(";", ",").split(","):
        p = chunk.strip()
        if p:
            parts.append(p)
    return parts


def run_oanda_unified_training_session(
    config_path: str,
    *,
    instruments: str,
    granularity: str = "M5",
    candles: int = 5000,
    resume_path: Optional[str] = None,
    all_features: bool = False,
) -> OandaTrainSessionResult:
    """Run one training session using fresh OANDA candles.

    `instruments` may be a comma-separated string, e.g. "EUR_USD,GBP_USD".
    Training will run sequentially per instrument; the model checkpoint from each
    run is used as the resume checkpoint for the next.

    Args:
        all_features: If True, use all engineered features (technical indicators,
                     statistical, time, lag, rolling) instead of just OHLCV.
    """
    from unified_multitask_training import train_unified_multitask

    inst_list = _split_instruments(instruments)
    if not inst_list:
        raise ValueError("No instruments provided")

    csv_paths: Dict[str, str] = {}

    last_model_path: Optional[str] = resume_path
    last_metrics_path: Optional[str] = None

    for inst in inst_list:
        csv_path = update_oanda_cache(
            config_path,
            instrument=inst,
            granularity=granularity,
            candles=int(candles),
            price="MBA",
        )
        csv_paths[inst] = str(csv_path)

        result: Dict[str, Any] = train_unified_multitask(
            config_path,
            csv_path=str(csv_path),
            resume_path=last_model_path,
            all_features=all_features,
        )
        last_model_path = str(result.get("model_path") or last_model_path or "")
        last_metrics_path = str(result.get("metrics_path") or last_metrics_path or "")

    if not last_model_path:
        raise RuntimeError("Unified training did not produce a model_path")

    return OandaTrainSessionResult(
        instruments=inst_list,
        csv_paths=csv_paths,
        model_path=last_model_path,
        metrics_path=last_metrics_path or "",
    )
