"""
SOTA Integration Layer — factory for drop-in replacement of legacy scanner components.

Provides:
  - `resolve_inference_engine(config, scanner)` → ModularEnsembleInference | SOTAInference
  - `resolve_agent_team(config)` → ScannerAgentTeam | NeuralAgentTeam

The scanner engine calls these factories during __init__ instead of directly
instantiating the legacy classes.  When config.use_sota_inference or
config.use_neural_agents is True, the SOTA/Neural variants are returned;
otherwise the legacy implementations are used unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)


def resolve_inference_engine(
    config: ScannerConfig,
    scanner: Optional[Any] = None,
) -> Any:
    """Return the inference engine for the scanner.

    If config.use_sota_inference is True, loads SOTAInference.
    Otherwise falls back to ModularEnsembleInference (legacy).
    """
    if getattr(config, "use_sota_inference", False):
        try:
            from src.sota_core import SOTAInference, SOTAInferenceConfig
            # MED-3 FIX: normalize min_confidence consistently to [0, 1]
            _raw_min_conf = float(getattr(config, "min_confidence", 55.0))
            _norm_min_conf = _raw_min_conf / 100.0 if _raw_min_conf > 1.0 else _raw_min_conf
            sota_cfg = SOTAInferenceConfig(
                model_path=getattr(config, "sota_model_path", "trained_data/models/sota_finetuned/sota_model.keras"),
                confidence_threshold=_norm_min_conf,
                min_tcn_probability=float(getattr(config, "min_tcn_probability", 0.60)),
                min_confidence=_norm_min_conf,
                max_drawdown_pct=float(getattr(config, "max_drawdown_pct", 0.025)),
                max_uncertainty_std=float(getattr(config, "max_uncertainty_std", 0.15)),
                use_tcn_volatility_filter=bool(getattr(config, "use_tcn_volatility_filter", True)),
                min_volatility_regime=int(getattr(config, "min_volatility_regime", 0)),
                tcn_required=bool(getattr(config, "require_tcn_volatility", False)),
                final_score_threshold=float(getattr(config, "final_score_threshold", 0.45)),
            )
            engine = SOTAInference(sota_cfg)
            logger.info("SOTA integration: using SOTAInference (raw-sequence DL model)")
            return engine
        except Exception as e:
            logger.warning(f"SOTA inference init failed ({e}) — falling back to legacy ensemble")

    # Legacy path
    try:
        from src.core.modular_inference import ModularEnsembleInference, InferenceConfig
        legacy_cfg = InferenceConfig(
            min_tcn_probability=float(getattr(config, "min_tcn_probability", 0.60)),
            min_confidence=float(getattr(config, "min_confidence", 0.55)),
            max_drawdown_pct=float(getattr(config, "max_drawdown_pct", 0.025)),
            max_uncertainty_std=float(getattr(config, "max_uncertainty_std", 0.15)),
            use_tcn_volatility_filter=bool(getattr(config, "use_tcn_volatility_filter", True)),
            min_volatility_regime=int(getattr(config, "min_volatility_regime", 0)),
            tcn_required=bool(getattr(config, "require_tcn_volatility", False)),
            final_score_threshold=float(getattr(config, "final_score_threshold", 0.45)),
            min_meta_confidence=float(getattr(config, "meta_labeler_threshold", 0.52)),
        )
        return ModularEnsembleInference(config=legacy_cfg)
    except Exception as e:
        logger.error(f"Legacy ensemble inference init failed: {e}")
        return None


def resolve_agent_team(config: ScannerConfig) -> Any:
    """Return the agent consensus team for the scanner.

    If config.use_neural_agents is True, returns NeuralAgentTeam.
    Otherwise falls back to ScannerAgentTeam (legacy rule-based agents).
    """
    if getattr(config, "use_neural_agents", False):
        try:
            from src.scanner.agents.neural import NeuralAgentTeam
            team = NeuralAgentTeam(config)
            logger.info("SOTA integration: using NeuralAgentTeam (learned policies)")
            return team
        except Exception as e:
            logger.warning(f"Neural agent team init failed ({e}) — falling back to legacy agents")

    # Legacy path
    try:
        from src.scanner.agents import ScannerAgentTeam
        return ScannerAgentTeam(config)
    except Exception as e:
        logger.error(f"Legacy agent team init failed: {e}")
        return None


def is_sota_active(config: ScannerConfig) -> bool:
    """Return True if either SOTA inference or neural agents are enabled."""
    return bool(
        getattr(config, "use_sota_inference", False)
        or getattr(config, "use_neural_agents", False)
    )
