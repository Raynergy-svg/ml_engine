"""
Scanner specialist agents.

Adds a lightweight deliberation layer on top of the existing scanner signal.
Each agent evaluates one aspect of the setup and emits a structured verdict
that can be combined into a weighted vote.
"""

from __future__ import annotations

import logging
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.scanner.results import PairAnalysis
from src.observability import (
    attributes as weave_attrs,
    init_weave,
    op as weave_op,
    patch_instance_methods,
)
from src.data.order_book import (
    extract_order_features as _ob_order_features,
    extract_position_features as _ob_position_features,
    snapshot as _ob_snapshot,
)
from src.scanner.agents.memory import (
    cache_verdicts_for_trade_open as _cache_verdicts_for_trade_open_mem,
    get_memory_store as _get_agent_memory,
    is_enabled as _agent_memory_enabled,
)

logger = logging.getLogger(__name__)

# US-503: Optional staleness import — wrapped so the module loads even if unavailable
try:
    from src.scanner.automation.model_freshness import get_model_freshness
except Exception:
    get_model_freshness = None  # type: ignore[assignment]




def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if result != result:
            return default
        return result
    except Exception:
        return default


def _last_value(df: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in df.columns or df.empty:
        return default
    return _safe_float(df[column].iloc[-1], default)


def _normalize_regime_name(regime: Any) -> str:
    """Normalize scanner volatility regime values to canonical names."""
    regime_names = ["LOW", "NORMAL", "HIGH", "EXTREME"]
    if regime is not None and str(regime).isdigit():
        regime_idx = int(str(regime))
        if 0 <= regime_idx < len(regime_names):
            return regime_names[regime_idx]
    name = str(regime or "NORMAL").upper()
    return name if name in regime_names else "NORMAL"


def _oanda_candles_to_frame(raw: Any) -> Optional[pd.DataFrame]:
    """Convert an OANDA candles response or DataFrame to canonical OHLCV."""
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        for column in ("open", "high", "low", "close"):
            if column not in df.columns:
                return None
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if "volume" not in df.columns:
            df["volume"] = 0
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        return df.dropna(subset=["open", "high", "low", "close"])

    candles = raw.get("candles", []) if isinstance(raw, dict) else []
    rows: list[dict[str, Any]] = []
    for candle in candles:
        if not candle.get("complete", True):
            continue
        mid = candle.get("mid") or candle.get("bid") or candle.get("ask") or {}
        try:
            rows.append({
                "time": candle.get("time"),
                "open": float(mid.get("o")),
                "high": float(mid.get("h")),
                "low": float(mid.get("l")),
                "close": float(mid.get("c")),
                "volume": int(candle.get("volume", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time")
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str, min_bars: int) -> Optional[pd.DataFrame]:
    """Aggregate lower-timeframe candles using proper OHLCV semantics."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    source = df.sort_index()
    required = {"open", "high", "low", "close"}
    if not required.issubset(source.columns):
        return None
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in source.columns:
        agg["volume"] = "sum"
    resampled = source.resample(rule, label="right", closed="right").agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    if len(resampled) < min_bars:
        return None
    return resampled


@dataclass
class AgentDecisionContext:
    """Normalized context shared by specialist agents."""

    analysis: PairAnalysis
    df_raw: pd.DataFrame
    df_feat: pd.DataFrame
    gate_details: Dict[str, Any]
    config: Any


@dataclass
class AgentVerdict:
    """Outcome from one specialist agent."""

    name: str
    score: float
    passed: bool
    weight: float
    reason: str
    reason_code: str
    confidence_delta: float = 0.0
    block_trade: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": _clip01(self.score),
            "passed": bool(self.passed),
            "weight": float(self.weight),
            "reason": self.reason,
            "reason_code": self.reason_code,
            "confidence_delta": float(self.confidence_delta),
            "block_trade": bool(self.block_trade),
            "metadata": dict(self.metadata),
        }


class ScannerAgentTeam:
    """Specialist-agent layer for scanner opportunity assessment."""

    _BASE_WEIGHTS: Dict[str, float] = {
        "trend": 1.15,
        "mean_reversion": 0.90,
        "volatility": 1.00,
        "risk_sentinel": 1.25,
        "uncertainty": 1.10,
        "execution_quality": 1.05,
        "news_risk": 0.95,
        "multi_timeframe": 1.10,
        "pair_performance": 0.85,
        "momentum": 1.05,
        "session_timing": 0.80,
        "support_resistance": 1.00,
        "order_flow": 0.95,  # Agent #15: OANDA order/position book contrarian signal
        "trader_readiness": 0.50,  # Agent #13: Aura human-side readiness signal
        "devil_advocate": 1.30,  # Agent #14: Adversarial bear-case evaluator (runs LAST)
    }

    _READINESS_SIGNAL_PATH = ".aura/bridge/readiness_signal.json"

    _WEIGHTS_FILE = "trained_data/models/agent_weights.json"
    _REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]

    def __init__(self, config: Any):
        self.config = config
        # Weave tracing: init is idempotent; safe even when Scanner bypasses Orchestrator.
        # Then wrap every _evaluate_* method so each agent call appears in the trace tree.
        # No-op when BUDDY_WEAVE_ENABLED is unset.
        init_weave()
        # Exclude _evaluate_body (the top-level dispatch wrapped by evaluate()).
        patch_instance_methods(self, prefix="_evaluate_", exclude={"_evaluate_body"})
        self._learned_weights: Dict[str, Any] = self._load_learned_weights()
        self._regime_weights: Dict[str, Dict[str, float]] = {}  # Cache of regime-specific weights
        self._global_weights: Dict[str, float] = {}  # Cross-regime running average
        self._regime_trade_counts: Dict[str, int] = {}  # Track trades per regime
        self._agent_lifecycle = None  # Phase 20 (US-122): injected from Scanner
        self._accuracy_matrix = None  # Phase 25 (US-155): injected from Scanner
        self._confidence_calibrator = None  # Phase 44 (US-280): Confidence calibration system
        self._meta_overrides: Dict[str, Dict[str, float]] = {}  # Tier 6: MetaLearner hyperparameter overrides
        self._meta_strategy_agent = None  # US-017: MetaStrategyAgent
        self._active_weight_regime: Optional[str] = None
        self._migrate_legacy_weights()

        # Phase 44 (US-280): Confidence Calibration System initialization
        try:
            from src.scanner.confidence_calibration import ConfidenceCalibrationSystem
            self._confidence_calibrator = ConfidenceCalibrationSystem()
            logger.info("Phase 44: Confidence calibration system initialized")
        except Exception as e:
            logger.debug(f"Confidence calibration init deferred: {e}")

        # Phase 52 (US-323): Isotonic Calibrator — replaces Platt scaling with PAVA-based calibration
        self._isotonic_calibrator = None
        try:
            from src.scanner.isotonic_calibrator import IsotonicCalibrator
            self._isotonic_calibrator = IsotonicCalibrator()
            # No persisted-model convention exists; calibrator falls back to
            # raw confidence until fit_from_journal() is invoked.
            logger.info("Phase 52 (US-323): Isotonic calibrator initialized (no model — will use fallback)")
        except Exception as e:
            logger.debug(f"Phase 52: Isotonic calibrator init deferred: {e}")

        # Phase 52 (US-324): Confidence Decomposer — directional/timing/magnitude breakdown
        self._confidence_decomposer = None
        try:
            from src.scanner.confidence_decomposer import ConfidenceDecomposer
            self._confidence_decomposer = ConfidenceDecomposer()
            logger.info("Phase 52 (US-324): Confidence decomposer initialized")
        except Exception as e:
            logger.debug(f"Phase 52: Confidence decomposer init deferred: {e}")

        # Phase 53 (US-331): Unified Confidence Pipeline — single-path calibration with monitoring
        self._confidence_pipeline = None
        try:
            from src.scanner.confidence_pipeline import UnifiedConfidencePipeline
            self._confidence_pipeline = UnifiedConfidencePipeline()
            logger.info("Phase 53 (US-331): Unified confidence pipeline initialized")
        except Exception as e:
            logger.debug(f"Phase 53: Confidence pipeline init deferred: {e}")

        # Phase 45 (US-283): Bayesian Agent Weights (Thompson Sampling)
        self._bayesian_weights = None
        try:
            from src.scanner.bayesian_agent_weights import (
                create_default_bayesian_weights,
            )
            self._bayesian_weights = create_default_bayesian_weights()

            # Load persisted state if available
            import os
            _bw_path = "trained_data/bayesian_weights.json"
            if os.path.exists(_bw_path):
                try:
                    self._bayesian_weights.load_state(_bw_path)
                    logger.info("Phase 45: Loaded Bayesian agent weights state")
                except Exception as _bw_load_err:
                    logger.warning(f"Phase 45: Could not load Bayesian weights: {_bw_load_err}")

            logger.info("Phase 45: Bayesian agent weights initialized (Thompson Sampling)")
        except Exception as e:
            logger.debug(f"Phase 45: Bayesian weights init deferred: {e}")

        # Phase 47 (US-297): Expectancy Tracker for per-agent per-regime weight modifiers
        self._expectancy_tracker = None
        try:
            from src.scanner.expectancy_tracker import (
                create_default_expectancy_tracker,
            )
            self._expectancy_tracker = create_default_expectancy_tracker()
            self._expectancy_tracker.load_state()
            logger.info("Phase 47: Expectancy tracker initialized for agent weight modifiers")
        except Exception as e:
            logger.debug(f"Phase 47: Expectancy tracker init deferred: {e}")

        # Phase 55 (US-343): Disagreement Tracker — correlates agent disagreement with outcomes
        self._disagreement_tracker = None
        try:
            from src.scanner.disagreement_tracker import DisagreementTracker
            self._disagreement_tracker = DisagreementTracker()
            logger.info("Phase 55 (US-343): Disagreement tracker initialized")
        except Exception as e:
            logger.debug(f"Phase 55: Disagreement tracker init deferred: {e}")

        # Phase 47 (US-294): Multi-Timeframe Confluence
        self._mtf_confluence = None
        try:
            from src.scanner.mtf_confluence import create_default_mtf_confluence
            self._mtf_confluence = create_default_mtf_confluence()
            logger.info("Phase 47: MTF confluence module initialized")
        except Exception as e:
            logger.debug(f"Phase 47: MTF confluence init deferred: {e}")
        # US-017: MetaStrategyAgent — dynamic strategy selection per regime
        if getattr(self.config, "enable_meta_learning", False):
            try:
                from src.autonomy.meta_agent import MetaStrategyAgent, MetaAgentConfig
                self._meta_strategy_agent = MetaStrategyAgent(MetaAgentConfig())
                logger.info("US-017: MetaStrategyAgent initialized")
            except Exception as e:
                logger.debug(f"US-017: MetaStrategyAgent init deferred: {e}")


        # Phase 47 (US-295): Ensemble Conflict Resolver
        self._ensemble_conflict = None
        try:
            from src.scanner.ensemble_conflict import create_default_conflict_resolver
            self._ensemble_conflict = create_default_conflict_resolver()
            logger.info("Phase 47: Ensemble conflict resolver initialized")
        except Exception as e:
            logger.debug(f"Phase 47: Ensemble conflict resolver init deferred: {e}")

        # Order flow agent: OANDA order/position book cache and client
        self._oanda_client = None
        self._position_book_cache: Dict[str, Any] = {}  # {pair: {"data": ..., "timestamp": datetime}}
        self._order_book_cache: Dict[str, Any] = {}  # {pair: {"data": ..., "timestamp": datetime}}

        # Episodic Market Memory: Record and query historical trade patterns
        self._episodic_memory = None
        try:
            from src.scanner.automation.episodic_memory import EpisodicMemory
            _max_episodes = getattr(config, "episodic_memory_max_episodes", 500)
            self._episodic_memory = EpisodicMemory(max_episodes=_max_episodes)
            logger.info("Episodic memory initialized (pattern suppression enabled)")
        except Exception as e:
            logger.warning(f"Episodic memory initialization failed: {e} (pattern suppression disabled)")
            self._episodic_memory = None
        # US-005: LLM Macro Agent initialization (local import to avoid circular dep)
        self._llm_macro_agent: Any = None
        try:
            from src.scanner.agents.llm_macro_agent import LLMMacroAgent, LLMMacroAgentConfig
            if getattr(self.config, "enable_llm_macro_agent", False):
                try:
                    shadow = getattr(self.config, "enable_llm_macro_shadow", True)
                    self._llm_macro_agent = LLMMacroAgent(
                        LLMMacroAgentConfig(shadow_mode=shadow)
                    )
                    logger.info("ScannerAgentTeam: LLM Macro Agent enabled (shadow=%s)", shadow)
                except Exception as exc:
                    logger.warning("ScannerAgentTeam: failed to init LLM Macro Agent: %s", exc)
            elif getattr(self.config, "enable_llm_macro_shadow", True):
                try:
                    self._llm_macro_agent = LLMMacroAgent(
                        LLMMacroAgentConfig(shadow_mode=True)
                    )
                    logger.info("ScannerAgentTeam: LLM Macro Agent shadow-logging enabled")
                except Exception as exc:
                    logger.warning("ScannerAgentTeam: failed to init LLM Macro Agent (shadow): %s", exc)
        except Exception as exc:
            logger.debug("ScannerAgentTeam: LLM Macro Agent import unavailable: %s", exc)

        # ── Neural Agent Team (lazy bridge) ────────────────────────────────
        # When use_neural_agents=True, we keep a reference to NeuralAgentTeam
        # so trade outcomes can be bridged into neural replay buffers.
        self._neural_team = None
        self._neural_update_counter = 0
        self._trade_verdict_cache: Dict[str, Any] = {}
        _neural_enabled = getattr(self.config, "use_neural_agents", False)
        if _neural_enabled:
            try:
                from src.scanner.agents.neural import NeuralAgentTeam
                self._neural_team = NeuralAgentTeam(self.config)
                logger.info(
                    "ScannerAgentTeam: NeuralAgentTeam initialized (%s mode)",
                    "shadow" if getattr(self.config, "shadow_neural_agents", True) else "live",
                )
            except Exception as exc:
                logger.warning("ScannerAgentTeam: NeuralAgentTeam init failed: %s", exc)
        # Graph Attention Consensus (optional, see end of file)

    # --- Persistent weight management ---

    def _migrate_legacy_weights(self) -> None:
        """Convert legacy flat weights dict to regime-aware format if needed."""
        if not self._learned_weights:
            return
        # If dict has agent names directly (no regime keys), migrate it
        if any(k in self._learned_weights for k in self._BASE_WEIGHTS.keys()):
            try:
                legacy_weights = dict(self._learned_weights)
                # Rebuild as regime-aware
                self._learned_weights = {
                    "_global": legacy_weights,
                    "NORMAL": dict(legacy_weights),
                    "HIGH": dict(legacy_weights),
                    "EXTREME": dict(legacy_weights),
                    "LOW": dict(legacy_weights),
                    "_meta": {"min_trades_per_regime": 10},
                }
                self._learned_weights, _ = self._ensure_complete_weight_schema(self._learned_weights)
                self._save_learned_weights()
                logger.info("Migrated agent weights to regime-aware format")
            except Exception as e:
                logger.warning(f"Failed to migrate legacy weights: {e}")
                self._learned_weights = self._init_regime_weights()

    def _ensure_complete_weight_schema(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Normalize regime-aware weights to include all baseline agent keys.

        This keeps persisted weights compatible with health checks that expect a
        complete agent map, while preserving learned values where present.
        """
        if not isinstance(data, dict):
            return self._init_regime_weights(), True

        changed = False

        for regime in self._REGIME_NAMES + ["_global"]:
            regime_weights = data.get(regime)
            if not isinstance(regime_weights, dict):
                data[regime] = dict(self._BASE_WEIGHTS)
                changed = True
                continue

            for agent_name, base_weight in self._BASE_WEIGHTS.items():
                if agent_name not in regime_weights:
                    regime_weights[agent_name] = base_weight
                    changed = True

        meta = data.get("_meta")
        if not isinstance(meta, dict):
            data["_meta"] = {"min_trades_per_regime": 10}
            changed = True
        elif "min_trades_per_regime" not in meta:
            meta["min_trades_per_regime"] = 10
            changed = True

        return data, changed

    def _load_learned_weights(self) -> Dict[str, Any]:
        """Load learned agent weights from disk with corruption recovery.

        Handles:
        - Legacy flat dict format
        - Regime-aware structure
        - JSON parse errors (falls back to baseline)
        - NaN/inf values (resets those agents to baseline)
        """
        path = Path(self._WEIGHTS_FILE)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                # Validate regime-aware structure
                if isinstance(data, dict) and any(regime in data for regime in self._REGIME_NAMES + ["_global", "_meta"]):
                    # Validate weight values
                    validated = self._validate_weights(data)
                    if validated:
                        normalized, _ = self._ensure_complete_weight_schema(validated)
                        return normalized
                # Legacy flat dict
                elif isinstance(data, dict) and any(k in data for k in self._BASE_WEIGHTS.keys()):
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in agent_weights.json: {e}. Using baseline weights.")
            except Exception as e:
                logger.warning(f"Failed to load agent weights: {e}")
        return self._init_regime_weights()

    def _validate_weights(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate all weight values are finite and in acceptable range.

        Args:
            data: Weights structure to validate

        Returns:
            Validated data, or None if validation failed (caller will use baseline)
        """
        import math

        valid_range = (0.1, 2.0)
        issues_found = False

        for regime in self._REGIME_NAMES + ["_global"]:
            if regime not in data or not isinstance(data[regime], dict):
                continue

            regime_weights = data[regime]
            for agent_name in list(regime_weights.keys()):
                if agent_name not in self._BASE_WEIGHTS:
                    logger.warning(
                        f"Unknown agent key {agent_name!r} in {regime} weights (not in "
                        "_BASE_WEIGHTS). Dropping -- persisted weight data should only ever "
                        "contain the canonical agent roster."
                    )
                    del regime_weights[agent_name]
                    issues_found = True
                    continue

                val = regime_weights[agent_name]

                # Check for NaN or inf
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval):
                        logger.warning(f"Invalid value ({val}) for {agent_name} in {regime}. Resetting to baseline.")
                        regime_weights[agent_name] = self._BASE_WEIGHTS.get(agent_name, 1.0)
                        issues_found = True
                    elif not (valid_range[0] <= fval <= valid_range[1]):
                        logger.warning(f"Out-of-range value ({fval}) for {agent_name} in {regime}. Clamping to valid range.")
                        regime_weights[agent_name] = max(valid_range[0], min(valid_range[1], fval))
                        issues_found = True
                except (ValueError, TypeError):
                    logger.warning(f"Non-numeric value for {agent_name} in {regime}. Resetting to baseline.")
                    regime_weights[agent_name] = self._BASE_WEIGHTS.get(agent_name, 1.0)
                    issues_found = True

        if issues_found:
            logger.info("Corrupted weights detected and recovered")

        return data

    def _init_regime_weights(self) -> Dict[str, Any]:
        """Initialize regime-aware weight structure."""
        return {
            "_global": dict(self._BASE_WEIGHTS),
            "NORMAL": dict(self._BASE_WEIGHTS),
            "HIGH": dict(self._BASE_WEIGHTS),
            "EXTREME": dict(self._BASE_WEIGHTS),
            "LOW": dict(self._BASE_WEIGHTS),
            "_meta": {"min_trades_per_regime": 10},
        }

    def reload_learned_weights(self) -> None:
        """Reload agent weights from disk and apply time-based decay.

        Runs once per session start. Applies:
        1. Load weights from disk
        2. Apply time-based decay toward baseline (1% per 24 hours, caps at 50%)
        3. Apply confidence scaling based on total trade count
        4. Clear regime cache to force fresh load

        This ensures weights don't persist indefinitely between sessions.
        """
        self._learned_weights = self._load_learned_weights()
        self._migrate_legacy_weights()
        self._learned_weights, schema_repaired = self._ensure_complete_weight_schema(self._learned_weights)
        self._apply_time_decay()
        self._apply_confidence_scaling()
        if schema_repaired:
            self._save_learned_weights()
        self._regime_weights = {}  # Clear cache to force fresh load

    def _apply_time_decay(self) -> None:
        """Apply time-based decay: weights drift 1% toward baseline per 24 hours.

        Formula: decayed = weight + (1.0 - weight) * min(hours_since / 2400, 0.5)
        - Caps at 50% drift toward baseline
        - Never fully resets
        - Logged for transparency
        """
        if not self._learned_weights or "_meta" not in self._learned_weights:
            return

        meta = self._learned_weights.get("_meta", {})
        last_updated_str = meta.get("last_updated")

        if not last_updated_str:
            # No timestamp yet, set it now
            meta["last_updated"] = datetime.now(timezone.utc).isoformat()
            return

        try:
            last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_since = (now - last_updated).total_seconds() / 3600.0

            if hours_since < 0.1:  # Skip if less than 5 minutes ago
                return

            # Cap drift at 50%
            decay_factor = min(hours_since / 2400.0, 0.5)

            changed = False
            for regime in self._REGIME_NAMES + ["_global"]:
                if regime not in self._learned_weights or not isinstance(self._learned_weights[regime], dict):
                    continue

                regime_weights = self._learned_weights[regime]
                for agent_name in list(regime_weights.keys()):
                    base = self._BASE_WEIGHTS.get(agent_name, 1.0)
                    current = _safe_float(regime_weights.get(agent_name), base)

                    # Apply time decay
                    decayed = current + (base - current) * decay_factor
                    decayed = round(decayed, 4)

                    if abs(decayed - current) > 1e-4:
                        regime_weights[agent_name] = decayed
                        changed = True

            if changed:
                meta["last_updated"] = now.isoformat()
                self._save_learned_weights()
                logger.info(f"Applied {hours_since:.0f}h time decay to agent weights (factor: {decay_factor:.2%})")

        except Exception as e:
            logger.debug(f"Time decay error: {e}")

    def _apply_confidence_scaling(self) -> None:
        """Scale learned weights by confidence based on trade count.

        Confidence scaling blends baseline and learned weights:
        - < 10 trades: 70% baseline + 30% learned
        - 10-50 trades: 30% baseline + 70% learned
        - > 50 trades: 100% learned weights

        Formula: final = baseline * (1 - confidence) + learned * confidence
        where confidence = min(total_trades / 50, 1.0)

        This prevents over-fitting to small sample sizes.
        """
        if not self._learned_weights or "_meta" not in self._learned_weights:
            return

        meta = self._learned_weights.get("_meta", {})
        total_trades = _safe_float(meta.get("total_trades", 0), 0)

        # Calculate confidence score (0.0 to 1.0)
        confidence = min(total_trades / 50.0, 1.0)

        if confidence >= 1.0:
            # Mature: use 100% learned weights, no scaling needed
            return

        changed = False
        for regime in self._REGIME_NAMES + ["_global"]:
            if regime not in self._learned_weights or not isinstance(self._learned_weights[regime], dict):
                continue

            regime_weights = self._learned_weights[regime]
            for agent_name in list(regime_weights.keys()):
                base = self._BASE_WEIGHTS.get(agent_name, 1.0)
                learned = _safe_float(regime_weights.get(agent_name), base)

                # Blend: final = baseline * (1 - confidence) + learned * confidence
                scaled = base * (1.0 - confidence) + learned * confidence
                scaled = round(scaled, 4)

                if abs(scaled - learned) > 1e-4:
                    regime_weights[agent_name] = scaled
                    changed = True

        if changed:
            self._save_learned_weights()
            confidence_pct = int(confidence * 100)
            logger.info(f"Applied confidence scaling ({confidence_pct}%) based on {int(total_trades)} trades")

    def get_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """Get agent weights for a specific volatility regime.

        Applies confidence scaling based on trade count at retrieval time.
        Returns weighted blend of baseline and learned weights if still in
        low-confidence phase (< 50 trades).

        Args:
            regime: Volatility regime name ("LOW", "NORMAL", "HIGH", "EXTREME", or regime index as string)

        Returns:
            Dict of agent name -> weight for this regime (or _global fallback if insufficient data)
        """
        # Normalize regime name
        regime = _normalize_regime_name(regime)

        # Return cached weights if available
        if regime in self._regime_weights:
            return dict(self._regime_weights[regime])

        # Ensure learned weights are regime-aware
        if not isinstance(self._learned_weights, dict) or "_global" not in self._learned_weights:
            self._learned_weights = self._init_regime_weights()

        # Check if regime has sufficient data
        min_trades = _safe_float(
            self._learned_weights.get("_meta", {}).get("min_trades_per_regime", 10),
            10,
        )
        regime_trades = _safe_float(self._learned_weights.get("_meta", {}).get(f"trades_{regime}", 0), 0)

        # Determine base weights for this regime
        if regime in self._learned_weights and regime_trades >= min_trades:
            weights = self._learned_weights.get(regime, {})
            if weights and isinstance(weights, dict):
                selected_weights = dict(weights)
            else:
                selected_weights = dict(self._BASE_WEIGHTS)
        else:
            # Fall back to global (cross-regime) weights
            global_weights = self._learned_weights.get("_global", dict(self._BASE_WEIGHTS))
            if global_weights and isinstance(global_weights, dict):
                selected_weights = dict(global_weights)
            else:
                selected_weights = dict(self._BASE_WEIGHTS)

        # Apply confidence scaling at retrieval time
        total_trades = _safe_float(self._learned_weights.get("_meta", {}).get("total_trades", 0), 0)
        confidence = min(total_trades / 50.0, 1.0)

        if confidence < 1.0:
            # Blend with baseline weights for immature samples
            scaled_weights = {}
            for agent_name, learned_val in selected_weights.items():
                base_val = self._BASE_WEIGHTS.get(agent_name, 1.0)
                blended = base_val * (1.0 - confidence) + learned_val * confidence
                scaled_weights[agent_name] = round(blended, 4)
            self._regime_weights[regime] = scaled_weights
            return dict(scaled_weights)

        # Mature weights: use learned directly
        self._regime_weights[regime] = selected_weights
        return dict(selected_weights)

    def _save_learned_weights(self, *, history_source: str = "team:learned") -> None:
        """Persist learned agent weights to disk with retry + atomic write.

        Phase 55 (US-342): Uses resilient_save for retry with exponential
        backoff and async fallback queue on total failure.

        Bucket D1: emits a versioned history entry to ``weight_history.jsonl``
        after a successful save. Callers can override ``history_source`` to
        differentiate writers (e.g. ``"team:rl_update"`` or ``"team:meta_overrides"``).
        """
        from pathlib import Path

        path = Path(self._WEIGHTS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_ok = False
        try:
            from src.scanner.safe_io import resilient_save
            resilient_save(str(path), self._learned_weights, context="agent_weights")
            save_ok = True
        except ImportError:
            # Fallback if safe_io not available
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(path, self._learned_weights)
                save_ok = True
            except ImportError:
                import json
                try:
                    with open(path, "w") as f:
                        json.dump(self._learned_weights, f, indent=2, default=str)
                    save_ok = True
                except Exception as e:
                    logger.error(f"Agent weights save failed completely: {e}")

        if save_ok:
            try:
                from src.scanner.weights_history import record_weight_version
                record_weight_version(
                    self._learned_weights,
                    source=history_source,
                )
            except (ImportError, OSError) as _hist_err:
                logger.debug("weight_history record skipped: %s", _hist_err)

    def apply_proposed_weights(
        self,
        proposal_path: Optional[str] = None,
        *,
        max_delta_pct: float = 0.05,
        blend_rate: float = 0.3,
        max_age_seconds: int = 7200,
    ) -> Dict[str, Any]:
        """Apply approved weight deltas from `.claude/proposed_weights.json`.

        This is the Tier-3 safety-gated path for autonomous self-improvement.
        Proposals should arrive through the meta-cybernetic approval path; this
        method validates, clamps, and blends them with current weights before
        saving.

        Safety gates (all non-negotiable):
          - Proposal must be recent (default: ≤2h old) — rejects stale files
          - Each agent delta clamped to ±5% of current weight per application
          - Final value clamped to valid range [0.05, 10.0] (same as existing
            `_validate_weights` bounds)
          - Unknown agent names rejected (typo prevention)
          - Blend rate of 0.3 (softer than trade-outcome EMA) dampens swings
          - After apply: proposal file archived to `.claude/proposed_weights_archive/`

        Expected proposal schema:
            {
              "proposed_at": "2026-04-14T12:34:56Z",
              "trade_id": "T-1234",
              "regime": "NORMAL" | "HIGH" | "EXTREME" | "LOW" | "_global",
              "deltas": {
                "<agent_name>": {
                  "target_weight": 0.05..10.0,
                  "reason": "short explanation"
                }
              }
            }

        Returns a summary dict:
            {
              "applied": bool,
              "skipped_reason": Optional[str],
              "changes": {agent_name: {"old": float, "new": float, "target": float}}
            }
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        default_path = Path(".claude/proposed_weights.json")
        path = Path(proposal_path) if proposal_path else default_path

        result: Dict[str, Any] = {"applied": False, "skipped_reason": None, "changes": {}}

        if not path.exists():
            result["skipped_reason"] = "no_proposal"
            return result

        # Read + parse (CLAUDE.md JSON-safety gate)
        try:
            with open(path, "r") as f:
                proposal = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"apply_proposed_weights: parse failed: {e}")
            result["skipped_reason"] = f"parse_error: {e}"
            return result

        if not isinstance(proposal, dict):
            result["skipped_reason"] = "schema_error: not a dict"
            return result

        # Staleness check
        proposed_at_raw = proposal.get("proposed_at")
        if not proposed_at_raw:
            result["skipped_reason"] = "schema_error: missing proposed_at"
            return result
        try:
            proposed_at = datetime.fromisoformat(str(proposed_at_raw).replace("Z", "+00:00"))
            if proposed_at.tzinfo is None:
                proposed_at = proposed_at.replace(tzinfo=timezone.utc)
            age_seconds = (datetime.now(timezone.utc) - proposed_at).total_seconds()
            if age_seconds > max_age_seconds:
                logger.warning(
                    f"apply_proposed_weights: stale proposal ({age_seconds:.0f}s > {max_age_seconds}s), skipping"
                )
                result["skipped_reason"] = f"stale: {age_seconds:.0f}s old"
                return result
        except (ValueError, TypeError) as e:
            result["skipped_reason"] = f"schema_error: bad proposed_at ({e})"
            return result

        regime = str(proposal.get("regime", "_global"))
        valid_regimes = set(self._REGIME_NAMES + ["_global"])
        if regime not in valid_regimes:
            result["skipped_reason"] = f"schema_error: unknown regime {regime}"
            return result

        deltas = proposal.get("deltas") or {}
        if not isinstance(deltas, dict) or not deltas:
            result["skipped_reason"] = "schema_error: no deltas"
            return result

        # Ensure structure exists
        if not self._learned_weights:
            self._learned_weights = self._init_regime_weights()
        self._learned_weights, _ = self._ensure_complete_weight_schema(self._learned_weights)

        known_agents = set(self._BASE_WEIGHTS.keys())
        regime_weights = self._learned_weights.setdefault(regime, dict(self._BASE_WEIGHTS))
        changes: Dict[str, Dict[str, float]] = {}

        valid_min, valid_max = 0.1, 2.0

        for agent_name, delta_spec in deltas.items():
            if agent_name not in known_agents:
                logger.warning(f"apply_proposed_weights: unknown agent '{agent_name}', skipping delta")
                continue
            if not isinstance(delta_spec, dict):
                continue
            try:
                target = float(delta_spec.get("target_weight"))
            except (TypeError, ValueError):
                logger.warning(f"apply_proposed_weights: bad target_weight for {agent_name}")
                continue

            # Clamp target to valid range
            target = max(valid_min, min(valid_max, target))

            current = _safe_float(regime_weights.get(agent_name), self._BASE_WEIGHTS.get(agent_name, 1.0))

            # Soft EMA blend (dampens swings)
            blended = current + blend_rate * (target - current)

            # Per-cycle hard cap: ±max_delta_pct of current
            max_step = abs(current) * max_delta_pct
            if max_step < 1e-6:  # Avoid zero-step when current is ~0
                max_step = valid_min * max_delta_pct
            if blended > current + max_step:
                blended = current + max_step
            elif blended < current - max_step:
                blended = current - max_step

            # Final range clamp
            blended = max(valid_min, min(valid_max, blended))
            blended = round(blended, 4)

            if abs(blended - current) >= 1e-4:
                regime_weights[agent_name] = blended
                changes[agent_name] = {"old": current, "new": blended, "target": target}

        if not changes:
            result["skipped_reason"] = "no_effective_changes"
            # Still archive to prevent re-processing on next cycle
            self._archive_proposal(path, proposal, status="noop")
            return result

        # Persist + invalidate cache
        self._regime_weights = {}
        self._save_learned_weights()

        # Archive the proposal so we don't re-apply it on the next cycle
        self._archive_proposal(path, proposal, status="applied", changes=changes)

        logger.info(
            f"apply_proposed_weights: applied {len(changes)} changes in regime {regime}: "
            f"{({k: (v['old'], v['new']) for k, v in changes.items()})}"
        )
        result["applied"] = True
        result["changes"] = changes
        return result

    def _archive_proposal(
        self,
        path,
        proposal: Dict[str, Any],
        status: str,
        changes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Move the proposal file into the archive dir and remove the live file.

        Archived filename: `.claude/proposed_weights_archive/{ISO_ts}_{status}.json`
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        try:
            archive_dir = Path(".claude/proposed_weights_archive")
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive_path = archive_dir / f"{ts}_{status}.json"
            archived = {
                "proposal": proposal,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "changes": changes or {},
            }
            archive_path.write_text(json.dumps(archived, indent=2, default=str))
            try:
                Path(path).unlink(missing_ok=True)
            except TypeError:
                # Python < 3.8 missing_ok fallback
                if Path(path).exists():
                    Path(path).unlink()
        except Exception as e:
            logger.warning(f"_archive_proposal failed: {e}")

    def apply_weight_decay(self, decay_rate: float = 0.02) -> Dict[str, float]:
        """Decay learned weights toward base weights across all regimes.

        Should be called once per scan cycle (e.g., from ContinuousScanner).
        Each call moves learned weights ``decay_rate`` fraction closer to the
        base weight.  This ensures recent RL adjustments don't persist
        indefinitely when the sample size is small.

        Uses AgentHealthMonitor (Phase 9) for selective decay rates when available:
        stale agents (win rate < 35% over 20+ trades) get 2x decay rate.

        Args:
            decay_rate: Fraction to move toward base weight per call (0.0–1.0).

        Returns:
            Dict of agent name -> new weight after decay (global weights).
        """
        if not self._learned_weights:
            return {}

        decay_rate = max(0.0, min(1.0, decay_rate))
        changed = False

        # Load selective decay rates from AgentHealthMonitor if available
        selective_rates: Dict[str, float] = {}
        try:
            from src.scanner.automation.agent_health import AgentHealthMonitor
            monitor = AgentHealthMonitor()
            selective_rates = monitor.get_selective_decay_rates()
        except Exception:
            pass

        # Ensure regime-aware structure
        if "_global" not in self._learned_weights:
            self._learned_weights = self._init_regime_weights()
        self._learned_weights, schema_repaired = self._ensure_complete_weight_schema(self._learned_weights)
        if schema_repaired:
            changed = True

        # Decay all regime weights
        for regime in self._REGIME_NAMES + ["_global"]:
            if regime not in self._learned_weights or not isinstance(self._learned_weights[regime], dict):
                continue

            regime_weights = self._learned_weights[regime]
            for name in list(regime_weights.keys()):
                base = self._BASE_WEIGHTS.get(name, 1.0)
                current = _safe_float(regime_weights.get(name), 1.0)
                if abs(current - base) < 1e-4:
                    # Keep baseline keys persisted for health-check completeness.
                    continue
                # Use selective decay rate if available, otherwise default
                agent_decay = selective_rates.get(name, decay_rate)
                new_weight = current + agent_decay * (base - current)
                new_weight = round(new_weight, 4)
                regime_weights[name] = new_weight
                changed = True

        if changed:
            self._save_learned_weights()
            self._regime_weights = {}  # Clear cache

        return dict(self._learned_weights.get("_global", self._BASE_WEIGHTS))

    def update_weights_from_outcome(
        self,
        agent_verdicts: List[Dict[str, Any]],
        trade_won: bool,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """Update agent weights based on trade outcome (RL feedback).

        Updates both regime-specific weights (if sufficient data) and global running average.

        Args:
            agent_verdicts: List of agent verdict dicts from the trade's PairAnalysis
            trade_won: True if trade hit TP, False if hit SL
            regime: Volatility regime the trade was entered in (e.g., "NORMAL", "HIGH")

        Returns:
            Dict of agent name -> new weight (global)
        """
        import logging

        logger = logging.getLogger(__name__)

        boost = _safe_float(getattr(self.config, "weight_boost_on_win", 0.10), 0.10)
        penalty = _safe_float(getattr(self.config, "weight_penalty_on_loss", 0.15), 0.15)
        min_w = _safe_float(getattr(self.config, "min_agent_weight", 0.1), 0.1)
        max_w = _safe_float(getattr(self.config, "max_agent_weight", 2.0), 2.0)

        # Normalize regime
        regime = _normalize_regime_name(regime)

        # Ensure regime-aware structure
        if "_global" not in self._learned_weights:
            self._learned_weights = self._init_regime_weights()

        # Update both regime-specific and global weights
        for verdict in agent_verdicts:
            name = str(verdict.get("name", ""))
            if not name:
                continue

            # Calculate weight delta
            if verdict.get("passed"):
                delta = boost if trade_won else -penalty
            else:
                delta = -boost * 0.5 if trade_won else penalty * 0.5

            # Update regime-specific weight
            regime_weights = self._learned_weights.get(regime, {})
            if not regime_weights or not isinstance(regime_weights, dict):
                regime_weights = dict(self._BASE_WEIGHTS)
                self._learned_weights[regime] = regime_weights

            current_regime = _safe_float(regime_weights.get(name), self._BASE_WEIGHTS.get(name, 1.0))
            _raw_regime = max(min_w, min(max_w, current_regime + delta))

            # Phase 26 (US-159): EMA damping to reduce weight oscillation
            # Check per-agent variance and increase EMA blending for oscillating agents
            _ema_alpha = 0.7  # Default: 70% new, 30% previous
            _var_key = f"weight_variance_{name}"
            _hist_key = f"weight_history_{name}"
            if "_meta" in self._learned_weights:
                _m = self._learned_weights["_meta"]
                _hist = _m.get(_hist_key, [])
                _hist.append(round(_raw_regime, 4))
                if len(_hist) > 20:
                    _hist = _hist[-20:]
                _m[_hist_key] = _hist
                if len(_hist) >= 5:
                    _mean_h = sum(_hist) / len(_hist)
                    _var = sum((_h - _mean_h) ** 2 for _h in _hist) / len(_hist)
                    _m[_var_key] = round(_var, 6)
                    if _var > 0.05:
                        _ema_alpha = 0.6  # More damping for oscillating agents
            new_regime_weight = round(_ema_alpha * _raw_regime + (1 - _ema_alpha) * current_regime, 4)
            regime_weights[name] = new_regime_weight

            # Update global weight (cross-regime running average)
            global_weights = self._learned_weights.get("_global", {})
            if not global_weights or not isinstance(global_weights, dict):
                global_weights = dict(self._BASE_WEIGHTS)
                self._learned_weights["_global"] = global_weights

            current_global = _safe_float(global_weights.get(name), self._BASE_WEIGHTS.get(name, 1.0))
            _raw_global = max(min_w, min(max_w, current_global + delta * 0.75))
            new_global_weight = round(_ema_alpha * _raw_global + (1 - _ema_alpha) * current_global, 4)
            global_weights[name] = new_global_weight

        # Track trade count per regime and total
        if "_meta" not in self._learned_weights:
            self._learned_weights["_meta"] = {"min_trades_per_regime": 10}
        meta = self._learned_weights["_meta"]
        if not isinstance(meta, dict):
            meta = {"min_trades_per_regime": 10}
            self._learned_weights["_meta"] = meta

        trades_key = f"trades_{regime}"
        meta[trades_key] = _safe_float(meta.get(trades_key, 0), 0) + 1

        # Increment total trade count
        total_trades = _safe_float(meta.get("total_trades", 0), 0) + 1
        meta["total_trades"] = int(total_trades)

        # Phase 25 (US-156): Track last trade per agent for staleness detection
        if "last_trade_per_agent" not in meta or not isinstance(meta.get("last_trade_per_agent"), dict):
            meta["last_trade_per_agent"] = {}
        _ltpa = meta["last_trade_per_agent"]
        for verdict in agent_verdicts:
            _vname = str(verdict.get("name", ""))
            if _vname and verdict.get("passed", False):
                _ltpa[_vname] = int(total_trades)

        # Phase 25 (US-156): Apply weight decay for stale agents every 50 trades
        _DECAY_INTERVAL = 50
        _STALENESS_THRESHOLD = 20  # No trades in last N outcomes = stale
        _DECAY_FACTOR = 0.95
        if int(total_trades) % _DECAY_INTERVAL == 0 and int(total_trades) > 0:
            _decayed = []
            _global_w = self._learned_weights.get("_global", {})
            for _agent_name, _w in list(_global_w.items()):
                _last_seen = _ltpa.get(_agent_name, 0)
                if (int(total_trades) - _last_seen) > _STALENESS_THRESHOLD:
                    # 2026-07-18 audit fix: decay toward the agent's own BASELINE,
                    # not multiplicatively toward a hard 0.5 floor — the old form
                    # permanently under-weighted idle hard-veto agents
                    # (devil_advocate 1.30, trend 1.15, risk_sentinel 1.25),
                    # inconsistent with the other decay mechanisms which all
                    # target _BASE_WEIGHTS. Staleness = forget what was learned,
                    # i.e. drift back to the design prior.
                    _base = float(self._BASE_WEIGHTS.get(_agent_name, 1.0))
                    _new_w = round(_base + (float(_w) - _base) * _DECAY_FACTOR, 4)
                    if abs(_new_w - float(_w)) > 0.001:
                        _global_w[_agent_name] = _new_w
                        _decayed.append(f"{_agent_name}: {_w:.3f}→{_new_w:.3f}")
            if _decayed:
                logger.info(
                    "US-156: Stale agent weight decay at trade %d: %s",
                    int(total_trades), ", ".join(_decayed),
                )

        # Update timestamp
        meta["last_updated"] = datetime.now(timezone.utc).isoformat()

        self._save_learned_weights()
        self._regime_weights = {}  # Clear cache

        # Save snapshot every 50 trades
        if int(total_trades) % 50 == 0:
            self._save_weight_snapshot(int(total_trades))

        # Phase 45 (US-283): Bayesian weight update via Thompson Sampling
        if self._bayesian_weights is not None:
            try:
                # Build agent_scores dict from verdicts
                _agent_scores = {}
                for verdict in agent_verdicts:
                    _name = str(verdict.get("name", ""))
                    if _name:
                        # Score: verdict score (already in [0, 1])
                        _agent_scores[_name] = _safe_float(verdict.get("score", 0.0), 0.0)

                # Outcome: "win" or "loss"
                _outcome = "win" if trade_won else "loss"

                self._bayesian_weights.update(
                    agent_scores=_agent_scores,
                    outcome=_outcome,
                    regime=regime,
                )

                # Persist atomically
                self._bayesian_weights.save_state("trained_data/bayesian_weights.json")

                logger.info(
                    f"Phase 45: Bayesian weights updated for regime={regime}, "
                    f"outcome={_outcome}, agents={len(_agent_scores)}"
                )
            except Exception as _bw_upd_err:
                logger.warning(f"Phase 45: Bayesian weight update failed: {_bw_upd_err}")

        # Phase 55 (US-343): Record trade outcome into DisagreementTracker
        if self._disagreement_tracker is not None:
            try:
                _dt_scores = {
                    str(v.get("name", "")): float(v.get("score", 0.0))
                    for v in agent_verdicts
                    if v.get("name")
                }
                if _dt_scores:
                    self._disagreement_tracker.record_trade(
                        agent_scores=_dt_scores,
                        won=trade_won,
                        pnl=0.0,  # PnL not available here; outcome (won/loss) drives learning
                    )
            except Exception as _dt_rec_err:
                logger.debug(f"Phase 55: Disagreement tracker recording failed: {_dt_rec_err}")

        return dict(self._learned_weights.get("_global", self._BASE_WEIGHTS))

    def _save_weight_snapshot(self, trade_count: int) -> None:
        """Save a snapshot of current weights every 50 trades.

        Snapshots enable rollback to any checkpoint. Keeps last 10 snapshots.

        Args:
            trade_count: Total trade count at snapshot time
        """
        snapshot_dir = Path("trained_data/models/weight_snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = snapshot_dir / f"{trade_count}.json"

        try:
            # Save snapshot with metadata
            snapshot_data = {
                "trade_count": trade_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "weights": dict(self._learned_weights),
            }

            with open(snapshot_path, "w") as f:
                json.dump(snapshot_data, f, indent=2, default=str)

            logger.info(f"Saved weight snapshot at {trade_count} trades: {snapshot_path.name}")

            # Clean up old snapshots (keep last 10)
            snapshots = sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem))
            if len(snapshots) > 10:
                for old_snapshot in snapshots[:-10]:
                    try:
                        old_snapshot.unlink()
                        logger.debug(f"Removed old snapshot: {old_snapshot.name}")
                    except Exception as e:
                        logger.debug(f"Failed to remove snapshot {old_snapshot.name}: {e}")

        except Exception as e:
            logger.error(f"Failed to save weight snapshot: {e}")

    def load_weight_snapshot(self, trade_count: int) -> bool:
        """Load weights from a specific snapshot.

        Enables rollback to a previous checkpoint.

        Args:
            trade_count: Trade count of the snapshot to load

        Returns:
            True if successful, False otherwise
        """
        snapshot_path = Path("trained_data/models/weight_snapshots") / f"{trade_count}.json"

        if not snapshot_path.exists():
            logger.warning(f"Snapshot not found: {snapshot_path}")
            return False

        try:
            with open(snapshot_path) as f:
                snapshot_data = json.load(f)

            self._learned_weights = snapshot_data.get("weights", self._init_regime_weights())
            self._regime_weights = {}  # Clear cache
            logger.info(f"Loaded weight snapshot from {trade_count} trades")
            return True

        except Exception as e:
            logger.error(f"Failed to load weight snapshot: {e}")
            return False

    def list_weight_snapshots(self) -> List[Tuple[int, str]]:
        """List all available weight snapshots.

        Returns:
            List of (trade_count, timestamp) tuples, sorted by trade count
        """
        snapshot_dir = Path("trained_data/models/weight_snapshots")
        if not snapshot_dir.exists():
            return []

        snapshots = []
        for snapshot_path in sorted(snapshot_dir.glob("*.json"), key=lambda p: int(p.stem)):
            try:
                with open(snapshot_path) as f:
                    data = json.load(f)
                    trade_count = data.get("trade_count", int(snapshot_path.stem))
                    timestamp = data.get("timestamp", "unknown")
                    snapshots.append((trade_count, timestamp))
            except Exception as e:
                logger.debug(f"Failed to read snapshot {snapshot_path.name}: {e}")

        return snapshots

    def apply_meta_overrides(
        self,
        meta_overrides: Dict[str, Dict[str, float]],
    ) -> None:
        """Apply MetaLearner hyperparameter overrides before scoring.

        This is called before each scan cycle. Overrides are per-agent per-regime.
        Only applies fields that exist in the override dict.

        Currently affects:
        - voting_veto_power: multiplied onto the agent's base weight
        - confidence_threshold: stored as floor for score clipping
        """
        self._meta_overrides = meta_overrides  # store for use during vote()
        logger.debug("AgentTeam: applied meta overrides for %d agents", len(meta_overrides))

    def evaluate(
        self,
        analysis: PairAnalysis,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        gate_details: Optional[Dict[str, Any]] = None,
    ) -> PairAnalysis:
        """Apply specialist agents to a scan result with regime-aware weights."""
        if analysis.error is not None or analysis.direction not in {"LONG", "SHORT"}:
            return analysis

        # Tag this entire evaluation (and every nested _evaluate_* call) with
        # filterable attributes for the Weave UI.
        trace_attrs = {
            "pair": str(getattr(analysis, "pair", "UNKNOWN")),
            "direction": str(getattr(analysis, "direction", "HOLD")),
            "regime": str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper(),
            "profile": str(getattr(self.config, "profile", "unknown")),
            "confidence": float(getattr(analysis, "confidence", 0.0) or 0.0),
        }

        with weave_attrs(trace_attrs):
            return self._evaluate_body(analysis, df_raw, df_feat, gate_details)

    def _evaluate_body(
        self,
        analysis: PairAnalysis,
        df_raw: pd.DataFrame,
        df_feat: pd.DataFrame,
        gate_details: Optional[Dict[str, Any]] = None,
    ) -> PairAnalysis:
        """Original evaluate() body, wrapped by attribute-tagged `evaluate()`."""
        ctx = AgentDecisionContext(
            analysis=analysis,
            df_raw=df_raw,
            df_feat=df_feat,
            gate_details=gate_details or {},
            config=self.config,
        )

        # Episodic Market Memory: Query pattern suppression signal
        if self._episodic_memory is not None and getattr(self.config, "enable_episodic_memory", True):
            # Hoisted out of the try block so the WARNING handler can
            # reference them safely without locals()-style lookups (which
            # semgrep flags as CWE-96 dynamic-code-injection risk).
            _ep_pair = getattr(analysis, "pair", "UNKNOWN")
            _ep_direction = getattr(analysis, "direction", "HOLD")
            _ep_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
            try:
                session = "london"  # TODO: derive session from UTC hour if available
                news_risk = _safe_float(getattr(analysis, "news_risk_score", 0.0), 0.0)
                uncertainty = _safe_float(getattr(analysis, "uncertainty_score", 0.0), 0.0)

                suppression = self._episodic_memory.get_suppression_signal(
                    pair=_ep_pair,
                    direction=_ep_direction,
                    regime=_ep_regime,
                    session=session,
                    news_risk_score=news_risk,
                    uncertainty_score=uncertainty,
                )
                ctx.gate_details["episodic_suppression"] = suppression

                if suppression:
                    logger.info(
                        "Episodic suppression ACTIVE for %s %s (pattern loss rate >= 70%%)",
                        _ep_pair, _ep_direction,
                    )
            except Exception as e:
                # Mythos audit 2026-04-28 — promoted-rule veto neighborhood:
                # episodic memory's get_suppression_signal returning True is
                # a hard veto on high-loss-rate patterns. A silent debug log
                # here was the same anti-pattern that hid the 04-15 MR-veto
                # signal under WVS arithmetic. Bump to WARNING with stack so
                # operator sees memory degradation, AND mark gate_details
                # with 'episodic_memory_degraded' so downstream consumers
                # can choose conservative behavior. We still default
                # suppression=False to avoid a memory hiccup blocking ALL
                # trades — degraded flag lets specific gates fail-closed.
                logger.warning(
                    "Episodic memory query failed pair=%s direction=%s "
                    "regime=%s — defaulting suppression=False with "
                    "degraded=True flag. Reason: %r",
                    _ep_pair, _ep_direction, _ep_regime,
                    e, exc_info=True,
                )
                ctx.gate_details["episodic_suppression"] = False
                ctx.gate_details["episodic_memory_degraded"] = True

        verdicts: List[AgentVerdict] = []

        # US-069: Determine regime-disabled agents for current volatility
        _regime_name = _normalize_regime_name(getattr(analysis, "volatility_regime", "UNKNOWN"))
        _regime_disabled = set(
            getattr(self.config, "regime_disabled_agents", {}).get(_regime_name, [])
        )
        self._active_weight_regime = _regime_name

        if getattr(self.config, "enable_trend_agent", True):
            verdicts.append(self._evaluate_trend(ctx))
        if getattr(self.config, "enable_mean_reversion_agent", True):
            verdicts.append(self._evaluate_mean_reversion(ctx))
        if getattr(self.config, "enable_volatility_agent", True):
            verdicts.append(self._evaluate_volatility(ctx))
        if getattr(self.config, "enable_risk_sentinel_agent", True):
            verdicts.append(self._evaluate_risk(ctx))
        if getattr(self.config, "enable_uncertainty_agent", True):
            verdicts.append(self._evaluate_uncertainty(ctx))
        if getattr(self.config, "enable_execution_quality_agent", True):
            verdicts.append(self._evaluate_execution_quality(ctx))
        if getattr(self.config, "enable_news_risk_agent", False):
            news_verdict = self._evaluate_news_risk(ctx)
            if news_verdict is not None:
                verdicts.append(news_verdict)
        if getattr(self.config, "enable_multi_timeframe_agent", False):
            verdicts.append(self._evaluate_multi_timeframe(ctx))
        if getattr(self.config, "enable_pair_performance_agent", False):
            perf_verdict = self._evaluate_pair_performance(ctx)
            if perf_verdict is not None:
                verdicts.append(perf_verdict)
        if self.config.enable_momentum_agent:
            verdicts.append(self._evaluate_momentum(ctx))
        if self.config.enable_session_timing_agent:
            verdicts.append(self._evaluate_session_timing(ctx))
        if self.config.enable_support_resistance_agent:
            sr_verdict = self._evaluate_support_resistance(ctx)
            if sr_verdict is not None:
                verdicts.append(sr_verdict)

        # Agent #16: LLM Macro Agent (macro reasoning via LLM)
        _llm_enabled = getattr(self.config, "enable_llm_macro_agent", False)
        _llm_shadow = getattr(self.config, "enable_llm_macro_shadow", True)
        if (_llm_enabled or _llm_shadow) and self._llm_macro_agent is not None:
            try:
                llm_verdict = self._llm_macro_agent.evaluate(ctx)
                if llm_verdict is not None:
                    verdicts.append(llm_verdict)
                    if _llm_shadow and not _llm_enabled:
                        # In shadow-only mode, tag the verdict so downstream knows
                        llm_verdict.metadata["shadow_only"] = True
            except Exception as exc:
                logger.warning("LLM Macro Agent evaluation failed: %s", exc)

        # Agent #15: Order Flow (OANDA order/position book contrarian signal)
        # Default True — backward compatible with configs/YAML dicts that omit the key.
        if getattr(self.config, "enable_order_flow_agent", True):
            of_verdict = self._evaluate_order_flow(ctx)
            if of_verdict is not None:
                verdicts.append(of_verdict)

        # Agent #13: Trader Readiness (Aura human-side intelligence)
        # 2026-05-12: default False — Aura writer not yet implemented; agent reads a file no one writes.
        # Code path retained so re-enable is a one-flag flip once the writer ships.
        if getattr(self.config, "enable_trader_readiness_agent", False):
            readiness_verdict = self._evaluate_trader_readiness(ctx)
            if readiness_verdict is not None:
                verdicts.append(readiness_verdict)

        # Agent #14: Devil's Advocate (Adversarial bear-case evaluator, runs LAST)
        # Canonical toggle: enable_devil_advocate_agent (new _agent-suffixed name).
        # Legacy toggle: enable_devil_advocate (kept for backward compatibility).
        # Both default True; disabling requires explicit False on the canonical flag.
        _da_enabled = (
            getattr(self.config, "enable_devil_advocate_agent", True)
            and getattr(self.config, "enable_devil_advocate", True)
        )
        if _da_enabled:
            try:
                from src.scanner.agents.agent_da import DevilsAdvocateAgent
                da_agent = DevilsAdvocateAgent()
                da_verdict = da_agent.evaluate(ctx)
                verdicts.append(da_verdict)
                logger.debug(
                    "Devil's Advocate evaluated for %s: bear_score=%.3f, passed=%s, block_trade=%s",
                    getattr(analysis, "pair", "?"),
                    da_verdict.metadata.get("bear_score", 0.0),
                    da_verdict.passed,
                    da_verdict.block_trade,
                )
            except Exception as da_err:
                logger.warning(f"Devil's Advocate agent failed: {da_err}")

        # US-069: Filter out regime-disabled agents (don't count toward vote totals).
        # 2026-07-18 audit fix: a verdict carrying block_trade=True is NEVER
        # dropped by this filter — trend passed=False and the devil's-advocate
        # bear case are HARD vetoes per trading.md, and this filter runs after
        # they were computed; dropping them silently discarded safety vetoes.
        # A disabled agent still doesn't vote; only its safety veto survives.
        if _regime_disabled:
            _skipped = [v for v in verdicts if v.name in _regime_disabled and not v.block_trade]
            verdicts = [v for v in verdicts if v.name not in _regime_disabled or v.block_trade]
            if _skipped:
                logger.info(
                    "Regime %s: skipped %d agents (%s) for %s",
                    _regime_name,
                    len(_skipped),
                    ", ".join(v.name for v in _skipped),
                    getattr(analysis, "pair", "UNKNOWN"),
                )

        if not verdicts:
            return analysis
        # US-017: MetaStrategyAgent — dynamic strategy selection
        if self._meta_strategy_agent is not None:
            try:
                _meta_selection = self._meta_strategy_agent.select(_regime_name)
                _selected_strategies = set(_meta_selection.get("selected_strategies", []))
                if _selected_strategies:
                    # 2026-07-18 audit fix (same rationale as the regime filter):
                    # MetaStrategy deselection must not discard a block_trade=True
                    # safety veto computed earlier this cycle.
                    _meta_skipped = [
                        v for v in verdicts
                        if v.name not in _selected_strategies and not v.block_trade
                    ]
                    verdicts = [
                        v for v in verdicts
                        if v.name in _selected_strategies or v.block_trade
                    ]
                    if _meta_skipped:
                        logger.info(
                            "MetaStrategy %s: deactivated %d agents (%s) for %s (reason=%s)",
                            _regime_name,
                            len(_meta_skipped),
                            ", ".join(v.name for v in _meta_skipped),
                            getattr(analysis, "pair", "UNKNOWN"),
                            _meta_selection.get("reason", "unknown"),
                        )
                    ctx.gate_details["meta_strategy_selection"] = _meta_selection
                    if _meta_selection.get("uncertain"):
                        ctx.gate_details["meta_strategy_uncertain"] = True
            except Exception as _meta_err:
                logger.debug("MetaStrategyAgent selection failed: %s", _meta_err)

        if not verdicts:
            return analysis

        # Apply regime-aware weight adjustments BEFORE voting
        regime_name = _regime_name
        verdicts = self._apply_regime_multipliers(verdicts, regime_name)

        # Agent memory — nudge each verdict's confidence_delta based on similar
        # past decisions for this agent + pair + regime. Read-only at scan time;
        # writes happen on TRADE_OPENED (below). Safe no-op unless
        # BUDDY_AGENT_MEMORY_ENABLED=1.
        if _agent_memory_enabled():
            try:
                _mem = _get_agent_memory()
                _pair_name = str(getattr(analysis, "pair", "UNKNOWN"))
                for v in verdicts:
                    mem_ctx = {
                        "score": v.score,
                        "confidence": getattr(analysis, "confidence", 0.0),
                        "adx": _last_value(df_feat, "adx_14"),
                        "rsi": _last_value(df_feat, "rsi_14"),
                        "atr_pips": _last_value(df_feat, "atr_pips"),
                        "spread_pips": getattr(analysis, "spread_pips", 0.0),
                        "uncertainty_score": getattr(analysis, "uncertainty_score", 0.0),
                        "model_disagreement": getattr(analysis, "model_disagreement", 0.0),
                        "weighted_vote_score": getattr(analysis, "weighted_vote_score", 0.0),
                        "regime": regime_name,
                        "direction": getattr(analysis, "direction", "HOLD"),
                    }
                    hits = _mem.query_and_score(
                        agent_name=v.name,
                        pair=_pair_name,
                        context=mem_ctx,
                        n_results=20,
                        scope="self",
                    )
                    if hits.confidence_delta != 0.0 and hits.n_with_outcome >= 5:
                        v.confidence_delta = float(v.confidence_delta) + hits.confidence_delta
                        v.metadata = dict(v.metadata or {})
                        v.metadata["memory_n_outcomes"] = hits.n_with_outcome
                        v.metadata["memory_avg_pnl"] = hits.avg_pnl_pips
                        v.metadata["memory_hit_rate"] = hits.hit_rate
                        v.metadata["memory_delta"] = hits.confidence_delta
                        logger.info(
                            "memory: agent=%s pair=%s nudged confidence_delta by %+.3f "
                            "(avg_pnl=%.1f, hit_rate=%.2f, n=%d)",
                            v.name, _pair_name, hits.confidence_delta,
                            hits.avg_pnl_pips or 0.0, hits.hit_rate, hits.n_with_outcome,
                        )
                # Cache verdicts keyed by pair+direction so the TRADE_OPENED
                # subscriber can persist them to memory once a trade is taken.
                _cache_verdicts_for_trade_open_mem(
                    pair=_pair_name,
                    direction=str(getattr(analysis, "direction", "HOLD")),
                    regime=regime_name,
                    context={
                        "adx": _last_value(df_feat, "adx_14"),
                        "rsi": _last_value(df_feat, "rsi_14"),
                        "atr_pips": _last_value(df_feat, "atr_pips"),
                        "spread_pips": getattr(analysis, "spread_pips", 0.0),
                        "uncertainty_score": getattr(analysis, "uncertainty_score", 0.0),
                        "model_disagreement": getattr(analysis, "model_disagreement", 0.0),
                        "weighted_vote_score": getattr(analysis, "weighted_vote_score", 0.0),
                        "confidence": getattr(analysis, "confidence", 0.0),
                    },
                    verdicts=verdicts,
                )
            except Exception as _mem_err:  # noqa: BLE001 — memory must never block trading
                logger.debug("agent memory nudge failed: %s", _mem_err)

        # US-076: Graph-attention heterogeneous agent consensus
        # 2026-05-12: default flipped False -> True to match dataclass default (was masking the flag for Mock configs).
        if getattr(self.config, "enable_graph_attention", True):
            try:
                _ga = GraphAttentionConsensus(temperature=1.0)
                verdicts = _ga.reweight_verdicts(
                    verdicts=verdicts,
                    regime=regime_name,
                )
            except Exception as _ga_err:
                logger.debug(f"Graph attention skipped: {_ga_err}")

        # Phase 20 (US-122) + Phase 25 (US-151): Apply agent lifecycle weight modifiers
        if self._agent_lifecycle is not None:
            _lifecycle_adjustments = []
            for v in verdicts:
                try:
                    modifier = self._agent_lifecycle.get_weight_modifier(v.name)
                    if modifier < 1.0:
                        _lifecycle_adjustments.append(
                            f"{v.name}={modifier:.2f}x"
                        )
                    v.weight = max(v.weight * modifier, 0.0)
                except Exception:
                    pass  # Non-blocking — use original weight
            if _lifecycle_adjustments:
                logger.info(
                    "US-151: Agent lifecycle modifiers applied for %s: %s",
                    getattr(analysis, "pair", "?"),
                    ", ".join(_lifecycle_adjustments),
                )

        # Phase 25 (US-155): Apply per-pair accuracy matrix weight adjustments
        if self._accuracy_matrix is not None:
            _pair_name = str(getattr(analysis, "pair", "") or "")
            if _pair_name:
                try:
                    _am_weights = self._accuracy_matrix.get_regime_weights(
                        regime=regime_name, pair=_pair_name,
                    )
                    _am_adjustments = []
                    for v in verdicts:
                        _am_mod = _am_weights.get(v.name, 1.0)
                        if abs(_am_mod - 1.0) > 0.001:
                            _am_adjustments.append(f"{v.name}={_am_mod:.2f}x")
                            v.weight = max(v.weight * _am_mod, 0.0)
                    if _am_adjustments:
                        logger.info(
                            "US-155: Accuracy matrix adjustments for %s/%s: %s",
                            _pair_name, regime_name,
                            ", ".join(_am_adjustments),
                        )
                except Exception as _am_err:
                    logger.debug(f"US-155: Accuracy matrix weights skipped: {_am_err}")

        # Phase 47 (US-297): Apply expectancy-based weight modifiers
        if self._expectancy_tracker is not None:
            _expectancy_adjustments = []
            try:
                for v in verdicts:
                    _exp_mod = self._expectancy_tracker.get_weight_modifier(v.name, regime_name)
                    if _exp_mod < 1.0:
                        _expectancy_adjustments.append(
                            f"{v.name}={_exp_mod:.2f}x"
                        )
                        v.weight = max(v.weight * _exp_mod, 0.0)
                if _expectancy_adjustments:
                    logger.info(
                        "Phase 47 (US-297): Expectancy modifiers for %s/%s: %s",
                        getattr(analysis, "pair", "?"), regime_name,
                        ", ".join(_expectancy_adjustments),
                    )
            except Exception as _exp_err:
                logger.debug(f"Phase 47: Expectancy weight modifier failed: {_exp_err}")

        # Phase 45 (US-283): Thompson Sampling weight overlay from Bayesian agent weights
        if self._bayesian_weights is not None:
            try:
                _regime_for_sampling = "NORMAL"  # Default
                # Try to get regime from analysis
                if hasattr(analysis, 'regime_name'):
                    _regime_for_sampling = analysis.regime_name
                elif hasattr(analysis, 'volatility_regime'):
                    _vr = analysis.volatility_regime
                    if isinstance(_vr, int) and 0 <= _vr < len(self._REGIME_NAMES):
                        _regime_for_sampling = self._REGIME_NAMES[_vr]
                    elif isinstance(_vr, str):
                        _regime_for_sampling = _vr.upper()

                _bayesian_weight_sample = self._bayesian_weights.sample_weights(_regime_for_sampling)
                logger.debug(
                    f"Phase 45: Thompson sampled weights for regime={_regime_for_sampling}, "
                    f"epsilon={_bayesian_weight_sample.epsilon:.3f}"
                )

                # Blend Bayesian weights with current verdict weights (70% Bayesian, 30% flat)
                if _bayesian_weight_sample is not None and hasattr(_bayesian_weight_sample, 'weights'):
                    _bw = _bayesian_weight_sample.weights
                    _blend_ratio = 0.7  # 70% Bayesian, 30% flat
                    for v in verdicts:
                        if v.name in _bw:
                            v.weight = (
                                _blend_ratio * _bw[v.name] * max(v.weight, 0.01) +
                                (1 - _blend_ratio) * max(v.weight, 0.0)
                            )
                            logger.debug(f"Phase 45: {v.name} weight blended to {v.weight:.4f}")
            except Exception as _bw_err:
                logger.debug(f"Phase 45: Bayesian weight sampling failed: {_bw_err}")

        # Tier 6: Apply MetaLearner hyperparameter overrides (voting_veto_power, confidence_threshold)
        if self._meta_overrides:
            _ml_adjustments = []
            for v in verdicts:
                _overrides = self._meta_overrides.get(v.name)
                if _overrides:
                    # voting_veto_power: multiplies weight (1.0 = no change, >1 = boost, <1 = dampen)
                    _veto = float(_overrides.get("voting_veto_power", 1.0))
                    if abs(_veto - 1.0) > 0.01:
                        v.weight = max(v.weight * _veto, 0.0)
                        _ml_adjustments.append(f"{v.name}:veto={_veto:.2f}x")
                    # confidence_threshold: if agent score below threshold, reduce score impact
                    _ct = float(_overrides.get("confidence_threshold", 0.0))
                    if _ct > 0.0 and v.score < _ct:
                        v.score = _clip01(v.score * 0.5)  # Halve sub-threshold scores
                        _ml_adjustments.append(f"{v.name}:ct={_ct:.2f}")
            if _ml_adjustments:
                logger.info(
                    "Tier 6: Meta-learner overrides for %s: %s",
                    getattr(analysis, "pair", "?"),
                    ", ".join(_ml_adjustments),
                )

        total_weight = sum(max(v.weight, 0.0) for v in verdicts) or 1.0
        weighted_vote_score = sum(_clip01(v.score) * max(v.weight, 0.0) for v in verdicts) / total_weight

        regime_thresholds = getattr(self.config, "regime_vote_thresholds", {}) or {}
        weighted_vote_threshold = max(
            _safe_float(getattr(self.config, "weighted_vote_threshold", 0.55), 0.55),
            _safe_float(regime_thresholds.get(regime_name), 0.0),
        )

        confidence_adjustment = sum(float(v.confidence_delta) * max(v.weight, 0.0) for v in verdicts) / total_weight
        block_trade = any(v.block_trade for v in verdicts)

        analysis.confidence = _clip01(float(analysis.confidence) + confidence_adjustment)
        analysis.weighted_vote_score = _clip01(weighted_vote_score)
        analysis.weighted_vote_threshold = _clip01(weighted_vote_threshold)

        # Phase 44 (US-280): Apply confidence calibration if available
        # Kill switch (2026-05-09): config.enable_phase44_calibration=False bypasses entirely.
        if self._confidence_calibrator is not None and bool(getattr(self.config, "enable_phase44_calibration", True)):
            try:
                _cal_verdicts = [
                    {"name": v.name, "score": _clip01(v.score), "weight": max(v.weight, 0.0), "passed": v.passed}
                    for v in verdicts
                ]
                _cal_regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
                _regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
                if _cal_regime.isdigit():
                    _cal_regime = _regime_names.get(int(_cal_regime), "NORMAL")

                _calibrated = self._confidence_calibrator.calibrate(_cal_verdicts, _cal_regime)

                # Override weighted_vote_score with calibrated score
                analysis.weighted_vote_score = _clip01(_calibrated.final_confidence)

                # Store calibration metadata in analysis
                if not hasattr(analysis, 'calibration_details'):
                    analysis.calibration_details = {}
                analysis.calibration_details = {
                    "raw_score": round(_calibrated.raw_weighted_score, 4),
                    "calibrated_score": round(_calibrated.final_confidence, 4),
                    "ensemble_disagreement": round(_calibrated.ensemble_disagreement, 4),
                    "disagreement_level": _calibrated.disagreement_level,
                    "agent_agreement_quality": round(_calibrated.agent_agreement, 4),
                    "platt_adjusted": round(_calibrated.platt_calibrated, 4),
                    "meta_confidence": round(_calibrated.meta_confidence, 4),
                }

                logger.debug(
                    "Phase 44: %s calibrated confidence: raw=%.3f → calibrated=%.3f (disagreement=%s, meta=%.2f)",
                    getattr(analysis, "pair", "?"),
                    _calibrated.raw_weighted_score,
                    _calibrated.final_confidence,
                    _calibrated.disagreement_level,
                    _calibrated.meta_confidence,
                )
            except Exception as _cal_err:
                logger.debug(f"Phase 44: Confidence calibration skipped: {_cal_err}")

        # Phase 52 (US-323): Isotonic calibration overlay — overrides Platt if valid model exists
        if self._isotonic_calibrator is not None:
            try:
                _raw_for_isotonic = float(analysis.weighted_vote_score)
                _isotonic_result = self._isotonic_calibrator.calibrate(_raw_for_isotonic)
                if _isotonic_result is not None and _isotonic_result != _raw_for_isotonic:
                    _prev_score = analysis.weighted_vote_score
                    analysis.weighted_vote_score = _clip01(_isotonic_result)
                    # Store isotonic metadata
                    if not hasattr(analysis, 'calibration_details'):
                        analysis.calibration_details = {}
                    analysis.calibration_details["isotonic_raw"] = round(_raw_for_isotonic, 4)
                    analysis.calibration_details["isotonic_calibrated"] = round(_isotonic_result, 4)
                    analysis.calibration_details["isotonic_applied"] = True
                    logger.debug(
                        "Phase 52 (US-323): %s isotonic calibration: %.3f → %.3f",
                        getattr(analysis, "pair", "?"), _prev_score, analysis.weighted_vote_score,
                    )
            except Exception as _iso_err:
                logger.debug(f"Phase 52: Isotonic calibration skipped: {_iso_err}")

        # Phase 52 (US-324): Confidence decomposition — directional/timing/magnitude breakdown
        if self._confidence_decomposer is not None:
            try:
                _agent_verdict_map = {v.name: v.passed for v in verdicts}
                _uncertainty = float(getattr(analysis, "uncertainty_score", 0.0) or 0.0)
                _disagreement = float(getattr(analysis, "model_disagreement", 0.0) or 0.0)
                _session = str(getattr(analysis, "session_name", "LONDON") or "LONDON").upper()
                _mtf = str(getattr(analysis, "mtf_signal", "PROCEED") or "PROCEED").upper()
                _regime = str(getattr(analysis, "volatility_regime", "NORMAL") or "NORMAL").upper()
                _atr_pct = float(getattr(analysis, "atr_percentile", 0.50) or 0.50)

                _decomp = self._confidence_decomposer.decompose(
                    raw_confidence=float(analysis.weighted_vote_score),
                    agent_verdicts=_agent_verdict_map,
                    uncertainty_score=_uncertainty,
                    model_disagreement=_disagreement,
                    session_name=_session,
                    mtf_signal=_mtf,
                    regime=_regime,
                    atr_percentile=_atr_pct,
                )

                # Apply recomposed confidence (replaces raw)
                analysis.weighted_vote_score = _clip01(_decomp.recomposed_confidence)

                if not hasattr(analysis, 'calibration_details'):
                    analysis.calibration_details = {}
                analysis.calibration_details["decomposition"] = {
                    "directional": _decomp.directional_strength,
                    "timing": _decomp.timing_quality,
                    "magnitude": _decomp.magnitude_estimate,
                    "recomposed": _decomp.recomposed_confidence,
                    "penalty_applied": _decomp.high_conf_penalty_applied,
                }

                logger.debug(
                    "Phase 52 (US-324): %s decomposed: dir=%.3f tim=%.3f mag=%.3f → %.3f%s",
                    getattr(analysis, "pair", "?"),
                    _decomp.directional_strength, _decomp.timing_quality, _decomp.magnitude_estimate,
                    _decomp.recomposed_confidence,
                    " (PENALTY)" if _decomp.high_conf_penalty_applied else "",
                )
            except Exception as _decomp_err:
                logger.debug(f"Phase 52: Confidence decomposition skipped: {_decomp_err}")

        # Phase 53 (US-331): Unified confidence pipeline monitoring
        # Tracks calibration error across deciles and flags degradation
        if self._confidence_pipeline is not None:
            try:
                _pipe_result = self._confidence_pipeline.calibrate(
                    raw_confidence=float(analysis.weighted_vote_score),
                    agent_verdicts={v.name: v.passed for v in verdicts},
                    regime=str(getattr(analysis, "volatility_regime", "NORMAL") or "NORMAL").upper(),
                    isotonic_calibrator=self._isotonic_calibrator,
                    decomposer=self._confidence_decomposer,
                    session_name=str(getattr(analysis, "session_name", "LONDON") or "LONDON").upper(),
                    mtf_signal=str(getattr(analysis, "mtf_signal", "PROCEED") or "PROCEED").upper(),
                    atr_percentile=float(getattr(analysis, "atr_percentile", 0.50) or 0.50),
                    uncertainty_score=float(getattr(analysis, "uncertainty_score", 0.0) or 0.0),
                    model_disagreement=float(getattr(analysis, "model_disagreement", 0.0) or 0.0),
                )
                if not hasattr(analysis, 'calibration_details'):
                    analysis.calibration_details = {}
                analysis.calibration_details["pipeline_path"] = _pipe_result.path_used
                analysis.calibration_details["calibration_error"] = round(_pipe_result.calibration_error, 4)
                analysis.calibration_details["degradation_flag"] = _pipe_result.degradation_flag
                if _pipe_result.degradation_flag:
                    logger.warning(
                        "Phase 53 (US-331): %s calibration degradation — error=%.3f",
                        getattr(analysis, "pair", "?"), _pipe_result.calibration_error,
                    )
            except Exception as _pipe_err:
                logger.debug(f"Phase 53: Confidence pipeline monitoring error: {_pipe_err}")

        # Phase 55 (US-343): Apply disagreement-based confidence penalty
        if self._disagreement_tracker is not None:
            try:
                _agent_scores = {v.name: _clip01(v.score) for v in verdicts}
                _disagree_penalty = self._disagreement_tracker.get_confidence_penalty(_agent_scores)
                if _disagree_penalty < 0:
                    _prev_score = analysis.weighted_vote_score
                    analysis.weighted_vote_score = _clip01(
                        float(analysis.weighted_vote_score) + _disagree_penalty
                    )
                    if not hasattr(analysis, 'calibration_details'):
                        analysis.calibration_details = {}
                    analysis.calibration_details["disagreement_penalty"] = round(_disagree_penalty, 4)
                    logger.info(
                        "Phase 55 (US-343): %s disagreement penalty=%.4f (%.3f → %.3f)",
                        getattr(analysis, "pair", "?"),
                        _disagree_penalty, _prev_score, analysis.weighted_vote_score,
                    )
            except Exception as _dt_err:
                logger.debug(f"Phase 55: Disagreement penalty skipped: {_dt_err}")

        # ── Neural agent shadow evaluation (non-voting, for replay learning) ──
        if getattr(self, "_neural_team", None) is not None:
            try:
                for _n_name, _n_agent in self._neural_team.agents.items():
                    _n_verdict = _n_agent.evaluate(ctx)
                    if _n_verdict is not None:
                        verdicts.append(_n_verdict)
            except Exception as _n_err:
                logger.debug("Neural shadow evaluation skipped: %s", _n_err)

        analysis.agent_reasons = [v.to_dict() for v in verdicts]
        analysis.agent_reason_codes = list(dict.fromkeys(v.reason_code for v in verdicts if v.reason_code))
        analysis.why_trade = [v.reason for v in verdicts if v.passed and v.reason]
        analysis.why_no_trade = [v.reason for v in verdicts if (not v.passed or v.block_trade) and v.reason]

        uncertainty_meta = next((v.metadata for v in verdicts if v.name == "uncertainty"), {})
        analysis.uncertainty_score = _clip01(_safe_float(uncertainty_meta.get("uncertainty_score"), analysis.uncertainty_score))
        analysis.confidence_variance = _clip01(_safe_float(uncertainty_meta.get("confidence_variance"), analysis.confidence_variance))
        analysis.model_disagreement = _clip01(_safe_float(uncertainty_meta.get("model_disagreement"), analysis.model_disagreement))
        # Phase 76: Pass max_model_disagreement to PairAnalysis for is_tradeable check
        analysis._max_model_disagreement = _clip01(_safe_float(
            getattr(self.config, "max_model_disagreement", 0.50), 0.50
        ))

        execution_meta = next((v.metadata for v in verdicts if v.name == "execution_quality"), {})
        analysis.execution_quality_score = _clip01(_safe_float(execution_meta.get("execution_quality_score"), analysis.execution_quality_score))
        analysis.execution_quality_passed = bool(execution_meta.get("execution_quality_passed", analysis.execution_quality_passed))
        analysis.spread_pips = _safe_float(execution_meta.get("spread_pips"), analysis.spread_pips)
        analysis.est_slippage_pips = _safe_float(execution_meta.get("est_slippage_pips"), analysis.est_slippage_pips)
        analysis.liquidity_score = _clip01(_safe_float(execution_meta.get("liquidity_score"), analysis.liquidity_score))

        if block_trade:
            analysis.blocked_by_circuit_breaker = True
            analysis.circuit_breakers_triggered = list(
                dict.fromkeys(v.reason_code for v in verdicts if v.block_trade and v.reason_code)
            )

        if analysis.weighted_vote_score < analysis.weighted_vote_threshold:
            analysis.why_no_trade.append(
                f"agent vote {analysis.weighted_vote_score:.2f} below {analysis.weighted_vote_threshold:.2f}"
            )

        if block_trade or analysis.weighted_vote_score < analysis.weighted_vote_threshold:
            analysis.gates_passed = False

        analysis.why_trade = list(dict.fromkeys(analysis.why_trade))
        analysis.why_no_trade = list(dict.fromkeys(analysis.why_no_trade))
        return analysis

    def _apply_regime_multipliers(
        self,
        verdicts: List[AgentVerdict],
        regime: str,
    ) -> List[AgentVerdict]:
        """Apply dynamic weight multipliers based on volatility regime.

        In EXTREME volatility: momentum weight *= 1.3, mean_reversion *= 0.7
        In HIGH volatility: volatility agent weight *= 1.2
        In NORMAL: no adjustment (baseline)

        Args:
            verdicts: List of agent verdicts to adjust
            regime: Volatility regime ("NORMAL", "HIGH", "EXTREME")

        Returns:
            List of verdicts with adjusted weights
        """
        multipliers: Dict[str, float] = {}

        if regime == "EXTREME":
            multipliers = {
                "momentum": 1.3,
                "mean_reversion": 0.7,
                "trend": 1.15,
            }
        elif regime == "HIGH":
            multipliers = {
                "volatility": 1.2,
                "momentum": 1.1,
            }
        # NORMAL regime: no multipliers (baseline)

        if not multipliers:
            return verdicts

        # Apply multipliers to verdict weights
        adjusted = []
        for v in verdicts:
            if v.name in multipliers:
                mult = multipliers[v.name]
                new_weight = round(v.weight * mult, 4)
                v.weight = new_weight
            adjusted.append(v)

        return adjusted

    def _weight_for(self, name: str, regime: Optional[str] = None) -> float:
        """Get weight for an agent, respecting regime if provided.

        Args:
            name: Agent name
            regime: Optional volatility regime (if not provided, uses global weights)

        Returns:
            Weight value for this agent
        """
        if regime is None:
            regime = self._active_weight_regime
        if regime:
            regime_weights = self.get_weights_for_regime(regime)
            return _safe_float(regime_weights.get(name), self._BASE_WEIGHTS.get(name, 1.0))

        # Fall back to global weights (cross-regime running average)
        if isinstance(self._learned_weights, dict) and "_global" in self._learned_weights:
            global_weights = self._learned_weights.get("_global", {})
            if isinstance(global_weights, dict) and name in global_weights:
                return _safe_float(global_weights.get(name), self._BASE_WEIGHTS.get(name, 1.0))

        # Legacy flat dict support
        if name in self._learned_weights:
            return float(self._learned_weights[name])

        return float(self._BASE_WEIGHTS.get(name, 1.0))

    def _direction_matches(self, signal_direction: float, expected: str) -> bool:
        if expected == "LONG":
            return signal_direction > 0
        if expected == "SHORT":
            return signal_direction < 0
        return False

    def _evaluate_trend(self, ctx: AgentDecisionContext) -> AgentVerdict:
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        sma_50 = _last_value(ctx.df_feat, "sma_50", sma_20)
        adx = _last_value(ctx.df_feat, "adx", _safe_float(ctx.analysis.ridge_confidence))

        trend_signal = 0.0
        if close > sma_20 >= sma_50:
            trend_signal = 1.0
        elif close < sma_20 <= sma_50:
            trend_signal = -1.0

        aligned = self._direction_matches(trend_signal, ctx.analysis.direction)
        contra = trend_signal != 0.0 and not aligned
        adx_support = _clip01((adx - 18.0) / 22.0)

        score = 0.48 + adx_support * 0.24
        if aligned:
            score += 0.22
        elif contra:
            score -= 0.24

        if aligned:
            reason = f"trend aligned (ADX {adx:.0f})"
            reason_code = "trend_align"
            confidence_delta = 0.04
        elif contra:
            reason = f"trend conflicts with {ctx.analysis.direction.lower()} bias"
            reason_code = "trend_contra"
            confidence_delta = -0.06
        else:
            reason = f"trend neutral (ADX {adx:.0f})"
            reason_code = "trend_neutral"
            confidence_delta = 0.0

        passed = score >= 0.55
        block_trade = False
        if (
            not passed
            and ctx.analysis.direction in ("LONG", "SHORT")
            and getattr(ctx.config, "enable_trend_agent_hard_veto", True)
        ):
            block_trade = True
            reason = reason + " (hard veto)"

        return AgentVerdict(
            name="trend",
            score=_clip01(score),
            passed=passed,
            weight=self._weight_for("trend"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_mean_reversion(self, ctx: AgentDecisionContext) -> AgentVerdict:
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        direction = ctx.analysis.direction

        if direction == "LONG":
            support = _clip01((50.0 - rsi) / 20.0)
            oppose = _clip01((rsi - 58.0) / 20.0)
        else:
            support = _clip01((rsi - 50.0) / 20.0)
            oppose = _clip01((42.0 - rsi) / 20.0)

        score = _clip01(0.50 + support * 0.25 - oppose * 0.30)
        if support >= 0.40:
            reason = f"mean reversion supports {direction.lower()} (RSI {rsi:.1f})"
            reason_code = "mean_reversion_support"
            confidence_delta = 0.02
        elif oppose >= 0.45:
            reason = f"RSI stretched against {direction.lower()} (RSI {rsi:.1f})"
            reason_code = "mean_reversion_contra"
            confidence_delta = -0.04
        else:
            reason = f"RSI neutral (RSI {rsi:.1f})"
            reason_code = "mean_reversion_neutral"
            confidence_delta = 0.0

        # H2 composite veto: MR voted NO on every losing trade in the 04-15 streak.
        # On its own, MR weight (0.90, lowest) is arithmetically drowned in the WVS.
        # The composite rule — MR fails AND models already disagree — catches the
        # "nobody knows what's happening here" state that produced MFE=0 losses.
        # Uses ctx.gate_details (populated upstream by gate evaluator), not
        # ctx.analysis.model_disagreement (which is assigned AFTER verdicts).
        passed = score >= 0.52
        block_trade = False
        if (
            not passed
            and direction in ("LONG", "SHORT")
            and getattr(ctx.config, "enable_mean_reversion_veto", True)
        ):
            disagree_floor = _safe_float(
                getattr(ctx.config, "mean_reversion_veto_disagree_floor", 0.25),
                0.25,
            )
            model_disagree = float(ctx.gate_details.get("model_disagreement", 0.0) or 0.0)
            if model_disagree > disagree_floor:
                block_trade = True
                reason = reason + f" (hard veto — disagreement {model_disagree:.2f} > {disagree_floor:.2f})"

        return AgentVerdict(
            name="mean_reversion",
            score=score,
            passed=passed,
            weight=self._weight_for("mean_reversion"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_volatility(self, ctx: AgentDecisionContext) -> AgentVerdict:
        atr_pips = _safe_float(ctx.analysis.atr_pips)
        vol_pct = _clip01(_safe_float(ctx.analysis.volatility_percentile, 0.5))
        regime = str(getattr(ctx.analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").upper()
        regime_score = {
            "LOW": 0.25,
            "NORMAL": 0.52,
            "HIGH": 0.72,
            "EXTREME": 0.85,
        }.get(regime, 0.55)

        min_atr = max(_safe_float(getattr(ctx.config, "min_atr_pips", 5.0), 5.0), 0.1)
        atr_score = _clip01(atr_pips / (min_atr * 1.5))
        score = _clip01(0.20 + atr_score * 0.45 + vol_pct * 0.20 + regime_score * 0.15)

        extreme_low_atr = atr_pips <= (min_atr * 0.50)
        block_trade = bool(getattr(ctx.analysis, "volatility_gate_passed", True) is False) or extreme_low_atr
        if block_trade:
            if extreme_low_atr:
                reason = f"volatility DEAD ({atr_pips:.1f} <= 0.5x {min_atr:.1f} pips) [HARD BLOCK]"
                reason_code = "volatility_dead"
                confidence_delta = -0.10
            else:
                reason = f"volatility too weak ({atr_pips:.1f} pips)"
                reason_code = "volatility_block"
                confidence_delta = -0.07
        elif score >= 0.60:
            reason = f"volatility supportive ({regime.lower()}, {atr_pips:.1f} pips)"
            reason_code = "volatility_support"
            confidence_delta = 0.03
        else:
            reason = f"volatility mixed ({regime.lower()}, {atr_pips:.1f} pips)"
            reason_code = "volatility_mixed"
            confidence_delta = -0.01

        return AgentVerdict(
            name="volatility",
            score=score,
            passed=score >= 0.52 and not block_trade,
            weight=self._weight_for("volatility"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_risk(self, ctx: AgentDecisionContext) -> AgentVerdict:
        drawdown = _safe_float(ctx.analysis.drawdown)
        max_drawdown = max(_safe_float(getattr(ctx.config, "max_drawdown_pct", 0.025), 0.025), 1e-6)
        risk_ratio = drawdown / max_drawdown

        score = _clip01(1.0 - min(risk_ratio, 2.0) / 2.0)
        if bool(ctx.analysis.risk_passed):
            score = _clip01(score + 0.12)

        block_trade = (not bool(ctx.analysis.risk_passed)) and risk_ratio >= 1.0
        if block_trade:
            reason = f"risk sentinel blocked ({drawdown:.2%} drawdown)"
            reason_code = "risk_block"
            confidence_delta = -0.08
        elif score >= 0.60:
            reason = f"risk acceptable ({drawdown:.2%} drawdown)"
            reason_code = "risk_ok"
            confidence_delta = 0.02
        else:
            reason = f"risk elevated ({drawdown:.2%} drawdown)"
            reason_code = "risk_elevated"
            confidence_delta = -0.03

        return AgentVerdict(
            name="risk_sentinel",
            score=score,
            passed=score >= 0.55 and not block_trade,
            weight=self._weight_for("risk_sentinel"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
        )

    def _evaluate_uncertainty(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Evaluate uncertainty with ensemble conflict resolution.

        Phase 47 (US-295): Integrates Ensemble Conflict Resolver for graduated penalties
        based on TCN/Ridge/RF model disagreement.
        """
        direction = ctx.analysis.direction
        confidence = _clip01(_safe_float(ctx.analysis.confidence))
        close = _last_value(ctx.df_raw, "close", _safe_float(ctx.analysis.current_price))
        sma_20 = _last_value(ctx.df_feat, "sma_20", close)
        rsi = _last_value(ctx.df_feat, "rsi", 50.0)
        ret_5 = _last_value(ctx.df_feat, "returns", 0.0)
        if abs(ret_5) < 1e-6 and "close" in ctx.df_raw.columns and len(ctx.df_raw) >= 6:
            base = ctx.df_raw["close"]
            ret_5 = _safe_float((base.iloc[-1] / base.iloc[-6]) - 1.0, 0.0)

        # Phase 76: Expanded from 3 to 5 heuristics for finer-grained disagreement.
        # With 3 heuristics, 1/3=0.33 > 0.30 hard floor always blocks.
        # With 5 heuristics, 1/5=0.20 passes, 2/5=0.40 blocks — more proportional.
        sma_50 = _last_value(ctx.df_feat, "sma_50", close)
        macd_hist = _last_value(ctx.df_feat, "macd_hist", 0.0)  # Feature is "macd_hist" not "macd_histogram"

        heuristics: List[int] = []
        if close != sma_20:
            heuristics.append(1 if close > sma_20 else -1)
        if abs(rsi - 50.0) >= 3.0:
            heuristics.append(1 if rsi < 50.0 else -1)
        if abs(ret_5) >= 0.0001:
            heuristics.append(1 if ret_5 > 0 else -1)
        # Phase 76: SMA_50 trend alignment — price above long MA = bullish
        if close != sma_50:
            heuristics.append(1 if close > sma_50 else -1)
        # Phase 76: MACD histogram — positive = bullish momentum
        if abs(macd_hist) >= 1e-6:
            heuristics.append(1 if macd_hist > 0 else -1)

        if direction == "SHORT":
            heuristics = [-1 * x for x in heuristics]

        oppose = sum(1 for h in heuristics if h < 0)
        total = len(heuristics)
        model_disagreement = (oppose / total) if total > 0 else 0.0
        confidence_uncertainty = _clip01((0.60 - confidence) / 0.10)
        confidence_variance = _clip01(abs(0.62 - confidence) / 0.20)
        uncertainty_score = _clip01(confidence_uncertainty * 0.60 + model_disagreement * 0.40)

        max_uncertainty = _clip01(_safe_float(getattr(ctx.config, "max_uncertainty_score", 0.40), 0.40))
        max_disagreement = _clip01(_safe_float(getattr(ctx.config, "max_model_disagreement", 0.50), 0.50))

        # US-503: Tighten uncertainty threshold when models are stale
        # 2026-04-20: Promoted-rule hardening (trading.md 2026-04-15 staleness_block,
        # 10-trade / -$2,637 loss streak). When models are stale AND uncertainty exceeds
        # the lowered threshold, HARD-BLOCK — soft_uncertainty_blocking must not downgrade
        # this to a confidence penalty.
        stale_hard_block = False
        stale_age_days = 0.0
        try:
            if get_model_freshness is not None:
                _freshness = get_model_freshness(
                    active_pairs=[getattr(ctx.analysis, "pair", "")],
                    use_master_pair_models=bool(
                        getattr(ctx.config, "use_master_pair_models", True)
                    ),
                )
                _oldest = _freshness.get("oldest_age_days") or 0.0
                _staleness_age = _safe_float(getattr(ctx.config, "staleness_age_threshold_days", 7.0), 7.0)
                _staleness_thresh = _safe_float(getattr(ctx.config, "staleness_uncertainty_threshold", 0.35), 0.35)
                if _oldest > _staleness_age:
                    stale_age_days = float(_oldest)
                    _effective = min(max_uncertainty, _staleness_thresh)
                    if _effective < max_uncertainty:
                        logger.info(
                            f"Uncertainty threshold tightened to {_effective:.2f} (models {_oldest:.1f}d stale)"
                        )
                        max_uncertainty = _effective
                    # Hard-block override — bypasses soft_uncertainty_blocking
                    if uncertainty_score > _staleness_thresh:
                        stale_hard_block = True
                        logger.warning(
                            f"STALENESS HARD BLOCK: uncertainty={uncertainty_score:.2f} > "
                            f"{_staleness_thresh:.2f} with models {_oldest:.1f}d stale "
                            f"(> {_staleness_age:.1f}d threshold)"
                        )
        except Exception as _stale_err:
            # Mythos audit 2026-04-28 — observability fix only. The bare
            # `except Exception: pass` previously here silently swallowed
            # every freshness probe failure (CWE-754); operators had no
            # signal that the staleness hard-block was bypassed. Now we
            # WARN + log stack trace so probe failures surface in logs.
            #
            # Behavior preserved: fail-OPEN (do NOT escalate
            # stale_hard_block on probe failure). This honors the US-503
            # AC-3 contract pinned by tests/test_phase91_staleness_block.py
            # ("falls back to max_uncertainty_score=0.40, no crash").
            #
            # The audit recommended fail-CLOSED here, citing the 10-trade
            # -$2,637 streak. That is a deliberate operator-level
            # decision because it can block all trades on a transient
            # probe wedge. Track-and-decide: see Mythos handoff
            # 2026-04-28.
            logger.warning(
                "Model-freshness probe failed — staleness hard-block "
                "defaulted OFF per US-503 AC-3 (uncertainty=%.3f). "
                "Reason: %r",
                uncertainty_score, _stale_err, exc_info=True,
            )
        soft_blocking = bool(getattr(ctx.config, "soft_uncertainty_blocking", False))
        ensemble_conflict_enabled = bool(getattr(ctx.config, "ensemble_conflict_enabled", False))

        # Phase 47 (US-295): Try ensemble conflict resolver for graduated penalties
        ensemble_conflict_result = None
        ensemble_penalty = 0.0
        should_block_ensemble = False

        if ensemble_conflict_enabled and self._ensemble_conflict is not None:
            try:
                # Extract TCN, Ridge, RF scores from analysis if available
                tcn_score = _safe_float(getattr(ctx.analysis, "tcn_score", 0.0), 0.0)
                ridge_score = _safe_float(getattr(ctx.analysis, "ridge_score", 0.0), 0.0)
                rf_score = _safe_float(getattr(ctx.analysis, "rf_score", 0.0), 0.0)

                # Create model predictions for ensemble conflict analysis
                from src.scanner.ensemble_conflict import ModelPrediction
                predictions = [
                    ModelPrediction(
                        model_name="TCN",
                        score=tcn_score,
                        direction="BUY" if tcn_score > 0.1 else ("SELL" if tcn_score < -0.1 else "HOLD"),
                        confidence=_clip01(abs(tcn_score)),
                    ),
                    ModelPrediction(
                        model_name="Ridge",
                        score=ridge_score,
                        direction="BUY" if ridge_score > 0.1 else ("SELL" if ridge_score < -0.1 else "HOLD"),
                        confidence=_clip01(abs(ridge_score)),
                    ),
                    ModelPrediction(
                        model_name="RF",
                        score=rf_score,
                        direction="BUY" if rf_score > 0.1 else ("SELL" if rf_score < -0.1 else "HOLD"),
                        confidence=_clip01(abs(rf_score)),
                    ),
                ]

                ensemble_conflict_result = self._ensemble_conflict.resolve(predictions)

                if ensemble_conflict_result is not None:
                    ensemble_penalty = ensemble_conflict_result.penalty
                    should_block_ensemble = ensemble_conflict_result.should_block

                    logger.debug(
                        f"Ensemble conflict: disagreement={ensemble_conflict_result.disagreement_score:.3f}, "
                        f"penalty={ensemble_penalty:.3f}, block={should_block_ensemble}"
                    )
            except Exception as e:
                # Fallback to legacy flat penalty on any error
                logger.debug(f"Ensemble conflict evaluation failed, using legacy logic: {e}")
                ensemble_conflict_result = None
                ensemble_penalty = 0.0

        exceeds_threshold = uncertainty_score > max_uncertainty or model_disagreement > max_disagreement

        # Hard floor: model_disagreement above threshold is a loss predictor per trading rules.
        # Phase 75: Now configurable (was hardcoded 0.30) to allow adaptive tuning.
        # Phase 76: When soft_uncertainty_blocking=True, disagreement between hard_floor
        # and max_model_disagreement applies graduated confidence penalty instead of
        # hard-blocking. Only hard-blocks when above max_model_disagreement.
        DISAGREEMENT_HARD_FLOOR = _clip01(_safe_float(
            getattr(ctx.config, "disagreement_hard_floor", 0.50), 0.50
        ))
        # Phase 91: >= not > — 0.50 means 50/50 heuristic split (coin flip), must not trade
        disagreement_above_floor = model_disagreement >= DISAGREEMENT_HARD_FLOOR
        # Hard block only when above max_model_disagreement, OR when soft blocking is off
        disagreement_dangerous = (
            disagreement_above_floor
            and (not soft_blocking or model_disagreement >= max_disagreement)
        )

        # Soft blocking: penalize confidence instead of hard-blocking for uncertainty.
        # Phase 76: disagreement between hard_floor and max_disagreement now soft-blocks
        # when soft_uncertainty_blocking=True (graduated penalty instead of hard block).
        # 2026-04-20: stale_hard_block bypasses soft_blocking — staleness + high uncertainty
        # is a hard block regardless of soft_uncertainty_blocking flag.
        block_trade = (
            should_block_ensemble
            or disagreement_dangerous
            or stale_hard_block
            or (exceeds_threshold and not soft_blocking)
        )

        if should_block_ensemble and ensemble_conflict_result is not None:
            # Ensemble conflict triggered a hard block
            reason = (
                f"ensemble direction conflict detected ({ensemble_conflict_result.disagreement_score:.2f} disagreement) "
                f"[HARD BLOCK]"
            )
            reason_code = "ensemble_direction_conflict"
            confidence_delta = -1.0  # Hard block: full penalty
        elif stale_hard_block:
            # Staleness + high uncertainty — promoted rule (trading.md 2026-04-15)
            reason = (
                f"staleness hard-block: uncertainty={uncertainty_score:.2f} with models "
                f"{stale_age_days:.1f}d stale [HARD BLOCK]"
            )
            reason_code = "staleness_hard_block"
            confidence_delta = -1.0  # Hard block: full penalty
        elif disagreement_dangerous:
            disagree_overshoot = model_disagreement - DISAGREEMENT_HARD_FLOOR
            reason = f"model disagreement dangerous ({model_disagreement:.2f} > {max_disagreement:.2f}) [HARD BLOCK]"
            reason_code = "disagreement_hard_block"
            confidence_delta = -0.10 - (disagree_overshoot * 0.50)
            if ensemble_conflict_result is not None and ensemble_penalty < 0.0:
                confidence_delta = min(confidence_delta, ensemble_penalty)
        elif disagreement_above_floor and soft_blocking:
            # Phase 76: Soft disagreement penalty — between hard_floor and max_disagreement
            disagree_overshoot = model_disagreement - DISAGREEMENT_HARD_FLOOR
            reason = f"model disagreement elevated ({model_disagreement:.2f} > {DISAGREEMENT_HARD_FLOOR:.2f}) [soft penalty]"
            reason_code = "disagreement_soft_penalty"
            confidence_delta = -0.05 - (disagree_overshoot * 0.30)
        elif exceeds_threshold and soft_blocking:
            # Scale penalty by how far over both thresholds we are
            uncert_overshoot = max(uncertainty_score - max_uncertainty, 0.0)
            disagree_overshoot = max(model_disagreement - max_disagreement, 0.0)
            overshoot = max(uncert_overshoot, disagree_overshoot)
            reason = f"uncertainty high ({uncertainty_score:.2f}) [soft penalty]"
            reason_code = "uncertainty_soft_penalty"
            # Use ensemble penalty if available, otherwise use flat penalty
            if ensemble_conflict_result is not None and ensemble_penalty < 0.0:
                confidence_delta = ensemble_penalty
            else:
                confidence_delta = -0.05 - (overshoot * 0.30)  # -5% to -14% penalty
        elif block_trade:
            reason = f"uncertainty high ({uncertainty_score:.2f})"
            reason_code = "uncertainty_block"
            confidence_delta = -0.10
        elif uncertainty_score <= max_uncertainty * 0.75:
            reason = f"uncertainty contained ({uncertainty_score:.2f})"
            reason_code = "uncertainty_ok"
            confidence_delta = 0.02
        else:
            reason = f"uncertainty borderline ({uncertainty_score:.2f})"
            reason_code = "uncertainty_watch"
            confidence_delta = -0.03

        metadata = {
            "uncertainty_score": uncertainty_score,
            "confidence_variance": confidence_variance,
            "model_disagreement": model_disagreement,
        }

        # Add ensemble conflict details if available
        if ensemble_conflict_result is not None:
            metadata["ensemble_conflict_disagreement"] = ensemble_conflict_result.disagreement_score
            metadata["ensemble_conflict_penalty"] = ensemble_penalty
            metadata["ensemble_direction_conflict"] = ensemble_conflict_result.has_direction_conflict
            metadata["ensemble_consensus_direction"] = ensemble_conflict_result.consensus_direction

        return AgentVerdict(
            name="uncertainty",
            score=_clip01(1.0 - uncertainty_score),
            passed=not block_trade,
            weight=self._weight_for("uncertainty"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata=metadata,
        )

    def _evaluate_execution_quality(self, ctx: AgentDecisionContext) -> AgentVerdict:
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)
        volume_ratio = _last_value(ctx.df_feat, "volume_ratio", 1.0)  # Phase 78: was "volume_ratio_20" (doesn't exist)
        liquidity_score = _clip01((volume_ratio - 0.35) / 1.15)
        if liquidity_score <= 0.0:
            liquidity_score = 0.45

        spread_pips = _safe_float(ctx.analysis.spread_pips)
        if spread_pips <= 0.0:
            spread_pips = max(0.1, min(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0) * 1.5, atr_pips * 0.04))

        slippage_pips = _safe_float(ctx.analysis.est_slippage_pips)
        if slippage_pips <= 0.0:
            slippage_pips = max(0.05, spread_pips * 0.45 + (1.0 - liquidity_score) * 0.9)

        max_spread = max(_safe_float(getattr(ctx.config, "max_spread_pips", 3.0), 3.0), 0.1)
        max_slippage = max(_safe_float(getattr(ctx.config, "max_slippage_pips", 2.0), 2.0), 0.1)
        min_liquidity = _clip01(_safe_float(getattr(ctx.config, "min_liquidity_score", 0.3), 0.3))

        spread_score = _clip01(1.0 - (spread_pips / (max_spread * 1.4)))
        slippage_score = _clip01(1.0 - (slippage_pips / (max_slippage * 1.4)))
        execution_quality_score = _clip01(spread_score * 0.35 + slippage_score * 0.30 + liquidity_score * 0.35)

        passed = spread_pips <= max_spread and slippage_pips <= max_slippage and liquidity_score >= min_liquidity
        extreme_spread = spread_pips >= (max_spread * 2.0)
        block_trade = extreme_spread or spread_pips > (max_spread * 1.25) or slippage_pips > (max_slippage * 1.25)

        if block_trade:
            if extreme_spread:
                reason = f"execution spread EXTREME ({spread_pips:.1f} >= 2x {max_spread:.1f} pips) [HARD BLOCK]"
                reason_code = "execution_spread_extreme"
                confidence_delta = -0.12
            else:
                reason = f"execution quality poor (spread {spread_pips:.1f} pips)"
                reason_code = "execution_block"
                confidence_delta = -0.08
        elif passed:
            reason = f"execution quality solid (liq {liquidity_score:.2f})"
            reason_code = "execution_ok"
            confidence_delta = 0.02
        else:
            reason = f"execution quality mixed (spread {spread_pips:.1f} pips)"
            reason_code = "execution_watch"
            confidence_delta = -0.02

        return AgentVerdict(
            name="execution_quality",
            score=execution_quality_score,
            passed=passed and not block_trade,
            weight=self._weight_for("execution_quality"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "execution_quality_score": execution_quality_score,
                "execution_quality_passed": passed and not block_trade,
                "spread_pips": spread_pips,
                "est_slippage_pips": slippage_pips,
                "liquidity_score": liquidity_score,
            },
        )

    def _evaluate_news_risk(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Evaluate news risk using live calendar, RSS headlines, and sentiment.

        Data sources (all lazy-imported, individually try/excepted):
          1. market_intelligence.EconomicCalendar  -- ForexFactory XML feed
          2. news_features.fetch_forex_news         -- RSS headlines (stdlib fallback)
          3. market_intelligence.NewsSentimentAnalyzer -- VADER sentiment scoring

        Combines calendar confidence penalty with directional sentiment alignment
        to produce a single AgentVerdict. Never crashes the scan pipeline.
        """
        import logging
        _log = logging.getLogger(__name__)

        pair = str(getattr(ctx.analysis, "pair", "")).upper().replace("/", "_")

        try:
            # ── 1. Live economic calendar events ─────────────────────────────
            raw_events = []
            try:
                from market_intelligence import EconomicCalendar as _MICal
                mi_cal = _MICal()
                mi_events = mi_cal.fetch_events(pair, hours_ahead=4)
                # Convert market_intelligence EconomicEvent objects to dicts
                # that NewsCalendarAgent._parse_api_entry() can consume
                for ev in mi_events:
                    raw_events.append({
                        "title": getattr(ev, "name", "") or getattr(ev, "title", ""),
                        "currency": getattr(ev, "currency", ""),
                        "impact": getattr(ev, "impact", "low"),
                        "date": getattr(ev, "time", datetime.now(timezone.utc)).isoformat(),
                    })
            except Exception as exc:
                _log.debug("EconomicCalendar fetch failed: %s", exc)

            # ── 2. Pass real events to NewsCalendarAgent ─────────────────────
            penalty = 0.0
            upcoming = []
            try:
                from src.scanner.agents.news_calendar import NewsCalendarAgent
                agent = NewsCalendarAgent(events=raw_events if raw_events else None)
                penalty = agent.get_confidence_penalty(pair)
                upcoming = agent.get_upcoming_events(pair, hours=2)
            except Exception as exc:
                _log.debug("NewsCalendarAgent failed: %s", exc)

            # ── 3. Fetch per-pair RSS headlines ──────────────────────────────
            headlines = []
            try:
                from news_features import fetch_forex_news, _rss_fallback_fetcher
                from news_features import _NEWS_FETCHER
                # If no fetcher is registered (feedparser missing), register
                # the stdlib fallback so fetch_forex_news actually works
                if _NEWS_FETCHER is None:
                    from news_features import set_news_fetcher
                    set_news_fetcher(_rss_fallback_fetcher)
                headlines = fetch_forex_news(pair, max_headlines=5)
            except Exception as exc:
                _log.debug("RSS headline fetch failed for %s: %s", pair, exc)

            # ── 4. Score headlines with VADER sentiment ──────────────────────
            pair_sentiment = 0.0
            sentiment_label = "NEUTRAL"
            if headlines:
                try:
                    from market_intelligence import NewsSentimentAnalyzer
                    analyzer = NewsSentimentAnalyzer(fast_mode=True)
                    scores = analyzer.analyze_batch(headlines)
                    if scores:
                        pair_sentiment = sum(
                            s.get("score", 0.0) for s in scores
                        ) / len(scores)
                        if pair_sentiment > 0.15:
                            sentiment_label = "BULLISH"
                        elif pair_sentiment < -0.15:
                            sentiment_label = "BEARISH"
                except Exception as exc:
                    _log.debug("Sentiment analysis failed: %s", exc)

            # ── 5. Directional alignment check ───────────────────────────────
            direction = str(getattr(ctx.analysis, "direction", "")).upper()
            sentiment_aligned = True
            sentiment_penalty = 0.0
            if sentiment_label == "BEARISH" and direction == "LONG":
                sentiment_penalty = -0.10
                sentiment_aligned = False
            elif sentiment_label == "BULLISH" and direction == "SHORT":
                sentiment_penalty = -0.10
                sentiment_aligned = False
            elif sentiment_label == "BULLISH" and direction == "LONG":
                sentiment_penalty = 0.05
            elif sentiment_label == "BEARISH" and direction == "SHORT":
                sentiment_penalty = 0.05

            # ── 6. Combine penalties ─────────────────────────────────────────
            total_penalty = penalty + sentiment_penalty

            # ── 7. Build metadata ────────────────────────────────────────────
            meta = {
                "pair_sentiment": round(pair_sentiment, 3),
                "sentiment_label": sentiment_label,
                "headline_count": len(headlines),
                "top_headline": headlines[0][:100] if headlines else None,
                "calendar_penalty": penalty,
                "sentiment_penalty": sentiment_penalty,
                "total_penalty": total_penalty,
                "upcoming_events": len(upcoming),
                "next_event": upcoming[0] if upcoming else {},
                "sentiment_aligned": sentiment_aligned,
            }

            # ── 8. Determine verdict ─────────────────────────────────────────
            next_ev = upcoming[0] if upcoming else {}

            if total_penalty < -0.15:
                # High calendar risk -- block
                reason_parts = []
                if next_ev:
                    reason_parts.append(
                        "%s impact event in %smin (%s %s)" % (
                            next_ev.get("impact", "HIGH"),
                            next_ev.get("minutes_until", "?"),
                            next_ev.get("currency", ""),
                            next_ev.get("title", "event"),
                        )
                    )
                else:
                    reason_parts.append(
                        "calendar penalty %.0f%%" % (penalty * 100,)
                    )
                if not sentiment_aligned:
                    reason_parts.append(
                        "sentiment %s opposes %s" % (sentiment_label, direction)
                    )
                return AgentVerdict(
                    name="news_risk",
                    score=0.30,
                    passed=False,
                    weight=self._weight_for("news_risk"),
                    reason="; ".join(reason_parts),
                    reason_code="news_block",
                    confidence_delta=float(total_penalty),
                    block_trade=True,
                    metadata=meta,
                )

            if not sentiment_aligned and penalty == 0.0:
                # Sentiment opposes direction but no calendar risk
                return AgentVerdict(
                    name="news_risk",
                    score=0.40,
                    passed=True,
                    weight=self._weight_for("news_risk"),
                    reason="sentiment %s opposes %s direction" % (
                        sentiment_label, direction
                    ),
                    reason_code="news_sentiment_opposed",
                    confidence_delta=float(total_penalty),
                    metadata=meta,
                )

            if sentiment_aligned and sentiment_label != "NEUTRAL" and penalty == 0.0:
                # Sentiment confirms direction, no calendar risk
                return AgentVerdict(
                    name="news_risk",
                    score=0.65,
                    passed=True,
                    weight=self._weight_for("news_risk"),
                    reason="sentiment %s confirms %s direction" % (
                        sentiment_label, direction
                    ),
                    reason_code="news_sentiment_aligned",
                    confidence_delta=float(total_penalty),
                    metadata=meta,
                )

            if total_penalty < 0.0:
                # Moderate calendar/sentiment penalty -- caution but pass
                reason_parts = []
                if penalty < 0.0 and next_ev:
                    reason_parts.append(
                        "event in %smin (%s)" % (
                            next_ev.get("minutes_until", "?"),
                            next_ev.get("title", "event"),
                        )
                    )
                if not sentiment_aligned:
                    reason_parts.append(
                        "sentiment %s opposes %s" % (sentiment_label, direction)
                    )
                if not reason_parts:
                    reason_parts.append(
                        "moderate news risk (penalty %.0f%%)" % (total_penalty * 100,)
                    )
                return AgentVerdict(
                    name="news_risk",
                    score=0.45,
                    passed=True,
                    weight=self._weight_for("news_risk"),
                    reason="; ".join(reason_parts),
                    reason_code="news_caution",
                    confidence_delta=float(total_penalty),
                    metadata=meta,
                )

            # No events, neutral sentiment
            return AgentVerdict(
                name="news_risk",
                score=0.60,
                passed=True,
                weight=self._weight_for("news_risk"),
                reason="no events, neutral sentiment",
                reason_code="news_clear",
                confidence_delta=float(total_penalty),
                metadata=meta,
            )

        except Exception as exc:
            # ── FALLBACK: UTC hour-based heuristic (last resort) ─────────
            _log.debug("News risk live data failed, using UTC fallback: %s", exc)
            now = datetime.now(timezone.utc)
            hour_utc = now.hour

            # Known recurring high-impact event hours in UTC:
            # NFP: Usually Friday 13:30 UTC
            # FOMC: Usually Wednesday 19:00 UTC
            # ECB/BOE: Usually Thursday 12-14 UTC
            high_risk_hours = {13, 14, 19}

            if hour_utc in high_risk_hours:
                return AgentVerdict(
                    name="news_risk",
                    score=0.48,
                    passed=True,
                    weight=self._weight_for("news_risk"),
                    reason="possible high-impact event window (no calendar data)",
                    reason_code="news_bootstrap_caution",
                    confidence_delta=-0.05,
                )

            return AgentVerdict(
                name="news_risk",
                score=0.55,
                passed=True,
                weight=self._weight_for("news_risk"),
                reason="outside major event hours (calendar unavailable)",
                reason_code="news_bootstrap_neutral",
                confidence_delta=0.0,
            )

    # MTF data cache: {(pair, granularity, count): (timestamp, df)}
    _mtf_cache: dict = {}
    _MTF_CACHE_TTL_M15 = 15 * 60
    _MTF_CACHE_TTL_H1 = 60 * 60
    _MTF_CACHE_TTL_H4 = 4 * 3600  # 4 hours
    _MTF_CACHE_TTL_D1 = 24 * 3600  # 24 hours

    def _fetch_mtf_candles(self, pair: str, granularity: str, count: int = 50) -> "Optional[pd.DataFrame]":
        """Fetch higher-timeframe candles from OANDA with caching.

        Returns DataFrame or None on failure (graceful degradation).
        """
        import time as _time

        granularity = str(granularity).upper()
        cache_key = (pair, granularity, int(count))
        ttl_by_granularity = {
            "M15": self._MTF_CACHE_TTL_M15,
            "H1": self._MTF_CACHE_TTL_H1,
            "H4": self._MTF_CACHE_TTL_H4,
            "D": self._MTF_CACHE_TTL_D1,
            "D1": self._MTF_CACHE_TTL_D1,
        }
        ttl = ttl_by_granularity.get(granularity, self._MTF_CACHE_TTL_H1)

        # Check cache
        if cache_key in self._mtf_cache:
            cached_time, cached_df = self._mtf_cache[cache_key]
            if _time.time() - cached_time < ttl:
                return cached_df

        try:
            if self._oanda_client is None:
                from src.utils.oanda_practice import OandaPracticeClient
                self._oanda_client = OandaPracticeClient.from_env()
            raw = self._oanda_client.get_candles(
                pair,
                granularity=granularity,
                count=count,
                price="M",
            )
            df = _oanda_candles_to_frame(raw)
            if df is not None and len(df) >= 5:
                self._mtf_cache[cache_key] = (_time.time(), df)
                return df
        except Exception as e:
            logger.debug(f"MTF fetch {pair}/{granularity} failed: {e}")

        return None

    @staticmethod
    def _mtf_direction_matches(direction: str, trend_direction: str) -> bool:
        if direction == "LONG":
            return trend_direction == "BULLISH"
        if direction == "SHORT":
            return trend_direction == "BEARISH"
        return False

    def _evaluate_multi_timeframe(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Multi-timeframe confluence: uses real H4/D1 data from OANDA when available.

        Fetches real H4 and D1 candles from OANDA API with caching (4h/24h TTL).
        Falls back to H1 approximation if OANDA fetch fails.

        Phase 47 (US-294): Integrates MTF Confluence module for Elder's Triple Screen scoring.
        """
        df = ctx.df_raw
        pair = str(getattr(ctx.analysis, "pair", "")).upper().replace("/", "_")
        direction = ctx.analysis.direction
        confluence_count = 0
        data_source = "synthetic"  # Track whether we used real data
        mtf_confluence_enabled = bool(getattr(ctx.config, "mtf_confluence_enabled", False))
        mtf_confluence_result = None

        # Phase 47 (US-294): Try Elder's Triple Screen via MTF Confluence module
        if mtf_confluence_enabled and self._mtf_confluence is not None:
            try:
                # Gather true 15M, 1H, 4H data. Scanner raw candles are not
                # assumed to be H1; default scanner granularity is M15.
                scanner_granularity = str(getattr(ctx.config, "granularity", "")).upper()
                data_15m = (
                    ctx.df_raw.copy()
                    if scanner_granularity == "M15" and len(ctx.df_raw) >= 20
                    else None
                )
                if data_15m is None and pair:
                    data_15m = self._fetch_mtf_candles(pair, "M15", count=120)

                data_1h = self._fetch_mtf_candles(pair, "H1", count=120) if pair else None
                if data_1h is None:
                    if scanner_granularity == "H1" and len(ctx.df_raw) >= 20:
                        data_1h = ctx.df_raw.copy()
                    else:
                        data_1h = _resample_ohlcv(ctx.df_raw, "1h", min_bars=20)

                h4_count = max(int(getattr(self._mtf_confluence.config, "sma_long_period", 200)), 220)
                data_4h = self._fetch_mtf_candles(pair, "H4", count=h4_count) if pair else None
                if data_4h is None and data_1h is not None:
                    data_4h = _resample_ohlcv(data_1h, "4h", min_bars=h4_count)

                # Only attempt confluence if we have reasonable data for all three screens
                if data_15m is not None and data_1h is not None and data_4h is not None:
                    mtf_confluence_result = self._mtf_confluence.score_confluence(
                        data_4h=data_4h,
                        data_1h=data_1h,
                        data_15m=data_15m,
                    )

                    if mtf_confluence_result is not None:
                        logger.debug(
                            f"MTF Confluence for {pair}: score={mtf_confluence_result.confluence_score:.3f}, "
                            f"recommendation={mtf_confluence_result.recommendation}, aligned={mtf_confluence_result.direction_aligned}"
                        )

                        trend_direction = str(
                            mtf_confluence_result.details.get("trend_direction", "")
                        )
                        if not trend_direction and mtf_confluence_result.screen_results:
                            trend_direction = str(mtf_confluence_result.screen_results[0].direction)
                        direction_matches = self._mtf_direction_matches(direction, trend_direction)

                        # Map recommendation to confluence_count override
                        if not direction_matches:
                            confluence_count = 0
                            score_override = min(
                                0.40,
                                0.25 + (mtf_confluence_result.confluence_score * 0.20),
                            )
                        elif mtf_confluence_result.recommendation == "PROCEED":
                            # High confidence confluenceence
                            confluence_count = 3
                            score_override = 0.75 + (mtf_confluence_result.confluence_score * 0.20)
                        elif mtf_confluence_result.recommendation == "CAUTION":
                            # Moderate confluence with warning
                            confluence_count = 2
                            score_override = 0.50 + (mtf_confluence_result.confluence_score * 0.25)
                        else:  # REJECT
                            # Poor confluence
                            confluence_count = 0
                            score_override = 0.30 + (mtf_confluence_result.confluence_score * 0.15)

                        data_source = "mtf_confluence"

                        # Use confluence score if valid
                        score = _clip01(score_override)
                        passed = (
                            direction_matches
                            and mtf_confluence_result.recommendation in ("PROCEED", "CAUTION")
                        )
                        confidence_delta = (confluence_count - 1.5) * 0.03

                        if direction_matches:
                            reason = (
                                f"MTF Confluence {mtf_confluence_result.recommendation} "
                                f"(score {mtf_confluence_result.confluence_score:.2f}, aligned={mtf_confluence_result.direction_aligned})"
                            )
                            reason_code = f"mtf_confluence_{mtf_confluence_result.recommendation.lower()}"
                        else:
                            reason = (
                                f"MTF Confluence trend {trend_direction.lower() or 'unknown'} "
                                f"conflicts with {direction.lower()} setup"
                            )
                            reason_code = "mtf_confluence_direction_mismatch"

                        return AgentVerdict(
                            name="multi_timeframe",
                            score=score,
                            passed=passed,
                            weight=self._weight_for("multi_timeframe"),
                            reason=reason,
                            reason_code=reason_code,
                            confidence_delta=confidence_delta,
                            metadata={
                                "confluence_count": confluence_count,
                                "timeframes": 3,
                                "multi_timeframe_data_source": data_source,
                                "mtf_confluence_score": mtf_confluence_result.confluence_score,
                                "mtf_confluence_aligned": mtf_confluence_result.direction_aligned,
                                "mtf_trend_direction": trend_direction,
                                "mtf_direction_matches_trade": direction_matches,
                                "mtf_screen_results": [s.to_dict() for s in mtf_confluence_result.screen_results],
                            },
                        )
            except Exception as e:
                # Fallback to legacy logic on any error
                logger.debug(f"MTF Confluence evaluation failed, falling back to legacy: {e}")
                mtf_confluence_result = None

        # Legacy MTF evaluation (fallback when confluence module unavailable or disabled)
        scanner_granularity = str(getattr(ctx.config, "granularity", "")).upper()
        legacy_h1_df = None
        if scanner_granularity == "H1" and len(df) >= 20:
            legacy_h1_df = df.copy()
        else:
            legacy_h1_df = _resample_ohlcv(df, "1h", min_bars=20)
        if legacy_h1_df is None and len(df) >= 20:
            legacy_h1_df = df.copy()

        # H1 trend (primary): SMA crossover
        if legacy_h1_df is not None and len(legacy_h1_df) >= 20:
            h1_close = float(legacy_h1_df["close"].iloc[-1])
            h1_sma20 = float(legacy_h1_df["close"].iloc[-20:].mean())
            if direction == "LONG" and h1_close > h1_sma20:
                confluence_count += 1
            elif direction == "SHORT" and h1_close < h1_sma20:
                confluence_count += 1

        # H4 trend: try real data first, fallback to calendar-aware OHLCV resample
        h4_df = self._fetch_mtf_candles(pair, "H4", count=60) if pair else None
        if h4_df is not None and len(h4_df) >= 10:
            data_source = "real"
            h4_close = float(h4_df["close"].iloc[-1])
            h4_sma = float(h4_df["close"].iloc[-min(10, len(h4_df)):].mean())
            if direction == "LONG" and h4_close > h4_sma:
                confluence_count += 1
            elif direction == "SHORT" and h4_close < h4_sma:
                confluence_count += 1
        else:
            h4_fallback = _resample_ohlcv(legacy_h1_df, "4h", min_bars=10) if legacy_h1_df is not None else None
            if h4_fallback is not None and len(h4_fallback) >= 10:
                data_source = "resampled"
                h4_last = float(h4_fallback["close"].iloc[-1])
                h4_sma = float(h4_fallback["close"].iloc[-min(10, len(h4_fallback)):].mean())
                if direction == "LONG" and h4_last > h4_sma:
                    confluence_count += 1
                elif direction == "SHORT" and h4_last < h4_sma:
                    confluence_count += 1
            else:
                confluence_count += 1  # Neutral fallback
                logger.debug("multi_timeframe: insufficient data for H4, treating as neutral")

        # D1 trend: try real data first, fallback to calendar-aware OHLCV resample
        d1_df = self._fetch_mtf_candles(pair, "D", count=20) if pair else None
        if d1_df is not None and len(d1_df) >= 5:
            data_source = "real"
            d1_close = float(d1_df["close"].iloc[-1])
            d1_sma = float(d1_df["close"].iloc[-min(5, len(d1_df)):].mean())
            if direction == "LONG" and d1_close > d1_sma:
                confluence_count += 1
            elif direction == "SHORT" and d1_close < d1_sma:
                confluence_count += 1
        else:
            d1_fallback = _resample_ohlcv(legacy_h1_df, "1D", min_bars=5) if legacy_h1_df is not None else None
            if d1_fallback is not None and len(d1_fallback) >= 5:
                data_source = "resampled"
                d1_last = float(d1_fallback["close"].iloc[-1])
                d1_sma = float(d1_fallback["close"].iloc[-min(5, len(d1_fallback)):].mean())
                if direction == "LONG" and d1_last > d1_sma:
                    confluence_count += 1
                elif direction == "SHORT" and d1_last < d1_sma:
                    confluence_count += 1
            else:
                confluence_count += 1
                logger.debug("multi_timeframe: insufficient data for D1, treating as neutral")

        # Score based on confluence
        score = 0.30 + confluence_count * 0.20
        passed = confluence_count >= 2
        confidence_delta = (confluence_count - 1.5) * 0.03

        if confluence_count >= 3:
            reason = f"full MTF confluence ({confluence_count}/3 timeframes agree)"
            reason_code = "mtf_full_confluence"
        elif confluence_count == 2:
            reason = f"partial MTF confluence ({confluence_count}/3)"
            reason_code = "mtf_partial"
        elif confluence_count == 1:
            reason = f"weak MTF support ({confluence_count}/3)"
            reason_code = "mtf_weak"
        else:
            reason = "no MTF confluence (all timeframes disagree)"
            reason_code = "mtf_none"

        return AgentVerdict(
            name="multi_timeframe",
            score=_clip01(score),
            passed=passed,
            weight=self._weight_for("multi_timeframe"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "confluence_count": confluence_count,
                "timeframes": 3,
                "multi_timeframe_data_source": data_source,
            },
        )

    def _evaluate_momentum(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Momentum agent: MACD crossover + rate-of-change assessment.

        Evaluates whether price momentum supports the trade direction using
        MACD signal alignment and short-term rate of change.
        """
        direction = ctx.analysis.direction
        df = ctx.df_feat if not ctx.df_feat.empty else ctx.df_raw

        # MACD components (normalized by price in feature engineering)
        macd = _last_value(df, "macd_norm", 0.0)
        macd_signal = _last_value(df, "macd_signal_norm", 0.0)
        macd_hist = _last_value(df, "macd_hist_norm", macd - macd_signal)

        # Rate of change (5-bar, try roc_5 first then roc)
        roc = _last_value(df, "roc_5", 0.0)
        if abs(roc) < 1e-8:
            roc = _last_value(df, "roc", 0.0)
        if abs(roc) < 1e-8 and "close" in ctx.df_raw.columns and len(ctx.df_raw) >= 6:
            closes = ctx.df_raw["close"]
            prev = float(closes.iloc[-6]) if len(closes) >= 6 else None  # M-3: belt-and-suspenders guard
            if prev is not None and prev != 0:
                roc = (float(closes.iloc[-1]) / prev - 1.0) * 100.0

        # Direction alignment
        if direction == "LONG":
            macd_aligned = macd_hist > 0
            roc_aligned = roc > 0
            macd_strength = _clip01(macd_hist / max(abs(macd) + 1e-6, 1e-6))
        else:
            macd_aligned = macd_hist < 0
            roc_aligned = roc < 0
            macd_strength = _clip01(-macd_hist / max(abs(macd) + 1e-6, 1e-6))

        # Composite score
        score = 0.45
        if macd_aligned:
            score += 0.25
        if roc_aligned:
            score += 0.15
        score += min(macd_strength * 0.15, 0.15)

        if macd_aligned and roc_aligned:
            reason = f"momentum aligned with {direction.lower()} (MACD hist {macd_hist:+.5f})"
            reason_code = "momentum_aligned"
            confidence_delta = 0.03
        elif macd_aligned or roc_aligned:
            reason = f"momentum mixed (MACD {'aligned' if macd_aligned else 'opposed'}, ROC {'aligned' if roc_aligned else 'opposed'})"
            reason_code = "momentum_mixed"
            confidence_delta = 0.0
        else:
            reason = f"momentum opposes {direction.lower()} bias"
            reason_code = "momentum_opposed"
            confidence_delta = -0.04

        return AgentVerdict(
            name="momentum",
            score=_clip01(score),
            passed=score >= 0.52,
            weight=self._weight_for("momentum"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "macd_hist": macd_hist,
                "roc": roc,
                "macd_aligned": macd_aligned,
                "roc_aligned": roc_aligned,
            },
        )

    def _evaluate_session_timing(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Session timing agent: forex session overlap awareness.

        Major pairs trade best during London/NY overlap (13:00-17:00 UTC).
        Scores higher during active sessions for the pair's currencies.
        """
        # Use last candle timestamp if available, otherwise system clock
        now = datetime.now(timezone.utc)
        if not ctx.df_raw.empty and ctx.df_raw.index.dtype.kind == 'M':
            try:
                last_ts = ctx.df_raw.index[-1]
                if hasattr(last_ts, 'to_pydatetime'):
                    now = last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
            except Exception:
                pass  # Fall back to system clock
        hour = now.hour

        # Define forex sessions (UTC hours)
        tokyo_active = 0 <= hour < 9
        london_active = 7 <= hour < 16
        ny_active = 13 <= hour < 22
        overlap_london_ny = 13 <= hour < 16

        pair = ctx.analysis.pair.upper()

        # Map currencies to their primary sessions
        jpy_pair = "JPY" in pair
        eur_gbp_pair = any(c in pair for c in ("EUR", "GBP", "CHF"))
        usd_cad_pair = any(c in pair for c in ("USD", "CAD"))
        aud_nzd_pair = any(c in pair for c in ("AUD", "NZD"))

        # Base score from session relevance
        score = 0.40
        active_sessions = []

        if overlap_london_ny:
            score += 0.35
            active_sessions.append("London/NY overlap")
        elif london_active:
            score += 0.25 if eur_gbp_pair else 0.15
            active_sessions.append("London")
        elif ny_active:
            score += 0.25 if usd_cad_pair else 0.15
            active_sessions.append("New York")
        elif tokyo_active:
            score += 0.25 if (jpy_pair or aud_nzd_pair) else 0.08
            active_sessions.append("Tokyo")
        else:
            active_sessions.append("off-hours")

        # Weekend/low-liquidity penalty
        weekday = now.weekday()
        if weekday >= 5:  # Saturday/Sunday
            score -= 0.25
            active_sessions.append("weekend")

        score = _clip01(score)
        passed = score >= 0.45

        if score >= 0.65:
            reason = f"optimal session timing ({', '.join(active_sessions)})"
            reason_code = "session_optimal"
            confidence_delta = 0.02
        elif passed:
            reason = f"acceptable session ({', '.join(active_sessions)})"
            reason_code = "session_ok"
            confidence_delta = 0.0
        else:
            reason = f"suboptimal session ({', '.join(active_sessions)})"
            reason_code = "session_weak"
            confidence_delta = -0.03

        return AgentVerdict(
            name="session_timing",
            score=score,
            passed=passed,
            weight=self._weight_for("session_timing"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "hour_utc": hour,
                "weekday": weekday,
                "active_sessions": active_sessions,
            },
        )

    def _evaluate_support_resistance(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Support/resistance agent: proximity to key price levels.

        Identifies recent swing highs/lows as S/R levels and checks if
        the current price is near a level that supports or opposes the trade.
        Falls back to simple min/max S/R from available candles if no swings found.
        """
        df = ctx.df_raw
        if "close" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            return None
        if len(df) < 10:
            return None

        close = float(df["close"].iloc[-1])
        highs = df["high"].values[-60:] if len(df) >= 60 else df["high"].values
        lows = df["low"].values[-60:] if len(df) >= 60 else df["low"].values
        atr_pips = max(_safe_float(ctx.analysis.atr_pips), 0.1)

        # Estimate pip value for the pair
        pip_value = 0.0001
        if "JPY" in ctx.analysis.pair.upper():
            pip_value = 0.01

        atr_price = atr_pips * pip_value

        # Find swing highs/lows (simple 5-bar pivot)
        resistance_levels: List[float] = []
        support_levels: List[float] = []

        for i in range(2, len(highs) - 2):
            if highs[i] >= max(highs[i - 2], highs[i - 1], highs[i + 1], highs[i + 2]):
                resistance_levels.append(float(highs[i]))
            if lows[i] <= min(lows[i - 2], lows[i - 1], lows[i + 1], lows[i + 2]):
                support_levels.append(float(lows[i]))

        # FALLBACK: bootstrapping without external data
        # If no swing pivots found, use simple min/max from recent candles
        if not resistance_levels and not support_levels:
            logger.debug("support_resistance: no swing pivots found, using min/max fallback")
            # Last 20 candles for recent levels
            window = min(20, len(highs))
            resistance_levels = [float(max(highs[-window:]))] if window > 0 else []
            support_levels = [float(min(lows[-window:]))] if window > 0 else []

            if not resistance_levels or not support_levels:
                return None

        # Find nearest S/R levels
        nearest_resistance = min(resistance_levels, key=lambda r: abs(r - close)) if resistance_levels else close + atr_price * 5
        nearest_support = min(support_levels, key=lambda s: abs(s - close)) if support_levels else close - atr_price * 5

        dist_to_resistance = (nearest_resistance - close) / atr_price if atr_price > 0 else 5.0
        dist_to_support = (close - nearest_support) / atr_price if atr_price > 0 else 5.0

        direction = ctx.analysis.direction
        score = 0.50

        if direction == "LONG":
            # Good: support nearby (bounce), resistance far (room to run)
            if dist_to_support < 1.5:
                score += 0.20  # Near support - good entry
            if dist_to_resistance > 2.0:
                score += 0.15  # Room to TP
            if dist_to_resistance < 0.5:
                score -= 0.25  # Resistance right above - bad
        else:  # SHORT
            # Good: resistance nearby (rejection), support far (room to run)
            if dist_to_resistance < 1.5:
                score += 0.20  # Near resistance - good entry
            if dist_to_support > 2.0:
                score += 0.15  # Room to TP
            if dist_to_support < 0.5:
                score -= 0.25  # Support right below - bad

        score = _clip01(score)
        passed = score >= 0.50

        if score >= 0.65:
            reason = f"S/R structure supports {direction.lower()} (R:{dist_to_resistance:.1f}x ATR, S:{dist_to_support:.1f}x ATR)"
            reason_code = "sr_support"
            confidence_delta = 0.03
        elif score >= 0.45:
            reason = f"S/R neutral (R:{dist_to_resistance:.1f}x ATR, S:{dist_to_support:.1f}x ATR)"
            reason_code = "sr_neutral"
            confidence_delta = 0.0
        else:
            reason = f"S/R opposes {direction.lower()} (price near {'resistance' if direction == 'LONG' else 'support'})"
            reason_code = "sr_opposed"
            confidence_delta = -0.04

        return AgentVerdict(
            name="support_resistance",
            score=score,
            passed=passed,
            weight=self._weight_for("support_resistance"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "dist_to_resistance_atr": round(dist_to_resistance, 2),
                "dist_to_support_atr": round(dist_to_support, 2),
            },
        )

    def _evaluate_order_flow(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Order flow agent: OANDA order/position book contrarian signal.

        Reads OANDA position book to compute retail long/short ratio,
        then applies contrarian logic: trading against the crowd is positive,
        trading with a heavily crowded side is a warning signal.

        Optionally reads order book to detect nearby order clusters that
        may act as support/resistance from pending flow.

        Always advisory (passed=True) — never blocks a trade.
        """
        from datetime import datetime, timezone

        pair = ctx.analysis.pair
        direction = ctx.analysis.direction
        _CACHE_TTL_SECONDS = 300  # 5-minute cache

        # ── Neutral fallback used on any failure ────────────────────────────
        def _neutral_verdict(reason: str = "order flow data unavailable") -> AgentVerdict:
            return AgentVerdict(
                name="order_flow",
                score=0.55,
                passed=True,
                weight=self._weight_for("order_flow"),
                reason=reason,
                reason_code="order_flow_unavailable",
                confidence_delta=0.0,
                metadata={},
            )

        # ── Lazy-init OANDA client ──────────────────────────────────────────
        if self._oanda_client is None:
            try:
                from src.utils.oanda_practice import OandaPracticeClient
                self._oanda_client = OandaPracticeClient.from_env()
            except Exception as e:
                logger.debug("order_flow: OANDA client init failed: %s", e)
                return _neutral_verdict("OANDA client unavailable")

        # ── Compute current_price once (reused by features + persisters) ─
        now = datetime.now(timezone.utc)
        current_price = _safe_float(
            ctx.analysis.current_price,
            _last_value(ctx.df_raw, "close"),
        )

        # ── Fetch position book (with 5-min cache) ─────────────────────────
        position_buckets = None

        cached = self._position_book_cache.get(pair)
        if cached and (now - cached["timestamp"]).total_seconds() < _CACHE_TTL_SECONDS:
            position_buckets = cached["data"]
        else:
            try:
                resp = self._oanda_client.get_position_book(pair)
                pb = resp.get("positionBook", {}) if isinstance(resp, dict) else {}
                position_buckets = pb.get("buckets")
                if position_buckets is not None:
                    self._position_book_cache[pair] = {
                        "data": position_buckets,
                        "timestamp": now,
                    }
                    # Persist snapshot for future training (best-effort, never blocks).
                    if current_price > 0:
                        try:
                            _ob_snapshot(pair, "position", position_buckets, current_price, timestamp=now)
                        except Exception as _e:  # noqa: BLE001 — persist must never raise
                            logger.debug("order_flow: position book persist failed: %s", _e)
                        # Also write to harvest parquet for bulk training
                        try:
                            from src.data.harvest import HarvestScheduler
                            hs = HarvestScheduler()
                            hs.harvest_position_book(pair)
                        except Exception:
                            pass  # Best-effort
            except Exception as e:
                logger.debug("order_flow: position book fetch failed for %s: %s", pair, e)

        if not position_buckets:
            return _neutral_verdict("position book data unavailable for " + pair)

        # ── Extract graded features from the position book ─────────────────
        # Returns dict of nine pb_* features; fallback-safe on empty input.
        try:
            position_features = _ob_position_features(
                position_buckets,
                current_price if current_price > 0 else 1.0,
            )
        except Exception as e:
            logger.debug("order_flow: position feature extraction failed: %s", e)
            return _neutral_verdict("position book parse error")

        long_ratio = position_features["pb_long_ratio"]
        total_imbalance = position_features["pb_total_imbalance"]
        # No usable counts? Treat as data-empty.
        if position_features["pb_n_buckets"] <= 0 or (
            position_features["pb_near_long_pct"] == 0.0
            and position_features["pb_near_short_pct"] == 0.0
            and total_imbalance == 0.0
            and long_ratio == 0.5
        ):
            # Could be genuinely balanced OR genuinely empty; the existing
            # contract returned _neutral_verdict on empty totals — preserve.
            if position_features["pb_n_buckets"] <= 0:
                return _neutral_verdict("position book empty for " + pair)

        # ── Graded contrarian signal (replaces prior binary 0.70/0.30) ────
        # alignment > 0 = going against the crowd (good contrarian signal);
        # alignment < 0 = going with the crowd (contrarian warning).
        # The old binary cut at long_ratio=0.70 corresponds to
        # |total_imbalance|≈0.40; this graded form preserves that score
        # range at the threshold and degrades smoothly between.
        direction_sign = 1.0 if direction == "LONG" else -1.0
        position_alignment = -direction_sign * total_imbalance

        score = _clip01(0.55 + 0.40 * position_alignment)
        confidence_delta = 0.10 * position_alignment

        crowd_long_pct = long_ratio * 100.0
        if abs(position_alignment) < 0.15:
            reason = "positioning balanced (crowd %.0f%% long)" % crowd_long_pct
            reason_code = "order_flow_neutral"
        elif position_alignment > 0:
            reason = "against crowd (%.0f%% long, going %s, align=%+.2f)" % (
                crowd_long_pct, direction, position_alignment,
            )
            reason_code = "order_flow_contrarian_aligned"
        else:
            reason = "with crowd (%.0f%% long, going %s, align=%+.2f)" % (
                crowd_long_pct, direction, position_alignment,
            )
            reason_code = "order_flow_contrarian_warning"

        # Preserve legacy metadata derivations for downstream consumers.
        crowd_is_long = long_ratio > 0.70
        crowd_is_short = long_ratio < 0.30

        # ── Order book enhancement (optional — reduce score if flow opposes) ─
        order_features: Dict[str, float] = {}  # defined here so metadata access is safe even on hard failure
        try:
            ob_cached = self._order_book_cache.get(pair)
            order_buckets = None
            if ob_cached and (now - ob_cached["timestamp"]).total_seconds() < _CACHE_TTL_SECONDS:
                order_buckets = ob_cached["data"]
            else:
                ob_resp = self._oanda_client.get_order_book(pair)
                ob = ob_resp.get("orderBook", {}) if isinstance(ob_resp, dict) else {}
                order_buckets = ob.get("buckets")
                if order_buckets is not None:
                    self._order_book_cache[pair] = {
                        "data": order_buckets,
                        "timestamp": now,
                    }
                    # Persist snapshot for future training (best-effort).
                    if current_price > 0:
                        try:
                            _ob_snapshot(pair, "order", order_buckets, current_price, timestamp=now)
                        except Exception as _e:  # noqa: BLE001 — persist must never raise
                            logger.debug("order_flow: order book persist failed: %s", _e)

            if order_buckets and current_price > 0:
                # ── Graded near-price flow penalty (replaces 2.0x binary) ─
                # ob_near_imbalance ∈ [-1, 1]: positive = heavy buy orders
                # near current price; negative = heavy sell orders near.
                # near_alignment > 0 = near-price flow agrees with direction;
                # near_alignment < 0 = opposing near-price flow (penalty).
                # The old binary cut (nearby_short > 2× nearby_long AND > 1.0)
                # corresponds to near_imbalance ≈ -0.33; this graded form
                # produces a comparable penalty at that threshold and degrades
                # smoothly above/below.
                try:
                    order_features = _ob_order_features(order_buckets, current_price)
                    near_imb = order_features["ob_near_imbalance"]
                    near_alignment = -direction_sign * near_imb
                    if near_alignment < -0.20:
                        # Opposing near-price flow; magnitude-scaled penalty.
                        penalty = 0.15 * min(abs(near_alignment), 1.0)
                        score = _clip01(score - penalty)
                        confidence_delta -= 0.05 * min(abs(near_alignment), 1.0)
                        side_word = "sell" if direction == "LONG" else "buy"
                        reason += " | heavy %s flow nearby (align=%+.2f)" % (side_word, near_alignment)
                except Exception as e:
                    logger.debug("order_flow: order feature extraction failed: %s", e)
                    order_features = {}
            else:
                order_features = {}
        except Exception as e:
            logger.debug("order_flow: order book enhancement failed for %s: %s", pair, e)
            # Order book is optional — position book signal is sufficient

        score = _clip01(score)

        return AgentVerdict(
            name="order_flow",
            score=score,
            passed=True,
            weight=self._weight_for("order_flow"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "long_ratio": round(long_ratio, 3),
                "short_ratio": round(1.0 - long_ratio, 3),
                "crowd_direction": "LONG" if long_ratio > 0.5 else "SHORT",
                "contrarian_warning": crowd_is_long or crowd_is_short,
                "institutional_risk": "HIGH" if (long_ratio > 0.75 or long_ratio < 0.25) else "LOW",
                # Graded features (added 2026-05-13 — order-book-features wiring):
                "position_alignment": round(position_alignment, 3),
                "total_imbalance": round(total_imbalance, 3),
                "near_imbalance": round(position_features.get("pb_near_imbalance", 0.0), 3),
                "top5_long_conc": round(position_features.get("pb_top5_long_concentration", 0.0), 3),
                "top5_short_conc": round(position_features.get("pb_top5_short_concentration", 0.0), 3),
                "ob_near_imbalance": round(order_features.get("ob_near_imbalance", 0.0), 3),
                "ob_total_imbalance": round(order_features.get("ob_total_imbalance", 0.0), 3),
            },
        )

    def _evaluate_pair_performance(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Check historical performance for this pair and adjust confidence.

        Reads ``trained_data/models/pair_performance.json`` (updated by
        :meth:`ExecutionManager._update_pair_performance`) and emits a verdict
        based on the pair's historical win rate and P/L.

        Returns a slightly cautious verdict when no history exists (bootstrap mode).
        """
        import json
        from pathlib import Path

        perf_path = Path("trained_data/models/pair_performance.json")
        pair = ctx.analysis.pair

        # Try to load performance data
        data = {}
        if perf_path.exists():
            try:
                data = json.loads(perf_path.read_text())
            except Exception as e:
                logger.debug(f"pair_performance: failed to load data ({e}), using bootstrap verdict")

        stats = data.get(pair) if data else None
        if not stats or stats.get("trades", 0) < 3:
            # FALLBACK: bootstrapping without external data
            # No history yet — return slightly cautious verdict to incentivize building history
            logger.debug(f"pair_performance: no history for {pair}, using bootstrap verdict")
            return AgentVerdict(
                name="pair_performance",
                score=0.45,  # Slightly below neutral
                passed=True,
                weight=self._weight_for("pair_performance"),
                reason=f"no trading history for {pair} yet (bootstrap)",
                reason_code="pair_no_history",
                confidence_delta=-0.02,
                metadata={
                    "trades_history": 0,
                    "recommendation": "trade_small_to_build_history",
                },
            )

        win_rate = _safe_float(stats.get("win_rate"), 0.5)
        total_pnl_pips = _safe_float(stats.get("total_pnl_pips"), 0.0)
        trades = int(stats.get("trades", 0))

        # Score: base 0.50, boost for good win rate, penalize for bad
        score = 0.50 + (win_rate - 0.50) * 0.60
        score = _clip01(score)

        # Confidence delta: up to +/- 5% based on historical edge
        confidence_delta = (win_rate - 0.50) * 0.10  # e.g., 60% WR -> +1%, 40% -> -1%
        confidence_delta = max(-0.05, min(0.05, confidence_delta))

        passed = win_rate >= 0.45

        if win_rate >= 0.60:
            reason = f"pair has strong history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_strong"
        elif win_rate >= 0.45:
            reason = f"pair has neutral history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_neutral"
        else:
            reason = f"pair has weak history ({win_rate:.0%} WR, {total_pnl_pips:+.0f} pips over {trades} trades)"
            reason_code = "pair_perf_weak"

        return AgentVerdict(
            name="pair_performance",
            score=score,
            passed=passed,
            weight=self._weight_for("pair_performance"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            metadata={
                "win_rate": win_rate,
                "total_pnl_pips": total_pnl_pips,
                "trades_history": trades,
            },
        )

    # ------------------------------------------------------------------
    # Agent #13: Trader Readiness (Aura → Buddy bridge signal)
    # ------------------------------------------------------------------

    def _evaluate_trader_readiness(self, ctx: AgentDecisionContext) -> Optional[AgentVerdict]:
        """Evaluate trader's cognitive/emotional readiness from Aura's signal.

        Reads the readiness signal JSON written by Aura's ReadinessComputer.
        This is the primary bridge point where human intelligence modulates
        market intelligence.

        DORMANT (2026-05-12 audit, Bucket A): No live writer for
        ``.aura/bridge/readiness_signal.json`` exists in this repo —
        ``BuddyAuraBridge.read_readiness`` has no ``write_readiness``
        counterpart, and the on-disk file is stale. Default toggle flipped
        OFF (``enable_trader_readiness_agent=False`` in ScannerConfig + all
        four profile dicts). This method's None-fallback path handles the
        missing-file case gracefully, so the agent does NOT silently
        neutral-vote on every trade — it simply abstains.
        Re-enable once Aura ships a writer for the signal file.

        Score mapping:
            readiness 80-100 → score 0.80-0.95 (full capacity)
            readiness 60-79  → score 0.60-0.79 (reduced, -20% position)
            readiness 40-59  → score 0.40-0.59 (significantly reduced)
            readiness 20-39  → score 0.20-0.39 (minimum positions)
            readiness 0-19   → score 0.05-0.19 (block trade)

        Returns None if no readiness signal file exists (Aura not running).
        """
        import json
        from pathlib import Path

        signal_path = Path(self._READINESS_SIGNAL_PATH)

        # If no signal file exists, Aura isn't running — skip gracefully
        if not signal_path.exists():
            logger.debug("trader_readiness: no signal file, Aura not active — skipping")
            return None

        try:
            data = json.loads(signal_path.read_text())
        except Exception as e:
            logger.warning(f"trader_readiness: failed to read signal ({e})")
            return None

        # Extract signal components
        readiness_score = _safe_float(data.get("readiness_score"), 70.0)
        cognitive_load = data.get("cognitive_load", "low")
        override_loss_rate = _safe_float(data.get("override_loss_rate_7d"), 0.0)
        emotional_state = data.get("emotional_state", "neutral")
        active_stressors = data.get("active_stressors", [])

        # --- Compute agent score from readiness (0-1 scale) ---
        score = readiness_score / 100.0

        # Apply override penalty: high override loss rate further reduces score
        if override_loss_rate > 0.5:
            score *= 0.7
            logger.info(
                f"trader_readiness: override penalty applied "
                f"(loss rate {override_loss_rate:.0%})"
            )
        elif override_loss_rate > 0.3:
            score *= 0.85

        # Cognitive load penalty
        if cognitive_load == "high":
            score *= 0.8
        elif cognitive_load == "medium":
            score *= 0.9

        score = _clip01(score)

        # --- Determine verdict ---
        block_trade = score < 0.2
        passed = score > 0.4

        # Confidence delta: readiness affects overall confidence
        if score >= 0.8:
            confidence_delta = 0.02
        elif score >= 0.6:
            confidence_delta = 0.0
        elif score >= 0.4:
            confidence_delta = -0.03
        else:
            confidence_delta = -0.06

        # Build reason string
        if block_trade:
            reason = (
                f"TRADER NOT READY — readiness {readiness_score:.0f}/100, "
                f"state: {emotional_state}, load: {cognitive_load}"
            )
            reason_code = "readiness_block"
        elif score < 0.4:
            stressor_text = f", stressors: {', '.join(active_stressors[:2])}" if active_stressors else ""
            reason = (
                f"low readiness {readiness_score:.0f}/100 "
                f"({emotional_state}, {cognitive_load} load{stressor_text})"
            )
            reason_code = "readiness_low"
        elif score < 0.7:
            reason = (
                f"moderate readiness {readiness_score:.0f}/100 "
                f"({emotional_state}) — reduced sizing recommended"
            )
            reason_code = "readiness_moderate"
        else:
            reason = f"readiness {readiness_score:.0f}/100 — trader in good cognitive state"
            reason_code = "readiness_good"

        return AgentVerdict(
            name="trader_readiness",
            score=score,
            passed=passed,
            weight=self._weight_for("trader_readiness"),
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "readiness_score": readiness_score,
                "cognitive_load": cognitive_load,
                "emotional_state": emotional_state,
                "override_loss_rate_7d": override_loss_rate,
                "active_stressors": active_stressors,
                "signal_timestamp": data.get("timestamp", ""),
            },
        )

    def cache_trade_verdicts(self, episode_id: str, verdicts: list) -> None:
        """Cache agent verdicts for later neural bridge retrieval."""
        if not episode_id:
            return
        self._trade_verdict_cache[episode_id] = {"verdicts": list(verdicts)}

    def record_trade_outcome(
        self, episode_id: str, outcome: str, pnl_pips: float
    ) -> bool:
        """Record trade outcome to episodic memory and neural agents.

        Called after a trade closes to update episodic memory with outcome.
        When neural agents are enabled, also pushes experience tuples into
        each agent's replay buffer for gradient-based learning.

        Args:
            episode_id: UUID returned from record_setup (stored during entry)
            outcome: Trade result ("WIN", "LOSS", or "BREAKEVEN")
            pnl_pips: Realized P/L in pips

        Returns:
            True if outcome was recorded, False if episode not found or episodic memory disabled
        """
        recorded = False
        if self._episodic_memory is not None:
            try:
                recorded = self._episodic_memory.record_outcome(
                    episode_id=episode_id,
                    outcome=outcome,
                    pnl_pips=float(pnl_pips),
                )
            except Exception as e:
                logger.warning(f"Failed to record trade outcome to episodic memory: {e}")

        # ── Bridge to neural agents ────────────────────────────────────────
        # If neural agents are active, push this trade's experience into
        # each agent's prioritized replay buffer for online learning.
        _neural_enabled = getattr(self.config, "use_neural_agents", False)
        if _neural_enabled and hasattr(self, "_neural_team") and self._neural_team is not None:
            try:
                self._bridge_outcome_to_neural(episode_id, outcome, pnl_pips)
            except Exception as e:
                logger.debug(f"Neural agent outcome bridge failed: {e}")

        # Evict to prevent unbounded growth
        self._trade_verdict_cache.pop(episode_id, None)

        return recorded

    def _bridge_outcome_to_neural(
        self, episode_id: str, outcome: str, pnl_pips: float
    ) -> None:
        """Push trade outcome into each neural agent's replay buffer.

        Requires the episode to have cached verdicts (written during scan).
        If no cached verdicts, silently returns.
        """
        if self._neural_team is None:
            return

        # Retrieve cached verdicts for this episode
        # (assumes the scan cycle cached them in a trade-entry buffer)
        cached = getattr(self, "_trade_verdict_cache", {}).get(episode_id)
        if cached is None:
            return

        trade_won = outcome == "WIN"
        for vd in cached.get("verdicts", []):
            agent_name = vd.get("name")
            agent = self._neural_team.agents.get(agent_name)
            if agent is None:
                continue
            try:
                agent.record_outcome(vd, trade_won)
            except Exception as e:
                logger.debug(f"Neural agent {agent_name} outcome recording failed: {e}")

        # Persist trainer state periodically
        if getattr(self, "_neural_update_counter", 0) % 50 == 0:
            try:
                self._neural_team.trainer.save_policies()
            except Exception as e:
                logger.debug(f"Neural agent periodic save failed: {e}")
        self._neural_update_counter = getattr(self, "_neural_update_counter", 0) + 1


# ==========================================================================
# US-076: Graph-Attention Heterogeneous Agent Consensus
# ==========================================================================

# Agent specialization categories — each maps to a "node type" in the
# attention graph, allowing correlation-aware weighting.
AGENT_SPECIALIZATIONS: Dict[str, str] = {
    "trend": "momentum",
    "momentum": "momentum",
    "mean_reversion": "momentum",
    "volatility": "regime",
    "risk_sentinel": "risk",
    "uncertainty": "risk",
    "execution_quality": "execution",
    "news_risk": "regime",
    "multi_timeframe": "momentum",
    "pair_performance": "execution",
    "trader_readiness": "risk",  # Agent #13: human-side risk signal
    "session_timing": "execution",
    "support_resistance": "momentum",
}


class GraphAttentionConsensus:
    """Graph-attention mechanism for heterogeneous agent consensus.

    Replaces equal-weight voting with correlation-aware attention:
    - Agents in the same specialization cluster share attention
    - Agents whose signals align with asset dependency structure
      get higher weight
    - Reduces double-counting correlated agent opinions

    The attention mechanism is a simplified single-head graph attention:
        attention(i) = softmax( score_i * alignment_i / sqrt(d) )
    where alignment_i measures how well agent i's specialization
    correlates with the current market state.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_attention: float = 0.05,
    ):
        self.temperature = max(temperature, 0.01)
        self.min_attention = min_attention

    def reweight_verdicts(
        self,
        verdicts: List[AgentVerdict],
        correlation_matrix: Optional[Dict[str, float]] = None,
        regime: str = "NORMAL",
    ) -> List[AgentVerdict]:
        """Apply graph-attention re-weighting to agent verdicts.

        Args:
            verdicts: Agent verdicts with scores and weights
            correlation_matrix: Optional pair correlation data
                (maps "pair1_pair2" -> correlation coefficient)
            regime: Current volatility regime

        Returns:
            Verdicts with attention-adjusted weights
        """
        if len(verdicts) <= 1:
            return verdicts

        # Step 1: Compute specialization alignment scores
        alignment_scores = self._compute_alignment_scores(verdicts, regime)

        # Step 2: Compute attention weights via softmax
        attention_weights = self._softmax_attention(
            verdicts, alignment_scores
        )

        # Step 3: Apply attention weights to verdict weights
        for i, verdict in enumerate(verdicts):
            old_weight = verdict.weight
            # Blend: 60% attention, 40% original weight
            verdict.weight = round(
                0.6 * attention_weights[i] * len(verdicts) * old_weight
                + 0.4 * old_weight,
                4,
            )
            verdict.weight = max(verdict.weight, self.min_attention)

        logger.debug(
            "GraphAttention: %d agents re-weighted (regime=%s, temp=%.2f)",
            len(verdicts), regime, self.temperature,
        )

        return verdicts

    def _compute_alignment_scores(
        self,
        verdicts: List[AgentVerdict],
        regime: str,
    ) -> List[float]:
        """Compute per-agent alignment scores based on specialization and regime.

        Higher scores for agents whose specialization is relevant to the
        current market regime.
        """
        # Regime-specialization alignment map
        regime_alignment: Dict[str, Dict[str, float]] = {
            "LOW": {"momentum": 0.6, "regime": 0.8, "risk": 1.0, "execution": 0.9},
            "NORMAL": {"momentum": 1.0, "regime": 1.0, "risk": 1.0, "execution": 1.0},
            "HIGH": {"momentum": 1.2, "regime": 1.3, "risk": 1.1, "execution": 0.8},
            "EXTREME": {"momentum": 0.8, "regime": 1.4, "risk": 1.5, "execution": 0.6},
        }

        current_alignment = regime_alignment.get(
            regime, regime_alignment["NORMAL"]
        )

        scores = []
        for v in verdicts:
            spec = AGENT_SPECIALIZATIONS.get(v.name, "momentum")
            base_alignment = current_alignment.get(spec, 1.0)
            # Modulate by agent's own confidence (score)
            score = base_alignment * (0.5 + 0.5 * _clip01(v.score))
            scores.append(score)

        return scores

    def _softmax_attention(
        self,
        verdicts: List[AgentVerdict],
        alignment_scores: List[float],
    ) -> List[float]:
        """Compute softmax attention weights.

        attention_i = exp(alignment_i / temperature) / sum(exp(...))
        """
        scaled = [s / self.temperature for s in alignment_scores]
        # Numerical stability: subtract max
        max_s = max(scaled) if scaled else 0.0
        exp_scores = [math.exp(s - max_s) for s in scaled]
        total = sum(exp_scores) or 1.0

        weights = [e / total for e in exp_scores]

        # Floor at min_attention
        weights = [max(w, self.min_attention) for w in weights]
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]

        return weights

    def compute_agent_graph_score(
        self,
        verdicts: List[AgentVerdict],
    ) -> Dict[str, Any]:
        """Compute graph-level consensus metrics.

        Returns:
            Dict with consensus_entropy, specialization_agreement,
            dominant_cluster, etc.
        """
        if not verdicts:
            return {"consensus_entropy": 0.0, "dominant_cluster": "none"}

        # Group verdicts by specialization
        clusters: Dict[str, List[float]] = {}
        for v in verdicts:
            spec = AGENT_SPECIALIZATIONS.get(v.name, "momentum")
            clusters.setdefault(spec, []).append(_clip01(v.score))

        # Cluster-level average scores (H-8: guard against empty cluster lists)
        cluster_scores = {
            k: sum(v) / len(v) if v else 0.0 for k, v in clusters.items()
        }

        # Dominant cluster
        dominant = max(cluster_scores, key=cluster_scores.get) if cluster_scores else "none"

        # Consensus entropy (lower = more agreement)
        all_scores = [_clip01(v.score) for v in verdicts]
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.5
        variance = sum((s - mean_score) ** 2 for s in all_scores) / max(len(all_scores), 1)
        entropy = math.sqrt(variance)

        return {
            "consensus_entropy": round(entropy, 4),
            "dominant_cluster": dominant,
            "cluster_scores": {k: round(v, 4) for k, v in cluster_scores.items()},
            "agent_count": len(verdicts),
        }
