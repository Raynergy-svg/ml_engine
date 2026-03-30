"""
Trade Cluster Analyzer — Phase 54 (US-336).

Clusters completed trades to identify high-win and low-win condition combinations.
Uses DBSCAN for density-based clustering and DecisionTree for interpretable rules.

Outputs:
  - Cluster registry: {cluster_id: {win_rate, count, conditions, confidence_adjustment}}
  - High-win clusters (win_rate > 0.50, count >= 20): +0.05 confidence boost
  - Low-win clusters (win_rate < 0.30, count >= 20): -0.10 confidence penalty

Re-clusters every 100 new trades (not every sync).

Usage:
    analyzer = TradeClusterAnalyzer()
    analyzer.fit(trades)
    registry = analyzer.get_cluster_registry()
    match = analyzer.classify_trade(trade_features)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/trade_clusters.json"

# Feature columns for clustering
_CLUSTER_FEATURES = [
    "session",        # 0=asian, 1=london, 2=newyork, 3=overlap
    "regime",         # 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    "agent_agreement",  # weighted_vote_score
    "trend_strength",
    "atr_level",      # normalized ATR
    "pair_encoded",   # integer-encoded pair
]

_SESSION_MAP = {"asian": 0, "london": 1, "newyork": 2, "overlap": 3}
_REGIME_MAP = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}


@dataclass
class ClusterInfo:
    """Metadata for a single cluster."""
    cluster_id: int = -1
    win_rate: float = 0.0
    count: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    confidence_adjustment: float = 0.0
    centroid: List[float] = field(default_factory=list)
    label: str = "neutral"  # "high_win", "low_win", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "win_rate": round(self.win_rate, 4),
            "count": self.count,
            "conditions": self.conditions,
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "centroid": [round(c, 6) for c in self.centroid],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClusterInfo":
        return cls(
            cluster_id=int(d.get("cluster_id", -1)),
            win_rate=float(d.get("win_rate", 0.0)),
            count=int(d.get("count", 0)),
            conditions=d.get("conditions", {}),
            confidence_adjustment=float(d.get("confidence_adjustment", 0.0)),
            centroid=[float(c) for c in d.get("centroid", [])],
            label=str(d.get("label", "neutral")),
        )


@dataclass
class TradeClusterConfig:
    """Configuration for trade cluster analyzer."""
    eps: float = 0.5                    # DBSCAN eps parameter
    min_samples: int = 15               # DBSCAN min_samples
    tree_max_depth: int = 4             # DecisionTree max_depth
    tree_min_samples_leaf: int = 20     # DecisionTree min_samples_leaf
    high_win_threshold: float = 0.50    # Win rate above this → high-win
    low_win_threshold: float = 0.30     # Win rate below this → low-win
    min_cluster_size: int = 20          # Minimum trades per cluster
    high_win_boost: float = 0.05        # Confidence boost for high-win clusters
    low_win_penalty: float = -0.10      # Confidence penalty for low-win clusters
    recluster_interval: int = 100       # Re-cluster every N new trades
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    pair_list: List[str] = field(default_factory=lambda: [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
        "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY",
        "USD_CHF", "EUR_AUD",
    ])


class TradeClusterAnalyzer:
    """Clusters trades to find high/low win-rate condition combinations."""

    def __init__(self, config: Optional[TradeClusterConfig] = None):
        self.config = config or TradeClusterConfig()
        self._registry: Dict[int, ClusterInfo] = {}
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._tree_rules: List[str] = []
        self._trades_since_cluster: int = 0
        self._last_fit_count: int = 0
        self._load_state()

    def extract_features(self, trade: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from a trade entry.

        Returns None if required features are missing.
        """
        try:
            # Session
            session = str(trade.get("session", "")).lower()
            session_val = _SESSION_MAP.get(session, 1)  # default NORMAL

            # Regime
            regime_raw = trade.get("regime", {})
            if isinstance(regime_raw, dict):
                regime_str = str(regime_raw.get("volatility_regime", "NORMAL"))
            else:
                regime_str = str(regime_raw)
            regime_val = _REGIME_MAP.get(regime_str, 1)

            # Agent agreement
            agents = trade.get("agents", {})
            agent_agreement = float(agents.get("weighted_vote_score", trade.get("weighted_vote_score", 0.5)))

            # Trend strength
            trend_strength = float(trade.get("trend_strength", agents.get("trend_strength", 0.5)))

            # ATR level (normalize to common range)
            atr = float(trade.get("atr", 0.001))
            atr_level = min(atr * 10000, 100)  # Normalize to ~0-100 range

            # Pair encoding
            pair = str(trade.get("pair", ""))
            pair_list = self.config.pair_list
            pair_val = pair_list.index(pair) if pair in pair_list else len(pair_list)

            return [session_val, regime_val, agent_agreement, trend_strength, atr_level, pair_val]
        except Exception as e:
            logger.debug(f"Phase 54 (US-336): Feature extraction failed: {e}")
            return None

    def fit(self, trades: List[Dict[str, Any]]) -> Dict[int, ClusterInfo]:
        """Cluster trades and build registry.

        Args:
            trades: List of trade entries with outcomes

        Returns:
            Cluster registry {cluster_id: ClusterInfo}
        """
        # Extract features
        features = []
        outcomes = []
        for t in trades:
            f = self.extract_features(t)
            if f is None:
                continue
            outcome = t.get("outcome", {})
            if not isinstance(outcome, dict):
                continue
            won = outcome.get("trade_won", False)
            features.append(f)
            outcomes.append(1 if won else 0)

        if len(features) < self.config.min_samples * 2:
            logger.info("Phase 54 (US-336): Insufficient trades for clustering (%d)", len(features))
            return self._registry

        X = np.array(features, dtype=np.float64)
        y = np.array(outcomes, dtype=np.int32)

        # Standardize features
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0)
        self._scaler_std[self._scaler_std < 1e-8] = 1.0  # Prevent division by zero
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        # DBSCAN clustering
        try:
            from sklearn.cluster import DBSCAN
            dbscan = DBSCAN(eps=self.config.eps, min_samples=self.config.min_samples)
            labels = dbscan.fit_predict(X_scaled)
        except ImportError:
            logger.warning("Phase 54 (US-336): sklearn not available, using simple binning")
            labels = self._simple_cluster(X_scaled)

        # Build cluster registry
        self._registry = {}
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for cid in unique_labels:
            mask = labels == cid
            cluster_y = y[mask]
            cluster_X = X[mask]
            count = int(mask.sum())

            if count < self.config.min_cluster_size:
                continue

            win_rate = float(cluster_y.mean())
            centroid = cluster_X.mean(axis=0).tolist()

            # Determine label and adjustment
            if win_rate > self.config.high_win_threshold:
                label = "high_win"
                adjustment = self.config.high_win_boost
            elif win_rate < self.config.low_win_threshold:
                label = "low_win"
                adjustment = self.config.low_win_penalty
            else:
                label = "neutral"
                adjustment = 0.0

            # Extract conditions from centroid
            conditions = self._centroid_to_conditions(centroid)

            self._registry[int(cid)] = ClusterInfo(
                cluster_id=int(cid),
                win_rate=win_rate,
                count=count,
                conditions=conditions,
                confidence_adjustment=adjustment,
                centroid=centroid,
                label=label,
            )

        # Fit decision tree for interpretable rules
        self._fit_decision_tree(X, y)

        self._last_fit_count = len(features)
        self._trades_since_cluster = 0

        n_high = sum(1 for c in self._registry.values() if c.label == "high_win")
        n_low = sum(1 for c in self._registry.values() if c.label == "low_win")
        logger.info(
            "Phase 54 (US-336): Clustered %d trades → %d clusters "
            "(%d high-win, %d low-win, %d neutral)",
            len(features), len(self._registry),
            n_high, n_low, len(self._registry) - n_high - n_low,
        )

        return self._registry

    def classify_trade(self, trade: Dict[str, Any]) -> Optional[ClusterInfo]:
        """Classify a trade against the cluster registry using nearest centroid.

        Returns ClusterInfo of the nearest cluster, or None if no clusters.
        """
        if not self._registry or self._scaler_mean is None:
            return None

        features = self.extract_features(trade)
        if features is None:
            return None

        X = np.array(features, dtype=np.float64)
        X_scaled = (X - self._scaler_mean) / self._scaler_std

        # Find nearest centroid
        best_cluster = None
        best_dist = float("inf")
        for cid, cluster in self._registry.items():
            if not cluster.centroid:
                continue
            centroid_scaled = (np.array(cluster.centroid) - self._scaler_mean) / self._scaler_std
            dist = float(np.linalg.norm(X_scaled - centroid_scaled))
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster

        return best_cluster

    def should_recluster(self) -> bool:
        """Whether enough new trades have accumulated for re-clustering."""
        return self._trades_since_cluster >= self.config.recluster_interval

    def record_trade(self) -> None:
        """Record that a new trade was synced (for recluster timing)."""
        self._trades_since_cluster += 1

    def get_cluster_registry(self) -> Dict[int, Dict[str, Any]]:
        """Get serializable cluster registry."""
        return {cid: c.to_dict() for cid, c in self._registry.items()}

    def get_tree_rules(self) -> List[str]:
        """Get interpretable rules from DecisionTree."""
        return list(self._tree_rules)

    # ── Internal methods ──

    def _simple_cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """Fallback clustering without sklearn — simple grid binning."""
        # Bin each dimension into 3 quantiles and assign cluster IDs
        n_dims = X_scaled.shape[1]
        labels = np.zeros(X_scaled.shape[0], dtype=np.int32)
        for d in range(n_dims):
            q33 = np.percentile(X_scaled[:, d], 33)
            q66 = np.percentile(X_scaled[:, d], 66)
            bins = np.digitize(X_scaled[:, d], [q33, q66])
            labels = labels * 3 + bins
        return labels

    def _centroid_to_conditions(self, centroid: List[float]) -> Dict[str, Any]:
        """Convert centroid values to readable conditions."""
        if len(centroid) < 6:
            return {}
        session_names = {0: "asian", 1: "london", 2: "newyork", 3: "overlap"}
        regime_names = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}
        return {
            "session": session_names.get(round(centroid[0]), "unknown"),
            "regime": regime_names.get(round(centroid[1]), "NORMAL"),
            "agent_agreement": round(centroid[2], 3),
            "trend_strength": round(centroid[3], 3),
            "atr_level": round(centroid[4], 2),
            "pair_index": round(centroid[5]),
        }

    def _fit_decision_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit shallow decision tree for interpretable rules."""
        self._tree_rules = []
        try:
            from sklearn.tree import DecisionTreeClassifier, export_text
            tree = DecisionTreeClassifier(
                max_depth=self.config.tree_max_depth,
                min_samples_leaf=self.config.tree_min_samples_leaf,
            )
            tree.fit(X, y)
            rules = export_text(tree, feature_names=_CLUSTER_FEATURES)
            self._tree_rules = [line for line in rules.split("\n") if line.strip()]
            logger.info(
                "Phase 54 (US-336): DecisionTree fit — %d rules, depth=%d",
                len(self._tree_rules), tree.get_depth(),
            )
        except ImportError:
            logger.debug("Phase 54 (US-336): sklearn not available for DecisionTree")
        except Exception as e:
            logger.debug(f"Phase 54 (US-336): DecisionTree fit error: {e}")

    def save_state(self) -> None:
        """Persist cluster registry with retry (US-342)."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "trades_since_cluster": self._trades_since_cluster,
                "last_fit_count": self._last_fit_count,
                "registry": {str(k): v.to_dict() for k, v in self._registry.items()},
                "scaler_mean": self._scaler_mean.tolist() if self._scaler_mean is not None else None,
                "scaler_std": self._scaler_std.tolist() if self._scaler_std is not None else None,
                "tree_rules": self._tree_rules,
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, data, context="trade_clusters")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 54 (US-336): Failed to save cluster state: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._trades_since_cluster = int(data.get("trades_since_cluster", 0))
                self._last_fit_count = int(data.get("last_fit_count", 0))

                registry = data.get("registry", {})
                if isinstance(registry, dict):
                    self._registry = {
                        int(k): ClusterInfo.from_dict(v)
                        for k, v in registry.items()
                        if isinstance(v, dict)
                    }

                mean = data.get("scaler_mean")
                if mean is not None:
                    self._scaler_mean = np.array(mean, dtype=np.float64)
                std = data.get("scaler_std")
                if std is not None:
                    self._scaler_std = np.array(std, dtype=np.float64)

                self._tree_rules = data.get("tree_rules", [])

                logger.info(
                    "Phase 54 (US-336): Loaded cluster analyzer — "
                    "%d clusters, %d trades since last fit",
                    len(self._registry), self._trades_since_cluster,
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 54 (US-336): Could not load cluster state: %s", e)
