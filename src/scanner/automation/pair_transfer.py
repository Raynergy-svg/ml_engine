"""Transfer learning across FX pair families.

US-101: Share learned feature representations between correlated FX pairs
to accelerate learning on low-data pairs. Group pairs into families
(USD-majors, crosses, JPY-pairs) by correlation. Transfer feature importance
weights and agent biases from high-data to low-data family members.

Based on 2024-2025 multi-task learning for financial time series.

Transfer approach:
1. Build pair families from correlation matrix (threshold-based clustering)
2. Within a family, high-data pairs are "teachers" for low-data pairs
3. Transfer blends feature weights: target = (1-ratio)*target + ratio*source
4. Confidence boost when family knowledge supports the signal
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TRANSFER_PATH = _PROJECT_ROOT / "trained_data" / "pair_transfer.json"

# Default pair families when no correlation matrix is available
DEFAULT_FAMILIES: Dict[str, List[str]] = {
    "usd_majors": ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD"],
    "jpy_pairs": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"],
    "chf_pairs": ["USD_CHF", "EUR_CHF", "GBP_CHF"],
    "cad_pairs": ["USD_CAD", "EUR_CAD", "GBP_CAD"],
    "crosses": ["EUR_GBP", "AUD_NZD", "EUR_AUD"],
}

# Transfer config
CORRELATION_THRESHOLD = 0.65  # Min |correlation| to group pairs
TRANSFER_RATIO = 0.30  # Blend ratio: how much to transfer from source
MIN_SOURCE_SAMPLES = 20  # Source pair needs this many samples to be a teacher
CONFIDENCE_BOOST_MAX = 0.10  # Maximum confidence boost from family knowledge
DECAY_FACTOR = 0.95  # Decay old transfer weights each cycle


@dataclass
class PairStats:
    """Statistics for a pair within its family."""

    pair: str = ""
    sample_count: int = 0
    avg_accuracy: float = 0.5
    feature_weights: Dict[str, float] = field(default_factory=dict)
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "sample_count": self.sample_count,
            "avg_accuracy": round(self.avg_accuracy, 4),
            "feature_weights": {
                k: round(v, 6) for k, v in self.feature_weights.items()
            },
            "last_updated": self.last_updated,
        }


class PairTransferLearning:
    """Transfer learned representations between correlated FX pairs.

    Groups pairs into families by correlation. High-data pairs within a
    family act as teachers for low-data pairs, transferring feature weights
    and providing confidence boosts.

    Usage:
        transfer = PairTransferLearning()
        transfer.update_pair_stats("EUR_USD", sample_count=100, accuracy=0.62,
                                    feature_weights={"momentum": 0.8, "trend": 0.6})
        boost = transfer.get_transfer_boost("AUD_USD", "usd_majors")
        blended = transfer.transfer_weights("EUR_USD", "AUD_USD", ratio=0.3)
    """

    def __init__(
        self,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        transfer_ratio: float = TRANSFER_RATIO,
        persistence_path: Optional[Path] = None,
    ):
        self.correlation_threshold = correlation_threshold
        self.transfer_ratio = transfer_ratio
        self.persistence_path = persistence_path or _TRANSFER_PATH

        # Family mappings: family_name → set of pairs
        self._families: Dict[str, Set[str]] = {}

        # Per-pair statistics
        self._pair_stats: Dict[str, PairStats] = {}

        # Pair → family reverse lookup
        self._pair_to_family: Dict[str, str] = {}

        # Transfer history for observability
        self._transfer_log: List[Dict[str, Any]] = []
        self._total_transfers: int = 0

        self._load_state()

        # Initialize default families if none loaded
        if not self._families:
            self._init_default_families()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_pair_families(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        threshold: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Group pairs into families based on correlation matrix.

        Uses single-linkage clustering: if any pair in a family has
        |correlation| >= threshold with a new pair, it joins that family.

        Args:
            correlation_matrix: {pair: {pair: correlation}} dict.
            threshold: Override correlation threshold.

        Returns:
            Dict mapping family_name → list of pairs.
        """
        _threshold = threshold or self.correlation_threshold
        pairs = list(correlation_matrix.keys())

        # Union-Find for clustering
        parent: Dict[str, str] = {p: p for p in pairs}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Cluster pairs by correlation
        for i, p1 in enumerate(pairs):
            for p2 in pairs[i + 1:]:
                corr = correlation_matrix.get(p1, {}).get(p2, 0.0)
                if abs(corr) >= _threshold:
                    union(p1, p2)

        # Build family groups
        clusters: Dict[str, List[str]] = defaultdict(list)
        for p in pairs:
            root = find(p)
            clusters[root].append(p)

        # Name families by dominant currency
        self._families.clear()
        self._pair_to_family.clear()
        family_idx = 0

        for _root, members in clusters.items():
            if len(members) < 2:
                continue  # Singles don't form families

            # Detect dominant currency
            currencies: Dict[str, int] = defaultdict(int)
            for pair in members:
                parts = pair.split("_")
                for c in parts:
                    currencies[c] += 1

            dominant = max(currencies, key=lambda c: currencies[c])
            family_name = f"{dominant.lower()}_family_{family_idx}"
            family_idx += 1

            self._families[family_name] = set(members)
            for p in members:
                self._pair_to_family[p] = family_name

        logger.info(
            f"PairTransfer: Built {len(self._families)} families "
            f"from {len(pairs)} pairs (threshold={_threshold})"
        )

        return {k: sorted(v) for k, v in self._families.items()}

    def update_pair_stats(
        self,
        pair: str,
        sample_count: int,
        accuracy: float,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update statistics for a pair.

        Args:
            pair: Currency pair name.
            sample_count: Total samples available.
            accuracy: Current accuracy for this pair.
            feature_weights: Feature importance weights.
        """
        stats = self._pair_stats.get(pair, PairStats(pair=pair))
        stats.sample_count = sample_count
        stats.avg_accuracy = accuracy
        if feature_weights:
            stats.feature_weights = dict(feature_weights)
        stats.last_updated = datetime.now(timezone.utc).isoformat()
        self._pair_stats[pair] = stats

    def transfer_weights(
        self,
        source_pair: str,
        target_pair: str,
        transfer_ratio: Optional[float] = None,
    ) -> Dict[str, float]:
        """Blend feature weights from source to target pair.

        Formula: target_w = (1 - ratio) * target_w + ratio * source_w

        Args:
            source_pair: High-data pair to learn from.
            target_pair: Low-data pair to transfer to.
            transfer_ratio: Blend ratio (0 = no transfer, 1 = full copy).

        Returns:
            Blended feature weights for target pair.
        """
        ratio = transfer_ratio or self.transfer_ratio

        source_stats = self._pair_stats.get(source_pair)
        target_stats = self._pair_stats.get(target_pair, PairStats(pair=target_pair))

        if not source_stats or not source_stats.feature_weights:
            return dict(target_stats.feature_weights)

        if source_stats.sample_count < MIN_SOURCE_SAMPLES:
            logger.debug(
                f"PairTransfer: {source_pair} has insufficient samples "
                f"({source_stats.sample_count} < {MIN_SOURCE_SAMPLES})"
            )
            return dict(target_stats.feature_weights)

        # Collect all feature keys
        all_features = set(source_stats.feature_weights.keys())
        all_features.update(target_stats.feature_weights.keys())

        blended: Dict[str, float] = {}
        for feat in all_features:
            src_w = source_stats.feature_weights.get(feat, 0.5)
            tgt_w = target_stats.feature_weights.get(feat, 0.5)
            blended[feat] = (1.0 - ratio) * tgt_w + ratio * src_w

        # Update target stats with blended weights
        target_stats.feature_weights = blended
        target_stats.last_updated = datetime.now(timezone.utc).isoformat()
        self._pair_stats[target_pair] = target_stats

        # Log transfer
        self._total_transfers += 1
        self._transfer_log.append({
            "source": source_pair,
            "target": target_pair,
            "ratio": ratio,
            "features_transferred": len(blended),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._transfer_log) > 200:
            self._transfer_log = self._transfer_log[-100:]

        logger.debug(
            f"PairTransfer: {source_pair}→{target_pair} "
            f"(ratio={ratio}, features={len(blended)})"
        )

        return blended

    def get_transfer_boost(
        self,
        pair: str,
        family: Optional[str] = None,
    ) -> float:
        """Get confidence boost from family knowledge.

        Computes a boost based on how well other family members are performing.
        If the family has high average accuracy, the target pair gets a small
        confidence boost (capped at CONFIDENCE_BOOST_MAX).

        Args:
            pair: Target pair.
            family: Family name (auto-detected if None).

        Returns:
            Confidence boost value (0.0 to CONFIDENCE_BOOST_MAX).
        """
        family_name = family or self._pair_to_family.get(pair)
        if not family_name or family_name not in self._families:
            return 0.0

        family_members = self._families[family_name]
        if len(family_members) < 2:
            return 0.0

        # Compute family accuracy excluding target pair
        accuracies = []
        for member in family_members:
            if member == pair:
                continue
            stats = self._pair_stats.get(member)
            if stats and stats.sample_count >= MIN_SOURCE_SAMPLES:
                accuracies.append(stats.avg_accuracy)

        if not accuracies:
            return 0.0

        family_accuracy = float(np.mean(accuracies))

        # Boost proportional to family accuracy above 0.50 (random baseline)
        if family_accuracy <= 0.50:
            return 0.0

        excess = family_accuracy - 0.50
        boost = min(CONFIDENCE_BOOST_MAX, excess * 0.20)

        return round(boost, 4)

    def get_best_teacher(
        self,
        target_pair: str,
    ) -> Optional[Tuple[str, float, int]]:
        """Find the best teacher pair for a target.

        Best teacher = highest accuracy in the same family with
        sufficient samples.

        Args:
            target_pair: Pair that needs knowledge transfer.

        Returns:
            Tuple of (teacher_pair, accuracy, sample_count) or None.
        """
        family_name = self._pair_to_family.get(target_pair)
        if not family_name:
            return None

        best: Optional[Tuple[str, float, int]] = None
        for member in self._families.get(family_name, set()):
            if member == target_pair:
                continue
            stats = self._pair_stats.get(member)
            if not stats or stats.sample_count < MIN_SOURCE_SAMPLES:
                continue
            if best is None or stats.avg_accuracy > best[1]:
                best = (member, stats.avg_accuracy, stats.sample_count)

        return best

    def auto_transfer(self) -> int:
        """Run automatic transfer for all low-data pairs.

        For each pair with fewer samples than MIN_SOURCE_SAMPLES,
        find the best teacher in its family and transfer weights.

        Returns:
            Number of transfers performed.
        """
        transfers = 0
        for pair, stats in list(self._pair_stats.items()):
            if stats.sample_count >= MIN_SOURCE_SAMPLES:
                continue  # Not a low-data pair

            teacher = self.get_best_teacher(pair)
            if teacher is None:
                continue

            teacher_pair, _acc, _count = teacher
            self.transfer_weights(teacher_pair, pair)
            transfers += 1

        if transfers > 0:
            logger.info(f"PairTransfer: Auto-transferred to {transfers} low-data pairs")

        return transfers

    def get_status(self) -> Dict[str, Any]:
        """Get transfer learning status for observability."""
        family_summary = {}
        for fname, members in self._families.items():
            family_summary[fname] = {
                "pairs": sorted(members),
                "n_pairs": len(members),
            }

        return {
            "n_families": len(self._families),
            "n_pairs_tracked": len(self._pair_stats),
            "total_transfers": self._total_transfers,
            "families": family_summary,
            "recent_transfers": self._transfer_log[-5:],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist transfer learning state."""
        try:
            data = {
                "version": 1,
                "families": {k: sorted(v) for k, v in self._families.items()},
                "pair_stats": {
                    k: v.to_dict() for k, v in self._pair_stats.items()
                },
                "total_transfers": self._total_transfers,
                "transfer_log": self._transfer_log[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"PairTransfer: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            # Restore families
            for fname, members in data.get("families", {}).items():
                self._families[fname] = set(members)
                for p in members:
                    self._pair_to_family[p] = fname

            # Restore pair stats
            for pair, stats_dict in data.get("pair_stats", {}).items():
                self._pair_stats[pair] = PairStats(
                    pair=stats_dict.get("pair", pair),
                    sample_count=stats_dict.get("sample_count", 0),
                    avg_accuracy=stats_dict.get("avg_accuracy", 0.5),
                    feature_weights=stats_dict.get("feature_weights", {}),
                    last_updated=stats_dict.get("last_updated", ""),
                )

            self._total_transfers = data.get("total_transfers", 0)
            self._transfer_log = data.get("transfer_log", [])

            logger.debug(
                f"PairTransfer: Loaded {len(self._families)} families, "
                f"{len(self._pair_stats)} pairs"
            )
        except Exception as e:
            logger.debug(f"PairTransfer: Failed to load: {e}")

    def _init_default_families(self) -> None:
        """Initialize with hardcoded FX pair families."""
        for fname, members in DEFAULT_FAMILIES.items():
            self._families[fname] = set(members)
            for p in members:
                self._pair_to_family[p] = fname
        logger.debug(
            f"PairTransfer: Initialized {len(self._families)} default families"
        )
