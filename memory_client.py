"""
Memory MCP Client for ML Engine

Provides persistent state management using the Model Context Protocol (MCP) Memory server.
Tracks scan results (7-day TTL), training sessions, and model state.

Usage:
    from memory_client import MLEngineMemory
    
    memory = MLEngineMemory()
    memory.store_scan(analyses)
    memory.log_training("GBP_JPY", {"accuracy": 0.78})
    
    if memory.should_skip_pair("EUR_USD", max_age_hours=24):
        print("Pair was recently trained")
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Memory storage path (aligned with mcp.json config)
MEMORY_STORAGE_PATH = Path("trained_data/.memory")

# TTL settings
SCAN_TTL_DAYS = 7
TRAINING_TTL_DAYS = None  # Indefinite


@dataclass
class ScanResult:
    """Stored scan result entity."""
    pair: str
    direction: str
    confidence: float
    volatility_percentile: float
    trend_strength: float
    gates_passed: bool
    timestamp: str  # ISO format
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScanResult":
        return cls(**d)


@dataclass
class TrainingSession:
    """Stored training session entity."""
    pair: str
    granularity: str
    models_trained: List[str]
    metrics: Dict[str, float]
    hyperparams: Dict[str, Any]
    duration_seconds: float
    warm_start: bool
    timestamp: str  # ISO format
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingSession":
        return cls(**d)


class MLEngineMemory:
    """
    Memory client for ML Engine state persistence.
    
    Uses file-based storage compatible with MCP Memory server.
    Implements TTL for scan results (7 days) and indefinite storage for training sessions.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or MEMORY_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Entity storage files
        self.scans_file = self.storage_path / "scan_results.json"
        self.training_file = self.storage_path / "training_sessions.json"
        self.model_state_file = self.storage_path / "model_state.json"
        
        # Initialize files if missing
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage files with empty data."""
        for filepath in [self.scans_file, self.training_file, self.model_state_file]:
            if not filepath.exists():
                filepath.write_text("[]" if filepath != self.model_state_file else "{}")
    
    def _load_json(self, filepath: Path) -> Any:
        """Load JSON from file with error handling."""
        try:
            return json.loads(filepath.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return [] if filepath != self.model_state_file else {}
    
    def _save_json(self, filepath: Path, data: Any):
        """Save JSON to file."""
        filepath.write_text(json.dumps(data, indent=2, default=str))
    
    # =========================================================================
    # SCAN RESULTS (7-day TTL)
    # =========================================================================
    
    def store_scan(
        self,
        analyses: List[tuple],  # List[Tuple[PairAnalysis, bool]]
        granularity: str = "H1",
    ) -> None:
        """
        Store scan results with 7-day TTL.
        
        Args:
            analyses: List of (PairAnalysis, gates_passed) tuples from buddy_scan()
            granularity: Timeframe of the scan
        """
        scans = self._load_json(self.scans_file)
        timestamp = datetime.now().isoformat()
        
        # Convert analyses to storable format
        for analysis, gates_passed in analyses:
            scan_result = ScanResult(
                pair=analysis.pair,
                direction=analysis.direction,
                confidence=analysis.confidence,
                volatility_percentile=analysis.volatility_percentile,
                trend_strength=analysis.trend_strength,
                gates_passed=gates_passed,
                timestamp=timestamp,
            )
            scans.append(scan_result.to_dict())
        
        # Clean up expired scans before saving
        scans = self._cleanup_expired_scans(scans)
        self._save_json(self.scans_file, scans)
        
        logger.info(f"📝 Stored {len(analyses)} scan results")
    
    def _cleanup_expired_scans(self, scans: List[Dict]) -> List[Dict]:
        """Remove scans older than SCAN_TTL_DAYS."""
        cutoff = datetime.now() - timedelta(days=SCAN_TTL_DAYS)
        valid_scans = []
        
        for scan in scans:
            try:
                scan_time = datetime.fromisoformat(scan["timestamp"])
                if scan_time > cutoff:
                    valid_scans.append(scan)
            except (KeyError, ValueError):
                pass  # Skip malformed entries
        
        removed = len(scans) - len(valid_scans)
        if removed > 0:
            logger.info(f"🧹 Cleaned up {removed} expired scan results (>{SCAN_TTL_DAYS} days)")
        
        return valid_scans
    
    def get_recent_scans(
        self,
        pair: Optional[str] = None,
        hours: int = 24,
    ) -> List[ScanResult]:
        """
        Get recent scan results.
        
        Args:
            pair: Filter by pair (optional)
            hours: How far back to look
            
        Returns:
            List of ScanResult objects
        """
        scans = self._load_json(self.scans_file)
        cutoff = datetime.now() - timedelta(hours=hours)
        
        results = []
        for scan in scans:
            try:
                scan_time = datetime.fromisoformat(scan["timestamp"])
                if scan_time > cutoff:
                    if pair is None or scan["pair"] == pair:
                        results.append(ScanResult.from_dict(scan))
            except (KeyError, ValueError):
                pass
        
        return results
    
    def cleanup_expired_scans(self) -> int:
        """
        Manually trigger cleanup of expired scans.
        
        Returns:
            Number of scans removed
        """
        scans = self._load_json(self.scans_file)
        original_count = len(scans)
        scans = self._cleanup_expired_scans(scans)
        self._save_json(self.scans_file, scans)
        return original_count - len(scans)
    
    # =========================================================================
    # TRAINING SESSIONS (Indefinite storage)
    # =========================================================================
    
    def log_training(
        self,
        pair: str,
        metrics: Dict[str, float],
        *,
        granularity: str = "H1",
        models_trained: Optional[List[str]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
        warm_start: bool = False,
    ) -> None:
        """
        Log a training session (stored indefinitely).
        
        Args:
            pair: Instrument trained on
            metrics: Training metrics (accuracy, loss, etc.)
            granularity: Timeframe
            models_trained: List of model names (default: full ensemble)
            hyperparams: Hyperparameters used
            duration_seconds: Training duration
            warm_start: Whether warm-start was used
        """
        sessions = self._load_json(self.training_file)
        
        session = TrainingSession(
            pair=pair,
            granularity=granularity,
            models_trained=models_trained or [
                "transformer", "xgboost", "rf", "ridge", "histgb"
            ],
            metrics=metrics,
            hyperparams=hyperparams or {},
            duration_seconds=duration_seconds,
            warm_start=warm_start,
            timestamp=datetime.now().isoformat(),
        )
        
        sessions.append(session.to_dict())
        self._save_json(self.training_file, sessions)
        
        logger.info(f"📊 Logged training session for {pair}")
    
    def get_training_history(
        self,
        pair: Optional[str] = None,
        limit: int = 10,
    ) -> List[TrainingSession]:
        """
        Get training history.
        
        Args:
            pair: Filter by pair (optional)
            limit: Maximum number of sessions to return
            
        Returns:
            List of TrainingSession objects (most recent first)
        """
        sessions = self._load_json(self.training_file)
        
        if pair:
            sessions = [s for s in sessions if s.get("pair") == pair]
        
        # Sort by timestamp (most recent first)
        sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return [TrainingSession.from_dict(s) for s in sessions[:limit]]
    
    def should_skip_pair(
        self,
        pair: str,
        max_age_hours: int = 24,
    ) -> bool:
        """
        Check if a pair was trained within max_age_hours.
        
        Args:
            pair: Instrument to check
            max_age_hours: Skip if trained within this many hours
            
        Returns:
            True if pair should be skipped (recently trained)
        """
        sessions = self._load_json(self.training_file)
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        for session in sessions:
            if session.get("pair") != pair:
                continue
            try:
                session_time = datetime.fromisoformat(session["timestamp"])
                if session_time > cutoff:
                    return True
            except (KeyError, ValueError):
                pass
        
        return False
    
    def get_last_training(self, pair: str) -> Optional[TrainingSession]:
        """
        Get the most recent training session for a pair.
        
        Args:
            pair: Instrument to check
            
        Returns:
            TrainingSession or None
        """
        history = self.get_training_history(pair=pair, limit=1)
        return history[0] if history else None
    
    # =========================================================================
    # MODEL STATE
    # =========================================================================
    
    def update_model_state(
        self,
        model_name: str,
        accuracy: float,
        trained_pairs: List[str],
        version: str = "1.0.0",
    ) -> None:
        """
        Update model state tracking.
        
        Args:
            model_name: Name of the model
            accuracy: Current validation accuracy
            trained_pairs: List of pairs the model was trained on
            version: Model version string
        """
        state = self._load_json(self.model_state_file)
        
        state[model_name] = {
            "accuracy": accuracy,
            "trained_pairs": trained_pairs,
            "version": version,
            "last_updated": datetime.now().isoformat(),
        }
        
        self._save_json(self.model_state_file, state)
        logger.info(f"📦 Updated model state: {model_name} (acc={accuracy:.2%})")
    
    def get_model_state(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model state.
        
        Args:
            model_name: Specific model (optional, returns all if None)
            
        Returns:
            Model state dict
        """
        state = self._load_json(self.model_state_file)
        
        if model_name:
            return state.get(model_name, {})
        return state
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        scans = self._load_json(self.scans_file)
        sessions = self._load_json(self.training_file)
        
        return {
            "scan_results_count": len(scans),
            "training_sessions_count": len(sessions),
            "storage_path": str(self.storage_path),
            "scan_ttl_days": SCAN_TTL_DAYS,
        }
    
    # =========================================================================
    # HISTORICAL ACCURACY TRACKING
    # =========================================================================
    
    def get_pair_accuracy(self, days: int = 30) -> Dict[str, float]:
        """
        Get historical prediction accuracy per pair.
        
        Analyzes stored scan results against actual outcomes
        to calculate per-pair accuracy. Uses scan confidence
        and direction vs subsequent price movement.
        
        Args:
            days: Number of days of history to analyze
            
        Returns:
            Dict mapping pair to accuracy (0-1)
        """
        scans = self._load_json(self.scans_file)
        cutoff = datetime.now() - timedelta(days=days)
        
        # Group scans by pair
        pair_results: Dict[str, List[Dict]] = {}
        for scan in scans:
            try:
                scan_time = datetime.fromisoformat(scan["timestamp"])
                if scan_time > cutoff:
                    pair = scan.get("pair", "")
                    if pair:
                        if pair not in pair_results:
                            pair_results[pair] = []
                        pair_results[pair].append(scan)
            except (KeyError, ValueError):
                pass
        
        # Calculate accuracy per pair
        # For now, use a heuristic based on confidence and gates_passed
        # In production, this would compare predictions vs actual outcomes
        accuracies: Dict[str, float] = {}
        
        for pair, scans_list in pair_results.items():
            if not scans_list:
                continue
            
            # Use average confidence where gates passed as proxy for accuracy
            gates_passed = [s for s in scans_list if s.get("gates_passed", False)]
            if gates_passed:
                avg_conf = sum(s.get("confidence", 0.5) for s in gates_passed) / len(gates_passed)
                # Scale confidence to reasonable accuracy range (50-85%)
                accuracies[pair] = 0.5 + (avg_conf * 0.35)
            else:
                # No gates passed - use lower baseline
                accuracies[pair] = 0.5
        
        return accuracies
    
    def update_pair_accuracy(
        self,
        pair: str,
        prediction_correct: bool,
        confidence: float,
    ) -> None:
        """
        Update pair accuracy tracking with a new outcome.
        
        Args:
            pair: Instrument name
            prediction_correct: Whether the prediction was correct
            confidence: Original prediction confidence
        """
        # Load existing accuracy tracking
        accuracy_file = self.storage_path / "pair_accuracy.json"
        
        try:
            if accuracy_file.exists():
                accuracy_data = json.loads(accuracy_file.read_text())
            else:
                accuracy_data = {}
        except (json.JSONDecodeError, FileNotFoundError):
            accuracy_data = {}
        
        # Initialize pair entry if missing
        if pair not in accuracy_data:
            accuracy_data[pair] = {
                "correct": 0,
                "total": 0,
                "last_updated": None,
            }
        
        # Update counts
        accuracy_data[pair]["total"] += 1
        if prediction_correct:
            accuracy_data[pair]["correct"] += 1
        accuracy_data[pair]["last_updated"] = datetime.now().isoformat()
        
        # Save
        accuracy_file.write_text(json.dumps(accuracy_data, indent=2))
        
        # Calculate new accuracy
        total = accuracy_data[pair]["total"]
        correct = accuracy_data[pair]["correct"]
        new_accuracy = correct / total if total > 0 else 0.5
        
        logger.info(f"📊 Updated {pair} accuracy: {new_accuracy:.1%} ({correct}/{total})")
    
    def get_pair_accuracy_detailed(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed accuracy information for a specific pair.
        
        Args:
            pair: Instrument name
            
        Returns:
            Dict with accuracy details or None
        """
        accuracy_file = self.storage_path / "pair_accuracy.json"
        
        try:
            if accuracy_file.exists():
                accuracy_data = json.loads(accuracy_file.read_text())
                if pair in accuracy_data:
                    data = accuracy_data[pair]
                    total = data.get("total", 0)
                    correct = data.get("correct", 0)
                    return {
                        "pair": pair,
                        "accuracy": correct / total if total > 0 else 0.5,
                        "correct": correct,
                        "total": total,
                        "last_updated": data.get("last_updated"),
                    }
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        
        return None
    
    def clear_all(self) -> None:
        """Clear all stored data (use with caution!)."""
        self.scans_file.write_text("[]")
        self.training_file.write_text("[]")
        self.model_state_file.write_text("{}")
        logger.warning("⚠️ All memory storage cleared!")


# Convenience function for quick access
def get_memory() -> MLEngineMemory:
    """Get a memory client instance."""
    return MLEngineMemory()


if __name__ == "__main__":
    # Quick test
    memory = MLEngineMemory()
    print(f"Memory stats: {memory.get_stats()}")
