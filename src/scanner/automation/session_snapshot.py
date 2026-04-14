"""Cross-session learning persistence via session snapshots.

Saves a structured snapshot at the end of each scan session (watch loop exit)
containing agent weights, pair accuracies, regime distribution, and trade stats.
Enables ConfigTuner to compare across sessions and blend parameters.

Usage:
    saver = SessionSnapshotManager()
    saver.save_snapshot(agent_weights, pair_accuracies, regime_counts, trade_stats)
    recent = saver.load_recent_snapshots(n=5)
    blended = saver.blend_agent_weights(current_weights)
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SNAPSHOT_DIR = _PROJECT_ROOT / "trained_data" / "session_snapshots"


@dataclass
class SessionSnapshot:
    """A single session's summary data."""

    timestamp: str = ""
    agent_weights: Dict[str, float] = field(default_factory=dict)
    pair_accuracies: Dict[str, float] = field(default_factory=dict)
    regime_counts: Dict[str, int] = field(default_factory=dict)
    trade_count: int = 0
    win_rate: float = 0.0
    avg_rr_ratio: float = 0.0
    scan_cycles: int = 0
    profile: str = "balanced"
    # Control plane health summaries (optional, for post-session review)
    control_plane_summary: Optional[Dict[str, Any]] = None
    queue_summary: Optional[Dict[str, Any]] = None
    policy_summary: Optional[Dict[str, Any]] = None


class SessionSnapshotManager:
    """Manages session snapshot persistence and cross-session blending."""

    def __init__(self, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else _SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(
        self,
        agent_weights: Dict[str, float],
        pair_accuracies: Optional[Dict[str, float]] = None,
        regime_counts: Optional[Dict[str, int]] = None,
        trade_count: int = 0,
        win_rate: float = 0.0,
        avg_rr_ratio: float = 0.0,
        scan_cycles: int = 0,
        profile: str = "balanced",
    ) -> Optional[Path]:
        """Save a session snapshot to disk.

        Args:
            agent_weights: Current agent weight dict {agent_name: weight}
            pair_accuracies: Per-pair accuracy dict {pair: accuracy}
            regime_counts: Regime observation counts {regime: count}
            trade_count: Total trades this session
            win_rate: Session win rate (0-1)
            avg_rr_ratio: Average risk/reward ratio
            scan_cycles: Number of scan cycles completed
            profile: Scanner profile used

        Returns:
            Path to saved snapshot, or None on failure
        """
        now = datetime.now(timezone.utc)
        snapshot = SessionSnapshot(
            timestamp=now.isoformat(),
            agent_weights=dict(agent_weights or {}),
            pair_accuracies=dict(pair_accuracies or {}),
            regime_counts=dict(regime_counts or {}),
            trade_count=trade_count,
            win_rate=win_rate,
            avg_rr_ratio=avg_rr_ratio,
            scan_cycles=scan_cycles,
            profile=profile,
        )

        filename = now.strftime("%Y%m%d_%H%M%S") + ".json"
        path = self.snapshot_dir / filename

        try:
            from src.scanner.automation.safe_json import safe_json_write
            success = safe_json_write(path, asdict(snapshot))
        except ImportError:
            import json
            try:
                path.write_text(json.dumps(asdict(snapshot), indent=2, default=str))
                success = True
            except Exception as e:
                logger.error(f"Failed to save session snapshot: {e}")
                success = False

        if success:
            logger.info(f"Session snapshot saved: {filename} ({trade_count} trades, {scan_cycles} cycles)")
            return path
        return None

    def load_recent_snapshots(self, n: int = 5) -> List[SessionSnapshot]:
        """Load the N most recent session snapshots.

        Args:
            n: Number of recent snapshots to load

        Returns:
            List of SessionSnapshot objects, most recent first
        """
        if not self.snapshot_dir.exists():
            return []

        # Sort by filename (timestamp-based) descending
        files = sorted(self.snapshot_dir.glob("*.json"), reverse=True)[:n]

        snapshots: List[SessionSnapshot] = []
        for f in files:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(f, default=None)
            except ImportError:
                import json
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    data = None

            if data and isinstance(data, dict):
                try:
                    snapshots.append(SessionSnapshot(
                        timestamp=data.get("timestamp", ""),
                        agent_weights=data.get("agent_weights", {}),
                        pair_accuracies=data.get("pair_accuracies", {}),
                        regime_counts=data.get("regime_counts", {}),
                        trade_count=int(data.get("trade_count", 0)),
                        win_rate=float(data.get("win_rate", 0.0)),
                        avg_rr_ratio=float(data.get("avg_rr_ratio", 0.0)),
                        scan_cycles=int(data.get("scan_cycles", 0)),
                        profile=data.get("profile", "balanced"),
                        control_plane_summary=data.get("control_plane_summary"),
                        queue_summary=data.get("queue_summary"),
                        policy_summary=data.get("policy_summary"),
                    ))
                except Exception as e:
                    logger.debug(f"Skipping malformed snapshot {f.name}: {e}")

        return snapshots

    def blend_agent_weights(
        self,
        current_weights: Dict[str, float],
        current_weight_factor: float = 0.7,
        n_sessions: int = 5,
    ) -> Dict[str, float]:
        """Blend current agent weights with historical session averages.

        Uses exponential moving average: new = factor * current + (1-factor) * historical_avg

        Args:
            current_weights: Current agent weights
            current_weight_factor: Weight given to current session (0-1)
            n_sessions: Number of historical sessions to average

        Returns:
            Blended weight dict. Returns current_weights unchanged if no history.
        """
        snapshots = self.load_recent_snapshots(n=n_sessions)
        if not snapshots:
            return dict(current_weights)

        # Average historical weights
        all_agents = set(current_weights.keys())
        for s in snapshots:
            all_agents.update(s.agent_weights.keys())

        historical_avg: Dict[str, float] = {}
        for agent in all_agents:
            values = [s.agent_weights.get(agent, 1.0) for s in snapshots if agent in s.agent_weights]
            if values:
                historical_avg[agent] = sum(values) / len(values)
            else:
                historical_avg[agent] = current_weights.get(agent, 1.0)

        # Blend
        hist_factor = 1.0 - current_weight_factor
        blended: Dict[str, float] = {}
        for agent in all_agents:
            curr = current_weights.get(agent, 1.0)
            hist = historical_avg.get(agent, 1.0)
            blended[agent] = round(current_weight_factor * curr + hist_factor * hist, 4)

        changed = sum(1 for a in blended if abs(blended[a] - current_weights.get(a, 1.0)) > 0.001)
        if changed > 0:
            logger.info(
                f"Cross-session weight blending: {changed} agents adjusted "
                f"(factor={current_weight_factor}, history={len(snapshots)} sessions)"
            )

        return blended

    def get_status(self) -> Dict[str, Any]:
        """Get snapshot manager status."""
        files = sorted(self.snapshot_dir.glob("*.json"), reverse=True) if self.snapshot_dir.exists() else []
        return {
            "snapshot_count": len(files),
            "latest": files[0].name if files else None,
            "snapshot_dir": str(self.snapshot_dir),
        }
