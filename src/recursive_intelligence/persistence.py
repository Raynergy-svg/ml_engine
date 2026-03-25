"""Domain-agnostic session persistence engine.

Extracted from Buddy's StateEngine (src/scanner/automation/state_engine.py).
Generalizes cross-session state management.

Buddy instantiation:
    state_schema = {goal, status, done, next, portfolio_snapshot, ...}
    snapshot_strategy = "on_trade_close"

Aura instantiation:
    state_schema = {emotional_state, readiness_score, active_patterns, last_conversation, ...}
    snapshot_strategy = "on_conversation_end"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SessionPersistence:
    """Manages cross-session state for any domain.

    Args:
        domain: Name of the domain ("market", "human", "bridge")
        state_path: Where to persist state
        default_state: Template for a fresh state
    """

    def __init__(
        self,
        domain: str,
        state_path: Optional[Path] = None,
        default_state: Optional[Dict[str, Any]] = None,
    ):
        self.domain = domain
        self.state_path = state_path or Path(f".aura/state_{domain}.json")
        self._default_state = default_state or {
            "domain": domain,
            "status": "ready",
            "last_updated": "",
            "session_count": 0,
        }

    def load(self) -> Dict[str, Any]:
        """Load state from disk, merging with defaults for missing keys."""
        if not self.state_path.exists():
            return dict(self._default_state)
        try:
            data = json.loads(self.state_path.read_text())
            # Merge with defaults for any missing keys
            merged = dict(self._default_state)
            merged.update(data)
            return merged
        except Exception as e:
            logger.warning(f"[{self.domain}] Failed to load state: {e}")
            return dict(self._default_state)

    def save(self, state: Dict[str, Any]) -> None:
        """Persist state to disk with timestamp."""
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        state["domain"] = self.domain

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.state_path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.error(f"[{self.domain}] Failed to save state: {e}")

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        """Load state, update specific fields, save back."""
        state = self.load()
        state.update(kwargs)
        self.save(state)
        return state

    def increment_session(self) -> Dict[str, Any]:
        """Increment session counter and update timestamp."""
        state = self.load()
        state["session_count"] = state.get("session_count", 0) + 1
        self.save(state)
        return state
