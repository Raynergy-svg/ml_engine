"""Memory manager with LRU cache and log rotation.

US-115: Prevent unbounded growth in watch mode. Provides LRU eviction
for in-memory caches, log rotation for append-only files, and memory
pressure monitoring.

Approach:
1. Register dict caches with max_size → evict oldest entries when full
2. Rotate large log files (observations.jsonl, scan_cycle_log.jsonl)
3. Monitor RSS memory usage, alert when above threshold
4. Run periodically (every N scan cycles) to keep system healthy
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "memory_manager_log.json"
_ARCHIVE_DIR = _PROJECT_ROOT / "trained_data" / "archive"

# Memory thresholds
RSS_WARNING_MB = 500   # Warn when RSS exceeds this
RSS_CRITICAL_MB = 1000  # Critical alert above this

# Log rotation defaults
DEFAULT_MAX_LINES = 10000
DEFAULT_KEEP_LINES = 5000  # Keep this many lines after rotation


class ManagedCache:
    """LRU-evicting dict wrapper."""

    def __init__(self, name: str, cache_dict: dict, max_size: int = 100):
        self.name = name
        self._cache = cache_dict
        self.max_size = max_size
        # OrderedDict gives O(1) move_to_end + popitem(last=False) vs O(n) list.remove
        self._access_order: OrderedDict = OrderedDict.fromkeys(cache_dict.keys())
        self.evictions: int = 0

    def touch(self, key: str) -> None:
        """Mark key as recently accessed — O(1) via OrderedDict.move_to_end."""
        if key in self._access_order:
            self._access_order.move_to_end(key)
        else:
            self._access_order[key] = None

    def evict_if_needed(self) -> int:
        """Evict oldest entries if cache exceeds max_size. Returns count evicted."""
        evicted = 0
        while len(self._cache) > self.max_size and self._access_order:
            oldest_key, _ = self._access_order.popitem(last=False)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                evicted += 1

        self.evictions += evicted
        return evicted

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_evictions": self.evictions,
            "utilization": round(len(self._cache) / max(self.max_size, 1), 4),
        }


class MemoryManager:
    """Manages memory pressure, cache eviction, and log rotation.

    Prevents unbounded growth in long-running watch mode by:
    - LRU-evicting oversized in-memory caches
    - Rotating large log files with archival
    - Monitoring RSS memory and alerting on pressure

    Usage:
        manager = MemoryManager()
        manager.register_cache("pair_returns", engine._pair_returns, max_size=50)
        manager.register_log("/path/to/observations.jsonl", max_lines=10000)
        manager.run_cleanup()  # Call every N scan cycles
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # Managed caches
        self._caches: Dict[str, ManagedCache] = {}

        # Managed log files
        self._log_files: Dict[str, Dict[str, Any]] = {}

        # Cleanup history — deque(maxlen) gives O(1) append with automatic eviction,
        # replacing the previous List + manual slicing that created a full copy each trim.
        self._cleanup_log: deque = deque(maxlen=50)
        self._total_cleanups: int = 0
        self._total_evictions: int = 0
        self._total_rotations: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_cache(
        self, name: str, cache_dict: dict, max_size: int = 100
    ) -> None:
        """Register a dict-like cache for LRU management.

        Args:
            name: Identifier for this cache.
            cache_dict: Reference to the actual dict (mutated in-place).
            max_size: Maximum number of entries before eviction.
        """
        self._caches[name] = ManagedCache(name, cache_dict, max_size)

    def register_log(
        self,
        path: str,
        max_lines: int = DEFAULT_MAX_LINES,
        keep_lines: int = DEFAULT_KEEP_LINES,
    ) -> None:
        """Register a log file for rotation management.

        Args:
            path: Path to the log file.
            max_lines: Rotate when file exceeds this many lines.
            keep_lines: Keep this many recent lines after rotation.
        """
        self._log_files[path] = {
            "max_lines": max_lines,
            "keep_lines": keep_lines,
            "rotations": 0,
        }

    def run_cleanup(self) -> Dict[str, Any]:
        """Run all cleanup tasks: evict caches, rotate logs.

        Returns:
            Cleanup report with evictions, rotations, memory stats.
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evictions": {},
            "rotations": {},
            "memory": self.get_memory_report(),
        }

        # 1. Evict oversized caches
        for name, cache in self._caches.items():
            evicted = cache.evict_if_needed()
            if evicted > 0:
                report["evictions"][name] = evicted
                self._total_evictions += evicted

        # 2. Rotate oversized log files
        for path, config in self._log_files.items():
            rotated = self._rotate_log(
                Path(path), config["max_lines"], config["keep_lines"]
            )
            if rotated:
                report["rotations"][path] = True
                config["rotations"] += 1
                self._total_rotations += 1

        # 3. Prune old backup directories (keep newest 10, delete the rest)
        pruned = self._prune_backup_dirs(_PROJECT_ROOT / "trained_data" / "backups", keep_n=10)
        if pruned:
            report["backup_dirs_pruned"] = pruned

        self._total_cleanups += 1
        self._cleanup_log.append(report)  # deque(maxlen=50) auto-evicts oldest

        return report

    def get_memory_report(self) -> Dict[str, Any]:
        """Get current memory usage report."""
        rss_mb = self._get_rss_mb()

        cache_stats = {
            name: cache.get_stats() for name, cache in self._caches.items()
        }

        return {
            "rss_mb": round(rss_mb, 1),
            "rss_warning": rss_mb > RSS_WARNING_MB,
            "rss_critical": rss_mb > RSS_CRITICAL_MB,
            "caches": cache_stats,
            "log_files": {
                path: {
                    "rotations": cfg["rotations"],
                    "exists": Path(path).exists(),
                }
                for path, cfg in self._log_files.items()
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Get manager status for observability."""
        return {
            "total_cleanups": self._total_cleanups,
            "total_evictions": self._total_evictions,
            "total_rotations": self._total_rotations,
            "registered_caches": list(self._caches.keys()),
            "registered_logs": list(self._log_files.keys()),
            "memory": self.get_memory_report(),
            "recent_cleanups": self._cleanup_log[-3:],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_backup_dirs(self, backup_root: Path, keep_n: int = 10) -> int:
        """Delete oldest timestamped backup directories, keeping the newest keep_n.

        backup_root/YYYYMMDD_HHMMSS/ directories accumulate indefinitely.
        Called every run_cleanup() cycle; safe to call when dir doesn't exist.
        Returns number of directories deleted.
        """
        if not backup_root.exists():
            return 0
        try:
            dirs = sorted(
                (d for d in backup_root.iterdir() if d.is_dir()),
                key=lambda d: d.name,
            )
            to_remove = dirs[:-keep_n] if len(dirs) > keep_n else []
            for d in to_remove:
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception as e:
                    logger.debug(f"Backup dir prune failed for {d}: {e}")
            if to_remove:
                logger.info("Pruned %d old backup dirs (kept newest %d)", len(to_remove), keep_n)
            return len(to_remove)
        except Exception as e:
            logger.debug(f"Backup dir prune error: {e}")
            return 0

    def _rotate_log(
        self, path: Path, max_lines: int, keep_lines: int
    ) -> bool:
        """Rotate a log file if it exceeds max_lines.

        Keeps the most recent keep_lines. Archives the rest.
        Returns True if rotation was performed.
        """
        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) <= max_lines:
                return False

            # Archive old lines
            archive_path = _ARCHIVE_DIR / f"{path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{path.suffix}"
            _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

            with open(archive_path, "w", encoding="utf-8") as f:
                f.writelines(lines[:-keep_lines])

            # Keep recent lines
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[-keep_lines:])

            logger.info(
                f"MemoryManager: Rotated {path.name}: "
                f"{len(lines)} → {keep_lines} lines (archived {len(lines) - keep_lines})"
            )
            return True

        except Exception as e:
            logger.debug(f"MemoryManager: Failed to rotate {path}: {e}")
            return False

    def _get_rss_mb(self) -> float:
        """Get current process RSS in MB."""
        try:
            # Read from /proc/self/status on Linux
            status_path = Path("/proc/self/status")
            if status_path.exists():
                with open(status_path, "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            return kb / 1024.0
            # Fallback: use resource module
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist manager state."""
        try:
            data = {
                "version": 1,
                "total_cleanups": self._total_cleanups,
                "total_evictions": self._total_evictions,
                "total_rotations": self._total_rotations,
                "cleanup_log": self._cleanup_log[-50:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"MemoryManager: Failed to save: {e}")

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

            if isinstance(data, dict):
                self._total_cleanups = data.get("total_cleanups", 0)
                self._total_evictions = data.get("total_evictions", 0)
                self._total_rotations = data.get("total_rotations", 0)
                self._cleanup_log = data.get("cleanup_log", [])
        except Exception as e:
            logger.debug(f"MemoryManager: Failed to load: {e}")
