"""Safe JSON read/write utilities with atomic writes and file locking.

Prevents corruption from concurrent access by using:
- fcntl file locking for mutual exclusion
- Temp file + fsync + os.rename for atomic writes
- Automatic .bak recovery on parse failure
"""
import json
import os
import tempfile
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Try to use fcntl for file locking (Unix-only)
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


class _FileLock:
    """Context manager for file locking using fcntl."""

    def __init__(self, path: Union[str, Path], exclusive: bool = True):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.exclusive = exclusive
        self._fd: Optional[int] = None

    def __enter__(self):
        if not _HAS_FCNTL:
            return self
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        try:
            flag = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
            fcntl.flock(self._fd, flag)
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception as _unlock_err:
                logger.warning(  # H-4: was silent; deadlocks on lock release failure must surface
                    "safe_json: lock release failed for %s: %s", self.lock_path, _unlock_err
                )
            os.close(self._fd)
            self._fd = None
        return False


# Phase 26 (US-160): Schema validators for critical JSON files
_VALIDATORS = {}  # filename -> validator_func mapping


def validate_agent_weights(data: Any) -> bool:
    """Validate agent_weights.json structure: must have _global dict with float values."""
    if not isinstance(data, dict):
        return False
    _global = data.get("_global")
    if not isinstance(_global, dict):
        return False
    # At least some agent keys with numeric values
    for k, v in _global.items():
        if k.startswith("_"):
            continue
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def validate_trade_journal(data: Any) -> bool:
    """Validate trade_journal_rl.json structure: list of entries with required fields."""
    if not isinstance(data, list):
        return False
    _REQUIRED = {"pair", "direction", "timestamp"}
    for entry in data[:5]:  # Spot-check first 5 entries
        if not isinstance(entry, dict):
            return False
        if not _REQUIRED.issubset(entry.keys()):
            return False
    return True


def validate_episodic_memory(data: Any) -> bool:
    """Validate episodic_memory.json structure: list of episodes with core fields."""
    if not isinstance(data, list):
        return False
    _REQUIRED = {
        "episode_id",
        "timestamp",
        "pair",
        "direction",
        "regime",
        "session",
        "confidence",
        "weighted_vote_score",
        "rr_ratio",
    }
    for entry in data[:5]:  # Spot-check first 5 entries
        if not isinstance(entry, dict):
            return False
        if not _REQUIRED.issubset(entry.keys()):
            return False
    return True


# Register validators by filename
_VALIDATORS["agent_weights.json"] = validate_agent_weights
_VALIDATORS["trade_journal_rl.json"] = validate_trade_journal
_VALIDATORS["episodic_memory.json"] = validate_episodic_memory


def safe_json_read(
    path: Union[str, Path],
    default: Any = None,
) -> Any:
    """Read a JSON file with locking and corruption recovery.

    If the file is corrupted, attempts to recover from .bak file.

    Args:
        path: Path to JSON file.
        default: Default value if file doesn't exist or is unrecoverable.

    Returns:
        Parsed JSON data, or default if unavailable.
    """
    path = Path(path)
    bak_path = path.with_suffix(path.suffix + ".bak")

    if not path.exists():
        return default

    with _FileLock(path, exclusive=False):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Phase 26 (US-160): Validate structure if validator registered
            _validator = _VALIDATORS.get(path.name)
            if _validator is not None and not _validator(data):
                logger.warning(
                    "US-160: Schema validation failed for %s — falling back to .bak",
                    path,
                )
                # Try backup
                if bak_path.exists():
                    try:
                        _bak_data = json.loads(bak_path.read_text(encoding="utf-8"))
                        if _validator(_bak_data):
                            logger.info("US-160: Recovered valid data from %s.bak", path)
                            path.write_text(json.dumps(_bak_data, indent=2, default=str), encoding="utf-8")  # M-9
                            return _bak_data
                    except Exception as _bak_err:
                        logger.warning(  # H-4: was silent except; now logs recovery failures
                            "US-160: backup recovery failed for %s: %s", path, _bak_err
                        )
                logger.warning(
                    "US-160: No valid backup for %s — returning default",
                    path,
                )
                return default
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Corrupted JSON in {path}: {e}")

            # Try .bak recovery
            if bak_path.exists():
                try:
                    data = json.loads(bak_path.read_text(encoding="utf-8"))
                    logger.info(f"Recovered {path} from .bak file")
                    # Restore from backup
                    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")  # M-9
                    return data
                except Exception as bak_err:
                    logger.warning(f"Backup recovery also failed for {path}: {bak_err}")

            return default
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return default


def safe_json_write(
    path: Union[str, Path],
    data: Any,
    indent: int = 2,
    create_backup: bool = True,
) -> bool:
    """Write JSON data atomically with locking.

    Uses temp file + fsync + os.rename to prevent partial writes.
    Creates a .bak of the previous version before overwriting.

    Args:
        path: Path to JSON file.
        data: Data to serialize as JSON.
        indent: JSON indentation level.
        create_backup: Whether to create .bak of previous version.

    Returns:
        True if write succeeded, False otherwise.
    """
    path = Path(path)
    bak_path = path.with_suffix(path.suffix + ".bak")

    path.parent.mkdir(parents=True, exist_ok=True)

    with _FileLock(path, exclusive=True):
        try:
            # Serialize first to catch errors before touching files
            json_str = json.dumps(data, indent=indent, default=str)

            # Create backup of current file
            if create_backup and path.exists():
                try:
                    bak_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                except Exception as bak_err:
                    logger.debug(f"Backup creation failed for {path}: {bak_err}")

            # Write to temp file in same directory (for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            try:
                os.write(fd, json_str.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                fd = -1  # Mark as closed

                # Atomic rename
                os.rename(tmp_path, str(path))
                return True

            except Exception:
                if fd >= 0:
                    os.close(fd)
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return False


def safe_jsonl_append(
    path: Union[str, Path],
    record: Any,
) -> bool:
    """Append a single JSON record to a JSONL file with locking.

    Args:
        path: Path to JSONL file.
        record: Data to serialize as a single JSON line.

    Returns:
        True if append succeeded, False otherwise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _FileLock(path, exclusive=True):
        try:
            line = json.dumps(record, default=str) + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error(f"Failed to append to {path}: {e}")
            return False
