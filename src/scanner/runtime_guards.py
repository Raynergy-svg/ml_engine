"""Runtime preconditions for scanner init and the meta scorecard.

These guards convert silent native-frame failures (SIGSEGV under disk pressure)
into observable, recoverable Python exceptions / scorecard skips.

Background
----------
On 2026-05-20, `tests/test_buddy_scan.py::test_scanner_initialization` crashed
with SIGSEGV inside `lightgbm.basic.Booster.__setstate__` while loading the
serialized `lgbm_momentum` model file. Initial theory: corrupt artifact or
lightgbm version mismatch.

On 2026-05-21, after `uv cache clean --force` freed ~5 GiB and lifted the
filesystem from 100% (~214 MiB free) to 99% (~3.2 GiB free), the SAME test
passed in isolation. The artifacts also loaded cleanly via `joblib.load(...)`
on their own. The single variable that changed across the crash↔pass boundary
was free disk space.

Diagnosis (HIGH confidence): when the volume backing `trained_data/` and the
process's mmap region is in the low hundreds of MB, native allocations inside
lightgbm / numpy / tensorflow / joblib's `_multiprocessing_helpers` fail in
non-recoverable ways and abort the process with SIGSEGV instead of raising a
Python exception. `joblib` already emits `[Errno 28] No space left on device`
and falls into serial mode, but lightgbm and numpy do not have graceful
fallbacks.

Load-bearing assumption: free disk space < ~500 MiB on the volume that holds
the model artifacts + process scratch space is the failure surface. The fix
detects that condition *before* the first heavy native allocation and converts
it into a `RuntimeError` (scanner) or a scorecard `low_disk_skipped` skip
(meta-pipeline) — both observable, both recoverable.

Threshold
---------
500 MiB is the default. Rationale:
- joblib's serial-mode warning fires at "no space left" (free ~0); we want
  margin above that floor.
- A single lightgbm Booster load is ~50-200 MiB of resident allocation; the
  full ensemble (transformer + tcn + 3x lightgbm + ridge + meta) is ~1 GiB
  peak.
- TensorFlow's eager-mode XLA cache + scratch buffers add another ~200 MiB.
- 500 MiB leaves a small but real headroom above the joblib-warning floor
  while not being so high that legitimate dev environments trip it. On the
  crashing day (2026-05-20) free was 214 MiB; on the passing day 3.2 GiB.
  500 MiB sits cleanly between the two.

Override via env var `BUDDY_DISK_PRESSURE_MIB` (integer MiB). Set to `0` to
disable the guard entirely (not recommended outside CI).

No mocks
--------
Tests inject disk-state via `monkeypatch.setattr` on the real `shutil.disk_usage`
function. That's a real callable returning a real `namedtuple`, not a
`MagicMock`. See `.claude/rules/improvement.md` "No-Mock Rule".
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_DISK_PRESSURE_MIB: int = 500
_ENV_VAR: str = "BUDDY_DISK_PRESSURE_MIB"


def get_disk_pressure_threshold_mib() -> int:
    """Return the configured low-disk threshold in MiB.

    Reads `BUDDY_DISK_PRESSURE_MIB` from the env. Returns
    `DEFAULT_DISK_PRESSURE_MIB` (500) when unset or unparseable. A value of `0`
    disables the guard (callers receive `ok=True` unconditionally).
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_DISK_PRESSURE_MIB
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "runtime_guards.invalid_threshold %s=%r — falling back to %d MiB",
            _ENV_VAR, raw, DEFAULT_DISK_PRESSURE_MIB,
        )
        return DEFAULT_DISK_PRESSURE_MIB


def check_disk_pressure(
    path: Optional[Path] = None,
    threshold_mib: Optional[int] = None,
) -> Tuple[bool, int, int, str]:
    """Check whether `path` has at least `threshold_mib` MiB free.

    Args:
        path: Filesystem path to check. Defaults to the project root (the dir
              two parents above this module — i.e. `src/scanner/` → repo root).
        threshold_mib: Threshold in MiB. Defaults to
                       `get_disk_pressure_threshold_mib()`.

    Returns:
        `(ok, free_mib, threshold_mib, reason)`:
          - `ok=True` → enough free space, `reason="ok"`.
          - `ok=False` → low disk, `reason="low_disk"` (or `"check_failed"` if
                          `shutil.disk_usage` raised).
          - `threshold_mib=0` → guard disabled, `ok=True`, `reason="disabled"`.

    Never raises. A `disk_usage` failure logs and returns `ok=False` with
    `reason="check_failed"` so callers can decide whether to skip or proceed.
    """
    if threshold_mib is None:
        threshold_mib = get_disk_pressure_threshold_mib()
    if path is None:
        # repo root = src/scanner/runtime_guards.py → ../../.. (= repo root)
        path = Path(__file__).resolve().parent.parent.parent

    if threshold_mib <= 0:
        # Guard disabled by operator. Report the free space anyway.
        try:
            usage = shutil.disk_usage(str(path))
            free_mib = int(usage.free // (1024 * 1024))
        except Exception:
            free_mib = -1
        return True, free_mib, 0, "disabled"

    try:
        usage = shutil.disk_usage(str(path))
    except Exception as e:
        logger.error(
            "runtime_guards.disk_usage_failed path=%s err=%s",
            path, e,
        )
        return False, -1, threshold_mib, "check_failed"

    free_mib = int(usage.free // (1024 * 1024))
    if free_mib < threshold_mib:
        return False, free_mib, threshold_mib, "low_disk"
    return True, free_mib, threshold_mib, "ok"


def assert_disk_ok_for_model_load(
    path: Optional[Path] = None,
    threshold_mib: Optional[int] = None,
    context: str = "model_load",
) -> None:
    """Raise `RuntimeError` if disk pressure would risk a SIGSEGV.

    Intended call sites:
      - `src/scanner/engine.py:Scanner.__init__` (before Phase 69 warm-up)
      - `src/core/modular_inference.py:ModularEnsembleInference.load_models`
        (before any model load)

    Logs a loud ERROR before raising so the operator sees the cause even if the
    exception is later swallowed by a catch-all.

    Args:
        path: Volume to check (defaults to repo root).
        threshold_mib: Threshold in MiB (defaults to env-configured value).
        context: Free-text tag used in the log + exception message (e.g.
                 `"scanner_init"`, `"modular_ensemble_load"`).

    Raises:
        RuntimeError: When `free < threshold` or the disk_usage call failed.
    """
    ok, free_mib, threshold, reason = check_disk_pressure(path, threshold_mib)
    if ok:
        return
    msg = (
        f"disk pressure ({reason}): {free_mib} MiB free < {threshold} MiB "
        f"threshold at {path or 'repo root'} — refusing to {context} to avoid "
        f"SIGSEGV in native model load. Free space and retry, or override via "
        f"{_ENV_VAR}=0 (not recommended)."
    )
    logger.error("runtime_guards.disk_pressure_block context=%s %s", context, msg)
    raise RuntimeError(msg)
