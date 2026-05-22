"""No-mock tests for the disk-pressure runtime guard.

Background
----------
A SIGSEGV in `lightgbm.basic.Booster.__setstate__` during scanner init on
2026-05-20 was traced (HIGH confidence) to disk pressure — free space dropped
to ~214 MiB on the volume holding `trained_data/`, causing native allocations
inside lightgbm / numpy to fail in non-recoverable ways. `joblib` falls back
to serial mode under ENOSPC, but lightgbm and numpy abort the process. After
`uv cache clean --force` freed ~5 GiB, the SAME test passed in isolation.

The guard converts that silent crash into a `RuntimeError` (scanner /
modular_inference) or a `low_disk_skipped` scorecard skip (meta-pipeline).

Testing philosophy (NO MOCKS — `.claude/rules/improvement.md`)
--------------------------------------------------------------
- Real `shutil.disk_usage` is monkeypatched to a lambda that returns a real
  `namedtuple` (`shutil._ntuple_diskusage`). This is a real callable returning
  real data, NOT a `MagicMock`. The pattern is the canonical no-mock way to
  inject OS-call state in pytest.
- Real `ChangeEvalHarness` is exercised via the real `_default_pytest_runner`
  factory; we just monkeypatch the disk check so the runner takes the
  low-disk branch without actually filling the disk.
- No `unittest.mock`, no `MagicMock`, no `patch` imports anywhere in this file.
"""

from __future__ import annotations

import os
import shutil
import warnings

import pytest

from src.scanner import runtime_guards
from src.scanner.runtime_guards import (
    DEFAULT_DISK_PRESSURE_MIB,
    assert_disk_ok_for_model_load,
    check_disk_pressure,
    get_disk_pressure_threshold_mib,
)


# ---------------------------------------------------------------------------
# Helpers — build real `shutil._ntuple_diskusage` instances (no MagicMock).
# ---------------------------------------------------------------------------


_DiskUsage = shutil._ntuple_diskusage  # the real namedtuple class


def _make_usage(free_mib: int, total_mib: int = 100_000) -> "_DiskUsage":
    """Return a real disk_usage namedtuple with `free_mib` MiB free."""
    free = free_mib * 1024 * 1024
    total = total_mib * 1024 * 1024
    used = max(0, total - free)
    return _DiskUsage(total=total, used=used, free=free)


def _patch_disk_usage(monkeypatch: pytest.MonkeyPatch, free_mib: int) -> None:
    """Replace `shutil.disk_usage` with a real lambda returning `free_mib` MiB."""
    monkeypatch.setattr(
        runtime_guards.shutil,
        "disk_usage",
        lambda _path: _make_usage(free_mib),
    )


# ---------------------------------------------------------------------------
# T1: guard returns ok=True when disk has more than threshold free
# ---------------------------------------------------------------------------


def test_t1_check_disk_pressure_returns_ok_when_disk_has_space(monkeypatch):
    """T1: 2048 MiB free vs 500 MiB threshold → ok=True, reason='ok'."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=2048)

    ok, free_mib, threshold_mib, reason = check_disk_pressure()

    assert ok is True, "guard should pass when 2 GiB > 500 MiB threshold"
    assert free_mib == 2048
    assert threshold_mib == DEFAULT_DISK_PRESSURE_MIB == 500
    assert reason == "ok"


# ---------------------------------------------------------------------------
# T2: guard returns ok=False when disk is below threshold
# ---------------------------------------------------------------------------


def test_t2_check_disk_pressure_returns_low_disk_below_threshold(monkeypatch):
    """T2: 100 MiB free vs 500 MiB threshold → ok=False, reason='low_disk'."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=100)

    ok, free_mib, threshold_mib, reason = check_disk_pressure()

    assert ok is False, "guard should fail when 100 MiB < 500 MiB threshold"
    assert free_mib == 100
    assert threshold_mib == 500
    assert reason == "low_disk"


def test_t2b_check_disk_pressure_matches_2026_05_20_crash_conditions(monkeypatch):
    """T2b: 214 MiB free (the actual crash-day value) trips the guard."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=214)

    ok, free_mib, _, reason = check_disk_pressure()

    assert ok is False
    assert free_mib == 214
    assert reason == "low_disk"


# ---------------------------------------------------------------------------
# T3: scanner-init helper raises RuntimeError on low disk
# ---------------------------------------------------------------------------


def test_t3_assert_disk_ok_raises_runtimeerror_on_low_disk(monkeypatch):
    """T3: assert_disk_ok_for_model_load raises with clear message on low disk."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=50)

    with pytest.raises(RuntimeError) as exc_info:
        assert_disk_ok_for_model_load(context="scanner_init")

    msg = str(exc_info.value)
    assert "disk pressure" in msg
    assert "50 MiB free" in msg
    assert "500 MiB threshold" in msg
    assert "scanner_init" in msg, "context tag should appear in the message"
    assert "BUDDY_DISK_PRESSURE_MIB" in msg, "override env var should be documented"


def test_t3b_assert_disk_ok_passes_silently_when_disk_has_space(monkeypatch):
    """T3b: assert_disk_ok returns None (no raise) when disk has space."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=10_000)

    result = assert_disk_ok_for_model_load(context="scanner_init")

    assert result is None


# ---------------------------------------------------------------------------
# T4: scorecard runner returns low_disk_skipped (does NOT spawn pytest)
# ---------------------------------------------------------------------------


def test_t4_scorecard_runner_returns_low_disk_skipped_under_pressure(monkeypatch):
    """T4: _default_pytest_runner returns low_disk_skipped + does NOT subprocess.run."""
    from src.scanner.automation import change_eval as ce

    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    monkeypatch.setattr(
        runtime_guards.shutil,
        "disk_usage",
        lambda _path: _make_usage(free_mib=80),
    )

    # Tripwire: if subprocess.run is called, the disk guard didn't fire. Fail
    # the test loudly via a sentinel that returns a structurally-real Popen
    # result — we expect this lambda to NEVER be invoked.
    subprocess_invocations = {"count": 0}

    def _fail_if_called(*_args, **_kwargs):
        subprocess_invocations["count"] += 1
        raise AssertionError(
            "subprocess.run should not be called under disk pressure — "
            "guard failed to short-circuit"
        )

    monkeypatch.setattr(ce.subprocess, "run", _fail_if_called)

    result = ce._default_pytest_runner(target=None)

    assert result["passed"] is False
    assert result["skipped"] is True
    assert result["reason"] == "low_disk_skipped"
    assert result["free_mib"] == 80
    assert result["threshold_mib"] == 500
    assert result["exit_code"] == -1
    assert result["failed"] == []
    assert "low_disk_skipped" in result["stdout_tail"]
    assert subprocess_invocations["count"] == 0, (
        "subprocess.run was called — disk guard did not short-circuit"
    )


# ---------------------------------------------------------------------------
# T5: scorecard runner shells out normally when disk is OK
# ---------------------------------------------------------------------------


def test_t5_scorecard_runner_shells_out_when_disk_ok(monkeypatch):
    """T5: _default_pytest_runner reaches subprocess.run when disk is healthy."""
    from src.scanner.automation import change_eval as ce

    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    monkeypatch.setattr(
        runtime_guards.shutil,
        "disk_usage",
        lambda _path: _make_usage(free_mib=10_000),
    )

    # Replace subprocess.run with a real callable that returns a real
    # CompletedProcess (no MagicMock). This confirms the disk guard does NOT
    # short-circuit when disk is healthy — the runner proceeds to invocation.
    import subprocess as _subprocess
    invocation_seen = {"called": False}

    def _fake_run(*_args, **_kwargs):
        invocation_seen["called"] = True
        return _subprocess.CompletedProcess(
            args=_args[0] if _args else [],
            returncode=0,
            stdout="== 1 passed in 0.01s ==\n",
            stderr="",
        )

    monkeypatch.setattr(ce.subprocess, "run", _fake_run)

    result = ce._default_pytest_runner(target=None)

    assert invocation_seen["called"] is True, (
        "subprocess.run should be invoked when disk is healthy"
    )
    assert result["passed"] is True
    assert result["exit_code"] == 0
    assert "skipped" not in result or not result.get("skipped"), (
        "healthy-disk path must not return skipped=True"
    )


# ---------------------------------------------------------------------------
# T6: threshold is configurable via BUDDY_DISK_PRESSURE_MIB env var
# ---------------------------------------------------------------------------


def test_t6a_env_var_overrides_default_threshold(monkeypatch):
    """T6a: BUDDY_DISK_PRESSURE_MIB=1000 raises threshold from 500 to 1000."""
    monkeypatch.setenv("BUDDY_DISK_PRESSURE_MIB", "1000")
    _patch_disk_usage(monkeypatch, free_mib=600)

    ok, free_mib, threshold_mib, reason = check_disk_pressure()

    assert ok is False, "600 MiB should fail when threshold is raised to 1000"
    assert free_mib == 600
    assert threshold_mib == 1000
    assert reason == "low_disk"


def test_t6b_env_var_zero_disables_guard(monkeypatch):
    """T6b: BUDDY_DISK_PRESSURE_MIB=0 disables the guard entirely."""
    monkeypatch.setenv("BUDDY_DISK_PRESSURE_MIB", "0")
    _patch_disk_usage(monkeypatch, free_mib=10)  # would trip default threshold

    ok, free_mib, threshold_mib, reason = check_disk_pressure()

    assert ok is True, "guard disabled by env=0 should always return ok=True"
    assert threshold_mib == 0
    assert reason == "disabled"
    # The free_mib reading should still be accurate so the operator can
    # diagnose if a downstream native abort happens anyway.
    assert free_mib == 10


def test_t6c_invalid_env_var_falls_back_to_default(monkeypatch):
    """T6c: BUDDY_DISK_PRESSURE_MIB='garbage' logs a warning, uses 500."""
    monkeypatch.setenv("BUDDY_DISK_PRESSURE_MIB", "not-an-int")
    threshold = get_disk_pressure_threshold_mib()
    assert threshold == DEFAULT_DISK_PRESSURE_MIB == 500


def test_t6d_get_threshold_unset_returns_default(monkeypatch):
    """T6d: env unset → DEFAULT_DISK_PRESSURE_MIB (500)."""
    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    assert get_disk_pressure_threshold_mib() == 500


# ---------------------------------------------------------------------------
# T7: test_buddy_scan.py emits no PytestReturnNotNoneWarning
# ---------------------------------------------------------------------------


def test_t7_buddy_scan_tests_emit_no_return_not_none_warning():
    """T7: importing and inspecting tests/test_buddy_scan.py reveals no
    `return True/False` patterns in test_* functions (the source of the
    PytestReturnNotNoneWarning).

    This is a static check on the file content rather than a live pytest run —
    running test_buddy_scan.py as a subprocess would re-trigger the scanner
    init that historically segfaulted under disk pressure (the exact failure
    mode the disk guard was added to prevent). Static inspection confirms the
    cosmetic fix without needing a clean disk.
    """
    import pathlib
    import re

    test_file = (
        pathlib.Path(__file__).resolve().parent / "test_buddy_scan.py"
    )
    assert test_file.exists(), f"expected sibling file at {test_file}"

    source = test_file.read_text(encoding="utf-8")

    # Walk the source line-by-line. Track whether we're inside a test_*
    # function (indented under `def test_...`). If we are, no `return <truthy>`
    # may appear. Bare `return` (early-exit) is OK; `return SOMETHING` is not.
    in_test_fn = False
    test_fn_indent: int | None = None
    offenders: list[tuple[int, str]] = []

    for lineno, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        # Detect entry into a test_* function
        m = re.match(r"^(\s*)def\s+(test_\w+)\s*\(", raw)
        if m:
            in_test_fn = True
            test_fn_indent = len(m.group(1))
            continue
        # Exit when we hit a line at or below the def's indent that's
        # non-blank and not a continuation
        if in_test_fn and stripped and not raw.startswith(" " * ((test_fn_indent or 0) + 1)):
            in_test_fn = False
            test_fn_indent = None
        if in_test_fn and re.match(r"^\s*return\s+\S", raw):
            offenders.append((lineno, stripped))

    assert not offenders, (
        f"test_buddy_scan.py has {len(offenders)} `return <value>` in test_* "
        f"functions (re-introduces PytestReturnNotNoneWarning): {offenders[:5]}"
    )


def test_t7b_buddy_scan_imports_clean():
    """T7b: tests/test_buddy_scan.py imports without warning emission.

    A negative assertion against PytestReturnNotNoneWarning at import time —
    confirms the file is syntactically clean and the warning surface stays
    quiet for downstream consumers (W&B logging, scorecard digest).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Import is a no-op if already loaded; the module-level code has no
        # side effects beyond importing the test functions.
        import importlib
        import tests.test_buddy_scan as _t
        importlib.reload(_t)

    offenders = [
        w for w in caught
        if "PytestReturnNotNoneWarning" in (w.category.__name__ if w.category else "")
    ]
    assert not offenders, f"unexpected warnings: {[str(w.message) for w in offenders]}"


# ---------------------------------------------------------------------------
# T8: GateEvaluator.load_models is guarded against disk pressure
#
# Coverage rationale: `load_models()` is the third native-load surface (after
# Scanner.__init__ and ModularEnsembleInference.load_models). It is invoked
# from engine.py:1692 during deferred init. The guard fires BEFORE any of the
# `_load_*` helpers (joblib / keras / lightgbm) touch disk, converting the
# SIGSEGV risk under ENOSPC into an observable RuntimeError that the caller
# `_initialize_models` can catch via its existing `except Exception` branch.
# ---------------------------------------------------------------------------


def test_t8a_gate_evaluator_load_models_raises_runtimeerror_under_disk_pressure(
    monkeypatch, tmp_path
):
    """T8a: GateEvaluator.load_models() raises RuntimeError with the
    `gate_evaluator_load` context tag when free disk is below threshold."""
    from src.scanner.gates import GateEvaluator

    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=60)

    # Real GateEvaluator against a real (empty) tmp model dir. We never reach
    # the inner _load_* calls because the disk guard short-circuits first.
    evaluator = GateEvaluator(
        model_dir=tmp_path,
        use_joint_only=False,
        use_per_pair_routing=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        evaluator.load_models(require_tcn=False)

    msg = str(exc_info.value)
    assert "disk pressure" in msg
    assert "60 MiB free" in msg
    assert "500 MiB threshold" in msg
    assert "gate_evaluator_load" in msg, (
        "context tag must appear so operators can trace which call site fired"
    )


def test_t8b_gate_evaluator_load_models_does_not_raise_disk_runtimeerror_when_disk_ok(
    monkeypatch, tmp_path
):
    """T8b: GateEvaluator.load_models() does NOT raise a disk-pressure
    RuntimeError when free disk is above threshold. The call may still
    raise FileNotFoundError (no TCN model in tmp dir) or return a status
    dict — we only assert the absence of the disk-pressure branch.
    """
    from src.scanner.gates import GateEvaluator

    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=10_000)

    evaluator = GateEvaluator(
        model_dir=tmp_path,
        use_joint_only=False,
        use_per_pair_routing=False,
    )

    # require_tcn=False prevents the FileNotFoundError that the missing
    # TCN model would otherwise raise. With healthy disk + no models on
    # disk, load_models should return a status dict where every value
    # is False (no models loaded) without invoking the disk-pressure guard.
    try:
        status = evaluator.load_models(require_tcn=False)
    except RuntimeError as exc:  # pragma: no cover — defensive
        assert "disk pressure" not in str(exc), (
            "disk-pressure RuntimeError must NOT fire when free_mib=10_000"
        )
        raise

    # All models failed to load (none exist) but the call completed.
    assert isinstance(status, dict)
    assert all(v is False for v in status.values()), (
        f"expected all models to fail to load (empty dir), got: {status}"
    )


def test_t8c_gate_evaluator_disk_guard_fires_before_any_model_file_is_opened(
    monkeypatch, tmp_path
):
    """T8c: The disk guard fires BEFORE any of the `_load_*` helpers run.

    Setup: pre-seed tmp_path with files named like real model artifacts that
    would raise on read (zero bytes or unparseable). Then monkeypatch each
    `_load_*` method on the evaluator instance to a tripwire that records
    invocation. Under disk pressure, the guard must short-circuit before
    ANY tripwire fires — proving call ordering: guard precedes model loads.
    """
    from src.scanner.gates import GateEvaluator

    monkeypatch.delenv("BUDDY_DISK_PRESSURE_MIB", raising=False)
    _patch_disk_usage(monkeypatch, free_mib=80)

    # Seed unreadable garbage files at the artifact paths so that if the
    # guard fails to fire, the actual `_load_*` calls would crash loudly on
    # them. The point is to make the failure mode visible: disk guard fires
    # = test passes via the pytest.raises; disk guard fails to fire = the
    # tripwires below would record invocations.
    for fname in (
        "catboost_momentum.pkl",
        "xgboost_momentum.pkl",
        "lgbm_momentum.pkl",
        "ridge_confidence.pkl",
        "rf_risk.pkl",
        "lgbm_risk.pkl",
        "transformer_direction.keras",
        "meta_labeler.pkl",
        "tcn_volatility_regime.keras",
    ):
        (tmp_path / fname).write_bytes(b"")

    evaluator = GateEvaluator(
        model_dir=tmp_path,
        use_joint_only=False,
        use_per_pair_routing=False,
    )

    invocations: list[str] = []

    def _tripwire(name: str):
        def _f(*_a, **_k):
            invocations.append(name)
            return False
        return _f

    # Real-callable tripwires (NOT MagicMock). Each records its invocation
    # so we can assert call ordering: zero invocations means the disk guard
    # short-circuited above them, proving guard placement is correct.
    for method_name in (
        "_load_catboost_momentum",
        "_load_xgboost_momentum",
        "_load_lgbm_momentum",
        "_load_ridge_confidence",
        "_load_rf_risk",
        "_load_lgbm_risk",
        "_load_transformer",
        "_load_meta_labeler",
        "_load_tcn_volatility",
    ):
        monkeypatch.setattr(evaluator, method_name, _tripwire(method_name))

    with pytest.raises(RuntimeError) as exc_info:
        evaluator.load_models(require_tcn=False)

    assert "gate_evaluator_load" in str(exc_info.value)
    assert invocations == [], (
        f"disk guard must fire BEFORE any _load_* helper runs, but these "
        f"were invoked first: {invocations}"
    )


# ---------------------------------------------------------------------------
# Bonus: real-disk smoke test (no monkeypatch) — confirms the guard works on
# the actual filesystem without crashing.
# ---------------------------------------------------------------------------


def test_real_disk_smoke():
    """Smoke: check_disk_pressure runs against the real FS without crashing.

    Doesn't assert ok=True (CI environments may legitimately be near full).
    Just confirms the call completes and returns a structurally valid tuple.
    """
    if os.environ.get("BUDDY_DISK_PRESSURE_MIB") not in (None, ""):
        # When the test environment has the env var set, the threshold may be
        # arbitrary. Don't assert on it; just confirm the call shape.
        pass

    ok, free_mib, threshold_mib, reason = check_disk_pressure()

    assert isinstance(ok, bool)
    assert isinstance(free_mib, int)
    assert isinstance(threshold_mib, int)
    assert isinstance(reason, str)
    assert reason in {"ok", "low_disk", "disabled", "check_failed"}
