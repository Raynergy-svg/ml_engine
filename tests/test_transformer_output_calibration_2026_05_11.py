"""Per-pair Transformer output-calibration threshold tests (added 2026-05-11).

Team B's SHORT-bias diagnostic (2026-05-11 ~04:30 EDT) identified a
train↔inference contract violation: the Transformer direction trainer
saves a per-pair calibrated threshold in
``transformer_direction.meta.pkl`` under the ``output_calibration`` key
(the empirical median of validation raw predictions), and its own
predictor at ``src/training/trainers/transformer_trainer.py:3061-3071``
uses that threshold. The runtime gates evaluator at
``src/scanner/gates.py:1902`` used a hardcoded ``prob_long > 0.5`` cut
and ignored the saved calibration entirely.

Result: with EUR_USD raw model mean=0.486 and threshold=0.4955, plus
several pairs whose calibrated threshold is above 0.5 (USD_JPY=0.536,
USD_CHF=0.565), runtime produced 83.5% SHORT on the live
virtual_trades ledger — even though every pair's training was balanced
(val_up_acc ≈ val_down_acc, gap ≤ 0.009). That bias triggered the
heuristic-vote ``disagreement_hard_block`` floor on essentially every
high-confidence scan and locked all trading out post-restart.

This module covers:
  * The pure helper ``_apply_calibrated_threshold`` end-to-end.
  * The contract sidecar load on a real on-disk artifact (EUR_USD).
  * Soft-required semantics: missing key / disabled flag / corrupt
    threshold all fall back to the 0.5 cut.

NO MOCKS rule (CLAUDE.md / .claude/rules/improvement.md "No-Mock Rule"):
the helper is pure (float + dict → str), so tests pass real Python
dicts. The artifact integration test reads the real on-disk meta
sidecar via ``joblib.load`` (the trainer also uses joblib at
transformer_trainer.py:1662; format-compatible with the meta sidecar's
serialization).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.scanner.gates import _apply_calibrated_threshold


# ---------------------------------------------------------------------------
# 1. The threshold actually shifts the cut
# ---------------------------------------------------------------------------


def test_calibrated_threshold_055_pushes_prob_054_to_short() -> None:
    """With a 0.55 threshold, prob_long=0.54 must be SHORT (would have been
    LONG under the hardcoded 0.5 cut). Locks in that the threshold IS read."""
    cal = {"threshold": 0.55, "enabled": True, "mean": 0.5, "std": 0.05}
    assert _apply_calibrated_threshold(0.54, cal) == "SHORT"


def test_calibrated_threshold_055_pushes_prob_056_to_long() -> None:
    """With a 0.55 threshold, prob_long=0.56 must be LONG."""
    cal = {"threshold": 0.55, "enabled": True}
    assert _apply_calibrated_threshold(0.56, cal) == "LONG"


def test_calibrated_threshold_below_05_admits_more_longs() -> None:
    """The EUR_USD real case: threshold=0.4955, mean=0.4857. A prob_long
    of 0.498 — which the 0.5 cut would have called SHORT — must now be
    LONG (it's above the 0.4955 calibrated boundary). This is the gap
    that produced systematic SHORT bias when the runtime used 0.5."""
    cal = {"threshold": 0.4955, "enabled": True}
    # Under old 0.5 cut: 0.498 > 0.5 is False → SHORT
    # Under new 0.4955 cut: 0.498 > 0.4955 is True → LONG (correct)
    assert _apply_calibrated_threshold(0.498, cal) == "LONG"
    # Below the calibrated threshold, still SHORT under both regimes.
    assert _apply_calibrated_threshold(0.49, cal) == "SHORT"


# ---------------------------------------------------------------------------
# 2. Soft-required fallback: missing / disabled / corrupt all → 0.5 cut
# ---------------------------------------------------------------------------


def test_none_calibration_falls_back_to_05_cut() -> None:
    """No calibration loaded (e.g. pre-Phase-2 artifact). Falls back to 0.5
    cut so old models still work."""
    assert _apply_calibrated_threshold(0.55, None) == "LONG"
    assert _apply_calibrated_threshold(0.45, None) == "SHORT"


def test_disabled_calibration_falls_back_to_05_cut() -> None:
    """``enabled: False`` is the trainer's way of saying "don't apply
    threshold drift". Inference must respect it."""
    cal = {"threshold": 0.7, "enabled": False}
    # prob_long=0.55 — would be SHORT if 0.7 were honored, LONG under 0.5.
    assert _apply_calibrated_threshold(0.55, cal) == "LONG"


def test_missing_threshold_key_falls_back_to_05_cut() -> None:
    """Partial / corrupt meta missing the ``threshold`` key. Don't crash."""
    cal = {"enabled": True, "mean": 0.5}  # threshold key absent
    assert _apply_calibrated_threshold(0.55, cal) == "LONG"


def test_nan_threshold_falls_back_to_05_cut() -> None:
    """A NaN threshold (e.g. trainer hiccup) must NOT be used as the cut —
    NaN comparisons always return False in Python, which would silently
    declare every prediction SHORT. Defensive guard required."""
    cal = {"threshold": float("nan"), "enabled": True}
    # With NaN passing through, ``prob_long > nan`` is False → SHORT for
    # every input. The defensive guard must kick in and use 0.5 instead.
    assert _apply_calibrated_threshold(0.55, cal) == "LONG"
    assert _apply_calibrated_threshold(0.45, cal) == "SHORT"


def test_out_of_range_threshold_falls_back_to_05_cut() -> None:
    """A threshold outside (0, 1) is invalid for a probability cut."""
    cal_high = {"threshold": 1.5, "enabled": True}
    cal_zero = {"threshold": 0.0, "enabled": True}
    cal_one = {"threshold": 1.0, "enabled": True}
    assert _apply_calibrated_threshold(0.55, cal_high) == "LONG"
    assert _apply_calibrated_threshold(0.55, cal_zero) == "LONG"
    assert _apply_calibrated_threshold(0.55, cal_one) == "LONG"


def test_non_numeric_threshold_falls_back_to_05_cut() -> None:
    """A string in the threshold slot (e.g. legacy serialization). No crash."""
    cal = {"threshold": "0.55", "enabled": True}  # string, not float
    # 0.55 string → not isinstance(int, float) → fallback to 0.5.
    assert _apply_calibrated_threshold(0.45, cal) == "SHORT"
    assert _apply_calibrated_threshold(0.55, cal) == "LONG"


# ---------------------------------------------------------------------------
# 3. Contract sidecar load — real on-disk artifact (EUR_USD)
# ---------------------------------------------------------------------------


def test_real_eur_usd_artifact_has_calibration_in_expected_range() -> None:
    """Locks the contract: the EUR_USD trained artifact on disk must carry
    output_calibration with threshold ≈ 0.4955 and enabled=True. If this
    test fails, the artifact was re-trained without the calibration
    contract — the gate should refuse it."""
    meta_path = Path("trained_data/models/EUR_USD/transformer_direction.meta.pkl")
    if not meta_path.exists():
        pytest.skip("EUR_USD transformer meta sidecar not present in this env")

    meta = joblib.load(meta_path)

    cal = meta.get("output_calibration")
    assert isinstance(cal, dict), (
        f"EUR_USD artifact missing output_calibration dict: got {type(cal)}"
    )
    assert cal.get("enabled") is True
    threshold = cal.get("threshold")
    assert isinstance(threshold, (int, float))
    # Team B's empirical: 0.4955 ± measurement noise. Range tolerant for
    # future re-trains that produce slightly different empirical medians.
    assert 0.40 < threshold < 0.60, (
        f"EUR_USD threshold {threshold} outside expected (0.40, 0.60) — "
        "either a contract regression or a model trained on very different data"
    )


def test_real_eur_usd_artifact_threshold_flips_prob_under_05_correctly() -> None:
    """End-to-end: load real EUR_USD calibration, call the helper, confirm
    the right behavior. Locks the user-facing contract (the bug-fix
    invariant): a prob_long strictly between the (saved threshold < 0.5)
    and 0.5 must now be LONG — pre-fix it would have been SHORT under
    the hardcoded 0.5 cut. This is the bias gap."""
    meta_path = Path("trained_data/models/EUR_USD/transformer_direction.meta.pkl")
    if not meta_path.exists():
        pytest.skip("EUR_USD transformer meta sidecar not present in this env")

    meta = joblib.load(meta_path)
    cal = meta.get("output_calibration")
    threshold = float(cal["threshold"])
    assert threshold < 0.5, (
        f"this test only meaningful when EUR_USD threshold < 0.5; got {threshold}"
    )

    # Pick a probe halfway between threshold and 0.5 — guaranteed in the
    # bias gap where old code said SHORT and new code says LONG.
    probe = (threshold + 0.5) / 2.0
    assert _apply_calibrated_threshold(probe, cal) == "LONG"


# ---------------------------------------------------------------------------
# 4. Behavior parity with the trainer's own predictor (math match)
# ---------------------------------------------------------------------------


def test_helper_matches_trainer_predictor_logic() -> None:
    """Trainer at transformer_trainer.py:3071 uses
    ``direction = 1 if prob_raw > threshold else 0`` (1 = LONG, 0 = SHORT).
    Our helper must produce the same output for the same inputs across a
    representative grid. This is the explicit train↔inference contract
    parity test."""
    cal = {"threshold": 0.55, "enabled": True}
    grid = [0.10, 0.30, 0.49, 0.54, 0.55, 0.5500001, 0.56, 0.70, 0.90]
    for prob in grid:
        trainer_says = "LONG" if prob > 0.55 else "SHORT"
        runtime_says = _apply_calibrated_threshold(prob, cal)
        assert trainer_says == runtime_says, (
            f"contract drift at prob={prob}: trainer={trainer_says}, "
            f"runtime={runtime_says}"
        )


# ---------------------------------------------------------------------------
# 5. Wiring: GateEvaluator carries the attribute slot
# ---------------------------------------------------------------------------


def test_gate_evaluator_initializes_output_calibration_attribute() -> None:
    """Three-layer wiring (layer 1): the attribute exists on a freshly
    constructed GateEvaluator. Defaults to None until a meta sidecar is
    loaded. If a future refactor drops the attribute, every
    ``_apply_calibrated_threshold(prob_long, self._transformer_output_calibration)``
    call at the predict site would AttributeError — this guards that."""
    from src.scanner.gates import GateEvaluator

    ge = GateEvaluator(model_dir="trained_data/models/joint")
    assert hasattr(ge, "_transformer_output_calibration")
    assert ge._transformer_output_calibration is None
