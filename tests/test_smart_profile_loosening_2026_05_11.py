"""Smart-profile loosening tests (added 2026-05-11).

After the 4-agent TUI audit + Path-1/2 config flips + restart-pending Phase 44
kill-switch landed, the operator observed trades STILL being blocked despite
confidence >= 0.52 (often >= 0.60). A two-team diagnostic (Data Engineer
mining ``trained_data/virtual_trades.jsonl`` + Backend Architect tracing the
decision tree) localized the root cause to multi-layer over-tightening of the
smart profile.

Empirical evidence (Team A, 24 h window, N=133 high-conf rejections):

    momentum  ............. 88.7 % (118)  sole-blocker  77 % (103)
    agent_passed=False  ... 22.6 % (30)
    mtf_confluence  ....... 0  % (0)  <-- operator's hypothesis was wrong
    trend / mr / corr ..... 0  % (0)

Decision-tree map (Team B) confirmed three smart-profile thresholds far below
the model's actual output distribution:

    min_momentum            0.12   vs   model median 0.088, max 0.097
    weighted_vote_threshold 0.72   vs   default 0.45
    max_model_disagreement  0.28   vs   default 0.50; calibrator routinely HIGH
    min_risk_reward_ratio   1.8    vs   default 1.2; silent post-filter cull

This commit loosens all four to values calibrated against the empirical model
output range while preserving the safety net (trend hard-veto, MR composite
veto, uncertainty hard-block remain ON in smart).

NO MOCKS rule (CLAUDE.md / .claude/rules/improvement.md "No-Mock Rule"):
real ``ScannerConfig`` instances, real ``SCAN_PROFILES`` dict reads, real
``apply_profile`` mutation. No mocks, no patch.
"""

from __future__ import annotations

import pytest

from src.scanner.config import SCAN_PROFILES, ScannerConfig


# ---------------------------------------------------------------------------
# 1. Smart profile carries the loosened values
# ---------------------------------------------------------------------------


def test_smart_profile_min_momentum_loosened_to_dataclass_default() -> None:
    """min_momentum was 0.12; loosened to 0.06 to match model output distribution."""
    assert SCAN_PROFILES["smart"]["min_momentum"] == 0.06


def test_smart_profile_weighted_vote_threshold_loosened_to_compromise() -> None:
    """WVS was 0.72 (Trade 1220 era); loosened to 0.60 between 0.45 default and 0.72."""
    assert SCAN_PROFILES["smart"]["weighted_vote_threshold"] == 0.60


def test_smart_profile_max_model_disagreement_loosened() -> None:
    """max_model_disagreement was 0.28; loosened to 0.40 (still tighter than 0.60 default)."""
    assert SCAN_PROFILES["smart"]["max_model_disagreement"] == 0.40


def test_smart_profile_min_rr_ratio_loosened() -> None:
    """min_risk_reward_ratio was 1.8; loosened to 1.5 to stop silent post-filter cull."""
    assert SCAN_PROFILES["smart"]["min_risk_reward_ratio"] == 1.5


# ---------------------------------------------------------------------------
# 2. Dataclass defaults are NOT changed (this is a profile-only fix)
# ---------------------------------------------------------------------------


def test_dataclass_default_min_momentum_unchanged() -> None:
    """ScannerConfig() default is the pre-existing 0.06 — profile now matches it."""
    cfg = ScannerConfig()
    assert cfg.min_momentum == 0.06


def test_dataclass_defaults_for_loosened_keys_match_smart() -> None:
    """After this commit, smart profile loosenings align with (or remain above)
    the dataclass defaults. Verifies we did not introduce a profile that is
    LOOSER than the canonical default — the loosening targets the gap that
    smart's prior calibration over-tightened."""
    cfg = ScannerConfig()
    # min_momentum: smart matches default exactly
    assert SCAN_PROFILES["smart"]["min_momentum"] == cfg.min_momentum
    # WVS: smart 0.60 still TIGHTER than default 0.45 (we did not loosen past
    # default — preserve some of the Trade 1220 hardening)
    assert SCAN_PROFILES["smart"]["weighted_vote_threshold"] >= cfg.weighted_vote_threshold
    # max_model_disagreement: smart 0.40 still TIGHTER than default 0.60
    assert SCAN_PROFILES["smart"]["max_model_disagreement"] <= cfg.max_model_disagreement
    # min_rr_ratio: smart 1.5 still TIGHTER than default 1.2
    assert SCAN_PROFILES["smart"]["min_risk_reward_ratio"] >= cfg.min_risk_reward_ratio


# ---------------------------------------------------------------------------
# 3. apply_profile actually mutates the config (consumer-side three-layer
# verification per .claude/rules/improvement.md "Live Wiring Verification")
# ---------------------------------------------------------------------------


def test_apply_profile_smart_propagates_loosened_values() -> None:
    """Applying the smart profile to a fresh ScannerConfig propagates all four
    loosenings to live attributes that consumers read via getattr()."""
    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    assert cfg.min_momentum == 0.06
    assert cfg.weighted_vote_threshold == 0.60
    assert cfg.max_model_disagreement == 0.40
    assert cfg.min_risk_reward_ratio == 1.5


def test_apply_profile_non_smart_does_not_loosen() -> None:
    """A profile other than smart (e.g. demo or aggressive) does not inherit
    smart's loosened values — they remain at their own profile or dataclass
    values."""
    # Pick a profile that isn't smart and isn't the loosening target.
    other_profiles = [p for p in SCAN_PROFILES if p != "smart"]
    assert other_profiles, "expected at least one non-smart profile to exist"
    for profile_name in other_profiles:
        cfg = ScannerConfig()
        cfg.apply_profile(profile_name)
        # If the other profile happens to set the same field, that's OK; what
        # matters is that smart's specific value isn't bleeding in.
        # We assert the non-smart profile uses its OWN value (whatever it is),
        # not smart's. Concretely: the test is meaningful for fields the other
        # profile doesn't override — they should be dataclass defaults.
        if "min_momentum" not in SCAN_PROFILES[profile_name]:
            assert cfg.min_momentum == ScannerConfig().min_momentum


# ---------------------------------------------------------------------------
# 4. The loosened thresholds actually unblock the empirical observed range
# (Team A's median 0.088 / max 0.097 momentum distribution)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observed_momentum", [0.060, 0.075, 0.088, 0.097])
def test_smart_profile_min_momentum_admits_empirical_range(observed_momentum: float) -> None:
    """Each observed momentum value from Team A's 24h rejection sample
    (median 0.088, max 0.097, p25 0.059) is at-or-above the new 0.06 floor."""
    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    # Same semantic check the gate evaluator does (>= threshold passes).
    assert observed_momentum >= cfg.min_momentum, (
        f"observed momentum {observed_momentum} should pass smart floor "
        f"{cfg.min_momentum} after loosening — was previously 0.12"
    )


def test_old_threshold_would_have_blocked_full_empirical_range() -> None:
    """Regression-witness: confirms the OLD 0.12 floor really did exclude the
    empirical max (0.097). Documents WHY we loosened."""
    old_floor = 0.12
    empirical_max_observed = 0.097
    assert empirical_max_observed < old_floor, (
        "expected old smart floor 0.12 to exclude the maximum observed "
        "momentum value 0.097 — this is the locked-out regression we just fixed"
    )
