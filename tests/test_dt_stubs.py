"""Tests for the Phase 1 Decision Transformer scaffolding.

Verifies:
  - All three modules import cleanly (no import-time side effects).
  - Config dataclasses validate their invariants at construction.
  - Stub methods raise REAL ``NotImplementedError`` (not silently passing).
  - Action constants and the public surface are intact.
  - Loader config defaults round-trip via real ``tmp_path`` (no mocks).

NO MOCKS. Per ``.claude/rules/improvement.md`` No-Mock Rule (promoted
2026-05-01): real classes against real disk via ``tmp_path``. Config-validation
exceptions are real ``ValueError``s; stub-method exceptions are real
``NotImplementedError``s. The whole point of this test file is to verify Phase
1 ships exception-raising stubs — mocking the exception would defeat the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Module imports — no side effects on import
# ---------------------------------------------------------------------------


def test_decision_transformer_module_imports() -> None:
    """``decision_transformer`` imports without torch and exposes its surface."""
    from src.training.rl import decision_transformer as dt_mod

    expected_names = {
        "ACTION_NO_TRADE",
        "ACTION_LONG",
        "ACTION_SHORT",
        "NUM_ACTIONS",
        "TOKEN_TYPE_RTG",
        "TOKEN_TYPE_STATE",
        "TOKEN_TYPE_ACTION",
        "DecisionTransformerConfig",
        "DecisionTransformer",
    }
    assert expected_names.issubset(set(dt_mod.__all__))
    for name in expected_names:
        assert hasattr(dt_mod, name), f"missing public export: {name}"


def test_trajectory_loader_module_imports() -> None:
    """``trajectory_loader`` imports and exposes its public surface."""
    from src.training.rl import trajectory_loader as tl_mod

    expected_names = {"Trajectory", "TrajectoryLoaderConfig", "TrajectoryLoader"}
    assert expected_names.issubset(set(tl_mod.__all__))


def test_offline_rl_trainer_module_imports() -> None:
    """``offline_rl_trainer`` imports and exposes its public surface."""
    from src.training.rl import offline_rl_trainer as tr_mod

    expected_names = {
        "DT_MODEL_PATH",
        "DT_CONFIG_PATH",
        "OfflineRLTrainerConfig",
        "OfflineRLTrainer",
    }
    assert expected_names.issubset(set(tr_mod.__all__))
    # Artifact paths point at trained_data, matching position_sizer.py convention.
    assert tr_mod.DT_MODEL_PATH.parts[0] == "trained_data"
    assert tr_mod.DT_CONFIG_PATH.parts[0] == "trained_data"


# ---------------------------------------------------------------------------
# Action / token-type constants
# ---------------------------------------------------------------------------


def test_action_constants_are_distinct_and_canonical() -> None:
    """Action ints are 0/1/2 and the count matches the spec."""
    from src.training.rl.decision_transformer import (
        ACTION_LONG,
        ACTION_NO_TRADE,
        ACTION_SHORT,
        NUM_ACTIONS,
        TOKEN_TYPE_ACTION,
        TOKEN_TYPE_RTG,
        TOKEN_TYPE_STATE,
    )

    assert (ACTION_NO_TRADE, ACTION_LONG, ACTION_SHORT) == (0, 1, 2)
    assert NUM_ACTIONS == 3
    assert {TOKEN_TYPE_RTG, TOKEN_TYPE_STATE, TOKEN_TYPE_ACTION} == {0, 1, 2}


# ---------------------------------------------------------------------------
# Config validation — real ValueErrors, no mocks
# ---------------------------------------------------------------------------


def test_dt_config_defaults_construct_cleanly() -> None:
    """Default DecisionTransformerConfig satisfies its invariants."""
    from src.training.rl.decision_transformer import DecisionTransformerConfig

    cfg = DecisionTransformerConfig()
    assert cfg.state_dim == 32
    assert cfg.num_actions == 3
    assert cfg.context_length == 32
    assert cfg.d_model % cfg.n_heads == 0
    assert 0.0 <= cfg.dropout < 1.0


def test_dt_config_rejects_bad_head_divisibility() -> None:
    """``d_model`` must be divisible by ``n_heads`` — real ValueError."""
    from src.training.rl.decision_transformer import DecisionTransformerConfig

    with pytest.raises(ValueError, match="divisible"):
        DecisionTransformerConfig(d_model=63, n_heads=4)


def test_dt_config_rejects_phase3_action_space() -> None:
    """Phase 2 hard-wires num_actions=3; spec §7 P3 will relax this."""
    from src.training.rl.decision_transformer import DecisionTransformerConfig

    with pytest.raises(ValueError, match="num_actions"):
        DecisionTransformerConfig(num_actions=5)


def test_dt_config_rejects_negative_context_length() -> None:
    from src.training.rl.decision_transformer import DecisionTransformerConfig

    with pytest.raises(ValueError, match="context_length"):
        DecisionTransformerConfig(context_length=0)


def test_loader_config_defaults_round_trip(tmp_path: Path) -> None:
    """LoaderConfig accepts real tmp_path filesystem locations.

    Real disk, no mock filesystem. The config does not yet read the file
    (Phase 2 work) but it stores the path for the loader to consume.
    """
    from src.training.rl.trajectory_loader import TrajectoryLoaderConfig

    journal = tmp_path / "trade_journal_rl.json"
    journal.write_text("[]")
    virtual = tmp_path / "virtual_trades.jsonl"
    virtual.write_text("")

    cfg = TrajectoryLoaderConfig(
        journal_path=journal, virtual_trades_path=virtual
    )
    assert cfg.journal_path == journal
    assert cfg.virtual_trades_path == virtual
    assert cfg.context_length == 32
    assert cfg.reward_clip_min < cfg.reward_clip_max


def test_loader_config_rejects_inverted_reward_clip() -> None:
    from src.training.rl.trajectory_loader import TrajectoryLoaderConfig

    with pytest.raises(ValueError, match="reward_clip"):
        TrajectoryLoaderConfig(reward_clip_min=5.0, reward_clip_max=-1.0)


def test_trainer_config_rejects_bad_promotion_pvalue() -> None:
    from src.training.rl.offline_rl_trainer import OfflineRLTrainerConfig

    with pytest.raises(ValueError, match="promotion_pvalue_threshold"):
        OfflineRLTrainerConfig(promotion_pvalue_threshold=1.5)


def test_trainer_config_defaults_match_spec() -> None:
    """Trainer config defaults match spec §6 decision rule."""
    from src.training.rl.offline_rl_trainer import OfflineRLTrainerConfig

    cfg = OfflineRLTrainerConfig()
    # Spec §6: ship only on >= 0.10R lift with p < 0.05.
    assert cfg.promotion_lift_threshold_R == pytest.approx(0.10)
    assert cfg.promotion_pvalue_threshold == pytest.approx(0.05)
    # Spec §6: 5-trade embargo between train/test.
    assert cfg.embargo == 5


# ---------------------------------------------------------------------------
# Stub methods raise NotImplementedError — real exceptions, no mocks
# ---------------------------------------------------------------------------


def test_dt_stub_methods_raise_not_implemented() -> None:
    """Phase 2 entry points raise loudly; drift off the plan fails."""
    from src.training.rl.decision_transformer import (
        DecisionTransformer,
        DecisionTransformerConfig,
    )

    model = DecisionTransformer(DecisionTransformerConfig())
    assert model.config.state_dim == 32  # construction succeeds
    assert model._layers_built is False

    with pytest.raises(NotImplementedError, match="Phase 2"):
        model._build_layers()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        model.save("/tmp/should_not_be_written")
    with pytest.raises(NotImplementedError, match="Phase 2"):
        DecisionTransformer.load("/tmp/nonexistent")


def test_loader_stub_methods_raise_not_implemented(tmp_path: Path) -> None:
    """Loader methods raise; construction with real tmp_path succeeds."""
    from src.training.rl.trajectory_loader import (
        TrajectoryLoader,
        TrajectoryLoaderConfig,
    )

    journal = tmp_path / "j.json"
    journal.write_text("[]")
    virtual = tmp_path / "v.jsonl"
    virtual.write_text("")

    loader = TrajectoryLoader(
        TrajectoryLoaderConfig(journal_path=journal, virtual_trades_path=virtual)
    )
    with pytest.raises(NotImplementedError, match="Phase 2"):
        loader.load()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        loader.get_pair_id("EUR_USD")
    with pytest.raises(NotImplementedError, match="Phase 2"):
        loader.train_holdout_split([])


def test_trainer_stub_methods_raise_not_implemented(tmp_path: Path) -> None:
    """Trainer raises on train/evaluate/promote; safety filter raises too.

    Wires real loader + real config — no mocks anywhere in the construction
    path. Confirms the trainer accepts the real types from the other stubs.
    """
    from src.training.rl.decision_transformer import DecisionTransformerConfig
    from src.training.rl.offline_rl_trainer import (
        OfflineRLTrainer,
        OfflineRLTrainerConfig,
    )
    from src.training.rl.trajectory_loader import (
        TrajectoryLoader,
        TrajectoryLoaderConfig,
    )

    journal = tmp_path / "j.json"
    journal.write_text("[]")
    virtual = tmp_path / "v.jsonl"
    virtual.write_text("")

    loader = TrajectoryLoader(
        TrajectoryLoaderConfig(journal_path=journal, virtual_trades_path=virtual)
    )
    trainer = OfflineRLTrainer(
        loader=loader,
        model_config=DecisionTransformerConfig(),
        trainer_config=OfflineRLTrainerConfig(),
    )

    with pytest.raises(NotImplementedError, match="Phase 2"):
        trainer.train()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        trainer.evaluate([])
    with pytest.raises(NotImplementedError, match="Phase 2"):
        trainer.maybe_promote({})
    # Critical-path safety filter: §6 catastrophic guard.
    with pytest.raises(NotImplementedError, match="Phase 2"):
        trainer._apply_runtime_safety_filter(predicted_action=1, gate_context={})
