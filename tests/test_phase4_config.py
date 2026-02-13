"""
Test Phase 4 configuration changes.

Validates:
1. Direction threshold increased to 0.0075
2. Default candles increased to 25000
3. Augmentation configuration present
4. RF target tracking enabled
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


def test_direction_threshold_updated():
    """Verify direction_threshold increased to 0.0075 (75 bps)."""
    config_path = Path("config/config_improved_H1.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    threshold = config.get("direction_threshold")
    assert threshold == 0.0075, f"Expected direction_threshold=0.0075, got {threshold}"
    print(f"✓ Direction threshold: {threshold} (75 bps)")


def test_default_candles_increased():
    """Verify default_candles increased to 25000."""
    config_path = Path("config/config_improved_H1.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    candles = config.get("buddy", {}).get("train_defaults", {}).get("default_candles")
    assert candles == 25000, f"Expected default_candles=25000, got {candles}"
    print(f"✓ Default candles: {candles} (~3 years H1 data)")


def test_augmentation_config_present():
    """Verify time-series augmentation configuration exists."""
    config_path = Path("config/config_improved_H1.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    transformer = config.get("transformer", {})
    
    # Check augmentation settings
    assert "use_augmentation" in transformer, "Missing use_augmentation config"
    assert transformer["use_augmentation"] is True, "Augmentation should be enabled"
    
    assert "augmentation_noise_std" in transformer, "Missing augmentation_noise_std"
    assert "augmentation_scale_range" in transformer, "Missing augmentation_scale_range"
    assert "augmentation_time_mask_prob" in transformer, "Missing augmentation_time_mask_prob"
    assert "augmentation_time_mask_max_len" in transformer, "Missing augmentation_time_mask_max_len"
    
    print("✓ Augmentation config present:")
    print(f"  - use_augmentation: {transformer['use_augmentation']}")
    print(f"  - noise_std: {transformer['augmentation_noise_std']}")
    print(f"  - scale_range: {transformer['augmentation_scale_range']}")
    print(f"  - time_mask_prob: {transformer['augmentation_time_mask_prob']}")
    print(f"  - time_mask_max_len: {transformer['augmentation_time_mask_max_len']}")


def test_trainer_config_augmentation():
    """Verify TrainerConfig has augmentation parameters (code inspection only)."""
    config_path = Path("src/training/trainers/config.py")
    content = config_path.read_text()
    
    # Check that augmentation attributes are defined
    assert "use_augmentation" in content, "Missing use_augmentation in TrainerConfig"
    assert "augmentation_noise_std" in content, "Missing augmentation_noise_std in TrainerConfig"
    assert "augmentation_scale_range" in content, "Missing augmentation_scale_range in TrainerConfig"
    assert "augmentation_time_mask_prob" in content, "Missing augmentation_time_mask_prob in TrainerConfig"
    assert "augmentation_time_mask_max_len" in content, "Missing augmentation_time_mask_max_len in TrainerConfig"
    
    print("✓ TrainerConfig has augmentation attributes in source code")


def test_transformer_trainer_has_augmentation_fn():
    """Verify TransformerDirectionTrainer has _create_augmentation_fn method (code inspection)."""
    trainer_path = Path("src/training/trainers/transformer_trainer.py")
    content = trainer_path.read_text()
    
    assert "_create_augmentation_fn" in content, "Missing _create_augmentation_fn method"
    assert "tf.data.Dataset" in content, "Missing tf.data.Dataset usage for augmentation"
    
    print("✓ TransformerDirectionTrainer has augmentation method in source code")


def test_rf_trainer_has_target_tracking():
    """Verify RandomForestTrainer tracks Phase 4 MAE target (code inspection)."""
    rf_path = Path("src/training/trainers/random_forest_trainer.py")
    content = rf_path.read_text()
    
    # Check for Phase 4 target tracking
    assert "PHASE 4" in content, "Missing Phase 4 reference in RF trainer"
    assert "target_achieved" in content, "Missing target_achieved tracking"
    assert "target_gap_bps" in content, "Missing target_gap_bps tracking"
    assert "10 bps" in content or "10.0" in content, "Missing 10 bps target reference"
    
    print("✓ RF trainer has Phase 4 target tracking in source code")


def test_multi_pair_guide_exists():
    """Verify multi-pair training guide exists."""
    guide_path = Path("docs/MULTI_PAIR_TRAINING_GUIDE.md")
    assert guide_path.exists(), "Missing MULTI_PAIR_TRAINING_GUIDE.md"
    
    content = guide_path.read_text()
    
    # Check key sections
    assert "JointMultiPairTrainer" in content, "Missing JointMultiPairTrainer documentation"
    assert "Phase 4" in content, "Missing Phase 4 reference"
    assert "Transfer Learning" in content, "Missing transfer learning explanation"
    
    print("✓ Multi-pair training guide exists")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4 Configuration Tests")
    print("=" * 60)
    
    test_direction_threshold_updated()
    test_default_candles_increased()
    test_augmentation_config_present()
    test_trainer_config_augmentation()
    test_transformer_trainer_has_augmentation_fn()
    test_rf_trainer_has_target_tracking()
    test_multi_pair_guide_exists()
    
    print("=" * 60)
    print("All Phase 4 tests passed! ✅")
    print("=" * 60)
