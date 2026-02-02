# Dependency Tree Analysis Summary

**Generated:** 2026-01-27T14:27:37.312616

## Overview

- **Total Modules:** 61
- **Average Direct Dependencies:** 1.25
- **Average Transitive Dependencies:** 0.0
- **Maximum Depth:** 1

## Module Type Distribution

- **Leaf Modules:** 34
- **Root Modules:** 27

## Top 10 Hub Modules (Most Imported)

| Rank | Module | Importers |
|------|--------|----------|
| 1 | `src.core.modular_data_loaders` | 13 |
| 2 | `src.data.feature_engineering` | 9 |
| 3 | `src.training.modular_trainers` | 7 |
| 4 | `src.utils.oanda_practice` | 6 |
| 5 | `src.utils.fx_paper` | 5 |
| 6 | `src.core.modular_inference` | 4 |
| 7 | `src.risk.confidence_calibration` | 4 |
| 8 | `src.utils.utils` | 4 |
| 9 | `src.utils` | 2 |
| 10 | `src.utils.text_features` | 2 |

## Top 10 Heavy Modules (Most Imports)

| Rank | Module | Direct Imports |
|------|--------|---------------|
| 1 | `main.py` | 10 |
| 2 | `buddy_scanner.py` | 7 |
| 3 | `src/training/buddy_training_helpers.py` | 4 |
| 4 | `src/training/walkforward_validation.py` | 4 |
| 5 | `src/utils/unified_talk.py` | 4 |
| 6 | `scripts/test_modular_ensemble.py` | 3 |
| 7 | `src/training/modular_trainers.py` | 3 |
| 8 | `src/training/tensorflow_data_pipeline.py` | 3 |
| 9 | `tests/test_overfitting_bias_fixes.py` | 3 |
| 10 | `check_class_dist.py` | 2 |

## Dependency Depth Distribution

| Depth | Count |
|-------|-------|
| 0 | 27 |
| 1 | 34 |

## Isolated Modules (Minimal Dependencies)

| Module | Imports | Importers |
|--------|---------|----------|
| `inspect_meta.py` | 1 | 0 |
| `pair_scanner.py` | 1 | 0 |
| `src/core/modular_inference.py` | 1 | 0 |
| `src/risk/position_sizing.py` | 1 | 0 |
| `src/risk/risk_management.py` | 1 | 0 |
| `src/risk/triple_barrier.py` | 1 | 0 |
| `src/utils/__init__.py` | 1 | 0 |
| `src/utils/enterprise_integration.py` | 1 | 0 |
| `src/utils/fx_paper.py` | 1 | 0 |
| `src/utils/training_improvements.py` | 1 | 0 |
| `test_calibration_integration.py` | 1 | 0 |
| `tests/test_confidence_calibration.py` | 1 | 0 |
| `tests/test_confidence_integration.py` | 1 | 0 |
| `tests/test_fx_paper.py` | 1 | 0 |
| `tests/test_modular_inference.py` | 1 | 0 |
| `check_class_dist.py` | 2 | 0 |
| `check_features.py` | 2 | 0 |
| `debug_joint_data.py` | 2 | 0 |
| `debug_joint_shape.py` | 2 | 0 |
| `src/core/modular_data_loaders.py` | 2 | 0 |

## Critical Structural Patterns


### Deepest Dependency Chains

Modules at the deepest levels of the dependency tree:

| Module | Depth |
|--------|-------|
| `buddy_scanner.py` | 1 |
| `src/utils/fx_paper.py` | 1 |
| `test_validation_preds.py` | 1 |
| `src/data/data_loader.py` | 1 |
| `src/risk/risk_management.py` | 1 |
| `src/training/buddy_training_helpers.py` | 1 |
| `check_class_dist.py` | 1 |
| `src/utils/unified_talk.py` | 1 |
| `pair_scanner.py` | 1 |
| `src/risk/triple_barrier.py` | 1 |
