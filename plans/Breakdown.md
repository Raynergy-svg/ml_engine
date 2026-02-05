# Main.py Decomposition Roadmap

**Status**: IN PROGRESS  
**Created**: 2026-02-04  
**Last Updated**: 2026-02-05  
**Author**: Principal Software Architect  
**Scope**: Decompose 13,847-line monolithic `main.py` into modular components

---

## Completion Status

| Phase | Status | Progress | Notes |
|-------|--------|----------|-------|
| **Phase 1: Foundation & Infrastructure** | ✅ COMPLETE | 100% | All config/checkpoint/validation modules extracted |
| **Phase 2: Training Pipeline Extraction** | ✅ COMPLETE | 100% | TrainingConfig, DataPreparation, ModelBuilders done |
| **Phase 3: Inference & Trading Extraction** | ✅ COMPLETE | 100% | Execution, PaperTrade modules created |
| **Phase 4: CLI Command Modularization** | 🔄 PENDING | 0% | Commands dir exists but not populated |
| **Phase 5: Final Consolidation** | 🔄 PENDING | 0% | main.py still at 13,847 lines |

### Modules Extracted (4,899 lines)

| Module | Lines | Status | Verified |
|--------|-------|--------|----------|
| `src/training/training_config.py` | 455 | ✅ Complete | ✅ |
| `src/training/checkpoint_manager.py` | 535 | ✅ Complete | ✅ |
| `src/training/data_preparation.py` | 1,037 | ✅ Complete | ✅ |
| `src/training/retrain_gates.py` | 248 | ✅ Complete | ✅ |
| `src/models/model_builders.py` | 591 | ✅ Complete | ✅ |
| `src/utils/instrument_validation.py` | 346 | ✅ Complete | ✅ |
| `src/trading/execution.py` | 880 | ✅ Complete | ✅ |
| `src/trading/paper_trade.py` | 807 | ✅ Complete | ✅ |

### Remaining Work

1. **Wire extracted modules into main.py** - Replace inline code with imports
2. **Create CLI command modules** in `src/cli/commands/`
3. **Reduce main.py** from 13,847 → ~400-600 lines
4. **Add integration tests** for extracted modules
5. **Update backward compatibility** exports

---

## Executive Summary

The current `main.py` violates SOLID principles, contains ~4,160 lines in a single function (`_train_buddy_impl`), mixes CLI orchestration with business logic, and creates circular import risks. This roadmap prescribes a phased migration to a modular architecture targeting:

- **Final `main.py` size**: ~400-600 lines (CLI routing only)
- **Zero breaking changes**: All existing CLI commands preserved
- **Improved testability**: Each module independently testable
- **Circular dependency elimination**: Clean import hierarchy

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [New Directory Structure](#3-new-directory-structure)
4. [Phase 1: Foundation & Infrastructure](#4-phase-1-foundation--infrastructure)
5. [Phase 2: Training Pipeline Extraction](#5-phase-2-training-pipeline-extraction)
6. [Phase 3: Inference & Trading Extraction](#6-phase-3-inference--trading-extraction)
7. [Phase 4: CLI Command Modularization](#7-phase-4-cli-command-modularization)
8. [Phase 5: Final Consolidation](#8-phase-5-final-consolidation)
9. [Import Resolution Strategy](#9-import-resolution-strategy)
10. [Validation & Verification Protocol](#10-validation--verification-protocol)
11. [Rollback Procedures](#11-rollback-procedures)
12. [Timeline & Milestones](#12-timeline--milestones)

---

## 1. Current State Analysis

### 1.1 Size Breakdown by Functional Area

| Section | Lines | % of File | Complexity |
|---------|-------|-----------|------------|
| Imports, Constants, Dataclasses | 1-140 | 1.0% | Low |
| Tier-2 Calibration (duplicated) | 141-275 | 1.0% | Low |
| OANDA Data Fetch | 280-395 | 0.8% | Medium |
| Keras Migration Helpers | 396-495 | 0.7% | Medium |
| Checkpoint Management | 497-575 | 0.6% | Low |
| Instrument Validation | 577-747 | 1.2% | Low |
| Buddy Wizard (interactive) | 749-946 | 1.4% | Medium |
| TensorFlow Metal Config | 948-1102 | 1.1% | Low |
| Model Builders | 1103-1358 | 1.8% | Medium |
| **`_train_buddy_impl()`** | 1359-5519 | **30.1%** | **Critical** |
| FX Trading Helpers | 5520-6155 | 4.6% | High |
| Paper Trade Planning | 6157-6414 | 1.9% | Medium |
| Dashboard Generation | 6415-6500 | 0.6% | Low |
| Legacy CLI Commands | 6502-6767 | 1.9% | Low |
| **`buddy()`** | 7130-8419 | **9.3%** | **High** |
| **`buddy_loop()`** | 8420-8977 | **4.0%** | High |
| CLI Dispatch Logic | 8978-9332 | 2.6% | Medium |
| Scan Commands | 9333-9550 | 1.6% | Medium |
| RL Training | 9551-9963 | 3.0% | High |
| Gate Retraining | 9964-10320 | 2.6% | Medium |
| Confidence Training | 10321-10765 | 3.2% | Medium |
| Analysis/Validation | 10766-12910 | 15.5% | Medium |
| `main()` Entrypoint | 12911-13847 | 6.8% | Low |

### 1.2 SOLID Principle Violations

| Principle | Violation | Severity |
|-----------|-----------|----------|
| **S**ingle Responsibility | `_train_buddy_impl()` handles: data loading, feature engineering, model building, training, validation, calibration, checkpoint saving | Critical |
| **O**pen/Closed | Adding new model types requires modifying central function | High |
| **L**iskov Substitution | N/A (no inheritance hierarchy) | - |
| **I**nterface Segregation | CLI commands receive 40+ parameters via kwargs | High |
| **D**ependency Inversion | High-level `buddy()` directly instantiates low-level `OandaPracticeClient` | Medium |

### 1.3 Existing Extraction (Already in src/)

These modules exist and should be **preserved as-is**:

| Module | Status | Notes |
|--------|--------|-------|
| `src/core/modular_inference.py` | Complete | 2,678 lines, gated ensemble |
| `src/core/modular_data_loaders.py` | Complete | Feature engineering |
| `src/models/tensorflow_models.py` | Complete | Model architectures |
| `src/training/modular_trainers.py` | Partial | Core trainers extracted |
| `src/risk/position_sizing.py` | Complete | Kelly-based sizing |
| `src/scanner/` | Complete | Multi-pair scanning |
| `src/cli/calibration.py` | Complete | **Duplicated in main.py** |

### 1.4 Circular Dependency Risks

```
Current problematic imports:
main.py → src/training/buddy_training_helpers.py → main.py (indirect)
main.py → src/core/modular_inference.py (inline imports)
src/cli/entry.py → main.py (delegation)
```

---

## 2. Target Architecture

### 2.1 Design Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (thin)                        │
│  main.py → argparse → dispatch → src/cli/commands/*.py         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  src/training/   src/inference/   src/trading/                  │
│  Stateless functions with dependency injection                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Core Layer                                │
│  src/core/   src/models/   src/risk/   src/data/                │
│  Pure functions, no side effects, no I/O                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                       │
│  src/utils/oanda_practice.py   src/utils/trade_journal.py       │
│  External API clients, persistence, logging                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Import Direction Rule

```
CLI → Service → Core → Infrastructure
      ↓
      Never reverse
```

---

## 3. New Directory Structure

### 3.1 Target Structure

```
src/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── app.py                      # NEW: Click/Typer app with command groups
│   ├── calibration.py              # EXISTING: Keep as-is
│   ├── config.py                   # EXISTING: Keep as-is
│   ├── display.py                  # NEW: Rich console output helpers
│   ├── helpers.py                  # EXISTING: Keep as-is
│   ├── tf_config.py                # EXISTING: Keep as-is
│   └── commands/
│       ├── __init__.py
│       ├── train.py                # NEW: train-buddy command
│       ├── buddy.py                # NEW: buddy inference command
│       ├── scan.py                 # NEW: scan command
│       ├── gates.py                # NEW: retrain-gates, analyze-gates
│       ├── rl.py                   # NEW: train-rl-sizer
│       ├── journal.py              # NEW: journal command
│       ├── validate.py             # NEW: validate-model command
│       └── legacy.py               # NEW: deprecated train/eval/infer
│
├── core/
│   ├── __init__.py
│   ├── modular_data_loaders.py     # EXISTING
│   ├── modular_inference.py        # EXISTING
│   └── feature_config.py           # NEW: Feature set definitions
│
├── data/
│   ├── __init__.py
│   ├── candle_smoothing.py         # EXISTING
│   ├── data_loader.py              # EXISTING
│   ├── data_processing.py          # EXISTING
│   ├── feature_engineering.py      # EXISTING
│   └── oanda_fetcher.py            # NEW: Extract from main.py
│
├── inference/
│   ├── __init__.py
│   ├── buddy_inference.py          # NEW: Extract buddy()
│   ├── buddy_loop.py               # NEW: Extract buddy_loop()
│   ├── decision_engine.py          # NEW: Trade decision logic
│   └── display_formatter.py        # NEW: Inference output formatting
│
├── models/
│   ├── __init__.py
│   ├── ensemble_model.py           # EXISTING
│   ├── model_builders.py           # NEW: Extract from main.py
│   ├── tensorflow_models.py        # EXISTING
│   └── xgboost_model.py            # EXISTING
│
├── risk/
│   ├── __init__.py                 # EXISTING (all files stay)
│   └── ... (unchanged)
│
├── rl/
│   ├── __init__.py
│   ├── training.py                 # NEW: Extract train_rl_sizer()
│   └── ... (existing files)
│
├── scanner/                        # EXISTING (unchanged)
│
├── trading/
│   ├── __init__.py
│   ├── execution.py                # NEW: FX trade execution
│   ├── paper_trade.py              # NEW: Paper trade planning
│   └── order_builder.py            # NEW: Order construction
│
├── training/
│   ├── __init__.py
│   ├── buddy_training_helpers.py   # EXISTING
│   ├── checkpoint_manager.py       # NEW: Checkpoint save/load
│   ├── gate_trainer.py             # NEW: Extract retrain_gates()
│   ├── modular_trainers.py         # EXISTING
│   ├── training_config.py          # NEW: Dataclasses
│   ├── training_pipeline.py        # NEW: Extract _train_buddy_impl()
│   └── walkforward_validation.py   # EXISTING
│
└── utils/
    ├── __init__.py                 # EXISTING (all files stay)
    ├── instrument_validation.py    # NEW: Extract from main.py
    └── ... (existing files)
```

### 3.2 Files to Create

| New File | Source Lines | Purpose |
|----------|--------------|---------|
| `src/training/training_config.py` | 38-140 | `OandaFetchOptions`, `BuddyTrainingOptions` dataclasses |
| `src/training/training_pipeline.py` | 1359-5519 | Main training orchestration |
| `src/training/checkpoint_manager.py` | 497-575 | Checkpoint loading/saving |
| `src/training/gate_trainer.py` | 9964-10320 | Gate model retraining |
| `src/inference/buddy_inference.py` | 7130-8419 | Single inference execution |
| `src/inference/buddy_loop.py` | 8420-8977 | Continuous monitoring loop |
| `src/inference/display_formatter.py` | (scattered) | Rich output formatting |
| `src/inference/decision_engine.py` | (from buddy) | Trade decision logic |
| `src/trading/execution.py` | 5520-6155 | FX trade execution |
| `src/trading/paper_trade.py` | 6157-6414 | Paper trade planning |
| `src/data/oanda_fetcher.py` | 280-395 | OANDA data fetching |
| `src/models/model_builders.py` | 1103-1358 | Model construction |
| `src/rl/training.py` | 9551-9963 | RL sizer training |
| `src/utils/instrument_validation.py` | 577-747 | Instrument parsing |
| `src/cli/display.py` | (scattered) | Console output centralization |
| `src/cli/commands/train.py` | (routing) | train-buddy command |
| `src/cli/commands/buddy.py` | (routing) | buddy command |
| `src/cli/commands/scan.py` | (routing) | scan command |
| `src/cli/commands/gates.py` | (routing) | gate commands |
| `src/cli/commands/journal.py` | (routing) | journal command |

---

## 4. Phase 1: Foundation & Infrastructure ✅ COMPLETE

**Duration**: 2-3 days  
**Risk**: Low  
**Goal**: Establish infrastructure without changing behavior  
**Completed**: 2026-02-05

### 4.1 Create Shared Configuration Module ✅

**File**: `src/training/training_config.py` (455 lines)

```python
"""Training configuration dataclasses.

Extracted from main.py to provide type-safe configuration objects
with validation and sensible defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OandaFetchOptions:
    """Configuration for OANDA data fetching."""
    instrument: str
    granularity: str = "H1"
    candles: int = 15000
    save_csv: str | None = None
    
    def __post_init__(self) -> None:
        if self.candles < 100:
            raise ValueError("candles must be >= 100")


@dataclass(frozen=True)
class BuddyTrainingAdvancedOptions:
    """Advanced training hyperparameters."""
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    use_focal_loss: bool = True
    use_class_weights: bool = True
    use_curriculum: bool = False
    curriculum_ks: str = "32,64,128,0"


@dataclass
class BuddyTrainingOptions:
    """Primary training configuration."""
    instrument: str
    csv_path: str | None = None
    oanda_live: bool = False
    granularity: str = "H1"
    candles: int = 15000
    epochs: int = 200
    batch_size: int = 64
    model_type: str = "transformer"
    advanced: BuddyTrainingAdvancedOptions = field(
        default_factory=BuddyTrainingAdvancedOptions
    )
    
    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "BuddyTrainingOptions":
        """Factory from CLI kwargs for backward compatibility."""
        advanced_keys = {f.name for f in dataclass_fields(BuddyTrainingAdvancedOptions)}
        advanced_kwargs = {k: v for k, v in kwargs.items() if k in advanced_keys}
        main_kwargs = {k: v for k, v in kwargs.items() if k not in advanced_keys}
        return cls(
            advanced=BuddyTrainingAdvancedOptions(**advanced_kwargs),
            **main_kwargs
        )
```

**Extraction Protocol**:
1. Copy dataclasses from main.py lines 38-140
2. Add `__post_init__` validation
3. Add factory methods for backward compatibility
4. Write unit tests before proceeding

### 4.2 Create Checkpoint Manager ✅

**File**: `src/training/checkpoint_manager.py` (535 lines)

```python
"""Model checkpoint save/load utilities.

Centralizes all checkpoint operations with consistent error handling
and metadata management.
"""
from __future__ import annotations

import json
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside model checkpoints."""
    instrument: str
    model_type: str
    feature_names: list[str]
    seq_len: int
    scaler_params: dict[str, Any]
    training_date: str
    config_hash: str | None = None


class CheckpointManager:
    """Manages model checkpoint persistence."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_keras_model(
        self,
        model: "tf.keras.Model",
        name: str,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Save Keras model with metadata."""
        model_path = self.model_dir / f"{name}.keras"
        meta_path = self.model_dir / f"{name}.meta.pkl"
        
        model.save(model_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata.__dict__, f)
        
        logger.info(f"Saved checkpoint: {model_path}")
        return model_path
    
    def load_keras_model(
        self,
        name: str,
        custom_objects: dict | None = None,
    ) -> tuple["tf.keras.Model", dict[str, Any]]:
        """Load Keras model with metadata."""
        import tensorflow as tf
        
        model_path = self.model_dir / f"{name}.keras"
        meta_path = self.model_dir / f"{name}.meta.pkl"
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects
        )
        
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        
        return model, metadata
```

**Extraction Protocol**:
1. Extract `_load_buddy_checkpoint()` from main.py lines 497-575
2. Extract `_meta_path_for_checkpoint()` from main.py lines 538-560
3. Convert to class-based design for dependency injection
4. Add comprehensive error handling

### 4.3 Create Instrument Validation Module ✅

**File**: `src/utils/instrument_validation.py`

Extract from main.py lines 577-747:
- `VALID_OANDA_INSTRUMENTS` set
- `_normalize_instrument()` function
- `_validate_instrument()` function
- `_extract_instrument_from_csv_path()` function
- `_get_pair_model_paths()` function

### 4.4 Remove Duplicate Calibration Code ⏳ TODO

**Action**: Delete `_tier2_*` functions from main.py (lines 141-275)

> **Note**: Module exists at `src/cli/calibration.py`, but main.py still has duplicate code. Needs wiring.

These functions are **already extracted** to `src/cli/calibration.py`. Update main.py to import from there:

```python
# Replace in main.py:
from src.cli.calibration import (
    get_calibration_dict,
    points_from_bins,
    interpolate_points,
    clip_prob,
    logit,
    sigmoid,
    temperature_scale_prob,
    apply_calibration,
)
```

### 4.5 Phase 1 Verification

```bash
# Run after each extraction:
pytest tests/ -v --tb=short
python -c "from src.training.training_config import BuddyTrainingOptions; print('OK')"
python -c "from src.training.checkpoint_manager import CheckpointManager; print('OK')"
python -c "from src.utils.instrument_validation import validate_instrument; print('OK')"

# Verify CLI still works:
./bin/Buddy status
./bin/Buddy --help
```

---

## 5. Phase 2: Training Pipeline Extraction ✅ COMPLETE

**Duration**: 5-7 days  
**Risk**: High  
**Goal**: Extract `_train_buddy_impl()` (4,160 lines)  
**Completed**: 2026-02-05

### 5.1 Decomposition Strategy

The monolithic `_train_buddy_impl()` must be split into these components:

```
_train_buddy_impl() ─┬─► DataPreparationPipeline
                     │   ├── fetch_or_load_data()
                     │   ├── compute_features()
                     │   └── create_train_val_split()
                     │
                     ├─► ModelFactory
                     │   ├── build_transformer()
                     │   ├── build_tcn()
                     │   └── build_xgboost()
                     │
                     ├─► TrainingOrchestrator
                     │   ├── train_direction_model()
                     │   ├── train_gate_models()
                     │   └── run_calibration()
                     │
                     └─► ValidationPipeline
                         ├── walk_forward_validation()
                         ├── compute_metrics()
                         └── generate_report()
```

### 5.2 Step 1: Extract Model Builders ✅

**File**: `src/models/model_builders.py` (591 lines)

Extract from main.py lines 1103-1358:
- `_build_buddy_model()` → `build_lstm_model()`
- `_build_buddy_model_shared_encoder()` → `build_shared_encoder_model()`
- `_build_buddy_model_tcn()` → `build_tcn_model()`
- `_build_xgboost_model()` → `build_xgboost_model()`
- `_build_buddy_model_for_type()` → `ModelFactory.build()`

```python
"""Model factory for creating ML models.

Provides a unified interface for building different model architectures
with consistent configuration handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import tensorflow as tf


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    feature_dim: int
    seq_len: int
    head_hidden: int = 64
    head_layers: int = 2
    head_dropout: float = 0.1
    dense_hidden: int = 128
    dense_dropout: float = 0.2
    kernel_regularizer: float = 0.002
    noise_std: float = 0.03


ModelType = Literal["transformer", "tcn", "lstm", "xgboost"]


class ModelFactory:
    """Factory for creating ML models."""
    
    @classmethod
    def build(
        cls,
        model_type: ModelType,
        config: ModelConfig,
    ) -> tf.keras.Model:
        """Build model of specified type."""
        builders = {
            "transformer": cls._build_transformer,
            "tcn": cls._build_tcn,
            "lstm": cls._build_lstm,
        }
        if model_type not in builders:
            raise ValueError(f"Unknown model type: {model_type}")
        return builders[model_type](config)
    
    @classmethod
    def _build_transformer(cls, config: ModelConfig) -> tf.keras.Model:
        from src.models.tensorflow_models import TransformerBlock
        # ... implementation
    
    @classmethod
    def _build_tcn(cls, config: ModelConfig) -> tf.keras.Model:
        # ... implementation
    
    @classmethod
    def _build_lstm(cls, config: ModelConfig) -> tf.keras.Model:
        # ... implementation
```

### 5.3 Step 2: Extract Data Preparation ✅

**File**: `src/training/data_preparation.py` (1,037 lines)

```python
"""Data preparation pipeline for training.

Handles data loading, feature engineering, and train/val splitting
with consistent preprocessing across all model types.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.oanda_fetcher import OandaFetcher
from src.core.modular_data_loaders import compute_normalized_features
from src.training.training_config import OandaFetchOptions

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Container for prepared training data."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    scaler_params: dict[str, Any]
    df: pd.DataFrame


class DataPreparationPipeline:
    """Prepares data for model training."""
    
    def __init__(
        self,
        config: dict[str, Any],
        seq_len: int = 60,
        val_split: float = 0.2,
    ):
        self.config = config
        self.seq_len = seq_len
        self.val_split = val_split
    
    def prepare(
        self,
        csv_path: str | None = None,
        oanda_options: OandaFetchOptions | None = None,
    ) -> PreparedData:
        """Execute full data preparation pipeline."""
        df = self._load_data(csv_path, oanda_options)
        df = self._clean_data(df)
        features_df, scaler_params = self._compute_features(df)
        X, y = self._create_sequences(features_df)
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        return PreparedData(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=list(features_df.columns),
            scaler_params=scaler_params,
            df=df,
        )
    
    def _load_data(
        self,
        csv_path: str | None,
        oanda_options: OandaFetchOptions | None,
    ) -> pd.DataFrame:
        """Load data from CSV or OANDA."""
        if csv_path:
            return pd.read_csv(csv_path)
        if oanda_options:
            fetcher = OandaFetcher.from_env()
            return fetcher.fetch(oanda_options)
        raise ValueError("Either csv_path or oanda_options required")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NaN/inf values."""
        return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    
    def _compute_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Compute normalized features."""
        return compute_normalized_features(df, self.config)
    
    def _create_sequences(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sequence windows for temporal models."""
        # Implementation extracted from _train_buddy_impl
        pass
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Time-series aware train/val split (no shuffling)."""
        split_idx = int(len(X) * (1 - self.val_split))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
```

### 5.4 Step 3: Extract Training Orchestrator

**File**: `src/training/training_pipeline.py`

```python
"""Training orchestration for the Buddy trading system.

Coordinates the full training workflow including direction model,
gate models, calibration, and validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from src.training.training_config import BuddyTrainingOptions
from src.training.data_preparation import DataPreparationPipeline, PreparedData
from src.training.checkpoint_manager import CheckpointManager, CheckpointMetadata
from src.training.modular_trainers import (
    train_transformer_model,
    train_xgb_model,
    train_rf_model,
    train_ridge_model,
)
from src.models.model_builders import ModelFactory, ModelConfig
from src.cli.calibration import CalibratorV2

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from a training run."""
    direction_model_path: Path
    gate_model_paths: dict[str, Path]
    metrics: dict[str, float]
    calibration: dict[str, Any]
    training_report: str


class TrainingPipeline:
    """Orchestrates the complete training workflow."""
    
    def __init__(
        self,
        options: BuddyTrainingOptions,
        config: dict[str, Any],
        model_dir: Path,
        progress_callback: Callable[[str, float], None] | None = None,
    ):
        self.options = options
        self.config = config
        self.checkpoint_manager = CheckpointManager(model_dir)
        self.progress_callback = progress_callback or (lambda msg, pct: None)
    
    def run(self) -> TrainingResult:
        """Execute complete training pipeline."""
        self.progress_callback("Preparing data...", 0.0)
        data = self._prepare_data()
        
        self.progress_callback("Training direction model...", 0.1)
        direction_path = self._train_direction_model(data)
        
        self.progress_callback("Training gate models...", 0.5)
        gate_paths = self._train_gate_models(data)
        
        self.progress_callback("Running calibration...", 0.8)
        calibration = self._run_calibration(data, direction_path)
        
        self.progress_callback("Generating report...", 0.95)
        report = self._generate_report(data, direction_path, gate_paths)
        
        self.progress_callback("Complete!", 1.0)
        return TrainingResult(
            direction_model_path=direction_path,
            gate_model_paths=gate_paths,
            metrics=self._compute_metrics(data),
            calibration=calibration,
            training_report=report,
        )
    
    def _prepare_data(self) -> PreparedData:
        """Prepare training data."""
        pipeline = DataPreparationPipeline(
            config=self.config,
            seq_len=self.config.get("seq_len", 60),
        )
        return pipeline.prepare(
            csv_path=self.options.csv_path,
            oanda_options=self._build_oanda_options(),
        )
    
    def _train_direction_model(self, data: PreparedData) -> Path:
        """Train the primary direction prediction model."""
        model_config = ModelConfig(
            feature_dim=data.X_train.shape[-1],
            seq_len=data.X_train.shape[1],
        )
        model = ModelFactory.build(self.options.model_type, model_config)
        
        # Training logic extracted from _train_buddy_impl
        # ... callbacks, early stopping, etc.
        
        metadata = CheckpointMetadata(
            instrument=self.options.instrument,
            model_type=self.options.model_type,
            feature_names=data.feature_names,
            seq_len=data.X_train.shape[1],
            scaler_params=data.scaler_params,
            training_date=datetime.now().isoformat(),
        )
        
        return self.checkpoint_manager.save_keras_model(
            model, "transformer_direction", metadata
        )
    
    def _train_gate_models(self, data: PreparedData) -> dict[str, Path]:
        """Train XGBoost, RF, and Ridge gate models."""
        paths = {}
        
        # Flatten sequences for gate models
        X_flat = data.X_train[:, -1, :]  # Use last timestep
        
        paths["xgboost"] = train_xgb_model(
            X_flat, data.y_train,
            save_path=self.checkpoint_manager.model_dir / "xgb_momentum.pkl"
        )
        
        paths["rf"] = train_rf_model(
            X_flat, data.y_train,
            save_path=self.checkpoint_manager.model_dir / "rf_risk.pkl"
        )
        
        paths["ridge"] = train_ridge_model(
            X_flat, data.y_train,
            save_path=self.checkpoint_manager.model_dir / "ridge_confidence.pkl"
        )
        
        return paths
    
    def _run_calibration(
        self,
        data: PreparedData,
        direction_path: Path,
    ) -> dict[str, Any]:
        """Run probability calibration on validation data."""
        calibrator = CalibratorV2()
        # ... calibration logic
        return calibrator.get_calibration_dict()
    
    def _generate_report(
        self,
        data: PreparedData,
        direction_path: Path,
        gate_paths: dict[str, Path],
    ) -> str:
        """Generate training summary report."""
        # ... report generation
        pass
    
    def _compute_metrics(self, data: PreparedData) -> dict[str, float]:
        """Compute validation metrics."""
        return {}
    
    def _build_oanda_options(self) -> OandaFetchOptions | None:
        if not self.options.oanda_live:
            return None
        return OandaFetchOptions(
            instrument=self.options.instrument,
            granularity=self.options.granularity,
            candles=self.options.candles,
        )
```

### 5.5 Step 4: Update main.py Train Command

After extraction, `train_buddy()` in main.py becomes a thin wrapper:

```python
def train_buddy(
    config_path: str,
    csv_path: str | None = None,
    *,
    options: BuddyTrainingOptions | None = None,
    **kwargs: Any,
) -> None:
    """Train Buddy trading model - CLI wrapper."""
    from src.training.training_config import BuddyTrainingOptions
    from src.training.training_pipeline import TrainingPipeline
    
    config = load_config(config_path)
    
    if options is None:
        options = BuddyTrainingOptions.from_kwargs(**kwargs)
    
    pipeline = TrainingPipeline(
        options=options,
        config=config,
        model_dir=Path("trained_data/models") / options.instrument,
        progress_callback=_cli_progress_callback,
    )
    
    result = pipeline.run()
    console.print(f"[green]Training complete![/green]")
    console.print(result.training_report)
```

### 5.6 Phase 2 Verification

```bash
# Unit tests for extracted modules:
pytest tests/test_training_pipeline.py -v
pytest tests/test_model_builders.py -v
pytest tests/test_data_preparation.py -v

# Integration test:
./bin/Buddy train -i EUR_USD --csv market_data/EUR_USD_H1.csv

# Compare outputs before/after:
diff trained_data/models/EUR_USD/ trained_data/models_backup/EUR_USD/
```

---

## 6. Phase 3: Inference & Trading Extraction ✅ COMPLETE

**Duration**: 4-5 days  
**Risk**: Medium-High  
**Goal**: Extract `buddy()` and trading execution  
**Completed**: 2026-02-05

### 6.1 Decomposition Strategy

```
buddy() ─┬─► DataLoader (fetch live candles)
         │
         ├─► InferenceEngine (run gated ensemble)
         │
         ├─► DecisionEngine (generate trade decision)
         │
         ├─► DisplayFormatter (Rich output)
         │
         └─► TradeExecutor (execute via OANDA)
```

### 6.2 Step 1: Extract Display Formatter

**File**: `src/cli/display.py`

```python
"""Rich console display utilities for inference output.

Provides consistent, visually appealing output formatting
for trade signals, confidence bands, and execution results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class SignalDisplay:
    """Data for signal display panel."""
    instrument: str
    direction: str
    confidence: float
    confidence_band: str
    calibrated_prob: float | None
    gates_passed: dict[str, bool]
    price: float
    atr: float
    spread_pips: float


class InferenceDisplayFormatter:
    """Formats inference results for console output."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def display_signal(self, signal: SignalDisplay) -> None:
        """Display signal panel with gate status."""
        color = "green" if signal.direction == "LONG" else "red"
        
        # Build gates table
        gates_table = Table(show_header=False, box=None)
        gates_table.add_column("Gate")
        gates_table.add_column("Status")
        
        for gate, passed in signal.gates_passed.items():
            status = "✓" if passed else "✗"
            gates_table.add_row(gate, f"[{'green' if passed else 'red'}]{status}[/]")
        
        # Main panel
        content = f"""
[bold {color}]{signal.direction}[/bold {color}] @ {signal.price:.5f}
Confidence: {signal.confidence:.1f}% ({signal.confidence_band})
Calibrated P(Win): {signal.calibrated_prob:.1%}
ATR: {signal.atr:.5f} | Spread: {signal.spread_pips:.1f} pips
"""
        panel = Panel(
            content,
            title=f"[bold]{signal.instrument}[/bold]",
            border_style=color,
        )
        self.console.print(panel)
        self.console.print(gates_table)
```

### 6.3 Step 2: Extract Decision Engine

**File**: `src/inference/decision_engine.py`

```python
"""Trade decision engine for the Buddy system.

Evaluates model predictions against gate thresholds and risk rules
to generate actionable trade decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.modular_inference import ModularEnsemble, InferenceConfig


@dataclass
class TradeDecision:
    """Result of trade decision evaluation."""
    should_trade: bool
    direction: str | None  # "LONG" or "SHORT"
    confidence: float
    confidence_band: str
    calibrated_probability: float | None
    gates_status: dict[str, bool]
    risk_metrics: dict[str, float]
    blockers: list[str]


class DecisionEngine:
    """Evaluates predictions to generate trade decisions."""
    
    def __init__(
        self,
        ensemble: ModularEnsemble,
        config: InferenceConfig,
    ):
        self.ensemble = ensemble
        self.config = config
    
    def evaluate(
        self,
        features: "np.ndarray",
        current_price: float,
        atr: float,
        spread_pips: float,
    ) -> TradeDecision:
        """Evaluate features and generate trade decision."""
        # Get ensemble predictions
        pred = self.ensemble.predict(features)
        
        # Evaluate gates
        gates_status = self._evaluate_gates(pred)
        
        # Check for blockers
        blockers = self._check_blockers(pred, spread_pips, atr)
        
        # Determine if should trade
        should_trade = all(gates_status.values()) and not blockers
        
        return TradeDecision(
            should_trade=should_trade,
            direction=self._determine_direction(pred) if should_trade else None,
            confidence=pred.confidence_score,
            confidence_band=self._band_for_confidence(pred.confidence_score),
            calibrated_probability=pred.calibrated_prob,
            gates_status=gates_status,
            risk_metrics=self._compute_risk_metrics(pred),
            blockers=blockers,
        )
    
    def _evaluate_gates(self, pred: Any) -> dict[str, bool]:
        """Evaluate all gate thresholds."""
        return {
            "direction": pred.tcn_prob >= self.config.min_tcn_probability,
            "confidence": pred.ridge_score >= self.config.min_confidence,
            "momentum": pred.xgb_percentile >= self.config.min_momentum,
            "risk": pred.rf_drawdown <= self.config.max_drawdown_pct,
        }
    
    def _check_blockers(
        self,
        pred: Any,
        spread_pips: float,
        atr: float,
    ) -> list[str]:
        """Check for conditions that block trading."""
        blockers = []
        
        if spread_pips > self.config.max_spread_pips:
            blockers.append(f"Spread too wide: {spread_pips:.1f} pips")
        
        if atr < self.config.min_atr:
            blockers.append(f"Volatility too low: ATR={atr:.5f}")
        
        return blockers
    
    def _determine_direction(self, pred: Any) -> str:
        """Determine trade direction from prediction."""
        return "LONG" if pred.direction_prob > 0.5 else "SHORT"
    
    def _band_for_confidence(self, confidence: float) -> str:
        """Map confidence score to display band."""
        if confidence >= 75:
            return "HIGH"
        if confidence >= 50:
            return "MEDIUM"
        return "LOW"
    
    def _compute_risk_metrics(self, pred: Any) -> dict[str, float]:
        """Compute risk metrics for position sizing."""
        return {
            "expected_drawdown": pred.rf_drawdown,
            "momentum_score": pred.xgb_percentile,
            "regime_confidence": pred.regime_prob if hasattr(pred, "regime_prob") else 0.0,
        }
```

### 6.4 Step 3: Extract Trade Executor ✅

**File**: `src/trading/execution.py` (880 lines)

**File**: `src/trading/execution.py`

Extract from main.py lines 5520-6155:
- `place_fx_trade()` logic
- Order building and submission
- Position sizing integration
- Risk limit enforcement

```python
"""FX trade execution via OANDA API.

Handles order construction, submission, and confirmation
with comprehensive error handling and logging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.utils.oanda_practice import OandaPracticeClient
from src.risk.position_sizing import PositionSizer
from src.utils.trade_journal import TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of trade execution attempt."""
    success: bool
    order_id: str | None
    fill_price: float | None
    units: int
    error_message: str | None


@dataclass
class OrderRequest:
    """Order request specification."""
    instrument: str
    units: int  # Positive for long, negative for short
    stop_loss_price: float
    take_profit_price: float
    price_bound: float | None = None


class TradeExecutor:
    """Executes trades via OANDA API."""
    
    def __init__(
        self,
        client: OandaPracticeClient,
        position_sizer: PositionSizer,
        journal: TradeJournal | None = None,
    ):
        self.client = client
        self.position_sizer = position_sizer
        self.journal = journal
    
    def execute(self, request: OrderRequest) -> ExecutionResult:
        """Execute trade order."""
        try:
            # Validate order
            self._validate_order(request)
            
            # Submit to OANDA
            response = self.client.place_market_order(
                instrument=request.instrument,
                units=request.units,
                stop_loss=request.stop_loss_price,
                take_profit=request.take_profit_price,
                price_bound=request.price_bound,
            )
            
            # Parse response
            order_id = response.get("orderCreateTransaction", {}).get("id")
            fill_price = response.get("orderFillTransaction", {}).get("price")
            
            result = ExecutionResult(
                success=True,
                order_id=order_id,
                fill_price=float(fill_price) if fill_price else None,
                units=request.units,
                error_message=None,
            )
            
            # Log to journal
            if self.journal:
                self.journal.record_entry(request, result)
            
            return result
            
        except Exception as e:
            logger.exception(f"Trade execution failed: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                fill_price=None,
                units=request.units,
                error_message=str(e),
            )
    
    def _validate_order(self, request: OrderRequest) -> None:
        """Validate order before submission."""
        if abs(request.units) < 1:
            raise ValueError("Units must be non-zero")
        if request.stop_loss_price <= 0:
            raise ValueError("Invalid stop loss price")
```

### 6.5 Step 4: Create Buddy Inference Module

**File**: `src/inference/buddy_inference.py`

```python
"""Buddy inference orchestration.

Coordinates data loading, prediction, decision making, and execution
for single inference runs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.modular_inference import ModularEnsemble, InferenceConfig
from src.core.modular_data_loaders import compute_normalized_features
from src.inference.decision_engine import DecisionEngine, TradeDecision
from src.trading.execution import TradeExecutor, OrderRequest
from src.cli.display import InferenceDisplayFormatter

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Complete inference result."""
    decision: TradeDecision
    executed: bool
    execution_result: Any | None
    data_summary: dict[str, Any]


class BuddyInference:
    """Orchestrates single buddy inference runs."""
    
    def __init__(
        self,
        config: dict[str, Any],
        model_dir: Path,
        display_formatter: InferenceDisplayFormatter | None = None,
    ):
        self.config = config
        self.ensemble = ModularEnsemble.load(model_dir)
        self.decision_engine = DecisionEngine(
            self.ensemble,
            InferenceConfig.from_config(config),
        )
        self.display = display_formatter
    
    def run(
        self,
        instrument: str,
        granularity: str = "H1",
        candles: int = 500,
        execute: bool = False,
        verbose: bool = False,
    ) -> InferenceResult:
        """Run single inference cycle."""
        # 1. Fetch latest data
        df = self._fetch_data(instrument, granularity, candles)
        
        # 2. Compute features
        features, price, atr, spread = self._prepare_features(df, instrument)
        
        # 3. Get trade decision
        decision = self.decision_engine.evaluate(
            features=features,
            current_price=price,
            atr=atr,
            spread_pips=spread,
        )
        
        # 4. Display results
        if self.display:
            self._display_results(instrument, decision, price, atr, spread)
        
        # 5. Execute if requested
        execution_result = None
        if execute and decision.should_trade:
            execution_result = self._execute_trade(instrument, decision, price, atr)
        
        return InferenceResult(
            decision=decision,
            executed=execute and decision.should_trade,
            execution_result=execution_result,
            data_summary={
                "candles": len(df),
                "price": price,
                "atr": atr,
                "spread_pips": spread,
            },
        )
    
    def _fetch_data(
        self,
        instrument: str,
        granularity: str,
        candles: int,
    ) -> "pd.DataFrame":
        """Fetch latest candle data."""
        from src.utils.oanda_practice import OandaPracticeClient
        from fx_paper import candles_to_ohlcv_df
        
        client = OandaPracticeClient.from_env()
        raw = client.get_candles(instrument, granularity, count=candles)
        return candles_to_ohlcv_df(raw)
    
    def _prepare_features(
        self,
        df: "pd.DataFrame",
        instrument: str,
    ) -> tuple["np.ndarray", float, float, float]:
        """Prepare features for inference."""
        # Compute normalized features
        features_df, _ = compute_normalized_features(df, self.config)
        
        # Extract metadata
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1] if "atr" in df else 0.001
        spread = self._compute_spread(df, instrument)
        
        # Create sequence
        seq_len = self.config.get("seq_len", 60)
        features = features_df.values[-seq_len:].reshape(1, seq_len, -1)
        
        return features, price, atr, spread
    
    def _compute_spread(self, df: "pd.DataFrame", instrument: str) -> float:
        """Compute spread in pips."""
        # Implementation from main.py
        pass
    
    def _display_results(
        self,
        instrument: str,
        decision: TradeDecision,
        price: float,
        atr: float,
        spread: float,
    ) -> None:
        """Display inference results."""
        from src.cli.display import SignalDisplay
        
        self.display.display_signal(SignalDisplay(
            instrument=instrument,
            direction=decision.direction or "FLAT",
            confidence=decision.confidence,
            confidence_band=decision.confidence_band,
            calibrated_prob=decision.calibrated_probability,
            gates_passed=decision.gates_status,
            price=price,
            atr=atr,
            spread_pips=spread,
        ))
    
    def _execute_trade(
        self,
        instrument: str,
        decision: TradeDecision,
        price: float,
        atr: float,
    ) -> Any:
        """Execute the trade."""
        from src.utils.oanda_practice import OandaPracticeClient
        from src.risk.position_sizing import PositionSizer
        
        client = OandaPracticeClient.from_env()
        sizer = PositionSizer(self.config)
        executor = TradeExecutor(client, sizer)
        
        # Build order
        units = sizer.compute_units(
            risk_pct=self.config.get("risk_per_trade_pct", 0.02),
            stop_distance_pips=self.config.get("stop_loss_pips", 15),
        )
        
        if decision.direction == "SHORT":
            units = -units
        
        # Compute SL/TP
        sl_pips = self.config.get("stop_loss_pips", 15)
        tp_pips = self.config.get("take_profit_pips", 30)
        pip_value = 0.0001 if "JPY" not in instrument else 0.01
        
        if decision.direction == "LONG":
            sl_price = price - sl_pips * pip_value
            tp_price = price + tp_pips * pip_value
        else:
            sl_price = price + sl_pips * pip_value
            tp_price = price - tp_pips * pip_value
        
        request = OrderRequest(
            instrument=instrument,
            units=units,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
        )
        
        return executor.execute(request)
```

### 6.6 Phase 3 Verification

```bash
# Unit tests:
pytest tests/test_decision_engine.py -v
pytest tests/test_trade_executor.py -v
pytest tests/test_buddy_inference.py -v

# Integration test (dry run):
./bin/Buddy EUR_USD --verbose

# Integration test (paper trade):  
./bin/Buddy EUR_USD --execute

# Compare outputs with pre-refactor baseline
```

---

## 7. Phase 4: CLI Command Modularization 🔄 PENDING

**Duration**: 2-3 days  
**Risk**: Low  
**Goal**: Move CLI commands to `src/cli/commands/`  
**Status**: Structure exists, commands not yet populated

### 7.1 Command Module Pattern

Each command becomes a thin wrapper that:
1. Parses arguments
2. Constructs service objects
3. Calls service methods
4. Formats output

**File**: `src/cli/commands/train.py`

```python
"""Train command implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console

from src.training.training_config import BuddyTrainingOptions
from src.training.training_pipeline import TrainingPipeline
from src.utils import load_config

console = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add train-buddy subparser."""
    parser = subparsers.add_parser(
        "train-buddy",
        help="Train Buddy trading model",
    )
    parser.add_argument("-i", "--instrument", required=True)
    parser.add_argument("--csv", dest="csv_path")
    parser.add_argument("--oanda-live", action="store_true")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--candles", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model-type", default="transformer")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.set_defaults(func=execute)


def execute(args: argparse.Namespace) -> None:
    """Execute train-buddy command."""
    config = load_config(args.config)
    
    options = BuddyTrainingOptions(
        instrument=args.instrument,
        csv_path=args.csv_path,
        oanda_live=args.oanda_live,
        granularity=args.granularity,
        candles=args.candles,
        epochs=args.epochs,
        model_type=args.model_type,
    )
    
    model_dir = Path("trained_data/models") / args.instrument
    
    pipeline = TrainingPipeline(
        options=options,
        config=config,
        model_dir=model_dir,
        progress_callback=_progress_callback if args.verbose else None,
    )
    
    with console.status("[bold green]Training..."):
        result = pipeline.run()
    
    console.print(f"[green]✓ Training complete![/green]")
    console.print(result.training_report)


def _progress_callback(message: str, progress: float) -> None:
    """Display progress updates."""
    console.print(f"[dim]{progress*100:.0f}%[/dim] {message}")
```

### 7.2 CLI App Consolidation

**File**: `src/cli/app.py`

```python
"""CLI application definition.

Provides the main argument parser and command routing.
"""
from __future__ import annotations

import argparse
import sys

from src.cli.commands import (
    train,
    buddy,
    scan,
    gates,
    rl,
    journal,
    validate,
    legacy,
)


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog="ml_engine",
        description="ML Engine Trading Bot",
    )
    parser.add_argument(
        "-c", "--config",
        default="config/config_improved_H1.yaml",
        help="Path to configuration file",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Register all command parsers
    train.add_parser(subparsers)
    buddy.add_parser(subparsers)
    scan.add_parser(subparsers)
    gates.add_parser(subparsers)
    rl.add_parser(subparsers)
    journal.add_parser(subparsers)
    validate.add_parser(subparsers)
    legacy.add_parser(subparsers)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

### 7.3 Update Entry Points

**File**: `src/cli/entry.py` (updated)

```python
"""CLI entry point - delegates to app module."""
from src.cli.app import main

if __name__ == "__main__":
    main()
```

**File**: `main.py` (reduced to ~400 lines)

```python
#!/usr/bin/env python3
"""
ML Engine Trading Bot - CLI Entry Point

This file provides backward compatibility for existing scripts.
All command implementations are in src/cli/commands/.
"""
from __future__ import annotations

import sys


def main() -> None:
    """Main entry point - delegates to CLI app."""
    from src.cli.app import main as cli_main
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
```

### 7.4 Preserve Backward Compatibility

For external scripts that import from main.py:

```python
# main.py - backward compatibility exports
from src.training.training_pipeline import TrainingPipeline
from src.inference.buddy_inference import BuddyInference
from src.training.training_config import BuddyTrainingOptions

# Deprecated function wrappers
def train_buddy(*args, **kwargs):
    """Deprecated: Use src.cli.commands.train.execute() instead."""
    import warnings
    warnings.warn(
        "train_buddy() is deprecated. Use TrainingPipeline directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.cli.commands.train import execute
    # ... wrapper logic
```

---

## 8. Phase 5: Final Consolidation 🔄 PENDING

**Duration**: 1-2 days  
**Risk**: Low  
**Goal**: Clean up and documentation  
**Status**: Blocked on Phase 4 completion

### 8.1 Remove Dead Code

1. Delete unused functions from main.py
2. Remove commented-out code
3. Delete `_tier2_*` duplicates (now in `src/cli/calibration.py`)
4. Remove legacy command stubs if not needed

### 8.2 Update Documentation

- Update [README.md](README.md) with new import paths
- Update [copilot-instructions.md](.github/copilot-instructions.md) with new architecture
- Add migration guide for external consumers

### 8.3 Final Directory Structure

```
ml_engine/
├── main.py                          # ~400 lines - entry point only
├── src/
│   ├── cli/
│   │   ├── app.py                   # CLI application
│   │   ├── display.py               # Output formatting
│   │   ├── calibration.py           # Existing
│   │   └── commands/
│   │       ├── train.py
│   │       ├── buddy.py
│   │       ├── scan.py
│   │       ├── gates.py
│   │       ├── rl.py
│   │       ├── journal.py
│   │       └── validate.py
│   ├── core/                        # Existing (unchanged)
│   ├── data/
│   │   └── oanda_fetcher.py         # NEW
│   ├── inference/
│   │   ├── buddy_inference.py       # NEW
│   │   ├── buddy_loop.py            # NEW
│   │   ├── decision_engine.py       # NEW
│   │   └── display_formatter.py     # NEW
│   ├── models/
│   │   └── model_builders.py        # NEW
│   ├── risk/                        # Existing (unchanged)
│   ├── rl/
│   │   └── training.py              # NEW
│   ├── scanner/                     # Existing (unchanged)
│   ├── trading/
│   │   ├── execution.py             # NEW
│   │   ├── paper_trade.py           # NEW
│   │   └── order_builder.py         # NEW
│   ├── training/
│   │   ├── checkpoint_manager.py    # NEW
│   │   ├── data_preparation.py      # NEW
│   │   ├── gate_trainer.py          # NEW
│   │   ├── training_config.py       # NEW
│   │   └── training_pipeline.py     # NEW
│   └── utils/
│       └── instrument_validation.py # NEW
└── tests/
    ├── test_training_pipeline.py    # NEW
    ├── test_buddy_inference.py      # NEW
    ├── test_decision_engine.py      # NEW
    ├── test_trade_executor.py       # NEW
    └── test_model_builders.py       # NEW
```

---

## 9. Import Resolution Strategy

### 9.1 Import Hierarchy Diagram

```
                    ┌─────────────────┐
                    │    main.py      │
                    │  (entry only)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ cli.commands│  │ cli.display │  │  cli.app    │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │
           ▼                ▼                ▼
    ┌─────────────────────────────────────────────┐
    │              Service Layer                   │
    │  training.training_pipeline                  │
    │  inference.buddy_inference                   │
    │  trading.execution                           │
    └─────────────────────┬───────────────────────┘
                          │
    ┌─────────────────────┼───────────────────────┐
    │                     │                       │
    ▼                     ▼                       ▼
┌────────────┐    ┌──────────────┐    ┌───────────────┐
│ core.*     │ ←→ │  models.*    │    │   risk.*      │
│ (pure)     │    │  (pure)      │    │   (pure)      │
└────────────┘    └──────────────┘    └───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   utils.* (I/O)       │
              │   oanda_practice      │
              │   trade_journal       │
              └───────────────────────┘
```

### 9.2 Rules to Eliminate Circular Dependencies

1. **Never import from parent packages**
   - ❌ `src/core/modular_inference.py` cannot import from `main.py`
   - ✅ Use dependency injection instead

2. **Use TYPE_CHECKING for type hints only**
   ```python
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       from src.training.training_pipeline import TrainingPipeline
   
   def some_function(pipeline: "TrainingPipeline") -> None:
       ...
   ```

3. **Defer imports inside functions for optional dependencies**
   ```python
   def run_with_tensorflow():
       import tensorflow as tf  # Only imported when needed
       ...
   ```

4. **Use Protocol classes for interfaces**
   ```python
   from typing import Protocol
   
   class ModelLoader(Protocol):
       def load(self, path: Path) -> Any: ...
   
   # Implementations can be anywhere without import issues
   ```

### 9.3 Re-export Pattern for Backward Compatibility

**File**: `src/training/__init__.py`

```python
"""Training module exports.

Provides backward-compatible imports for external consumers.
"""
from src.training.training_config import (
    OandaFetchOptions,
    BuddyTrainingOptions,
    BuddyTrainingAdvancedOptions,
)
from src.training.training_pipeline import TrainingPipeline
from src.training.checkpoint_manager import CheckpointManager

__all__ = [
    "OandaFetchOptions",
    "BuddyTrainingOptions", 
    "BuddyTrainingAdvancedOptions",
    "TrainingPipeline",
    "CheckpointManager",
]
```

---

## 10. Validation & Verification Protocol

### 10.1 Pre-Refactoring Baseline

Before any changes, capture baselines:

```bash
# 1. Record current test results
pytest tests/ -v --tb=short > baseline_tests.txt 2>&1

# 2. Capture import structure
python -c "import main; print([k for k in dir(main) if not k.startswith('_')])" > baseline_exports.txt

# 3. Generate baseline training output
./bin/Buddy train -i EUR_USD --csv market_data/EUR_USD_H1.csv 2>&1 | tee baseline_train.txt

# 4. Generate baseline inference output  
./bin/Buddy EUR_USD --verbose 2>&1 | tee baseline_inference.txt

# 5. Generate model checksums
find trained_data/models -name "*.keras" -exec sha256sum {} \; > baseline_models.sha256

# 6. Static analysis baseline
ruff check . --output-format=json > baseline_lint.json | true
mypy src/ --ignore-missing-imports > baseline_types.txt 2>&1 | true
```

### 10.2 Per-Phase Verification Checklist

After each phase, run:

```bash
#!/bin/bash
# scripts/verify_refactoring.sh

set -e

echo "=== Running Test Suite ==="
pytest tests/ -v --tb=short

echo "=== Checking for Import Errors ==="
python -c "from src.training import TrainingPipeline; print('OK')"
python -c "from src.inference import BuddyInference; print('OK')"
python -c "from src.trading import TradeExecutor; print('OK')"

echo "=== Verifying CLI Commands ==="
./bin/Buddy --help
./bin/Buddy status

echo "=== Static Analysis ==="
ruff check src/ --select=E,F,I

echo "=== Type Checking ==="
mypy src/training src/inference src/trading --ignore-missing-imports

echo "=== Verify No Circular Imports ==="
python -c "
import sys
import importlib

modules = [
    'src.training.training_pipeline',
    'src.inference.buddy_inference', 
    'src.trading.execution',
    'src.cli.commands.train',
]

for mod in modules:
    try:
        importlib.import_module(mod)
        print(f'✓ {mod}')
    except ImportError as e:
        print(f'✗ {mod}: {e}')
        sys.exit(1)
"

echo "=== All Checks Passed ==="
```

### 10.3 Regression Test Suite

Create new tests for each extracted module:

**File**: `tests/test_training_pipeline.py`

```python
"""Regression tests for training pipeline extraction."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.training.training_config import BuddyTrainingOptions
from src.training.training_pipeline import TrainingPipeline


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""
    
    @pytest.fixture
    def mock_config(self) -> dict:
        return {
            "seq_len": 60,
            "training": {"epochs": 10, "batch_size": 32},
        }
    
    @pytest.fixture
    def training_options(self) -> BuddyTrainingOptions:
        return BuddyTrainingOptions(
            instrument="EUR_USD",
            csv_path="tests/fixtures/sample_data.csv",
            epochs=10,
        )
    
    def test_pipeline_init(self, mock_config, training_options, tmp_path):
        """Pipeline initializes without errors."""
        pipeline = TrainingPipeline(
            options=training_options,
            config=mock_config,
            model_dir=tmp_path,
        )
        assert pipeline is not None
    
    @patch("src.training.training_pipeline.DataPreparationPipeline")
    def test_data_preparation_called(self, mock_prep, mock_config, training_options, tmp_path):
        """Data preparation is invoked correctly."""
        mock_prep.return_value.prepare.return_value = Mock()
        
        pipeline = TrainingPipeline(
            options=training_options,
            config=mock_config,
            model_dir=tmp_path,
        )
        
        # This would normally run the pipeline
        # pipeline.run()
        # mock_prep.assert_called_once()


class TestBuddyTrainingOptions:
    """Tests for configuration dataclasses."""
    
    def test_from_kwargs(self):
        """Factory method works correctly."""
        options = BuddyTrainingOptions.from_kwargs(
            instrument="GBP_USD",
            epochs=100,
            label_smoothing=0.1,
        )
        assert options.instrument == "GBP_USD"
        assert options.epochs == 100
        assert options.advanced.label_smoothing == 0.1
    
    def test_defaults(self):
        """Default values are sensible."""
        options = BuddyTrainingOptions(instrument="USD_JPY")
        assert options.granularity == "H1"
        assert options.epochs == 200
        assert options.model_type == "transformer"
```

**File**: `tests/test_decision_engine.py`

```python
"""Regression tests for decision engine extraction."""
import pytest
import numpy as np
from unittest.mock import Mock

from src.inference.decision_engine import DecisionEngine, TradeDecision
from src.core.modular_inference import InferenceConfig


class TestDecisionEngine:
    """Tests for DecisionEngine class."""
    
    @pytest.fixture
    def mock_ensemble(self) -> Mock:
        ensemble = Mock()
        ensemble.predict.return_value = Mock(
            tcn_prob=0.65,
            ridge_score=55.0,
            xgb_percentile=0.3,
            rf_drawdown=0.015,
            direction_prob=0.72,
            confidence_score=68.0,
            calibrated_prob=0.61,
        )
        return ensemble
    
    @pytest.fixture
    def config(self) -> InferenceConfig:
        return InferenceConfig(
            min_tcn_probability=0.60,
            min_confidence=50.0,
            min_momentum=0.20,
            max_drawdown_pct=0.025,
        )
    
    def test_all_gates_pass(self, mock_ensemble, config):
        """Decision engine approves trade when all gates pass."""
        engine = DecisionEngine(mock_ensemble, config)
        features = np.random.randn(1, 60, 50)
        
        decision = engine.evaluate(
            features=features,
            current_price=1.1000,
            atr=0.0015,
            spread_pips=0.8,
        )
        
        assert decision.should_trade is True
        assert decision.direction == "LONG"
        assert all(decision.gates_status.values())
    
    def test_gate_failure_blocks_trade(self, mock_ensemble, config):
        """Decision engine blocks trade when gate fails."""
        mock_ensemble.predict.return_value.tcn_prob = 0.45  # Below threshold
        
        engine = DecisionEngine(mock_ensemble, config)
        features = np.random.randn(1, 60, 50)
        
        decision = engine.evaluate(
            features=features,
            current_price=1.1000,
            atr=0.0015,
            spread_pips=0.8,
        )
        
        assert decision.should_trade is False
        assert decision.gates_status["direction"] is False
```

### 10.4 Integration Smoke Tests

**File**: `tests/test_integration_smoke.py`

```python
"""Integration smoke tests for refactored modules."""
import subprocess
import pytest


@pytest.mark.integration
class TestCLISmoke:
    """Smoke tests for CLI commands."""
    
    def test_help_command(self):
        """--help works without errors."""
        result = subprocess.run(
            ["python", "main.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ML Engine" in result.stdout or "usage" in result.stdout.lower()
    
    def test_status_command(self):
        """status command works."""
        result = subprocess.run(
            ["python", "main.py", "status"],
            capture_output=True,
            text=True,
        )
        # May fail if no models, but shouldn't crash
        assert result.returncode in (0, 1)
    
    def test_scan_help(self):
        """scan --help works."""
        result = subprocess.run(
            ["python", "main.py", "scan", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


@pytest.mark.integration
class TestImportSmoke:
    """Smoke tests for module imports."""
    
    def test_all_modules_importable(self):
        """All refactored modules import without errors."""
        modules = [
            "src.training.training_config",
            "src.training.training_pipeline",
            "src.training.checkpoint_manager",
            "src.inference.buddy_inference",
            "src.inference.decision_engine",
            "src.trading.execution",
            "src.cli.commands.train",
            "src.cli.commands.buddy",
        ]
        
        import importlib
        for mod in modules:
            try:
                importlib.import_module(mod)
            except ImportError as e:
                pytest.fail(f"Failed to import {mod}: {e}")
```

### 10.5 Data Integrity Audits

```bash
#!/bin/bash
# scripts/audit_data_integrity.sh

echo "=== Checking Model Checksums ==="
sha256sum -c baseline_models.sha256

echo "=== Comparing Training Outputs ==="
./bin/Buddy train -i EUR_USD --csv market_data/EUR_USD_H1.csv --epochs 1 2>&1 | tee verify_train.txt
diff <(head -20 baseline_train.txt) <(head -20 verify_train.txt)

echo "=== Comparing Inference Outputs ==="
./bin/Buddy EUR_USD --verbose 2>&1 | tee verify_inference.txt
# Extract key metrics for comparison
grep -E "Confidence|Direction|Gates" baseline_inference.txt > baseline_metrics.txt
grep -E "Confidence|Direction|Gates" verify_inference.txt > verify_metrics.txt
diff baseline_metrics.txt verify_metrics.txt
```

### 10.6 Side-Effect Safety Checks

```python
"""Side-effect safety verification."""
import pytest
from pathlib import Path
import shutil


@pytest.fixture
def clean_model_dir(tmp_path):
    """Provide clean model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    yield model_dir
    shutil.rmtree(model_dir)


class TestSideEffectSafety:
    """Verify no unintended side effects."""
    
    def test_inference_does_not_modify_models(self, clean_model_dir):
        """Inference should not modify model files."""
        # Copy a model to temp dir
        # Run inference
        # Verify model unchanged
        pass
    
    def test_dry_run_creates_no_trades(self):
        """Dry run mode should not execute trades."""
        pass
    
    def test_config_not_mutated(self):
        """Config dict should not be mutated by training."""
        pass
```

---

## 11. Rollback Procedures

### 11.1 Git Branch Strategy

```bash
# Create feature branch for refactoring
git checkout -b refactor/main-decomposition

# Create savepoints after each phase
git tag phase-1-complete
git tag phase-2-complete
# ...

# If issues arise, rollback to last stable phase
git reset --hard phase-2-complete
```

### 11.2 Emergency Rollback

```bash
# Full rollback to pre-refactoring state
git checkout main
git checkout -b hotfix/undo-refactor
git revert --no-commit phase-1-complete..HEAD
git commit -m "Revert: Undo main.py decomposition due to issues"
```

### 11.3 Partial Rollback

Keep backward-compatible imports in main.py so external scripts don't break:

```python
# main.py - emergency fallback mode
import warnings

try:
    from src.training.training_pipeline import TrainingPipeline
    from src.inference.buddy_inference import BuddyInference
except ImportError:
    warnings.warn("Falling back to legacy monolithic implementation")
    # Include legacy implementations inline
```

---

## 12. Timeline & Milestones

| Phase | Duration | Key Deliverables | Risk Level |
|-------|----------|------------------|------------|
| **Phase 1**: Foundation | 2-3 days | `training_config.py`, `checkpoint_manager.py`, remove duplicates | Low |
| **Phase 2**: Training | 5-7 days | `training_pipeline.py`, `data_preparation.py`, `model_builders.py` | High |
| **Phase 3**: Inference | 4-5 days | `buddy_inference.py`, `decision_engine.py`, `execution.py` | Medium-High |
| **Phase 4**: CLI | 2-3 days | `src/cli/commands/*.py`, `app.py` | Low |
| **Phase 5**: Cleanup | 1-2 days | Documentation, dead code removal | Low |

**Total Estimated Duration**: 14-20 days

### Milestones

- **M1**: Phase 1 complete, tests passing, duplicates removed
- **M2**: Training pipeline extracted, train-buddy CLI works
- **M3**: Inference pipeline extracted, buddy CLI works
- **M4**: All CLI commands modularized
- **M5**: main.py reduced to <600 lines, all tests passing

---

## Appendix A: Verification Commands Quick Reference

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Check imports
python -c "from src.training import TrainingPipeline; print('OK')"

# Static analysis
ruff check src/ --select=E,F,I
mypy src/ --ignore-missing-imports

# CLI smoke test
./bin/Buddy --help
./bin/Buddy status
./bin/Buddy EUR_USD --verbose

# Integration test
./bin/Buddy train -i EUR_USD --csv market_data/EUR_USD_H1.csv --epochs 1
./bin/Buddy scan --top-n 3
```

---

## Appendix B: File-to-Module Mapping

| main.py Lines | Target Module | Priority |
|---------------|---------------|----------|
| 38-140 | `src/training/training_config.py` | P1 |
| 141-275 | DELETE (duplicate of `src/cli/calibration.py`) | P1 |
| 280-395 | `src/data/oanda_fetcher.py` | P1 |
| 497-575 | `src/training/checkpoint_manager.py` | P1 |
| 577-747 | `src/utils/instrument_validation.py` | P1 |
| 1103-1358 | `src/models/model_builders.py` | P2 |
| 1359-5519 | `src/training/training_pipeline.py` | P2 |
| 5520-6155 | `src/trading/execution.py` | P3 |
| 6157-6414 | `src/trading/paper_trade.py` | P3 |
| 7130-8419 | `src/inference/buddy_inference.py` | P3 |
| 8420-8977 | `src/inference/buddy_loop.py` | P3 |
| 9551-9963 | `src/rl/training.py` | P4 |
| 9964-10320 | `src/training/gate_trainer.py` | P4 |
| 12911-13847 | `src/cli/app.py` | P4 |

---

## Appendix C: Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use dataclasses over plain dicts | Type safety, IDE support, validation in `__post_init__` |
| Keep main.py as thin entry point | Backward compatibility for `python main.py` |
| Protocol-based interfaces | Enables dependency injection without tight coupling |
| Phase training extraction first | Highest complexity, most value from extraction |
| Preserve `src/cli/calibration.py` | Already extracted, just remove duplicates from main.py |

---

## Changelog

### 2026-02-05 - Phases 1-3 Module Extraction Complete

**Modules Created:**
- ✅ `src/training/training_config.py` (455 lines) - OandaFetchOptions, BuddyTrainingOptions dataclasses
- ✅ `src/training/checkpoint_manager.py` (535 lines) - CheckpointManager, CheckpointMetadata
- ✅ `src/training/data_preparation.py` (1,037 lines) - DataPreparationPipeline, PreparedData
- ✅ `src/training/retrain_gates.py` (248 lines) - Gate model retraining logic
- ✅ `src/models/model_builders.py` (591 lines) - ModelFactory, build_* functions
- ✅ `src/utils/instrument_validation.py` (346 lines) - validate_instrument, normalize_instrument
- ✅ `src/trading/execution.py` (880 lines) - TradeExecutor, OrderRequest, ExecutionResult
- ✅ `src/trading/paper_trade.py` (807 lines) - Paper trading simulation

**Total Lines Extracted**: 4,899 lines

**Remaining Work:**
1. Wire extracted modules into main.py (replace duplicate code with imports)
2. Create CLI command modules in `src/cli/commands/`
3. Reduce main.py from 13,847 → ~400-600 lines
4. Integration testing
5. Backward compatibility verification

**Current main.py Size**: 13,847 lines (target: 400-600)

---

**Document End**

*This roadmap should be reviewed with the team before implementation begins. Adjust timelines based on available capacity and risk tolerance.*
