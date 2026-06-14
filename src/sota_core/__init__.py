"""
SOTA Core — End-to-end deep learning signal generator.

Replaces classical feature-engineering + ensemble voting with a single
neural network that consumes raw OHLCV sequences and learns its own
representations.  Trained in two phases:

1. Self-supervised pre-training on massive unlabeled historical data
   (masked candle reconstruction + next-bar return prediction).
2. Supervised fine-tuning on labeled trade outcomes (direction + regime).

Inference exposes the same interface as ModularEnsembleInference so the
scanner can swap it in without changing orchestration code.
"""

from __future__ import annotations

from .inference import SOTAInference, SOTAInferenceConfig
from .raw_sequence_model import RawSequenceModel, ModelConfig
from .trainer import SOTATrainer, TrainerConfig

__all__ = [
    "SOTAInference",
    "SOTAInferenceConfig",
    "RawSequenceModel",
    "ModelConfig",
    "SOTATrainer",
    "TrainerConfig",
]
