# AXIOM System Map

```text
DATA
  -> src/data + src/data_platform
  -> FEATURE/LABEL CONTRACT
     src/data/feature_engineering.py
     src/core/modular_data_loaders.py
  -> TRAINING
     src/training
     callers in scripts/
  -> ATOMIC CANDIDATE ARTIFACT
     model + feature_names + scaler + feature_indices
     + regime metadata + pipeline version + calibration
  -> INDEPENDENT OOS / EVIDENCE
     src/research + src/evidence
  -> RUNTIME INFERENCE
     src/scanner/gates.py
  -> ensemble/meta/risk
  -> broker boundary
```

## Contract that must be identical
Feature semantics; ordered feature names; label horizon/timestamp alignment; temporal split/purge/embargo; scaling/selection order; sequence length; output calibration; artifact ownership; abstention semantics; evaluation costs/baselines.

## Historical breaks already documented in repo history
- RobustScaler/StandardScaler ownership mismatch.
- Scaler refit producing identity-like saved transform.
- Training/inference feature-set and ordering mismatch.
- Missing regime contract fields.
- Runtime ignoring saved output calibration.
- Holdout using narrower feature builder than training.
- Runtime sequence length differing from artifact.
- Abstention converted into momentum/technical direction.
- Frame-anchored features changing predictions for identical bars.
- OOS assembly overwriting direction scaler/indices.
- Label positional misalignment after feature warmup.
- "Full stack" OOS omitting sibling heads.

## Target
One versioned `FeatureContract` is produced once and consumed by training, verification, and runtime. No path reconstructs preprocessing assumptions independently.
