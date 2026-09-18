# Disconnected, Suspect, and Harmful-Code Queue

## P0 — training/inference integrity
**A-001 Canonical feature contract.** Compare `src/data/feature_engineering.py`, `src/core/modular_data_loaders.py`, `scripts/train_single_model_m1.py`, `scripts/scheduled_retrain.py`, both holdout scripts, and `src/scanner/gates.py`. Result: one versioned contract.

**A-002 OOS trainer mismatch.** `path4_true_oos_holdout.py::train_master_on_split` historically enters through `compute_normalized_features`, while its evaluator repair says the expected head contract uses `FeatureEngineering(include_all=True)`. Treat Path4 as suspect until reconciled.

**A-003 Artifact ownership.** Direction model, scaler, indices, metadata and calibration must be one atomic package. Never build a candidate by copying ambiguous sibling files from an incumbent model directory.

**A-004 Silent fallback.** Missing contract/exception => ABSTAIN + observable reason. Never RSI/momentum/technical LONG/SHORT fabrication.

## P1 — code ownership
**A-010 Training entry points.** Classify every `scripts/train*.py`: canonical, wrapper, research, retired, quarantine. No two canonical scripts may independently build features/labels.

**A-011 Root Python.** Root should contain launch/config/project files, not domain implementations. Re-export shims are migration-only; large implementations need caller analysis.

**A-012 Inference namespaces.** Audit `src/inference/`, `src/core/modular_inference.py`, root `multi_pair_inference.py`, and `src/scanner/gates.py`.

**A-013 SOTA core.** Prove active call path for `src/sota_core/`; otherwise quarantine it.

## P1 — documentation truth
**A-020 Runtime identity.** Reconcile README's active FX runtime with `docs/fx-directional-path-retired.md`.

**A-021 Archive historical docs.** Checkpoint contains 229 docs, 167 directly under `docs/`. Target: `architecture/`, `runbooks/`, `reference/`, `research/`, `incidents/`, and `archive/YYYY-MM/`. Do not bulk-move before link/caller checks.

## P2 — repository hygiene
**A-030 Shim retirement.** Replace imports of root shims with `src.*`, then remove/move only after zero active callers.

**A-031 Script taxonomy.** Target folders: `operations/`, `training/`, `evaluation/`, `research/`, `diagnostics/`, `migration/`, `deprecated/`. Move atomically with CI/docs/subprocess updates.

**A-032 Artifact taxonomy.** Separate candidate, promoted, quarantined, evidence, logs, caches, and temporary output.

## "Bad code hurting us" test
Quarantine when proven to: silently change train/infer contract; fabricate direction after abstention; mutate another model's artifacts; bypass validation; be unreachable but presented active; duplicate an authoritative implementation and drift; or swallow correctness-critical exceptions and continue as success.
