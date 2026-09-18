# AXIOM Component Status

| Surface | Classification | Action |
|---|---|---|
| `src/data/` | CONNECTED | Canonical data/feature domain. |
| `src/core/modular_data_loaders.py` | CONNECTED / HIGH-RISK | Protect with parity tests. |
| `src/training/` | CONNECTED / HIGH-RISK | Canonical trainers; callers must converge. |
| `src/scanner/gates.py` | CONNECTED / HIGH-RISK | Contract consumer; fail closed. |
| `src/research/`, `src/evidence/` | CONNECTED | Research/evidence only; no runtime authority. |
| `src/data_platform/` | CONNECTED | Dataset provenance/forward capture. |
| `src/equity/`, `src/crypto/` | CONNECTED LANES | Keep lane boundaries explicit. |
| FX directional under `src/scanner/` | RETIRED/RESEARCH per retirement doc | Do not silently reactivate. |
| `src/sota_core/` | NEEDS OWNERSHIP AUDIT | Prove active callers or quarantine. |
| `src/inference/` vs `src/core/modular_inference.py` | NEEDS OWNERSHIP AUDIT | Assign explicit responsibilities. |
| `scripts/train_single_model_m1.py` | ACTIVE CANDIDATE / HIGH-RISK | Prove parity before calling canonical. |
| `scripts/scheduled_retrain.py` | ACTIVE CANDIDATE / HIGH-RISK | Must use same dataset contract. |
| `scripts/path4_true_oos_holdout.py` | SUSPECT | Repeated feature/artifact/label defects. |
| `scripts/per_pair_holdout_eval.py` | SUSPECT/RESEARCH | Historical overlap problem documented. |
| `scripts/train.py`, `train_single_model.py`, `train_full_ensemble.py` | NEEDS OWNERSHIP AUDIT | Multiple ambiguous training entry points. |
| `scripts/debug_*`, `diagnose_*`, `_diag_*` | DIAGNOSTIC | Quarantine candidates after caller audit. |
| `scripts/experiment_*`, `probe_*` | RESEARCH | Must not be production entry points. |
| root re-export shims | COMPATIBILITY-ONLY | Inventory callers then retire. |
| large root Python implementations | NEEDS OWNERSHIP AUDIT | Move only after call graph. |
| `legacy_quarantine/` | QUARANTINED | Correct isolation concept. |
| `trained_data/` | STATE/ARTIFACTS | Not source code; needs artifact taxonomy. |
| `tmp/` | TEMPORARY | Never architecture truth. |
| `docs/` root | DOCUMENTATION DEBT | 167 root docs at checkpoint; archive by status/domain. |
| `README.md` | CONTRADICTORY | Reconcile FX-active vs FX-retired claims. |

Definitions: CONNECTED = intentional role; HIGH-RISK = connected but historically contract-sensitive; RESEARCH = no runtime authority; COMPATIBILITY-ONLY = migration shim; NEEDS OWNERSHIP AUDIT = unclear callers; SUSPECT = documented mismatch; RETIRED = intentionally inactive; QUARANTINED = isolated.
