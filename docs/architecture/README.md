# AXIOM Architecture Control Center

> Status: audit map created from `main@efa66d481a37edbef3399ae0bfe188d50ac92028`.
> This is the human entry point for understanding ownership. It does not declare mapped code correct.

## Start here
1. [SYSTEM_MAP.md](SYSTEM_MAP.md) — intended end-to-end boundaries.
2. [COMPONENT_STATUS.md](COMPONENT_STATUS.md) — connected, retired, compatibility-only, and suspect surfaces.
3. [DISCONNECTED_AND_SUSPECT.md](DISCONNECTED_AND_SUSPECT.md) — investigation queue.
4. [../archive/README.md](../archive/README.md) — historical documentation policy.
5. [../../legacy_quarantine/README.md](../../legacy_quarantine/README.md) — already-quarantined code.

## Canonical ownership
| Concern | Canonical location |
|---|---|
| Feature engineering | `src/data/`, `src/core/modular_data_loaders.py` |
| Trainer implementations | `src/training/` |
| Runtime inference/gates | `src/scanner/` |
| Evidence/research governance | `src/evidence/`, `src/research/`, `src/data_platform/` |
| Runtime lanes | `src/equity/`, `src/crypto/`, `src/scanner/` |
| Broker boundary | `src/brokers/`, `src/utils/oanda_practice.py` |
| Operator/autonomy | `src/axiom_operator/`, `src/autonomy/`, `src/brain_loop/` |
| Tests | `tests/` |
| Generated/state artifacts | `trained_data/` |
| Historical/incompatible code | `legacy_quarantine/` |

## Rules
- Do not add another top-level implementation when an owner exists under `src/`.
- Training code must identify the canonical feature builder, labels, artifact contract, evaluator, and inference consumer.
- Model artifacts are invalid unless preprocessing is serialized and replayable.
- Never convert ML refusal into manufactured LONG/SHORT.
- Research scripts cannot silently become runtime authority.
- Compatibility shims at repository root are migration surfaces, not canonical code.
- Quarantine suspect code only after call-site/import analysis proves the move is safe.

## Contradiction to resolve
`README.md` describes the FX 15-agent system as Buddy's runtime, while `docs/fx-directional-path-retired.md` says that path was retired on 2026-06-21. Until operator intent is reconciled, neither narrative is the sole architecture truth.
