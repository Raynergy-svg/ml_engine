# CI Safety Gates

Three blocking gates protect the P0/P1 safety contract of the trading bot. They run locally via `pre-commit` and in CI via `.github/workflows/code-quality.yml` (`p0-safety-gates` job). Both invoke the same hooks — single source of truth.

## One-time setup

```bash
pip install pre-commit
pre-commit install
```

## The three gates

| Gate | Script | Blocks on |
| ---- | ------ | --------- |
| 1. Orphan-key guard | `src/scanner/tools/validate_config_adjustments.py` | ScannerConfig keys in `.claude/config_adjustments.json`, `logs/reflection_staging/config_adjustments.json`, or `.claude/pending_adjustments.json` that do not match real dataclass fields. Exit 1 = orphan. Exit 2 = validator broken. |
| 2. P0 py_compile | `python3 -m py_compile` | Syntax errors in the 5 files carrying the P0 safety contract: `dynamic_sl_tp.py`, `agents/_team.py`, `engine.py`, `execution.py`, `automation/orchestrator.py`. |
| 3. Circuit-breaker regression | `pytest tests/scanner/test_circuit_breaker_enforcement.py -q` | Regression in trend-veto, staleness hard-block, or RL-sync enforcement (7 tests). |

## Manual run

```bash
pre-commit run --all-files                    # run every hook
pre-commit run config-adjustments-orphans     # gate 1 only
pre-commit run p0-py-compile                  # gate 2 only
pre-commit run circuit-breaker-regression     # gate 3 only
```

Config lives in `.pre-commit-config.yaml`.

## Developer setup

The `.pre-commit-config.yaml` is inert locally until you install the git hook. Run these once per clone:

```bash
pip install pre-commit                       # if not already installed
pre-commit install                           # installs commit-time hook (required)
pre-commit install --hook-type pre-push      # optional: also fire on push
pre-commit run --all-files                   # sanity check — should exit 0
```

Without `pre-commit install`, the YAML does nothing on your machine. CI runs every hook regardless of your local setup; the local install only protects your own git workflow from pushing breakage upstream.
