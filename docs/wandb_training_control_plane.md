# W&B Training Control Plane

Operator-facing guide for the W&B-backed training control plane that
supersedes hardcoded Python defaults across all 7 ML training heads.

**Status:** active as of 2026-05-02. Source of truth: `src/training/wandb_control_plane.py`.

---

## What changed

Yesterday (2026-04-30) confidence training was the only head with W&B
logging — and even there it was logging only, not config-driven. Today
all 7 heads route their *operator-tunable* settings through versioned
W&B config artifacts.

| Head              | Model class      | Trainer file                                                  | W&B artifact name                  |
| ----------------- | ---------------- | -------------------------------------------------------------- | ---------------------------------- |
| direction         | Tiny Transformer | `src/training/trainers/transformer_trainer.py`                | `direction_training_config`        |
| confidence        | LightGBM         | `src/training/trainers/ridge_trainer.py`                      | `confidence_training_config`       |
| momentum          | LightGBM         | `src/training/trainers/lightgbm_trainers.py`                  | `momentum_training_config`         |
| risk              | LightGBM         | `src/training/trainers/lightgbm_trainers.py`                  | `risk_training_config`             |
| volatility_regime | TCN              | `src/training/trainers/tcn_trainer.py`                        | `volatility_regime_training_config`|
| trend_regime      | Transformer      | `src/training/trainers/transformer_regime_trainer.py`         | `trend_regime_training_config`     |
| meta_labeler      | XGBoost          | `src/training/meta_labeling.py`                                | `meta_labeler_training_config`     |

---

## Strategic vs hidden

**Exposed (you tweak via W&B UI / CLI):**

- `label_generation`: `lookahead_bars`, `sl_atr_mult`, `tp_atr_mult`, `label_mode`, `winsorize_bounds`
- `training_behavior`: `enable_per_pair_finetune`, `enable_joint_training`, `expected_r2_band`, `min_real_labels_per_pair`, plus `min_confidence_threshold` for `meta_labeler`
- `hyperparameters` (top-3 per model class):
  - LightGBM (confidence/momentum/risk): `n_estimators`, `learning_rate`, `max_depth`
  - Transformer (direction/trend_regime): `epochs`, `learning_rate`, `dropout`
  - XGBoost (meta_labeler): `n_estimators`, `max_depth`, `dropout_rate`
  - TCN (volatility_regime): `epochs`, `learning_rate`, `dropout`

**Hidden (stays in code — change requires PR review):**

- Architectures: `d_model`, `num_heads`, `num_layers`, conv kernel sizes
- Validation: walk-forward window, embargo gap, k-folds (`walkforward_validation.py` is untouched)
- LightGBM internals: `num_leaves`, `reg_alpha`, `reg_lambda`, `min_child_weight`
- Loss / optimizer / EMA decay

---

## Layout of a config artifact

The artifact contains a single JSON file
(`<head>_training_config.json`). Schema version 1:

```json
{
  "schema_version": 1,
  "head": "confidence",
  "model_class": "lightgbm",
  "label_generation": {
    "lookahead_bars": 12,
    "sl_atr_mult": 1.0,
    "tp_atr_mult": 1.5,
    "label_mode": "real",
    "winsorize_bounds": [-0.02, 0.02]
  },
  "training_behavior": {
    "enable_per_pair_finetune": true,
    "enable_joint_training": true,
    "min_real_labels_per_pair": 50,
    "expected_r2_band": [0.05, 0.30]
  },
  "hyperparameters": {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6
  },
  "ranges": {
    "lookahead_bars": [4, 48],
    "n_estimators": [20, 1000],
    "...": "..."
  }
}
```

Local seeds live in `src/training/training_defaults/`; they are the
fallback when W&B is offline AND the auto-seed source on first run.

---

## Operator workflow

### 1. View configs in W&B UI

Project: whatever `WANDB_PROJECT` resolves to (default
`buddy-training-control-plane`). Navigate to **Artifacts →
training-config** for the head you want.

Each push lands as a new version (`v0`, `v1`, ...). The `:latest` alias
follows the most recent push.

### 2. Edit a config

Easiest path is to download → edit → re-upload via the W&B CLI:

```bash
# Download current config
wandb artifact get <entity>/<project>/confidence_training_config:latest --root /tmp/conf
# Edit
$EDITOR /tmp/conf/confidence_training_config.json
# Upload as new version
wandb artifact put /tmp/conf/confidence_training_config.json \
  --name <entity>/<project>/confidence_training_config \
  --type training-config
```

Alternatively from Python:

```python
from src.training.wandb_control_plane import push_config, pull_config
cfg = pull_config("confidence")
cfg["hyperparameters"]["n_estimators"] = 250
push_config("confidence", cfg)
```

### 3. Trigger pickup

| Source                              | Picks up changes                            |
| ----------------------------------- | ------------------------------------------- |
| `online_retrainer.py` (drift loop)  | Next retrain cycle (cooldown-protected)     |
| `scripts/retrain_confidence_*.py`   | Next manual run                             |
| `scripts/train_full_ensemble.py`    | Next manual run                             |
| RL position sizer / agent weights   | Not affected — those have separate stores   |

Auto-retrains are tagged `source=auto_retrain`. Manual runs are tagged
`source=manual`. Filter in W&B UI to compare.

---

## Strategic config ranges (sane defaults)

| Key                      | Recommended       | Hard min/max     | Notes                                     |
| ------------------------ | ----------------- | ---------------- | ----------------------------------------- |
| `lookahead_bars`         | 12 (H1)           | 4 / 48           | Higher = smoother labels, less data       |
| `sl_atr_mult`            | 1.0               | 0.5 / 3.0        | LOW regime must stay >= 1.2 (trading rule)|
| `tp_atr_mult`            | 1.5               | 0.8 / 4.0        | Keep TP/SL >= 1.2 (R:R gate)              |
| `label_mode`             | `"real"`          | real/pseudo/blend| `pseudo` only when journal too small      |
| `min_real_labels_per_pair`| 50–200           | 20 / 5000        | Below 50 = noisy gradients                |
| `expected_r2_band`       | `[0.05, 0.30]`    | —                | R² > upper = leak detection ERROR         |
| LightGBM `n_estimators`  | 100–300           | 20 / 1000        |                                           |
| LightGBM `learning_rate` | 0.05–0.10         | 0.01 / 0.5       |                                           |
| LightGBM `max_depth`     | 4–6               | 2 / 12           |                                           |
| Transformer `epochs`     | 50                | 10 / 200         |                                           |
| Transformer `learning_rate` | 3e-4           | 1e-5 / 1e-2      | Adam optimizer                            |
| Transformer `dropout`    | 0.2–0.4           | 0.0 / 0.7        | Aliased to `transformer_dropout` + `tcn_dropout` |

---

## Auto-retrain integration (online_retrainer.py)

`OnlineRetrainer.trigger_retrain()` now:

1. Calls `_safe_pull_config(head)` for each model it retrains
   (`momentum`, `risk`, `confidence`).
2. Applies the head's `hyperparameters` to the local model fit.
3. Calls `_safe_log_run(head=..., extra_config={"source": "auto_retrain"})`
   so every auto-retrain shows up in W&B with full config visibility.

Cooldowns + drift triggers are unchanged — only the config source moved.

If W&B is unreachable during a retrain, `_safe_pull_config` logs a
WARNING and returns `{}`; the retrainer falls back to the previous
in-code defaults so retraining never blocks on observability.

---

## Environment variables

Same conventions as `src/training/wandb_confidence.py`:

| Var                  | Effect                                                                     |
| -------------------- | -------------------------------------------------------------------------- |
| `WANDB_DISABLED=1`   | Top-priority kill switch — every control-plane call no-ops.               |
| `WANDB_API_KEY`      | Set → online mode. Unset → offline (`./wandb/` dir).                      |
| `WANDB_MODE`         | Explicit override (`online`/`offline`/`disabled`).                        |
| `WANDB_PROJECT`      | Default `buddy-training-control-plane`.                                    |
| `WANDB_ENTITY`       | Org / team. Optional.                                                      |
| `WANDB_LOG_ARTIFACTS=1` | Upload model artifacts (.pkl/.keras) alongside metrics on `log_run`. |

Auth: `wandb login` on the operator shell writes to `~/.netrc`; the
buddy venv inherits that. Confirm with `python -c "import wandb;
wandb.login()"`. The `./buddy` launcher activates `.venv` automatically.

---

## Troubleshooting

**Q: My retrain logs say "auto-tweaking won't apply this cycle".**
A: Network or auth issue. Helper logged a warning; retrain proceeded with
script defaults. Fix W&B access and the next cycle will pick up the
config. Check `WANDB_API_KEY` and `wandb status`.

**Q: I edited the artifact but the next retrain still uses the old HPs.**
A: Three checks:
1. Did you push a new artifact *version*, or just edit a draft? `wandb artifact ls <project>` should show a new vN.
2. Is the `:latest` alias correct? `wandb artifact get <name>:latest --root /tmp/x && cat /tmp/x/*.json`.
3. Was the retrain run before your push (timestamp check)?

**Q: Schema mismatch on pull.**
A: `pull_config` validates against `REQUIRED_TOP_KEYS`. Mismatches fall
back to the local default and log a WARNING. To fix, re-upload a config
that matches the schema (use `pull_config(head)` to get the right
shape).

**Q: Offline mode — where are the runs?**
A: `WANDB_DIR/wandb/offline-run-<timestamp>-<hash>/`. Default
`WANDB_DIR` is `./wandb/`. Sync them with `wandb sync wandb/offline-run-*`
once online.

**Q: Operator runbook for re-onlining a stale offline run?**
A: `wandb login` (once), then `wandb sync wandb/offline-run-*`. The
runs will appear in the project with their original metadata and tags
preserved.

**Q: Test that needs real W&B API.**
A: Mark `@pytest.mark.skipif(not os.getenv("WANDB_API_KEY"), reason="needs online W&B")`. We currently have none — every existing test mocks the `Api` / `init` boundary.

---

## File map

| Path                                                         | Role                                       |
| ------------------------------------------------------------ | ------------------------------------------ |
| `src/training/wandb_control_plane.py`                        | The helper. `pull/push/log_run/seed_default`. |
| `src/training/training_defaults/<head>_training_config.json` | Local defaults / fallback / first-time seed. |
| `src/training/wandb_confidence.py`                           | Legacy confidence-only logger (kept; runs alongside). |
| `online_retrainer.py`                                        | Auto-retrain consumer of the control plane. |
| `scripts/retrain_confidence_leak_fix.py`                     | Manual joint confidence retrain (control-plane wired). |
| `scripts/retrain_per_pair_confidence.py`                     | Manual per-pair confidence retrain. |
| `scripts/train_full_ensemble.py`                             | Multi-head launcher (per-head wiring at construction). |
| `tests/test_wandb_control_plane.py`                          | Helper unit tests. |
| `tests/test_online_retrainer_wandb_integration.py`           | Online retrainer ↔ control plane tests. |

---

## Compatibility / pinning

Tested on `wandb==0.26.x`. Older releases (<0.23) lack
`reinit="finish_previous"` — `wandb_observatory.py` already handles that;
the control plane uses plain `reinit=True` which works on every release
≥0.16.
