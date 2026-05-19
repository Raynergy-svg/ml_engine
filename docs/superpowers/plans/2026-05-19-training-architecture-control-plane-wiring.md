# Training Architecture — W&B Control Plane Wiring

**Status**: Proposed (plan only; no code in this commit)
**Author**: Software Architect agent, 2026-05-19
**Trigger**: Operator callout 2026-05-18 23:14 EDT — *"why am i reaching on my own 60% with wandb and you cant even hit past 55"* and *"we need the way buddy trains, updated, since we now are autonomous quant loop, with agents"*.
**Load-bearing question this answers**: Why do agent-driven retrains hit val_acc=51–52% on M15 EUR_USD/GBP_USD while operator's manual wandb-tuned runs hit 60%? Hypothesis (MEDIUM confidence): the two paths use **different hyperparameters and a different lookahead**, and the agent path never reads the operator's tuned config.

---

## 1. Audit Table — Training Entry Points vs Control Plane

Verified by grep `pull_config\|wandb_control_plane` across `/scripts/` and `/src/training/` (excluding `__pycache__`). HIGH confidence per entry — every row is backed by a citation.

| # | Entry point | Pulls control plane? | Reads `training_defaults/`? | Hardcoded HP source | Confidence |
|---|---|---|---|---|---|
| 1 | `scripts/train_single_model_m1.py` | **NO** | NO | `MASTER_PARAMS` dict + `DEFAULT_PARAMS` dict at `scripts/train_single_model_m1.py:127-183` | HIGH — grep returns 0 hits for `pull_config` in this file |
| 2 | `scripts/stage_4a_news_retrain.py` | **NO** | NO | `argparse` defaults at `scripts/stage_4a_news_retrain.py:40-52` + `TrainerConfig()` defaults at line 118 | HIGH — grep returns 0 hits |
| 3 | `scripts/train_full_ensemble.py` | YES (best-effort) | YES via `pull_config` | `MASTER_PARAMS` at `scripts/train_full_ensemble.py:152-158` ALSO present and wins for `seq_len`, `batch_size`, `lr`, `patience` | HIGH — `_control_plane_apply` at line 71, called per head; but module-level `MASTER_PARAMS` still controls sequencing/optimizer HPs not in the JSONs |
| 4 | `scripts/retrain_per_pair_confidence.py` | YES | YES | `TrainerConfig()` at use, then `apply_config_to_trainer` overrides | HIGH — `pull_config("confidence")` at line 129 |
| 5 | `scripts/retrain_confidence_leak_fix.py` | YES | YES | `argparse --lookahead-bars` default `24` at line 158 **overrides** JSON's `12` | HIGH — `pull_config("confidence")` at line 269, but CLI default ships 24 |
| 6 | `scripts/retrain_volatility_regime_leak_fix.py` | **NO** | NO | `TrainerConfig()` at line 294 (no params passed) | HIGH — grep returns 0 hits |
| 7 | `scripts/train_transfer_learning.py` | **NO** | NO | Imports `MASTER_PARAMS, DEFAULT_PARAMS` from `train_single_model_m1` at line 296 | HIGH — chain of hardcoded inheritance |
| 8 | `scripts/train_tcn_warmstart.py` | **NO** | NO | Inline `TrainerConfig(epochs=200, batch_size=64, learning_rate=0.0001, …)` at line 92-94 | HIGH |
| 9 | `scripts/train_planner_model.py` | NO | NO | YAML file passed via `--config` (separate planner system) | HIGH |
| 10 | `scripts/train_meta_calibrator.py` | NO | NO | Calibration script — no head HPs needed | MEDIUM — confirmed no control plane import |
| 11 | `scripts/train_rl_*.py` (4 files) | NO | NO | RL has its own HP loop; not in the 7-head control plane scope | HIGH |
| 12 | `scripts/scheduled_retrain.py` | NO directly | Indirectly via the subprocess scripts it calls | Calls `train_single_model_m1.py` as subprocess → inherits #1's bypass; also hardcodes `HOLDOUT_LOOKAHEAD = 24` at line 489 | HIGH |
| 13 | `online_retrainer.py` (repo root) | YES | YES | `_safe_pull_config(head)` at lines 499, 578, 652 for `momentum`/`risk`/`confidence`. **No `direction` pull** | HIGH |
| 14 | `src/training/buddy_training_helpers.py::train_joint_multi_pair_ensemble` | **NO** | NO | `TrainerConfig(**kwargs)` from caller; `lookahead: int = 24` hardcoded at line 536 | HIGH |
| 15 | `src/training/retrain_agent.py` / `retrain_worker.py` | NO | NO | Pass-through; calls scripts/subprocess | MEDIUM — grep shows no direct training config wiring |
| 16 | `src/training/meta_labeler_retrainer.py` | NO | NO | Dataclass `learning_rate: float = 0.03` at line 64 | HIGH |

**Summary**: of 16 entry points, **3 wire the control plane correctly** (online_retrainer for 3 heads, retrain_per_pair_confidence, train_full_ensemble for HPs the JSON exposes). **The two operational paths agents actually trigger — `train_single_model_m1.py` and `stage_4a_news_retrain.py` — bypass it completely.**

The contract gap explains the 51 vs 60 spread: when operator manually tunes via wandb UI, the JSONs and W&B artifacts update; but `train_single_model_m1.py` never reads them, so the agent's `--model transformer` run uses 2026-04-vintage `MASTER_PARAMS` values frozen in the script.

---

## 2. Bypass Divergence Table — Hardcoded vs Canonical

Canonical = `src/training/training_defaults/<head>_training_config.json` (the local fallback the control plane uses when W&B is offline; operator-tuned).

| Hyperparam | Canonical (direction JSON) | `train_single_model_m1.py` value | `stage_4a_news_retrain.py` | `load_direction_data` default | Divergence magnitude |
|---|---|---|---|---|---|
| `lookahead_bars` | **12** | not present (loader's default `lookahead=24` wins) | `argparse default=24` | `DIRECTION_DEFAULTS['lookahead']` (read at `modular_data_loaders.py:1830`) | **2× — labels predict 24-bar moves but model is JSON-targeted at 12. THIS IS PROBABLY THE LARGEST SOURCE OF THE 51 vs 60 GAP.** |
| `sl_atr_mult` | 1.0 | n/a (script doesn't reach pseudo-label path) | n/a | n/a | label-gen knob, only fires in `compute_alternative_targets`; irrelevant for `train_single_model_m1`'s `real` label mode |
| `tp_atr_mult` | 1.5 | n/a | n/a | n/a | same |
| `label_mode` | `"real"` | implicit `real` (no override) | implicit `real` | implicit `real` | aligned — no divergence |
| `winsorize_bounds` | `[-0.02, 0.02]` | not applied (script doesn't call winsorize) | same | not applied | MEDIUM — JSON exposes it but no consumer reads it; likely dead key |
| `epochs` | **50** | `200` (hardcoded at `train_single_model_m1.py:297`) | `TrainerConfig()` default `100` (config.py:20) | n/a | **4× over canonical. Combined with patience differences this is mostly absorbed by early-stopping, but burns training-budget on misconfigured runs.** |
| `learning_rate` | **0.0003** | `MASTER_PARAMS[pair]["lr"]` (0.000376–0.000630) or `DEFAULT_PARAMS["lr"]=0.000415` | `TrainerConfig()` default `0.001` (config.py:22) | n/a | EUR_USD falls to DEFAULT (0.000415) → 38% above canonical; GBP_USD falls to DEFAULT too; AUD_USD uses 0.000630 → 110% above canonical |
| `dropout` | **0.4** | `MASTER_PARAMS[pair]["dropout_rate"]` (0.183–0.468) or `DEFAULT_PARAMS=0.363` | `TrainerConfig()` defaults `transformer_dropout=0.2` (config.py:51) | n/a | per pair varies; stage_4a uses 0.2 — half the canonical |
| `seq_len` | not exposed (architecture-internal) | `MASTER_PARAMS[pair]["seq_len"]` 90 or 120 | `TrainerConfig()` default `60` (config.py:27) | n/a | **stage_4a uses 60 vs script's 90-120 — direct comparability broken between price-only and news-fused baselines** |
| `batch_size` | not exposed | `MASTER_PARAMS=128` for all 5 pairs | `TrainerConfig()` default `64` | n/a | 2× — affects gradient noise, possibly accuracy |
| `patience` | not exposed | 10–25 depending on pair | `TrainerConfig()` default `20` | n/a | varies |
| `n_estimators` (confidence) | **100** | n/a | n/a | n/a | aligned via `pull_config` in `retrain_*confidence*.py` (HIGH confidence wired) |
| `max_depth` (confidence) | **6** | n/a | n/a | n/a | aligned |

**Critical claim (MEDIUM confidence, requires empirical confirmation)**: the `lookahead=24` default baked into the data loader and the `train_single_model_m1` subprocess path is producing labels that are 2× the horizon the operator-tuned JSON targets. If operator's manual wandb runs use `lookahead=12` (because they call paths that *do* honor the JSON, or because the JSON came from wandb sweeps that converged on 12), the auto-retrains have been training a different model class entirely. Verification next session: grep operator's recent wandb run configs for the actual `lookahead_bars` used.

**Tagging**: this is MEDIUM not HIGH because I have not opened a recent operator wandb run to read the `lookahead_bars` it logged. The asymmetric path divergence is HIGH-confidence; the **causal attribution to the 51-vs-60 gap** is MEDIUM until that grep is done.

---

## 3. Control Plane API — How It Works

Verified by reading `src/training/wandb_control_plane.py` end-to-end (576 lines).

### Public API surface

```
pull_config(head: str) -> Dict[str, Any]        # operator-tuned config, falls back to local JSON
push_config(head, config) -> Optional[dict]      # uploads a new version
seed_default(head, defaults=None) -> Optional[dict]   # idempotent first-time setup
log_run(head, config, metrics, artifacts=…) -> Optional[dict]
apply_config_to_trainer(trainer, config) -> None # walks config and sets trainer.<key> or trainer.config.<key>
```

### Behavior contract (verified at file:line)

- `pull_config(head)` tries W&B first (`_try_pull_from_wandb` at line 233), falls back to `load_local_default(head)` reading `src/training/training_defaults/<head>_training_config.json` (line 139).
- Offline path is ALWAYS available — `_resolve_mode()` returns `"offline"` when `WANDB_API_KEY` unset (line 100).
- `WANDB_DISABLED=1` makes everything no-op (line 96-97).
- Schema validation enforces 6 required keys (line 80-87): `schema_version, head, model_class, label_generation, training_behavior, hyperparameters`.
- `apply_config_to_trainer` does double-targeting (line 548-572): sets attrs on `trainer.<key>` AND `trainer.config.<key>`, with a special `dropout` → `transformer_dropout/tcn_dropout` alias.

### Heads supported (line 65-73)

`direction, confidence, momentum, risk, volatility_regime, trend_regime, meta_labeler`. RL is **not** in scope. Planner/calibrator are **not** in scope.

### Known limitation (HIGH confidence)

The JSONs only expose top-3 HPs per head. `seq_len`, `batch_size`, `patience`, `mixed_precision`, TCN-specific filters — none of these are control-plane-exposed today. So even after Tier 1 wiring is done, `MASTER_PARAMS`-style per-pair-specific overrides for `seq_len=90 vs 120` will need EITHER a JSON schema bump OR per-pair routing inside the control plane.

---

## 4. Tier 1 Plan — Wire Existing Entry Points to Control Plane

Goal: every training entry point reads `pull_config(<head>)` at startup. CLI flags continue to override (operator agency preserved). No JSON schema change in Tier 1.

**Ordering rationale**: ship the unblock for the operator's complaint first (direction head on operational paths), then the long tail. Each PR is a small isolated diff.

### PR-1: `train_single_model_m1.py` reads control plane for direction + volatility_regime + momentum + risk + confidence

**Why first**: this is the script `scheduled_retrain.py` runs as subprocess, AND the script the autonomous loop calls. Fixing this single file unblocks every agent-triggered retrain.

**Diff sketch (do NOT implement in this commit)**:

1. After `apply_features(df)` at line 614, before the per-model loop at line 621:
   ```
   from src.training.wandb_control_plane import pull_config, apply_config_to_trainer
   cp_configs = {
       "direction": pull_config("direction"),
       "volatility_regime": pull_config("volatility_regime"),
       "momentum": pull_config("momentum"),
       "risk": pull_config("risk"),
       "confidence": pull_config("confidence"),
   }
   ```
2. In `train_transformer` at line 290, after `TrainerConfig(...)` constructor at line 296: call `apply_config_to_trainer(trainer, cp_configs["direction"])` AFTER trainer instantiation at line 312. The control plane's `dropout` alias correctly maps to `transformer_dropout`.
3. **CRITICAL**: change `load_direction_data(df_feat)` at line 305 to thread `lookahead=cp_configs["direction"]["label_generation"]["lookahead_bars"]` (with safe `dict.get` chain). This is the single largest divergence-fix.
4. Each `train_<head>` function picks up its corresponding `cp_configs[head]` after trainer instantiation.
5. **Preserve operator CLI overrides**: add `--lookahead-override`, `--lr-override`, `--dropout-override` flags that, if set, win over the control plane value. This is what protects the autonomous-quant-loop's ability to A/B test.

**Test surface (no mocks per `.claude/rules/improvement.md` No-Mock Rule)**:
- Run `python scripts/train_single_model_m1.py --instrument EUR_USD --model transformer --candles 5000` under `WANDB_DISABLED=1`. Verify log line `pull_config(direction)` falls back to local JSON and `lookahead=12` (the JSON value) shows up in `load_direction_data` log message (currently logs `lookahead=24`).
- Verify the saved `transformer_direction.meta.pkl` contains `lookahead_bars=12` in its sidecar.
- Verify the 10% gap rule (`_quarantine_if_overshipped`) still fires by training a deliberately-overfit run.

**Rollback**: revert the PR; the hardcoded `MASTER_PARAMS` path still works. Zero blast radius outside this file.

### PR-2: `stage_4a_news_retrain.py` reads control plane for direction

**Why second**: Stage 4-A news fusion is the P1 modernization lever per CLAUDE.md. The current `argparse --lookahead default=24` defeats the JSON's `lookahead=12`. To compare news-fused vs price-only fairly, both must use the same lookahead.

**Diff sketch**:
1. After `args = ap.parse_args()` at line 53: `cp_direction = pull_config("direction")`.
2. If `--lookahead` was NOT passed on CLI (use argparse SUPPRESS or sentinel), use `cp_direction["label_generation"]["lookahead_bars"]`.
3. Same for `TrainerConfig()` at line 118 — pass `apply_config_to_trainer(trainer, cp_direction)` after instantiation.

**Test**: smoke run `--pair EUR_USD --candles 5000`, assert log shows `lookahead=12` (JSON) not `lookahead=24` (current argparse default), and `transformer_dropout=0.4` not `0.2`.

### PR-3: `scheduled_retrain.py` HOLDOUT_LOOKAHEAD reads the JSON

**Why**: line 489 hardcodes `HOLDOUT_LOOKAHEAD = 24` for the post-retrain holdout eval. This must match whatever the trainer used or the eval is comparing apples to oranges.

**Diff sketch**:
```
from src.training.wandb_control_plane import pull_config
HOLDOUT_LOOKAHEAD = int(
    pull_config("direction").get("label_generation", {}).get("lookahead_bars", 24)
)
```

### PR-4: `train_full_ensemble.py` lets control plane HPs win over `MASTER_PARAMS`

The script already calls `_control_plane_apply` (`scripts/train_full_ensemble.py:71`). But `MASTER_PARAMS` at line 152 is read BEFORE the apply call and used for `seq_len`, `batch_size`, `lr`, `patience`. Tier 1 move: keep `MASTER_PARAMS` as floor defaults, but let the control plane's `hyperparameters` block override `learning_rate` and `dropout`. Document the precedence chain in the docstring.

### PR-5: `retrain_confidence_leak_fix.py` — flip the argparse default

Line 158: `--lookahead-bars default=24`. Change to `default=None`; if None, read from JSON's `lookahead_bars=12`. Same protective pattern as PR-2.

### PR-6: `retrain_volatility_regime_leak_fix.py` and `train_transfer_learning.py`

Add `pull_config` calls. Lower priority — these are not on the autonomous agent's hot path.

### PR-7: `train_tcn_warmstart.py` — pull `volatility_regime`

Already uses `TrainerConfig` directly with hardcoded values (line 92-94). Wire `pull_config("volatility_regime")` + `apply_config_to_trainer`.

---

## 5. Tier 2 — Architectural Recommendation for the Autonomous-Quant-Loop-with-Agents Era

The Tier 1 fix closes the contract gap. Tier 2 makes the system maintainable as the agent population grows.

### Recommendation: `buddy retrain` as the single CLI surface

Today: agents call `python scripts/train_single_model_m1.py --instrument X --model transformer --candles 25000` directly. Tomorrow: they call `python -m buddy retrain --pair X --granularity M15`. Internally that CLI does:

1. **Pull from control plane** (`pull_config` for every head being trained).
2. **Apply operator safety floors** (HIGH confidence per operator directive 2026-05-18 23:18): `candles >= 25000` enforced regardless of what argparse defaulted to. If the agent passes `--candles 10000`, the CLI logs a warning and bumps to 25000. If the agent passes `--candles 100000`, that's honored.
3. **Per-pair fan-out**: if `--pair all`, iterate the master pair list.
4. **Per-head fan-out**: if `--model all`, iterate the 7 heads.
5. **Quarantine guard** (HARD_MAX_GAP=0.10) is enforced inside the CLI, not the underlying script — no agent can bypass it by calling the underlying script directly because the underlying scripts get demoted to private (move to `src/training/_internal/`).
6. **Log every run** via `log_run(head, …)` to W&B — every agent retrain becomes a wandb run that operator can A/B compare against their manual sweep runs.

### Trade-offs

| Option | Pros | Cons |
|---|---|---|
| **A. Add `buddy retrain` CLI, demote scripts** (recommended) | One surface; safety floors enforced uniformly; agents and operator share the same path | Bigger refactor; existing automations (`com.mlengine.retrain.plist`) need updating |
| B. Add `pull_config` calls to every script, leave the surface fragmented | Tier 1 alone; smaller blast radius | Every new training script is a fresh contract bug surface; the 16-entry-point audit table grows |
| C. Promote `online_retrainer.py` to be the single surface | Already wires the control plane for 3 heads | It's an in-process retrainer with no subprocess isolation; mixing autonomous-loop training with M1-Metal-heavy subprocess training kills memory budget |

**Pick A.** The autonomous-quant-loop's promise — "agents do training without coding sessions" — only holds when the CLI surface is stable AND the safety invariants live there.

### Sequencing Tier 2

- **Tier 2.A**: ship Tier 1 (PRs 1-7). System works.
- **Tier 2.B**: write `src/cli/retrain.py` (new file) wrapping the existing trainers. Make `buddy retrain` an alias added to `buddy_scanner.py`'s subcommand router. Existing scripts continue to work.
- **Tier 2.C**: migrate `scheduled_retrain.py` to call `buddy retrain` instead of `train_single_model_m1.py` subprocess.
- **Tier 2.D**: migrate agent harnesses (autonomous-loop, ralph, code-repair) to call `buddy retrain`. Document the contract in `docs/runbook_retrain.md`.
- **Tier 2.E** (deferred): demote `scripts/train_single_model_m1.py` → `src/training/_internal/train_single_model_m1.py`. Direct invocation prints a deprecation warning pointing to `buddy retrain`.

### Schema bump for per-pair routing (Tier 2.F, optional)

If operator wants per-pair `seq_len` and `batch_size` in the control plane (not just architecture-internal), bump `training_defaults/direction_training_config.json` to `schema_version: 2` with an optional `pair_overrides: {EUR_USD: {seq_len: 90, ...}, ...}` block. The control plane's `_migrate_if_needed` (line 169) already has a migration hook. Don't ship this until operator asks for it.

---

## 6. Risks + Rollback

| Risk | Likelihood | Cost if it happens | Mitigation |
|---|---|---|---|
| Tier 1 PR-1 changes `lookahead` from 24 → 12 and breaks per-pair holdout eval | MEDIUM | Holdout numbers drop / become incomparable across the cutover | Run BOTH before merging: train one pair with new path, eval with both `HOLDOUT_LOOKAHEAD=24` and `=12`. Don't merge if both numbers degrade. |
| Operator's manual wandb sweeps used `lookahead=24` not `12`, JSON is stale | MEDIUM | The Tier 1 unblock has the opposite sign — we make things worse | Before merging PR-1, grep `wandb run config` for the operator's last 5 manual sweeps, confirm `lookahead_bars`. Update the JSON to whatever operator's manual runs use; THEN merge the wiring. |
| `apply_config_to_trainer` silently no-ops on key it doesn't recognize | LOW | Confidence in the wiring drops; "wired" claim becomes a lie (per the honesty protocol) | Every Tier 1 PR adds a `logger.info("Applied control plane config for %s: %s", head, list(cfg.keys()))` line that the operator can grep in `logs/buddy_debug.log` to confirm the wiring fired. |
| W&B online mode hangs the training startup | LOW (timeout=15s built into `wandb_control_plane.py:_try_pull_from_wandb`) | 15s delay per head per training run | Acceptable; the fallback to local JSON is automatic. |
| Tier 2 `buddy retrain` CLI accidentally bypasses HARD_MAX_GAP | HIGH if not tested | A model with gap > 10% gets shipped to `trained_data/models/<pair>/` and gates pick it up | Tier 2.B PR MUST include a test that deliberately overfits then asserts the file landed in `_quarantine/`. Operator's 10% rule is non-negotiable per `train_single_model_m1.py:58`. |

**Rollback plan for each PR**: every diff is single-file and reverts cleanly. The Tier 1 fixes are additive (`pull_config` returns `{}` on any failure → behavior degrades to today's hardcoded path). The Tier 2 CLI is a new file; revert by removing it from the subcommand router.

---

## 7. Test Strategy (No-Mock Compliance)

Per `.claude/rules/improvement.md` No-Mock Rule promoted 2026-05-01.

### Tier 1 tests (per PR)
1. **Real disk, real JSON**: each PR's test reads `src/training/training_defaults/<head>_training_config.json` from the actual filesystem. No `MagicMock`, no `patch`.
2. **`WANDB_DISABLED=1`** for the test environment. The control plane falls back to local JSON. Verify the right values land in `trainer.config` after `apply_config_to_trainer`.
3. **Smoke retrain at small scale**: `--candles 5000` (not 25000) so the test runs in ~30s. Assert the resulting `meta.pkl` sidecar has the expected `lookahead_bars` and `learning_rate` values.
4. **Integration assertion**: after the retrain, `grep "pull_config(direction)" logs/buddy_debug.log` shows the call fired. This is the "Live Wiring Verification Gate" from `.claude/rules/improvement.md`.

### Tier 2 tests (when Tier 2.B lands)
1. `python -m buddy retrain --pair EUR_USD --candles 10000 --dry-run` logs that `candles` was bumped to 25000 (operator floor).
2. End-to-end: `buddy retrain --pair EUR_USD --candles 5000 --model transformer` with `WANDB_DISABLED=1` produces the same artifacts as `python scripts/train_single_model_m1.py …` would have, but with control-plane HPs in the meta sidecar.
3. Quarantine test: force a high-gap model (small `--candles 500`, no early stopping), assert file lands in `trained_data/models/EUR_USD/_quarantine/`.

### Manual verification (operator-side)
After Tier 1 PR-1 + PR-2 ship:
1. `python scripts/train_single_model_m1.py --instrument EUR_USD --model transformer --candles 25000`
2. Compare resulting val_acc to the operator's manual wandb-tuned run. If gap closes from 9pp (51 vs 60) to <2pp, the hypothesis is confirmed.
3. If gap remains >5pp, the load-bearing question shifts: it's not the control plane wiring; the operator's manual runs are doing something else (e.g. different feature set, different label_mode, different data window). Re-audit.

---

## 8. Confidence Calibration on Load-Bearing Claims

Per `.claude/rules/improvement.md` confidence-calibration rule (operator feedback 2026-05-11):

| Claim | Confidence | Evidence / how to falsify |
|---|---|---|
| `train_single_model_m1.py` does not call `pull_config` | **HIGH** | `grep "pull_config" scripts/train_single_model_m1.py` returns 0 hits |
| `stage_4a_news_retrain.py` does not call `pull_config` | **HIGH** | same grep, same result |
| The lookahead divergence (12 JSON vs 24 loader) is the largest source of the 51-vs-60 gap | **MEDIUM** | Untested. Falsify by reading operator's manual wandb run config for `lookahead_bars`. If it shows 24, this claim is wrong and Tier 1 PR-1 needs the JSON updated to 24 BEFORE merging. |
| Tier 1 will close the gap by ≥3pp | **LOW** | Pure hypothesis until a smoke retrain runs post-merge. The claim "this fix is causal" is exactly what I got wrong in the 2026-05-11 `gates.py:1902` incident; not making that mistake again. |
| `buddy retrain` CLI is the right Tier 2 surface | **MEDIUM** | The architectural argument is sound. The actual ergonomics survive contact with the operator using it; might need iteration. |
| The control plane's offline fallback works | **HIGH** | Read `wandb_control_plane.py:194-230` end-to-end — `pull_config` returns `{}` on total failure, falls back to local JSON on partial failure. |

---

## 9. Done Criteria (this document)

- [x] Audit table with file:line citations
- [x] Bypass divergence table with magnitudes
- [x] Tier 1 ordered PR list (7 PRs, each single-file, with diff sketch)
- [x] Tier 2 architectural recommendation with trade-off table
- [x] Risks + rollback per Tier
- [x] No-mock test strategy
- [x] Confidence calibration on every load-bearing claim
- [x] HARD_MAX_GAP / 10% rule preserved (never touched in any PR)
- [x] `direction_training_config.json` values preserved (never changed in any PR)

---

## 10. Open Questions (operator decision before Tier 1 PR-1 merges)

1. **Is the JSON's `lookahead_bars=12` what operator's wandb sweeps actually converged on?** If not, the JSON is stale and Tier 1 ships the wrong unblock. Verification: `wandb run` page or `wandb api` for last 5 successful sweeps.
2. **Should the autonomous loop's `--candles 25000` floor be hard (CLI rejects lower) or soft (CLI bumps and warns)?** Operator's 2026-05-18 23:18 directive said "no less than 25k". Default to HARD unless operator says otherwise.
3. **Should Tier 2.E (script demotion) ship at all?** It's a breaking change for anyone running the old commands. Default to NO; ship 2.B/C/D, leave the old scripts as compatibility shims with a `DeprecationWarning`.

Resolve 1 first — that's the only one blocking Tier 1.
