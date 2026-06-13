# Review queue — Danger Zone proposals for the human

Pre-seeded from the 2026-06-11 production audit. These are OUTSIDE the loop's writable
surface (src/scanner, src/risk, .claude, CI) — human or supervised-session work only.

## P0 — financial correctness (audit: Code Reviewer agent, all file:line verified)

1. `src/scanner/gates.py:1587,1889` — scaler.transform failure silently falls back to RAW
   unscaled features before predict (`except Exception: pass  # Use raw features`).
   Reproduces the C1 silent-skew class. Proposal: refuse (return None) + log contract-gap
   warning, mirroring the existing zero-fill refusal rule.
2. `src/scanner/automation/state_engine.py:136,183,193` — three direct `write_text` calls to
   `.claude/state.json` bypass the `_save_atomic` defined at :101 (one runs every scan cycle).
   Proposal: route all three through `_save_atomic`.
3. `src/scanner/execution.py:4300` — SL/TP cache read swallowed; sl/tp stay 0.0 and portfolio
   risk is silently understated. Proposal: log + treat pair as max-risk (conservative) or
   surface as rejection.
4. `src/scanner/engine.py` — 22 `except Exception: pass` sites; the per-pair ATR SL/TP config
   read (cluster 3713-4286) silently overrides tuned SL/TP with defaults. Proposal: add
   logger.warning at minimum on every financial-path swallow.
5. `src/scanner/agents/_team.py:956,1561` — agent-weight loads fail silently (learned weights
   silently wrong). Proposal: log + surface in analysis.agent_reason_codes.
6. `src/scanner/agents/_team.py:1338` — episodic-memory suppression hardcodes session="london".

## P1 — structural (audit: DevOps agent)

7. Runtime-state untrack: 563 ignored-but-tracked files, 181 model binaries, 311 CSVs, .git
   at 680MB. Proposal: one `git rm -r --cached` commit + W&B artifacts (already wired); no LFS,
   no history rewrite. ~half day, reversible.
8. No lockfile: 22 range-pinned deps resolve fresh in CI. Proposal: uv lock or pip-compile.
9. CI: ruff configured but never run; no mypy gate; coverage `|| true`; actions tag-pinned.
10. `scripts/run_full_training.sh` + 2 others have no `set -euo pipefail`.

## P1 — loop environment (2026-06-12 red-base iteration; loop cannot self-fix these)

11. **flake8 is not installed in the loop interpreter** (`/opt/homebrew/Caskroom/miniforge/base`).
    `python -m flake8` exits "No module named flake8", so the contract's lint gate reports FAIL
    on every iteration regardless of code quality. Checked: base env, tf-metal env,
    /opt/homebrew/bin, ~/.local/bin — no flake8 anywhere; pip install is outside the allowlist.
    Proposal: operator runs `pip install flake8==7.3.0` (the CI-pinned version) into the base
    interpreter, OR the runner's verify step uses a dedicated venv.
12. **pytest-asyncio is not installed in the loop interpreter.** `@pytest.mark.asyncio` tests
    fail (not skip) with "async def functions are not natively supported". The two
    test_command_palette.py modal tests were converted to sync `asyncio.run()` drivers this
    iteration, so nothing currently depends on it — but any future async test will silently
    re-introduce the failure mode. Proposal: `pip install pytest-asyncio` alongside flake8.
13. **Stale docstrings in Danger Zone files** (can't edit from the loop): 
    `src/scanner/agents/_team.py:739` still documents the weight clamp as `[0.05, 10.0]`
    (actual: `[0.1, 2.0]` since 66aca32); `src/scanner/automation/event_handlers.py:1045`
    still says "60s timeout" (actual: 90s lightweight / 420s deep since 96fb666).
14. **The runner's STATE.md pytest run was KeyboardInterrupted at 36.71s**
    (`continuous.py:1426` in the traceback) — the suite never completed, so the failure list
    in STATE.md was a truncated snapshot. If the runner imposes a hard timeout on the
    state-snapshot pytest, consider raising it or noting truncation in STATE.md.
    *(2026-06-12 21:41 update: recurred — same `continuous.py:1426` KeyboardInterrupt signature
    at 58.43s with `1152 passed, 23 skipped, 4 xfailed` and zero failures listed; STATE.md
    recorded "pytest FAIL (exit 2)" purely from the interrupt exit code. Still open.)*

*(2026-06-12 21:41 status: items 11 and 12 are RESOLVED — operator installed flake8 7.3.0 +
pytest-asyncio into the base interpreter at ~17:40; evidence: STATE.md 21:41 snapshot contains
real flake8 violation output, impossible if the module were missing.)*

## P2 — lint fix outside the loop's SAFE surface (2026-06-12 21:41 red-base iteration)

15. `src/utils/oanda_practice.py:23` — flake8 `F821 undefined name 'Path'`. The annotation
    `path: "Path"` references `Path`, which is only imported inside `_load_project_dotenv()`
    (line 68), never at module scope. Runtime-harmless today (`from __future__ import
    annotations` at :6 means the annotation is never evaluated), but it keeps `flake8 src/`
    permanently red and would break under `typing.get_type_hints()`. `src/utils/**` is outside
    the loop's SAFE surface (src/tui, tests, docs, notebooks), so proposing instead of editing.
    Exact patch (behavior-neutral, import-only):
    add `from pathlib import Path` to the module-level imports after `from typing import ...`
    (line 10), and drop the quotes on the annotation (`path: Path`). pathlib is stdlib and
    side-effect-free; no execution/risk/signal logic is touched. Until this lands, the loop's
    flake8 gate will show exactly 1 residual error from this file.
