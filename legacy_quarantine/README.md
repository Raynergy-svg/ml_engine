# Legacy quarantine

This folder holds legacy/retired/stale materials that are intentionally removed from the primary "supported" surface of the repo.

## Why this exists
- Prevents running or following outdated training entrypoints (e.g. `train_visual.py`, `train_enhanced.py`).
- Keeps older artifacts available for reference or future recovery.

## Supported entrypoints (current)
- Training: `python main.py train-buddy ...`
- Trading/runtime: `python main.py buddy ...`

## What goes here
- Stale docs referencing removed entrypoints.
- Tests that are explicitly retired (e.g. unconditional `SkipTest("Retired: ...")`).
- Retired modules/scripts that should not be used.
