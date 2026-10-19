# Axiom 2.0 FX Baseline Manifest

**Status:** Frozen pre-migration reference  
**Recorded from:** `main`  
**Baseline commit:** `4c6ace9aaf6ea6a79327fc9f025e618f30645285`  
**Baseline parent:** `1e3b1d04220f62ff94e7eb5787c0b01144b24fb6`  
**Recorded on:** 2026-09-30

## Purpose

This manifest freezes the reproducible FX/OANDA baseline before Axiom 2.0
equities-first work begins. It is an index of existing code and evidence
locations, not a migration or a replacement evidence store.

The FX/OANDA lane remains unchanged and reproducible. No files in the
locations below are moved, deleted, or made part of the Axiom 2.0 default
execution path by Task 1.

## Existing FX/OANDA implementation locations

- `src/brokers/oanda.py`
- `src/brokers/oanda_v20.py`
- `src/utils/oanda_practice.py`
- `src/utils/oanda_streaming.py`
- `src/scanner/fx_retired.py`
- `cli/fx_trading.py`
- `oanda_practice.py`

## Existing FX/OANDA evidence locations

- `src/evidence/` — canonical, hashing, signing, store, and evidence
  contract infrastructure.
- `trained_data/backtests/oanda_sentiment_result.json` — committed OANDA
  backtest result artifact.
- `docs/implementation-notes/MANIFEST.txt` — existing implementation
  manifest/index.
- `docs/fx-edge-search-final-verdict-2026-06-18.md` — recorded FX edge-search
  verdict.
- `docs/fx-directional-path-retired.md` — recorded retirement status for the
  prior FX directional path.

## Axiom 2.0 namespace boundary

The new namespace begins at:

- `src/axiom2/`
- `src/axiom2/contracts/`

These initializers intentionally import no broker modules. Later Axiom 2.0
tasks must preserve this boundary while keeping the FX/OANDA locations above
available for legacy research and reproducibility.
