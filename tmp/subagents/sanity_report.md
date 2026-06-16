# SOTA RawSequenceModel Sanity Report

Generated: 2026-06-15T17:43:12Z

## 1. Tensor Compilation Test

| Metric     | Value     |
|------------|-----------|
| Status     | **PASSED** |
| Duration   | ~7.4s     |
| Input shape| (1, 128, 5)|
| Direction output shape | (1, 1) ✅ |
| Regime output shape    | (1, 4) ✅ |

Model instantiates cleanly with:
- `seq_len=128`, `d_model=128`, `num_layers=3`, `num_heads=4`
- `build()` succeeds
- Forward pass on dummy NumPy tensor produces expected shapes
- No TensorFlow/Keras crashes

## 2. Data Availability

| Source                  | Status       |
|-------------------------|--------------|
| `trained_data/harvest/*.parquet` | **MISSING** |
| `trained_data/csv/*.csv`         | **MISSING** |

No training data is present in the workspace. The overfit sanity test requires at least one data file.

## 3. Dry-Run Output Summary

```
2026-06-15 17:43:12,937 INFO __main__ Discovered 0 data files
2026-06-15 17:43:12,937 INFO __main__ Using 0 data paths
```

Script structure validates successfully:
- Entry point executes without traceback
- Logger configured correctly
- Data discovery logic runs (finds 0 files, exits gracefully)
- Exit code: 0

## 4. Overfit Test

**SKIPPED** — No training data available.

Preconditions not met:
- No parquet files in `trained_data/harvest/`
- No CSV files in `trained_data/csv/`

To run the overfit test, populate one of the above directories with labeled price-window data, then re-run.

## 5. Errors / Warnings

- None. All executed tests completed without error.
- Zero deprecation warnings emitted during tensor test.
- Dry-run exits gracefully with no crash.

## Recommendation

Training infrastructure is structurally sound. Next step is to generate or import training data (e.g., via the harvest pipeline or CSV ingestion) before attempting the 1K-window overfit test.
