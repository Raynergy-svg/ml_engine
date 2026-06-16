#!/usr/bin/env python3
"""Merge yearly Dukascopy M15 parquets into a single harvest file."""

import glob
import sys
from pathlib import Path

import pandas as pd

def main() -> None:
    files = sorted(glob.glob("trained_data/dukascopy/yearly/*/EUR_USD_M15.parquet"))
    if not files:
        print("No M15 parquets found to merge.")
        sys.exit(0)
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    out = Path("trained_data/harvest/EUR_USD_M15_dukascopy.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd")
    print(f"Merged M15: {len(df)} bars -> {out}")

if __name__ == "__main__":
    main()
