#!/usr/bin/env python3
"""Task 8 campaign entrypoint: explicit local inputs only; no sealed-holdout capability."""
import argparse
from pathlib import Path
import pandas as pd
from src.axiom2.research.ranker import run_registered_campaign

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--features",type=Path,required=True); parser.add_argument("--labels",type=Path,required=True)
    parser.parse_args()
    raise SystemExit("campaign authority/context must be supplied by the research service; CLI cannot create signing authority")

if __name__=="__main__": main()
