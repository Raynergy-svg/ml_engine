"""Helper utilities for CLI module.

This module provides parsing functions, path utilities, and other
helper functions used throughout the CLI commands.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.cli.config import (
    VALID_OANDA_INSTRUMENTS,
    TRANSFORMER_DIRECTION_FILENAME,
    TCN_DIRECTION_FILENAME,
    TRANSFORMER_REGIME_FILENAME,
    XGB_MOMENTUM_FILENAME,
    RF_RISK_FILENAME,
    RIDGE_CONFIDENCE_FILENAME,
)

logger = logging.getLogger(__name__)


def parse_bool_answer(s: str | None, *, default: bool) -> bool:
    """Parse a string answer as boolean.

    Args:
        s: Input string (y/n/yes/no/true/false/1/0).
        default: Default value if input is empty or None.

    Returns:
        Parsed boolean value.
    """
    if s is None:
        return bool(default)
    v = str(s).strip().lower()
    if not v:
        return bool(default)
    if v in {"y", "yes", "true", "1", "on"}:
        return True
    if v in {"n", "no", "false", "0", "off"}:
        return False
    return bool(default)


def parse_float_answer(s: str | None, *, default: float) -> float:
    """Parse a string answer as float.

    Args:
        s: Input string containing a number.
        default: Default value if parsing fails.

    Returns:
        Parsed float value.
    """
    if s is None:
        return float(default)
    v = str(s).strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except (ValueError, TypeError):
        return float(default)


def parse_int_answer(s: str | None, *, default: int) -> int:
    """Parse a string answer as integer.

    Args:
        s: Input string containing a number.
        default: Default value if parsing fails.

    Returns:
        Parsed integer value.
    """
    if s is None:
        return int(default)
    v = str(s).strip()
    if not v:
        return int(default)
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return int(default)


def normalize_instrument(s: str) -> str:
    """Normalize instrument name to OANDA format (e.g., EUR_USD).

    Args:
        s: Input instrument string (e.g., "eurusd", "EUR-USD", "EUR/USD").

    Returns:
        Normalized instrument string (e.g., "EUR_USD").
    """
    v = (s or "").strip().upper().replace("-", "_").replace("/", "_")
    v = "_".join([p for p in v.split("_") if p])
    if "_" not in v and len(v) == 6:
        v = v[:3] + "_" + v[3:]
    return v


def extract_instrument_from_csv_path(csv_path: str) -> str:
    """Extract instrument from CSV filename for pair-specific model training.

    Patterns recognized:
    - market_data/EUR_USD_H1.csv -> EUR_USD
    - market_data/oanda_GBP_USD_H1_live_*.csv -> GBP_USD
    - market_data/USD_JPY_*.csv -> USD_JPY
    - market_data/EURUSD.csv -> EUR_USD (6-char format)

    Args:
        csv_path: Path to CSV file.

    Returns:
        Normalized instrument or "GENERIC" if not extractable.
    """
    if not csv_path:
        return "GENERIC"

    filename = Path(csv_path).stem.upper()

    # Pattern 1: Direct match for XXX_YYY format
    for instrument in VALID_OANDA_INSTRUMENTS:
        if instrument in filename:
            return instrument

    # Pattern 2: 6-char format like "EURUSD"
    match = re.search(r"([A-Z]{6})", filename)
    if match:
        candidate = match.group(1)[:3] + "_" + match.group(1)[3:]
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate

    # Pattern 3: Try to extract XXX_YYY pattern from filename
    match = re.search(r"([A-Z]{3})_([A-Z]{3})", filename)
    if match:
        candidate = f"{match.group(1)}_{match.group(2)}"
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate

    return "GENERIC"


def get_pair_model_paths(
    model_dir: Path, instrument: str, model_type: str = "transformer"
) -> dict[str, Any]:
    """Get pair-specific model paths for saving/loading.

    Args:
        model_dir: Base directory for models.
        instrument: Instrument name (e.g., "EUR_USD") or "GENERIC".
        model_type: Model type ("transformer" or "tcn").

    Returns:
        Dict with paths for direction, regime, xgboost, rf, ridge, histgb.

    Example for EUR_USD:
        - trained_data/models/EUR_USD/transformer_direction.keras
        - trained_data/models/EUR_USD/xgb_momentum.pkl
    """
    # Use GENERIC for generic models, pair name for pair-specific
    if instrument and instrument != "GENERIC":
        pair_dir = model_dir / instrument
    else:
        pair_dir = model_dir

    # Ensure directory exists
    pair_dir.mkdir(parents=True, exist_ok=True)

    direction_filename = (
        TRANSFORMER_DIRECTION_FILENAME
        if model_type == "transformer"
        else TCN_DIRECTION_FILENAME
    )

    return {
        "direction": pair_dir / direction_filename,
        "regime": pair_dir / TRANSFORMER_REGIME_FILENAME,
        "xgboost": pair_dir / XGB_MOMENTUM_FILENAME,
        "rf": pair_dir / RF_RISK_FILENAME,
        "ridge": pair_dir / RIDGE_CONFIDENCE_FILENAME,
        "histgb": pair_dir / "histgb_direction.pkl",
        "pair_dir": pair_dir,
    }


def validate_instrument(instrument: str) -> str:
    """Validate and normalize an instrument name.

    Args:
        instrument: Input instrument string.

    Returns:
        Normalized instrument string.

    Raises:
        ValueError: If instrument is invalid with suggestions.
    """
    # Special case: "all" means scan all major pairs
    if instrument.upper() == "ALL":
        return "ALL"

    normalized = normalize_instrument(instrument)

    if normalized in VALID_OANDA_INSTRUMENTS:
        return normalized

    # Try to suggest a correction for common typos
    suggestions = []
    for valid in VALID_OANDA_INSTRUMENTS:
        # Check if it's a simple character swap or typo
        if len(normalized) == len(valid):
            diff = sum(1 for a, b in zip(normalized, valid) if a != b)
            if diff <= 2:
                suggestions.append(valid)

    error_msg = f"Invalid instrument: '{instrument}' (normalized: '{normalized}')"
    if suggestions:
        error_msg += f"\n  Did you mean: {', '.join(sorted(suggestions))}?"
    error_msg += (
        "\n  Valid examples: EUR_USD, GBP_USD, USD_JPY, AUD_USD, EUR_GBP, "
        "or 'all' for all major pairs"
    )

    raise ValueError(error_msg)


# Aliases for backward compatibility with main.py naming convention
_parse_bool_answer = parse_bool_answer
_parse_float_answer = parse_float_answer
_parse_int_answer = parse_int_answer
_normalize_instrument = normalize_instrument
_extract_instrument_from_csv_path = extract_instrument_from_csv_path
_get_pair_model_paths = get_pair_model_paths
_validate_instrument = validate_instrument
