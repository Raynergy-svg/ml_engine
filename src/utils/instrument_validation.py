"""Instrument validation utilities for OANDA FX trading.

This module provides validation, normalization, and utility functions for
OANDA trading instruments.

Includes:
- Complete set of valid OANDA FX instruments (majors, minors, exotics)
- Instrument name normalization (handles various input formats)
- CSV path parsing for automatic instrument detection
- Pair-specific model path generation

Example usage:
    from src.utils.instrument_validation import (
        validate_instrument,
        normalize_instrument,
        extract_instrument_from_csv_path,
        get_pair_model_paths,
        VALID_OANDA_INSTRUMENTS,
    )

    # Validate user input
    instrument = validate_instrument("eur-usd")  # Returns "EUR_USD"

    # Normalize various formats
    normalize_instrument("EURUSD")  # Returns "EUR_USD"
    normalize_instrument("eur/usd")  # Returns "EUR_USD"

    # Extract from CSV path
    pair = extract_instrument_from_csv_path("market_data/GBP_USD_H1.csv")  # "GBP_USD"

    # Get model paths for a pair
    paths = get_pair_model_paths(Path("trained_data/models"), "EUR_USD")
"""

from __future__ import annotations

import re
from pathlib import Path


# Valid OANDA FX instruments (major, minor, and exotic pairs)
VALID_OANDA_INSTRUMENTS: frozenset[str] = frozenset({
    # Major pairs
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    # Cross pairs
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CHF", "NZD_CAD",
    # Exotic pairs
    "USD_SGD", "USD_HKD", "USD_MXN", "USD_ZAR", "USD_TRY", "USD_SEK", "USD_NOK", "USD_DKK",
    "USD_PLN", "USD_HUF", "USD_CZK", "USD_THB", "USD_INR", "USD_CNH",
    "EUR_SEK", "EUR_NOK", "EUR_DKK", "EUR_PLN", "EUR_HUF", "EUR_CZK", "EUR_TRY", "EUR_ZAR",
    "GBP_SGD", "GBP_PLN", "GBP_ZAR",
    "AUD_SGD", "AUD_HKD",
    "SGD_JPY", "SGD_CHF", "HKD_JPY", "TRY_JPY", "ZAR_JPY", "MXN_JPY",
})


# Major pairs for scanning (most liquid)
MAJOR_PAIRS: tuple[str, ...] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
)


# Cross pairs (commonly traded)
CROSS_PAIRS: tuple[str, ...] = (
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "GBP_AUD",
    "EUR_CAD", "GBP_CAD", "AUD_CAD", "NZD_JPY", "CAD_JPY", "CHF_JPY",
)


def normalize_instrument(instrument: str) -> str:
    """Normalize an instrument name to OANDA format.

    Handles various input formats:
    - "EUR_USD" -> "EUR_USD" (already normalized)
    - "eur_usd" -> "EUR_USD" (lowercase)
    - "EUR-USD" -> "EUR_USD" (hyphen)
    - "EUR/USD" -> "EUR_USD" (slash)
    - "EURUSD" -> "EUR_USD" (6-char no separator)
    - "  EUR_USD  " -> "EUR_USD" (whitespace)

    Args:
        instrument: Instrument name in any common format.

    Returns:
        Normalized instrument string in "XXX_YYY" format.
    """
    # Strip whitespace, uppercase, normalize separators
    v = (instrument or "").strip().upper().replace("-", "_").replace("/", "_")

    # Remove any duplicate or leading/trailing underscores
    v = "_".join([p for p in v.split("_") if p])

    # Handle 6-char format without separator
    if "_" not in v and len(v) == 6:
        v = v[:3] + "_" + v[3:]

    return v


def validate_instrument(instrument: str) -> str:
    """Validate and normalize an instrument name.

    Args:
        instrument: Instrument name to validate.

    Returns:
        Normalized instrument name if valid.

    Raises:
        ValueError: If instrument is not a valid OANDA FX instrument.
            Includes suggestions for common typos.
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
    error_msg += "\n  Valid examples: EUR_USD, GBP_USD, USD_JPY, AUD_USD, EUR_GBP, or 'all' for all major pairs"

    raise ValueError(error_msg)


def is_valid_instrument(instrument: str) -> bool:
    """Check if an instrument is valid without raising an exception.

    Args:
        instrument: Instrument name to check.

    Returns:
        True if valid, False otherwise.
    """
    try:
        validate_instrument(instrument)
        return True
    except ValueError:
        return False


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
        Normalized instrument name if found, "GENERIC" otherwise.
        Returns "GENERIC" for files that don't contain a recognizable pair.
    """
    if not csv_path:
        return "GENERIC"

    filename = Path(csv_path).stem.upper()  # e.g., "EUR_USD_H1" or "oanda_GBP_USD_H1_live_20250114"

    # Pattern 1: Direct match for XXX_YYY format
    for instrument in VALID_OANDA_INSTRUMENTS:
        if instrument in filename:
            return instrument

    # Pattern 2: 6-char format like "EURUSD"
    match = re.search(r'([A-Z]{6})', filename)
    if match:
        candidate = match.group(1)[:3] + "_" + match.group(1)[3:]
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate

    # Pattern 3: Try to extract XXX_YYY pattern from filename
    match = re.search(r'([A-Z]{3})_([A-Z]{3})', filename)
    if match:
        candidate = f"{match.group(1)}_{match.group(2)}"
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate

    return "GENERIC"


def get_pair_model_paths(
    model_dir: Path,
    instrument: str,
    model_type: str = "transformer",
) -> dict[str, Path]:
    """Get pair-specific model paths for saving/loading.

    Returns dict with paths for all model components:
    - direction: Transformer/TCN direction model
    - regime: Regime detection model
    - xgboost: XGBoost momentum model
    - rf: Random Forest risk model
    - ridge: Ridge confidence model
    - histgb: HistGB hybrid voting model (optional)
    - pair_dir: Directory containing all pair models

    Example for EUR_USD:
    - trained_data/models/EUR_USD/transformer_direction.keras
    - trained_data/models/EUR_USD/xgb_momentum.pkl

    Args:
        model_dir: Base directory for model storage.
        instrument: Trading instrument (e.g., "EUR_USD").
        model_type: Model architecture type ("transformer" or "tcn").

    Returns:
        Dictionary mapping model component names to their file paths.
    """
    # Use GENERIC for generic models, pair name for pair-specific
    if instrument and instrument != "GENERIC":
        pair_dir = model_dir / instrument
    else:
        pair_dir = model_dir

    # Ensure directory exists
    pair_dir.mkdir(parents=True, exist_ok=True)

    direction_filename = "transformer_direction.keras" if model_type == "transformer" else "tcn_direction.keras"

    return {
        'direction': pair_dir / direction_filename,
        'regime': pair_dir / "transformer_regime.keras",
        'xgboost': pair_dir / "xgb_momentum.pkl",
        'rf': pair_dir / "rf_risk.pkl",
        'ridge': pair_dir / "ridge_confidence.pkl",
        'histgb': pair_dir / "histgb_direction.pkl",
        'pair_dir': pair_dir,
    }


def get_base_currency(instrument: str) -> str:
    """Get the base currency of an instrument.

    Args:
        instrument: Normalized instrument (e.g., "EUR_USD").

    Returns:
        Base currency code (e.g., "EUR").
    """
    normalized = normalize_instrument(instrument)
    return normalized.split("_")[0] if "_" in normalized else normalized[:3]


def get_quote_currency(instrument: str) -> str:
    """Get the quote currency of an instrument.

    Args:
        instrument: Normalized instrument (e.g., "EUR_USD").

    Returns:
        Quote currency code (e.g., "USD").
    """
    normalized = normalize_instrument(instrument)
    return normalized.split("_")[1] if "_" in normalized else normalized[3:]


def is_jpy_pair(instrument: str) -> bool:
    """Check if instrument contains JPY (affects pip calculation).

    Args:
        instrument: Instrument name.

    Returns:
        True if JPY is base or quote currency.
    """
    normalized = normalize_instrument(instrument)
    return "JPY" in normalized


def get_pip_value(instrument: str) -> float:
    """Get the pip value for an instrument.

    Most pairs: 0.0001 (1 pip = 4th decimal place)
    JPY pairs: 0.01 (1 pip = 2nd decimal place)

    Args:
        instrument: Instrument name.

    Returns:
        Pip value (0.0001 or 0.01).
    """
    return 0.01 if is_jpy_pair(instrument) else 0.0001


def format_pair(instrument: str, style: str = "oanda") -> str:
    """Format an instrument in different styles.

    Args:
        instrument: Instrument name.
        style: Output style - "oanda" (EUR_USD), "slash" (EUR/USD),
               "compact" (EURUSD).

    Returns:
        Formatted instrument string.
    """
    normalized = normalize_instrument(instrument)

    if style == "oanda":
        return normalized
    elif style == "slash":
        return normalized.replace("_", "/")
    elif style == "compact":
        return normalized.replace("_", "")
    else:
        raise ValueError(f"Unknown format style: {style}")


__all__ = [
    # Constants
    "VALID_OANDA_INSTRUMENTS",
    "MAJOR_PAIRS",
    "CROSS_PAIRS",
    # Core functions
    "normalize_instrument",
    "validate_instrument",
    "is_valid_instrument",
    "extract_instrument_from_csv_path",
    "get_pair_model_paths",
    # Utility functions
    "get_base_currency",
    "get_quote_currency",
    "is_jpy_pair",
    "get_pip_value",
    "format_pair",
]
