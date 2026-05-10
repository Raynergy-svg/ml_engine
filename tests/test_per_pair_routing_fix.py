"""Per-pair model routing fix — verify Phase 5.D tuned models load.

Bug (2026-05-09): every production pair routed transformer inference to
trained_data/models/joint/transformer_direction.keras (the 11-day-old joint
fallback) or to trained_data/models/EUR_GBP/ (the historical correlation
master). Phase 5.D shipped tuned per-pair transformers (EUR_USD 58.3% OOS,
USD_JPY 63.2% OOS, GBP_USD/EUR_JPY/USD_CAD/AUD_USD/NZD_USD/USD_CHF/GBP_JPY)
and they were silently shadowed by the master alias in
``Scanner._resolve_model_source_pair``.

Root cause: ``use_master_pair_models=True`` (default) caused EUR_USD →
master EUR_GBP redirect even when ``trained_data/models/EUR_USD/`` held
freshly-tuned models. EUR_GBP itself happened to be a master, so it was the
only pair whose per-pair models EVER loaded — masking the bug as "infrastructure
works for one pair but not 14".

Fix: ``_resolve_model_source_pair`` now probes for
``trained_data/models/{PAIR}/transformer_direction.keras`` first; only falls
through to the correlation-group master when no per-pair training output
exists for the requested instrument.

Tests use REAL ``ScannerConfig``, REAL ``Scanner`` instances, and REAL disk
under ``tmp_path``. No ``unittest.mock``, no ``MagicMock``, no ``patch``
(see ``.claude/rules/improvement.md`` "No-Mock Rule").
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.scanner.config import ScannerConfig
from src.scanner.engine import Scanner


@pytest.fixture
def scanner() -> Scanner:
    """Real Scanner instance with default config — no broker, no oanda."""
    return Scanner(config=ScannerConfig())


# ---------------------------------------------------------------------------
# Production-disk assertions: confirm the live tree wires per-pair correctly.
# These rely on Phase 5.D-trained per-pair transformers being on disk at
# trained_data/models/{PAIR}/transformer_direction.keras (operator-managed).
# ---------------------------------------------------------------------------

PRODUCTION_PAIRS_WITH_PER_PAIR_MODELS = [
    "EUR_USD",   # tuned 2026-05-08 — was redirected to EUR_GBP master
    "USD_JPY",   # tuned 2026-05-08 — was redirected to USD_CAD master
    "GBP_USD",   # tuned 2026-05-08 — was redirected to EUR_GBP master
    "EUR_JPY",   # tuned 2026-05-08 — was redirected to EUR_GBP master
    "USD_CAD",   # tuned (also a master, so path returned itself but not via per-pair check)
    "AUD_USD",   # tuned (also a master)
    "NZD_USD",   # tuned 2026-05-08 — was redirected to AUD_USD master
    "USD_CHF",   # tuned 2026-05-08 — was redirected to USD_CAD master
    "GBP_JPY",   # tuned 2026-05-08 — was redirected to EUR_GBP master
]


@pytest.mark.parametrize("pair", PRODUCTION_PAIRS_WITH_PER_PAIR_MODELS)
def test_production_pair_routes_to_itself_when_per_pair_model_exists(
    scanner: Scanner, pair: str
) -> None:
    """When trained_data/models/{PAIR}/transformer_direction.keras exists,
    _resolve_model_source_pair MUST return the pair, not its master.

    Pre-fix this returned EUR_GBP / USD_CAD / AUD_USD for non-master pairs
    in their correlation group, silently shadowing the freshly-tuned per-pair
    models with stale master models.
    """
    pair_marker = Path("trained_data/models") / pair / "transformer_direction.keras"
    if not pair_marker.exists():
        pytest.skip(f"Per-pair transformer not on disk for {pair} — operator must train")

    resolved = scanner._resolve_model_source_pair(pair)
    assert resolved == pair, (
        f"BUG: {pair} resolved to {resolved!r}; per-pair model exists at "
        f"{pair_marker} but routing redirected to master. "
        f"Phase 5.D tuned models would be dead code in production."
    )


# ---------------------------------------------------------------------------
# Isolated tmp_path tests — verify the per-pair-first / master-fallback /
# strict-per-pair branches independently of the production disk layout.
# ---------------------------------------------------------------------------


def _make_synthetic_models_dir(root: Path, pair: str) -> Path:
    """Create a real per-pair model dir with the marker file the routing
    function probes (transformer_direction.keras). Empty file is fine — the
    routing function only checks ``Path.exists``."""
    pair_dir = root / "trained_data" / "models" / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    marker = pair_dir / "transformer_direction.keras"
    marker.write_bytes(b"")  # real file on real disk
    return marker


def test_per_pair_marker_present_returns_pair(tmp_path: Path) -> None:
    """In an isolated tree, EUR_USD with a per-pair marker MUST return EUR_USD,
    NOT its correlation-group master EUR_GBP."""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        _make_synthetic_models_dir(tmp_path, "EUR_USD")
        scanner = Scanner(config=ScannerConfig())
        assert scanner._resolve_model_source_pair("EUR_USD") == "EUR_USD"
    finally:
        os.chdir(cwd)


def test_per_pair_marker_absent_falls_through_to_master(tmp_path: Path) -> None:
    """Without a per-pair marker, the legacy correlation-group master
    redirect MUST still apply (don't break pairs that lack dedicated
    training, e.g. EUR_AUD / EUR_CHF / AUD_NZD-grouped instruments)."""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # No per-pair marker created for EUR_USD; expected to fall through.
        scanner = Scanner(config=ScannerConfig())
        # EUR_USD's correlation-group master is EUR_GBP per
        # src/training/correlation_group_config.py.
        assert scanner._resolve_model_source_pair("EUR_USD") == "EUR_GBP"
    finally:
        os.chdir(cwd)


def test_use_master_pair_models_false_disables_both_redirects(tmp_path: Path) -> None:
    """``use_master_pair_models=False`` MUST force the pair through, regardless
    of disk layout — escape hatch for operators who want strict per-pair
    routing for every instrument."""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Don't create a per-pair marker; should still return the pair itself
        # because the master fallback is disabled.
        cfg = ScannerConfig()
        cfg.use_master_pair_models = False
        scanner = Scanner(config=cfg)
        assert scanner._resolve_model_source_pair("EUR_USD") == "EUR_USD"
    finally:
        os.chdir(cwd)


def test_unknown_pair_returns_itself(tmp_path: Path) -> None:
    """A pair outside any correlation group MUST return itself unchanged."""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        scanner = Scanner(config=ScannerConfig())
        # Synthetic ticker the correlation_group_config doesn't know.
        assert scanner._resolve_model_source_pair("XXX_YYY") == "XXX_YYY"
    finally:
        os.chdir(cwd)


def test_pair_normalization_handles_slash_form(tmp_path: Path) -> None:
    """Operator-facing forms like 'EUR/USD' MUST normalize to the
    underscore form before path probing."""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        _make_synthetic_models_dir(tmp_path, "EUR_USD")
        scanner = Scanner(config=ScannerConfig())
        assert scanner._resolve_model_source_pair("EUR/USD") == "EUR_USD"
    finally:
        os.chdir(cwd)
