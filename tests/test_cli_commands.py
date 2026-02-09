"""
Integration tests for CLI commands module.

Validates:
1. train_buddy re-export from cli.training matches the original
2. _MAJOR_PAIRS constant has expected FX pairs
3. Key buddy functions exist and are callable
4. Dispatch helpers have correct signatures
"""

from __future__ import annotations

import pytest

# Pre-load cli.commands to handle potential cascading import failures
# (cli/__init__.py -> candle_optimizer -> modular_trainers -> tensorflow).
# When tensorflow is not installed, the first import of any cli submodule
# fails because cli/__init__.py triggers the cascade. However, cli.commands
# itself gets loaded before the failure point, so subsequent imports work.
try:
    import cli.commands  # noqa: F401
except ModuleNotFoundError:
    pass


# ---------------------------------------------------------------------------
# Test: train_buddy re-export
# ---------------------------------------------------------------------------

class TestTrainBuddyReExport:
    """Verify train_buddy is re-exported from cli.training for backward compat."""

    def test_import_train_buddy_from_commands(self):
        """train_buddy should be importable from cli.commands."""
        from cli.commands import train_buddy
        assert callable(train_buddy), "train_buddy should be callable"

    def test_import_train_buddy_from_training(self):
        """train_buddy should be importable from cli.training."""
        from cli.training import train_buddy
        assert callable(train_buddy), "train_buddy should be callable"

    def test_same_function_object(self):
        """cli.commands.train_buddy should be the same object as cli.training.train_buddy."""
        from cli.commands import train_buddy as tb_commands
        from cli.training import train_buddy as tb_training
        assert tb_commands is tb_training, (
            "cli.commands.train_buddy should be re-exported from cli.training"
        )


# ---------------------------------------------------------------------------
# Test: _MAJOR_PAIRS constant
# ---------------------------------------------------------------------------

class TestMajorPairs:
    """Verify the _MAJOR_PAIRS constant."""

    def test_major_pairs_exist(self):
        """_MAJOR_PAIRS should be a non-empty list."""
        from cli.commands import _MAJOR_PAIRS
        assert isinstance(_MAJOR_PAIRS, list)
        assert len(_MAJOR_PAIRS) > 0

    def test_major_pairs_count(self):
        """Should have 7 major FX pairs."""
        from cli.commands import _MAJOR_PAIRS
        assert len(_MAJOR_PAIRS) == 7, (
            f"Expected 7 major pairs, got {len(_MAJOR_PAIRS)}: {_MAJOR_PAIRS}"
        )

    def test_eur_usd_in_pairs(self):
        """EUR_USD must be in the major pairs list."""
        from cli.commands import _MAJOR_PAIRS
        assert "EUR_USD" in _MAJOR_PAIRS

    def test_all_pairs_have_underscore_format(self):
        """All pairs should use OANDA underscore format (e.g., EUR_USD)."""
        from cli.commands import _MAJOR_PAIRS
        for pair in _MAJOR_PAIRS:
            assert "_" in pair, f"Pair {pair} missing underscore separator"
            parts = pair.split("_")
            assert len(parts) == 2, f"Pair {pair} should have exactly 2 currencies"
            assert len(parts[0]) == 3, f"Base currency in {pair} should be 3 chars"
            assert len(parts[1]) == 3, f"Quote currency in {pair} should be 3 chars"

    def test_major_pairs_content(self):
        """Verify all expected major pairs are present."""
        from cli.commands import _MAJOR_PAIRS

        expected = {"EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
                    "AUD_USD", "USD_CAD", "NZD_USD"}
        assert set(_MAJOR_PAIRS) == expected, (
            f"Unexpected pairs. Got {set(_MAJOR_PAIRS)}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Test: Key function existence
# ---------------------------------------------------------------------------

class TestKeyFunctions:
    """Verify key functions exist in cli.commands and have expected signatures."""

    def test_buddy_function_exists(self):
        """buddy() function should exist and be callable."""
        from cli.commands import buddy
        assert callable(buddy)

    def test_buddy_loop_function_exists(self):
        """buddy_loop() function should exist."""
        from cli.commands import buddy_loop
        assert callable(buddy_loop)

    def test_buddy_validate_function_exists(self):
        """buddy_validate() function should exist."""
        from cli.commands import buddy_validate
        assert callable(buddy_validate)

    def test_buddy_test_function_exists(self):
        """buddy_test() legacy function should exist."""
        from cli.commands import buddy_test
        assert callable(buddy_test)

    def test_dispatch_helpers_exist(self):
        """_dispatch_buddy and _dispatch_train_buddy should exist."""
        from cli.commands import _dispatch_buddy, _dispatch_train_buddy
        assert callable(_dispatch_buddy)
        assert callable(_dispatch_train_buddy)

    def test_normalize_command_args_exists(self):
        """_normalize_command_args helper should exist."""
        from cli.commands import _normalize_command_args
        assert callable(_normalize_command_args)

    def test_compute_force_units_exists(self):
        """_compute_force_units helper should exist."""
        from cli.commands import _compute_force_units
        assert callable(_compute_force_units)


# ---------------------------------------------------------------------------
# Test: _compute_force_units logic
# ---------------------------------------------------------------------------

class TestComputeForceUnits:
    """Test _compute_force_units helper function."""

    def test_none_inputs_return_none(self):
        """Both None -> None."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=None, force_margin_raw=None)
        assert result is None

    def test_force_units_raw_passthrough(self):
        """force_units_raw should be converted to int and returned."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=5000, force_margin_raw=None)
        assert result == 5000

    def test_force_units_raw_as_string(self):
        """String force_units_raw should be converted to int."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw="2000", force_margin_raw=None)
        assert result == 2000

    def test_force_margin_conversion(self):
        """force_margin_raw should be converted to units via /0.05 formula."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=None, force_margin_raw=100.0)
        assert result == int(round(100.0 / 0.05))

    def test_force_units_takes_priority(self):
        """force_units_raw takes priority over force_margin_raw."""
        from cli.commands import _compute_force_units
        result = _compute_force_units(force_units_raw=3000, force_margin_raw=100.0)
        assert result == 3000


# ---------------------------------------------------------------------------
# Test: _normalize_command_args
# ---------------------------------------------------------------------------

class TestNormalizeCommandArgs:
    """Test command argument normalization."""

    def test_normalizes_uppercase_buddy(self):
        """'Buddy' command should be lowercased to 'buddy'."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="Buddy")
        _normalize_command_args(args)
        assert args.command == "buddy"

    def test_leaves_lowercase_unchanged(self):
        """'buddy' command should not be modified."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="buddy")
        _normalize_command_args(args)
        assert args.command == "buddy"

    def test_non_buddy_command_unchanged(self):
        """Other commands should not be modified."""
        from cli.commands import _normalize_command_args
        from types import SimpleNamespace

        args = SimpleNamespace(command="train")
        _normalize_command_args(args)
        assert args.command == "train"


# ---------------------------------------------------------------------------
# Test: Buddy interactive wizard check
# ---------------------------------------------------------------------------

class TestBuddyWizardCheck:
    """Test _maybe_run_buddy_interactive_wizard existence."""

    def test_wizard_check_function_exists(self):
        """_maybe_run_buddy_interactive_wizard should exist and be callable."""
        from cli.commands import _maybe_run_buddy_interactive_wizard
        assert callable(_maybe_run_buddy_interactive_wizard)

    def test_repl_launcher_exists(self):
        """_maybe_launch_buddy_repl should exist and be callable."""
        from cli.commands import _maybe_launch_buddy_repl
        assert callable(_maybe_launch_buddy_repl)
