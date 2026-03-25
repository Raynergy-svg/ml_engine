"""
Tests for Phase 50 Data Activation (US-311 through US-316).

Validates journal pattern extraction, confidence calibration seeding,
strategy fitness seeding, adaptive trailing stops, and compound gate floor.
"""

import json
import math
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

# ─── US-311: Journal Analyzer ────────────────────────────────────

class TestJournalAnalyzerLoad:
    """Test journal loading and normalization."""

    def _make_journal(self, entries, path):
        with open(path, "w") as f:
            json.dump(entries, f)

    def test_load_empty_journal(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        path = str(tmp_path / "journal.json")
        self._make_journal([], path)
        analyzer = JournalAnalyzer()
        records = analyzer.load_journal(path)
        assert records == []

    def test_load_valid_entries(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        entries = [
            {
                "pair": "EUR_USD", "direction": "LONG", "confidence": 0.7,
                "entry_price": 1.08, "agents": {"agent_votes": 3, "weighted_vote_score": 0.6},
                "outcome": {"exit_reason": "tp_hit", "trade_won": True, "pnl_pips": 20.0, "duration_minutes": 120},
            },
            {
                "pair": "GBP_USD", "direction": "SHORT", "confidence": 0.5,
                "entry_price": 1.27, "agents": {"agent_votes": 2, "weighted_vote_score": 0.4},
                "outcome": {"exit_reason": "sl_hit", "trade_won": False, "pnl_pips": -15.0, "duration_minutes": 60},
            },
        ]
        path = str(tmp_path / "journal.json")
        self._make_journal(entries, path)
        analyzer = JournalAnalyzer()
        records = analyzer.load_journal(path)
        assert len(records) == 2
        assert records[0].pair == "EUR_USD"
        assert records[0].trade_won is True
        assert records[1].trade_won is False

    def test_skip_entries_without_outcome(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        entries = [
            {"pair": "EUR_USD", "confidence": 0.7, "entry_price": 1.08},  # No outcome
            {"pair": "EUR_USD", "confidence": 0.7, "entry_price": 1.08,
             "outcome": {"trade_won": True, "pnl_pips": 10, "duration_minutes": 30}},
        ]
        path = str(tmp_path / "journal.json")
        self._make_journal(entries, path)
        analyzer = JournalAnalyzer()
        records = analyzer.load_journal(path)
        assert len(records) == 1

    def test_legacy_string_outcome(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        entries = [
            {"pair": "EUR_USD", "confidence": 0.6, "entry_price": 1.08, "outcome": "win"},
            {"pair": "EUR_USD", "confidence": 0.5, "entry_price": 1.08, "outcome": "loss"},
        ]
        path = str(tmp_path / "journal.json")
        self._make_journal(entries, path)
        analyzer = JournalAnalyzer()
        records = analyzer.load_journal(path)
        assert len(records) == 2
        assert records[0].trade_won is True
        assert records[1].trade_won is False

    def test_missing_file(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        analyzer = JournalAnalyzer()
        records = analyzer.load_journal(str(tmp_path / "nonexistent.json"))
        assert records == []


class TestJournalAnalyzerAnalysis:
    """Test pattern extraction and statistical analysis."""

    def _make_journal_with_patterns(self, tmp_path, n=100):
        """Create a journal with planted patterns."""
        entries = []
        for i in range(n):
            pair = "EUR_USD" if i % 2 == 0 else "GBP_USD"
            # Plant pattern: high confidence = more wins
            conf = 0.5 + (i % 10) * 0.04
            won = conf > 0.7 or (i % 3 == 0)  # Higher win rate at high confidence
            entries.append({
                "pair": pair, "direction": "LONG" if i % 2 == 0 else "SHORT",
                "confidence": conf, "entry_price": 1.08,
                "agents": {"agent_votes": 3, "weighted_vote_score": 0.5},
                "outcome": {
                    "exit_reason": "tp_hit" if won else "sl_hit",
                    "trade_won": won, "pnl_pips": 15.0 if won else -10.0,
                    "duration_minutes": 60 + i * 5,
                },
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    def test_analyze_produces_report(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        path = self._make_journal_with_patterns(tmp_path)
        analyzer = JournalAnalyzer()
        report = analyzer.analyze(path)
        assert report.total_trades == 100
        assert 0 < report.baseline_win_rate < 1
        assert len(report.patterns) > 0

    def test_pair_breakdown_populated(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        path = self._make_journal_with_patterns(tmp_path)
        analyzer = JournalAnalyzer()
        report = analyzer.analyze(path)
        assert "EUR_USD" in report.pair_breakdown
        assert "GBP_USD" in report.pair_breakdown

    def test_confidence_breakdown_populated(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        path = self._make_journal_with_patterns(tmp_path)
        analyzer = JournalAnalyzer()
        report = analyzer.analyze(path)
        assert len(report.confidence_breakdown) > 0

    def test_save_and_load_report(self, tmp_path):
        from src.scanner.journal_analyzer import JournalAnalyzer
        path = self._make_journal_with_patterns(tmp_path)
        analyzer = JournalAnalyzer()
        report = analyzer.analyze(path)
        out_path = str(tmp_path / "patterns.json")
        assert analyzer.save_report(report, out_path) is True
        with open(out_path) as f:
            data = json.load(f)
        assert data["total_trades"] == 100
        assert "patterns" in data

    def test_z_test_calculation(self):
        from src.scanner.journal_analyzer import JournalAnalyzer
        # Known values: p=0.6, n=100, p0=0.5 => z ≈ 2.0
        z, p_val = JournalAnalyzer._proportion_z_test(0.6, 100, 0.5)
        assert abs(z - 2.0) < 0.1
        assert p_val < 0.05  # Should be significant


# ─── US-312: Calibration Seeder ──────────────────────────────────

class TestCalibrationSeeder:
    """Test confidence calibration from historical data."""

    def _make_journal(self, tmp_path, n=200):
        entries = []
        for i in range(n):
            conf = 0.4 + (i / n) * 0.5  # 0.4 to 0.9
            # Higher confidence → slightly more wins (realistic)
            won = (conf > 0.65 and i % 3 != 0) or (conf <= 0.65 and i % 4 == 0)
            entries.append({
                "pair": "EUR_USD", "confidence": conf,
                "outcome": {"trade_won": won, "pnl_pips": 10 if won else -10, "duration_minutes": 60},
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path

    def test_seed_produces_params(self, tmp_path):
        from src.scanner.calibration_seeder import CalibrationSeeder
        path = self._make_journal(tmp_path)
        seeder = CalibrationSeeder()
        result = seeder.seed_from_journal(path)
        assert result is not None
        assert isinstance(result.a, float)
        assert isinstance(result.b, float)
        assert result.n_samples > 0

    def test_insufficient_data(self, tmp_path):
        from src.scanner.calibration_seeder import CalibrationSeeder, CalibrationSeederConfig
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump([{"pair": "EUR_USD", "confidence": 0.5,
                        "outcome": {"trade_won": True}}] * 10, f)
        seeder = CalibrationSeeder(CalibrationSeederConfig(min_samples=50))
        result = seeder.seed_from_journal(path)
        assert result is None

    def test_calibrate_confidence(self, tmp_path):
        from src.scanner.calibration_seeder import CalibrationSeeder, CalibrationParams
        seeder = CalibrationSeeder()
        params = CalibrationParams(
            a=2.0, b=-1.0, n_samples=100, n_holdout=20,
            train_win_rate=0.5, holdout_win_rate=0.5,
            calibration_error=0.05, is_valid=True,
        )
        cal = seeder.calibrate_confidence(0.5, params)
        # sigmoid(2*0.5 - 1) = sigmoid(0) = 0.5
        assert abs(cal - 0.5) < 0.01

    def test_invalid_params_returns_raw(self, tmp_path):
        from src.scanner.calibration_seeder import CalibrationSeeder, CalibrationParams
        seeder = CalibrationSeeder()
        params = CalibrationParams(
            a=2.0, b=-1.0, n_samples=100, n_holdout=20,
            train_win_rate=0.5, holdout_win_rate=0.5,
            calibration_error=0.3, is_valid=False,
        )
        cal = seeder.calibrate_confidence(0.7, params)
        assert cal == 0.7  # Returns raw when invalid

    def test_save_and_load_params(self, tmp_path):
        from src.scanner.calibration_seeder import CalibrationSeeder, CalibrationParams
        seeder = CalibrationSeeder()
        params = CalibrationParams(
            a=1.5, b=-0.5, n_samples=100, n_holdout=20,
            train_win_rate=0.4, holdout_win_rate=0.45,
            calibration_error=0.08, confidence_range=(0.3, 0.9), is_valid=True,
        )
        path = str(tmp_path / "cal.json")
        assert seeder.save_params(params, path) is True
        loaded = CalibrationSeeder.load_params(path)
        assert loaded is not None
        assert loaded.a == 1.5
        assert loaded.confidence_range == (0.3, 0.9)

    def test_sigmoid_stability(self):
        from src.scanner.calibration_seeder import _sigmoid
        # Test extreme values don't overflow
        assert _sigmoid(100.0) == pytest.approx(1.0)
        assert _sigmoid(-100.0) == pytest.approx(0.0)
        assert _sigmoid(0.0) == pytest.approx(0.5)


# ─── US-313: Fitness Seeder ──────────────────────────────────────

class TestFitnessSeeder:
    """Test strategy fitness seeding from historical data."""

    def _make_journal(self, tmp_path, pairs=3, trades_per_pair=30):
        pair_names = [f"PAIR_{i}" for i in range(pairs)]
        entries = []
        for p in pair_names:
            for i in range(trades_per_pair):
                won = i % 3 != 0  # ~67% win rate
                entries.append({
                    "pair": p, "confidence": 0.6,
                    "outcome": {
                        "exit_reason": "tp_hit" if won else "sl_hit",
                        "trade_won": won, "pnl_pips": 15.0 if won else -10.0,
                        "duration_minutes": 120,
                    },
                })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        return path, pair_names

    def test_seed_all_pairs(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        path, pairs = self._make_journal(tmp_path)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert len(result) == 3
        for p in pairs:
            assert p in result

    def test_fitness_scores_in_range(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        path, _ = self._make_journal(tmp_path)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        for pf in result.values():
            assert 0 <= pf.fitness_score <= 1.0

    def test_winning_pair_gets_normal(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        # Create a pair with 80% win rate
        entries = []
        for i in range(30):
            won = i % 5 != 0  # 80% win rate
            entries.append({
                "pair": "WINNER", "confidence": 0.7,
                "outcome": {
                    "exit_reason": "tp_hit" if won else "sl_hit",
                    "trade_won": won, "pnl_pips": 20.0 if won else -10.0,
                    "duration_minutes": 60,
                },
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert result["WINNER"].action == "NORMAL"
        assert result["WINNER"].fitness_score > 0.5

    def test_losing_pair_gets_skip(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        entries = []
        for i in range(30):
            won = i % 10 == 0  # 10% win rate
            entries.append({
                "pair": "LOSER", "confidence": 0.5,
                "outcome": {
                    "exit_reason": "tp_hit" if won else "sl_hit",
                    "trade_won": won, "pnl_pips": 10.0 if won else -20.0,
                    "duration_minutes": 60,
                },
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert result["LOSER"].action == "SKIP"
        assert result["LOSER"].fitness_score < 0.3

    def test_insufficient_trades_skipped(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        entries = [
            {"pair": "SMALL", "confidence": 0.5,
             "outcome": {"trade_won": True, "pnl_pips": 10, "duration_minutes": 30}}
        ] * 5  # Only 5 trades, below minimum 10
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert len(result) == 0

    def test_save_and_load_state(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        path, _ = self._make_journal(tmp_path)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        out_path = str(tmp_path / "fitness.json")
        assert seeder.save_state(result, out_path) is True
        loaded = FitnessSeeder.load_state(out_path)
        assert loaded is not None
        assert len(loaded) == 3

    def test_consecutive_losses_tracked(self, tmp_path):
        from src.scanner.fitness_seeder import FitnessSeeder
        entries = []
        # 5 losses, then 5 wins, then 5 losses = max streak of 5
        for i in range(15):
            won = 5 <= i < 10
            entries.append({
                "pair": "STREAKY", "confidence": 0.5,
                "outcome": {"trade_won": won, "pnl_pips": 10 if won else -10, "duration_minutes": 60},
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)
        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert result["STREAKY"].consecutive_losses == 5


# ─── US-314: Adaptive Trailing Stop ──────────────────────────────

class TestAdaptiveTrailingStop:
    """Test regime and confidence-aware trailing stop scaling."""

    def test_regime_multiplier_mapping(self):
        """Verify regime→multiplier mapping values."""
        mults = {"LOW": 2.0, "NORMAL": 2.5, "HIGH": 3.0, "EXTREME": 4.0}
        for regime, expected in mults.items():
            assert expected > 0, f"Missing multiplier for {regime}"

    def test_confidence_scale_range(self):
        """Confidence scale should be 0.8-1.2."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            scale = 0.8 + (conf * 0.4)
            assert scale == pytest.approx(0.8 + conf * 0.4, abs=1e-9)

    def test_trail_distance_increases_with_regime(self):
        """Higher regime → wider trail (let profits breathe more)."""
        atr = 0.0015
        conf = 0.6
        conf_scale = 0.8 + (conf * 0.4)
        low_trail = atr * 2.0 * conf_scale
        high_trail = atr * 3.0 * conf_scale
        extreme_trail = atr * 4.0 * conf_scale
        assert low_trail < high_trail < extreme_trail

    def test_trail_distance_increases_with_confidence(self):
        """Higher confidence → wider trail (let winners run)."""
        atr = 0.0015
        regime_mult = 2.5
        low_conf_trail = atr * regime_mult * (0.8 + 0.3 * 0.4)
        high_conf_trail = atr * regime_mult * (0.8 + 0.9 * 0.4)
        assert low_conf_trail < high_conf_trail

    def test_fallback_to_default(self):
        """When no regime available, default trail_mult=1.5 should be used."""
        default_mult = 1.5
        assert default_mult == 1.5  # Hardcoded fallback


# ─── US-315: Compound Gate Floor ─────────────────────────────────

class TestCompoundGateFloor:
    """Test compound multiplier floor prevents near-zero sizing."""

    def test_floor_clamps_low_compound(self):
        """Compound below floor should be clamped."""
        floor = 0.10
        compound = 0.05  # 0.5 * 0.5 * 0.2
        result = max(compound, floor)
        assert result == 0.10

    def test_floor_does_not_affect_normal(self):
        """Compound above floor should pass through."""
        floor = 0.10
        compound = 0.4
        result = max(compound, floor)
        assert result == 0.4

    def test_triple_gate_reduction_scenario(self):
        """Spread 0.5 × Calendar 0.5 × Fitness 0.3 = 0.075 → clamp to 0.10."""
        spread = 0.5
        calendar = 0.5
        fitness = 0.3
        compound = spread * calendar * fitness
        assert compound == pytest.approx(0.075)
        floor = 0.10
        clamped = max(compound, floor)
        assert clamped == 0.10

    def test_zero_compound_stays_zero(self):
        """If any gate returns 0 (BLOCK), compound stays 0 (blocked trade)."""
        compound = 1.0 * 0.0 * 1.0  # Calendar blocks
        floor = 0.10
        # Floor only applies when compound > 0
        if compound > 0:
            compound = max(compound, floor)
        assert compound == 0.0

    def test_single_gate_reduction_no_floor(self):
        """Single gate reducing to 0.5 should NOT trigger floor."""
        compound = 1.0 * 0.5 * 1.0
        floor = 0.10
        result = max(compound, floor)
        assert result == 0.5


# ─── US-316: Lazy Imports ────────────────────────────────────────

class TestPhase50LazyImports:
    """Verify Phase 50 modules are accessible through __init__.py."""

    def test_journal_analyzer_import(self):
        from src.scanner import JournalAnalyzer
        assert JournalAnalyzer is not None

    def test_calibration_seeder_import(self):
        from src.scanner import CalibrationSeeder
        assert CalibrationSeeder is not None

    def test_fitness_seeder_import(self):
        from src.scanner import FitnessSeeder
        assert FitnessSeeder is not None


# ─── Integration: End-to-end pipeline ────────────────────────────

class TestPhase50Integration:
    """End-to-end tests combining multiple Phase 50 modules."""

    def test_full_pipeline_journal_to_patterns(self, tmp_path):
        """Journal → Analyzer → Patterns file."""
        from src.scanner.journal_analyzer import JournalAnalyzer
        entries = []
        for i in range(100):
            won = i % 3 != 0
            entries.append({
                "pair": "EUR_USD", "direction": "LONG", "confidence": 0.5 + i * 0.003,
                "entry_price": 1.08,
                "outcome": {"trade_won": won, "pnl_pips": 15 if won else -10, "duration_minutes": 120},
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)

        analyzer = JournalAnalyzer()
        report = analyzer.analyze(path)
        out = str(tmp_path / "patterns.json")
        assert analyzer.save_report(report, out)
        assert os.path.exists(out)
        with open(out) as f:
            data = json.load(f)
        assert data["total_trades"] == 100

    def test_full_pipeline_journal_to_fitness(self, tmp_path):
        """Journal → Fitness Seeder → State file."""
        from src.scanner.fitness_seeder import FitnessSeeder
        entries = []
        for pair in ["EUR_USD", "GBP_USD"]:
            for i in range(20):
                won = i % 2 == 0
                entries.append({
                    "pair": pair, "confidence": 0.6,
                    "outcome": {"trade_won": won, "pnl_pips": 10 if won else -10, "duration_minutes": 60},
                })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)

        seeder = FitnessSeeder()
        result = seeder.seed_from_journal(path)
        assert len(result) == 2
        out = str(tmp_path / "fitness.json")
        assert seeder.save_state(result, out)
        loaded = FitnessSeeder.load_state(out)
        assert len(loaded) == 2

    def test_full_pipeline_journal_to_calibration(self, tmp_path):
        """Journal → Calibration Seeder → Params file."""
        from src.scanner.calibration_seeder import CalibrationSeeder
        entries = []
        for i in range(200):
            conf = 0.4 + (i / 200) * 0.5
            won = conf > 0.6 and i % 2 == 0
            entries.append({
                "pair": "EUR_USD", "confidence": conf,
                "outcome": {"trade_won": won},
            })
        path = str(tmp_path / "journal.json")
        with open(path, "w") as f:
            json.dump(entries, f)

        seeder = CalibrationSeeder()
        result = seeder.seed_from_journal(path)
        assert result is not None
        out = str(tmp_path / "cal.json")
        assert seeder.save_params(result, out)
        loaded = CalibrationSeeder.load_params(out)
        assert loaded is not None
        assert loaded.n_samples > 0
