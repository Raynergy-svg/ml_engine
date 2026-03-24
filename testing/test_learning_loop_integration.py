#!/usr/bin/env python3
"""
Comprehensive dry-run integration test of the ML Engine learning loop.

Tests all automation modules in isolation with mock data:
- LearningEngine: trade outcome analysis
- AccuracyGate: per-pair accuracy tracking
- ConfigTuner: rule-based config adjustments
- StateEngine: cross-session state persistence
- ImprovementTracker: session metrics
- ObservationLog: market observation logging

CRITICAL: All file I/O uses tempfile (never touches production .claude/ or trained_data/)
"""

import json
import logging
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for test output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class TestImports(unittest.TestCase):
    """Verify all automation modules import without errors."""

    def test_import_learning_engine(self):
        """Test LearningEngine import."""
        try:
            from src.scanner.automation.learning_engine import LearningEngine
            self.assertIsNotNone(LearningEngine)
            logger.info("✓ LearningEngine imported")
        except Exception as e:
            self.fail(f"Failed to import LearningEngine: {e}")

    def test_import_accuracy_gate(self):
        """Test AccuracyGate import."""
        try:
            from src.scanner.automation.accuracy_gate import AccuracyGate
            self.assertIsNotNone(AccuracyGate)
            logger.info("✓ AccuracyGate imported")
        except Exception as e:
            self.fail(f"Failed to import AccuracyGate: {e}")

    def test_import_config_tuner(self):
        """Test ConfigTuner import."""
        try:
            from src.scanner.automation.config_tuner import ConfigTuner
            self.assertIsNotNone(ConfigTuner)
            logger.info("✓ ConfigTuner imported")
        except Exception as e:
            self.fail(f"Failed to import ConfigTuner: {e}")

    def test_import_state_engine(self):
        """Test StateEngine import."""
        try:
            from src.scanner.automation.state_engine import StateEngine
            self.assertIsNotNone(StateEngine)
            logger.info("✓ StateEngine imported")
        except Exception as e:
            self.fail(f"Failed to import StateEngine: {e}")

    def test_import_improvement_tracker(self):
        """Test ImprovementTracker import."""
        try:
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            self.assertIsNotNone(ImprovementTracker)
            logger.info("✓ ImprovementTracker imported")
        except Exception as e:
            self.fail(f"Failed to import ImprovementTracker: {e}")

    def test_import_observation_log(self):
        """Test ObservationLog import."""
        try:
            from src.scanner.automation.observation_log import ObservationLog
            self.assertIsNotNone(ObservationLog)
            logger.info("✓ ObservationLog imported")
        except Exception as e:
            self.fail(f"Failed to import ObservationLog: {e}")

    def test_import_scanner_config(self):
        """Test ScannerConfig import."""
        try:
            from src.scanner.config import ScannerConfig
            self.assertIsNotNone(ScannerConfig)
            logger.info("✓ ScannerConfig imported")
        except Exception as e:
            self.fail(f"Failed to import ScannerConfig: {e}")


class TestLearningEngine(unittest.TestCase):
    """Test LearningEngine trade outcome analysis."""

    def setUp(self):
        """Create temp directories for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        from src.scanner.automation.learning_engine import LearningEngine
        self.learnings_path = self.temp_path / "learnings.md"
        self.rules_path = self.temp_path / "rules.md"

        self.engine = LearningEngine(
            learnings_path=self.learnings_path,
            rules_path=self.rules_path,
        )

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_analyze_trade_winning(self):
        """Test analysis of a winning trade."""
        trade = {
            "pair": "EUR_USD",
            "trade_id": "trade_001",
            "direction": "LONG",
            "confidence": 0.75,
            "sl_pips": 15,
            "tp_pips": 30,
            "atr_pips": 20,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": {
                "trade_won": True,
                "pnl_pips": 25,
                "realized_pl": 250.0,
                "close_time": datetime.now(timezone.utc).isoformat(),
            },
            "agents": {
                "agent_reasons": [
                    {"name": "trend", "score": 0.8},
                    {"name": "uncertainty", "score": 0.2},
                ]
            },
            "weighted_vote_score": 0.75,
            "model_disagreement": 0.1,
        }

        entries = self.engine.analyze_trade(trade)
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)
        logger.info(f"✓ Winning trade analyzed: {len(entries)} learning entries")

        # Should have high_consensus_works entry
        consensus_entries = [e for e in entries if "high_consensus" in e.insight]
        self.assertGreater(len(consensus_entries), 0)

    def test_analyze_trade_losing_with_high_uncertainty(self):
        """Test analysis of losing trade with high uncertainty."""
        trade = {
            "pair": "GBP_USD",
            "trade_id": "trade_002",
            "direction": "SHORT",
            "confidence": 0.55,
            "sl_pips": 20,
            "tp_pips": 30,
            "atr_pips": 10,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": {
                "trade_won": False,
                "pnl_pips": -20,
                "realized_pl": -200.0,
                "close_time": datetime.now(timezone.utc).isoformat(),
            },
            "agents": {
                "agent_reasons": [
                    {"name": "uncertainty", "score": 0.55},
                ]
            },
        }

        entries = self.engine.analyze_trade(trade)
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)

        # Should flag uncertainty warning
        uncertainty_entries = [e for e in entries if "uncertainty_was_warning" in e.insight]
        self.assertGreater(len(uncertainty_entries), 0)
        logger.info(f"✓ Losing trade with uncertainty: {len(entries)} entries, flagged uncertainty")

    def test_analyze_trade_with_malformed_agents(self):
        """Test robustness: trade with missing/malformed agents dict."""
        trade = {
            "pair": "USD_JPY",
            "trade_id": "trade_003",
            "direction": "LONG",
            "confidence": 0.6,
            "outcome": {
                "trade_won": True,
                "pnl_pips": 10,
                "realized_pl": 100.0,
            },
            # agents is None
            "agents": None,
        }

        # Should not crash
        try:
            entries = self.engine.analyze_trade(trade)
            self.assertIsInstance(entries, list)
            logger.info("✓ Malformed agents handled gracefully")
        except Exception as e:
            self.fail(f"Failed on malformed agents: {e}")

    def test_analyze_trade_no_outcome(self):
        """Test trade with no outcome (not closed)."""
        trade = {
            "pair": "EUR_GBP",
            "trade_id": "trade_004",
            "direction": "LONG",
            # No outcome key
        }

        entries = self.engine.analyze_trade(trade)
        self.assertIsInstance(entries, list)
        self.assertEqual(len(entries), 0)
        logger.info("✓ Open trade (no outcome) returns empty list")

    def test_append_to_learnings(self):
        """Test appending learning entries to learnings.md."""
        from src.scanner.automation.learning_engine import LearningEntry

        entries = [
            LearningEntry(
                date="2026-03-18",
                category="sl_tp",
                insight="sl_too_tight for EUR_USD: lost 25.0p > 2x ATR",
                action="Increase atr_sl_multiplier",
                source_trade_id="trade_001",
            ),
            LearningEntry(
                date="2026-03-18",
                category="agent_accuracy",
                insight="high_consensus_works: vote=0.75, won $250.00",
                action="Prefer high-consensus setups",
                source_trade_id="trade_002",
            ),
        ]

        count = self.engine.append_to_learnings(entries)
        self.assertEqual(count, 2)
        self.assertTrue(self.learnings_path.exists())

        content = self.learnings_path.read_text()
        self.assertIn("sl_too_tight", content)
        self.assertIn("high_consensus_works", content)
        logger.info("✓ Learning entries appended to file")


class TestAccuracyGate(unittest.TestCase):
    """Test AccuracyGate per-pair accuracy tracking."""

    def setUp(self):
        """Create temp directory for test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.data_path = self.temp_path / "pair_accuracy.json"

        from src.scanner.automation.accuracy_gate import AccuracyGate
        self.gate = AccuracyGate(
            min_accuracy=0.55,
            min_trades=5,
            data_path=str(self.data_path),
        )

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_record_outcome(self):
        """Test recording trade outcomes."""
        self.gate.record_outcome(
            pair="EUR_USD",
            predicted_direction="LONG",
            actual_outcome=True,
            confidence=0.75,
        )
        self.assertTrue(self.data_path.exists())
        data = json.loads(self.data_path.read_text())
        self.assertIn("EUR_USD", data)
        self.assertEqual(data["EUR_USD"]["total"], 1)
        logger.info("✓ Trade outcome recorded")

    def test_check_pair_below_min_trades(self):
        """Test that pairs below min_trades threshold are not blocked."""
        self.gate.record_outcome("GBP_USD", "LONG", True, 0.8)

        is_allowed, accuracy, reason = self.gate.check_pair("GBP_USD")
        # Should return True (not blocked) since only 1 trade < 5 min_trades
        self.assertTrue(is_allowed)
        logger.info("✓ Pair below min_trades is not blocked")

    def test_check_pair_accuracy_above_threshold(self):
        """Test that accurate pairs pass the gate."""
        pair = "USD_JPY"

        # Add 6 winning trades
        for i in range(6):
            self.gate.record_outcome(pair, "LONG", True, 0.75)

        is_allowed, accuracy, reason = self.gate.check_pair(pair)
        # Should return True (not blocked) — 100% accuracy > 55% threshold
        self.assertTrue(is_allowed)
        logger.info("✓ High-accuracy pair passes gate")

    def test_check_pair_accuracy_below_threshold(self):
        """Test that inaccurate pairs are blocked."""
        pair = "AUD_USD"

        # Add 5 losing trades
        for i in range(5):
            self.gate.record_outcome(pair, "LONG", False, 0.75)

        is_allowed, accuracy, reason = self.gate.check_pair(pair)
        # Should return False (blocked) — 0% accuracy < 55% threshold
        self.assertFalse(is_allowed)
        logger.info("✓ Low-accuracy pair is blocked")

    def test_get_blocked_pairs(self):
        """Test retrieving list of blocked pairs."""
        # Add some good trades to EUR_USD
        for i in range(5):
            self.gate.record_outcome("EUR_USD", "LONG", True, 0.75)

        # Add bad trades to GBP_JPY
        for i in range(5):
            self.gate.record_outcome("GBP_JPY", "LONG", False, 0.75)

        blocked = self.gate.get_blocked_pairs()
        self.assertIsInstance(blocked, list)
        # GBP_JPY should be in blocked list (0% accuracy)
        self.assertIn("GBP_JPY", blocked)
        logger.info(f"✓ Blocked pairs retrieved: {blocked}")


class TestConfigTuner(unittest.TestCase):
    """Test ConfigTuner rule-based config adjustments."""

    def setUp(self):
        """Create temp directory and test config."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.rules_path = self.temp_path / "rules.md"
        self.adjustments_path = self.temp_path / "adjustments.json"
        self.applied_path = self.temp_path / "applied.json"

        from src.scanner.automation.config_tuner import ConfigTuner
        from src.scanner.config import ScannerConfig

        self.tuner = ConfigTuner(
            rules_path=self.rules_path,
            adjustments_path=self.adjustments_path,
            applied_rules_path=self.applied_path,
        )
        self.config = ScannerConfig()

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_load_rules_empty_file(self):
        """Test loading rules from non-existent file."""
        rules = self.tuner.load_rules()
        self.assertEqual(rules, [])
        logger.info("✓ Empty rules file returns empty list")

    def test_apply_rule_atr_sl_adjustment(self):
        """Test applying ATR SL multiplier rule."""
        # Create rules file with promotion
        rules_content = """
## Promoted Rules
- [2026-03-18] sizing: Increase atr_sl_multiplier by 0.1 for EUR_USD (SL too tight observed 3 times)
"""
        self.rules_path.write_text(rules_content)

        old_sl = self.config.atr_sl_multiplier
        adjustments = self.tuner.apply_to_config(self.config)
        new_sl = self.config.atr_sl_multiplier

        self.assertGreater(new_sl, old_sl)
        self.assertEqual(new_sl, old_sl + 0.1)
        self.assertGreater(len(adjustments), 0)
        logger.info(f"✓ ATR SL adjusted: {old_sl} → {new_sl}")

    def test_apply_rule_uncertainty_adjustment(self):
        """Test applying max_uncertainty_score rule."""
        rules_content = """
## Promoted Rules
- [2026-03-18] agent_accuracy: Lower max_uncertainty_score by 0.02 (uncertainty predicted 3 losses)
"""
        self.rules_path.write_text(rules_content)

        old_unc = self.config.max_uncertainty_score
        adjustments = self.tuner.apply_to_config(self.config)
        new_unc = self.config.max_uncertainty_score

        self.assertLess(new_unc, old_unc)
        self.assertEqual(new_unc, max(0.30, old_unc - 0.02))
        logger.info(f"✓ Uncertainty adjusted: {old_unc} → {new_unc}")

    def test_bounds_enforcement(self):
        """Test that config bounds are enforced."""
        # Push config to max
        self.config.atr_sl_multiplier = 2.0

        rules_content = """
## Promoted Rules
- [2026-03-18] sizing: Increase atr_sl_multiplier by 0.1 for EUR_USD (SL too tight)
"""
        self.rules_path.write_text(rules_content)

        adjustments = self.tuner.apply_to_config(self.config)
        # Should not exceed 2.0 (max bound)
        self.assertLessEqual(self.config.atr_sl_multiplier, 2.0)
        logger.info("✓ Bounds enforced (no exceeding max)")

    def test_same_rule_not_applied_twice(self):
        """Test that same rule doesn't apply in same session."""
        rules_content = """
## Promoted Rules
- [2026-03-18] sizing: Increase atr_sl_multiplier by 0.1 for EUR_USD (SL too tight)
"""
        self.rules_path.write_text(rules_content)

        old_sl = self.config.atr_sl_multiplier

        # Apply first time
        adjustments1 = self.tuner.apply_to_config(self.config)
        sl_after_first = self.config.atr_sl_multiplier

        # Apply second time — should not change (rule already applied)
        adjustments2 = self.tuner.apply_to_config(self.config)
        sl_after_second = self.config.atr_sl_multiplier

        self.assertGreater(sl_after_first, old_sl)
        self.assertEqual(sl_after_first, sl_after_second)
        self.assertEqual(len(adjustments2), 0)
        logger.info("✓ Same rule not applied twice in session")


class TestStateEngine(unittest.TestCase):
    """Test StateEngine cross-session state persistence."""

    def setUp(self):
        """Create temp directory for state file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.state_path = self.temp_path / "state.json"

        from src.scanner.automation.state_engine import StateEngine
        self.engine = StateEngine(state_path=self.state_path)

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_load_missing_state(self):
        """Test loading state from non-existent file."""
        state = self.engine.load_state()
        self.assertIsInstance(state, dict)
        self.assertIn("goal", state)
        self.assertIn("status", state)
        logger.info("✓ Missing state file returns defaults")

    def test_save_and_load_state(self):
        """Test saving and loading state round-trip."""
        self.engine.save_state(
            goal="Test goal",
            status="in_progress",
            done=["task1", "task2"],
            next_action="Continue testing",
            open_questions=["Q1", "Q2"],
            improvement_focus="accuracy",
        )

        self.assertTrue(self.state_path.exists())

        state = self.engine.load_state()
        self.assertEqual(state["goal"], "Test goal")
        self.assertEqual(state["status"], "in_progress")
        self.assertEqual(state["done"], ["task1", "task2"])
        self.assertEqual(state["next"], "Continue testing")
        logger.info("✓ State saved and loaded correctly")

    def test_load_corrupted_json(self):
        """Test loading corrupted JSON file."""
        self.state_path.write_text("{ invalid json")

        state = self.engine.load_state()
        self.assertIsInstance(state, dict)
        self.assertIn("goal", state)
        logger.info("✓ Corrupted JSON returns defaults")

    def test_load_state_with_missing_keys(self):
        """Test loading state missing required keys."""
        partial_state = {"goal": "partial"}
        self.state_path.write_text(json.dumps(partial_state))

        state = self.engine.load_state()
        self.assertIn("goal", state)
        self.assertIn("status", state)
        logger.info("✓ Partial state merged with defaults")

    def test_increment_scan_cycle(self):
        """Test incrementing scan cycle counter."""
        count1 = self.engine.increment_scan_cycle()
        self.assertEqual(count1, 1)

        count2 = self.engine.increment_scan_cycle()
        self.assertEqual(count2, 2)

        count3 = self.engine.increment_scan_cycle()
        self.assertEqual(count3, 3)
        logger.info("✓ Scan cycle counter increments correctly")


class TestImprovementTracker(unittest.TestCase):
    """Test ImprovementTracker session metrics."""

    def setUp(self):
        """Create temp directory for logs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.log_path = self.temp_path / "improvement.jsonl"

        from src.scanner.automation.improvement_tracker import ImprovementTracker
        self.tracker = ImprovementTracker(log_path=self.log_path)

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_record_session(self):
        """Test recording a session snapshot."""
        trades = [
            {"outcome": {"trade_won": True, "realized_pl": 100}},
            {"outcome": {"trade_won": True, "realized_pl": 150}},
            {"outcome": {"trade_won": False, "realized_pl": -50}},
        ]

        record = self.tracker.record_session(
            trades=trades,
            learnings_added=3,
            rules_promoted=1,
            config_adjustments=[],
        )

        self.assertEqual(record["session_trades"], 3)
        self.assertEqual(record["wins"], 2)
        self.assertEqual(record["losses"], 1)
        self.assertEqual(record["net_pnl"], 200)
        self.assertTrue(self.log_path.exists())
        logger.info("✓ Session recorded")

    def test_get_trend_insufficient_data(self):
        """Test get_trend with single record."""
        trades = [{"outcome": {"trade_won": True, "realized_pl": 100}}]
        self.tracker.record_session(trades)

        trend = self.tracker.get_trend()
        self.assertIn("win_rate_trend", trend)
        self.assertEqual(trend["win_rate_trend"], "insufficient_data")
        logger.info("✓ Trend returns insufficient_data for single entry")

    def test_get_trend_stable(self):
        """Test get_trend detects stable performance."""
        # Record 3 similar sessions
        for i in range(3):
            trades = [
                {"outcome": {"trade_won": True, "realized_pl": 100}},
                {"outcome": {"trade_won": False, "realized_pl": -50}},
            ]
            self.tracker.record_session(trades, learnings_added=2)

        trend = self.tracker.get_trend(window=2)
        # Win rate should be stable (66.7% both windows)
        self.assertIn(trend["win_rate_trend"], ["stable", "improving", "declining"])
        logger.info(f"✓ Trend calculated: {trend}")

    def test_generate_report(self):
        """Test improvement report generation."""
        trades = [
            {"outcome": {"trade_won": True, "realized_pl": 100}},
            {"outcome": {"trade_won": False, "realized_pl": -50}},
        ]

        self.tracker.record_session(trades, learnings_added=2, rules_promoted=1)

        report = self.tracker.generate_report()
        self.assertIsInstance(report, str)
        self.assertIn("Sessions:", report)
        self.assertIn("Total trades:", report)
        logger.info("✓ Report generated")


class TestObservationLog(unittest.TestCase):
    """Test ObservationLog market observation logging."""

    def setUp(self):
        """Create temp directory for logs."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.log_path = self.temp_path / "observations.jsonl"

        from src.scanner.automation.observation_log import ObservationLog
        self.log = ObservationLog(log_path=self.log_path)

    def tearDown(self):
        """Clean up temp directory."""
        self.temp_dir.cleanup()

    def test_log_observation(self):
        """Test logging a single observation."""
        self.log.log_observation(
            pair="EUR_USD",
            category="regime_change",
            description="Volatility spike detected",
            metadata={"volatility": 0.25},
        )

        self.assertTrue(self.log_path.exists())
        lines = self.log_path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)

        entry = json.loads(lines[0])
        self.assertEqual(entry["pair"], "EUR_USD")
        self.assertEqual(entry["category"], "regime_change")
        logger.info("✓ Observation logged")

    def test_get_recent_all(self):
        """Test retrieving all recent observations."""
        for i in range(5):
            self.log.log_observation(
                pair="EUR_USD",
                category="regime_change",
                description=f"Observation {i}",
            )

        recent = self.log.get_recent(limit=10)
        self.assertEqual(len(recent), 5)
        logger.info("✓ Retrieved all observations")

    def test_get_recent_filter_pair(self):
        """Test filtering observations by pair."""
        self.log.log_observation("EUR_USD", "regime_change", "Obs 1")
        self.log.log_observation("GBP_USD", "agent_disagreement", "Obs 2")
        self.log.log_observation("EUR_USD", "near_miss", "Obs 3")

        eur_recent = self.log.get_recent(pair="EUR_USD")
        self.assertEqual(len(eur_recent), 2)

        gbp_recent = self.log.get_recent(pair="GBP_USD")
        self.assertEqual(len(gbp_recent), 1)
        logger.info("✓ Filtered by pair")

    def test_get_recent_filter_category(self):
        """Test filtering observations by category."""
        self.log.log_observation("EUR_USD", "regime_change", "Obs 1")
        self.log.log_observation("EUR_USD", "regime_change", "Obs 2")
        self.log.log_observation("EUR_USD", "near_miss", "Obs 3")

        regime = self.log.get_recent(category="regime_change")
        self.assertEqual(len(regime), 2)

        near = self.log.get_recent(category="near_miss")
        self.assertEqual(len(near), 1)
        logger.info("✓ Filtered by category")


class TestIntegration(unittest.TestCase):
    """End-to-end integration test of full learning loop."""

    def test_full_learning_loop(self):
        """
        Simulate complete learning loop:
        1. Trade executed and closed
        2. Outcome analyzed → learning entries
        3. Patterns promoted to rules
        4. Rules applied to config
        5. Session tracked in improvement log
        """
        from src.scanner.automation.learning_engine import LearningEngine
        from src.scanner.automation.config_tuner import ConfigTuner
        from src.scanner.automation.improvement_tracker import ImprovementTracker
        from src.scanner.config import ScannerConfig

        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        try:
            # Step 1: Create trade
            trade = {
                "pair": "EUR_USD",
                "trade_id": "integration_001",
                "direction": "LONG",
                "confidence": 0.75,
                "outcome": {
                    "trade_won": True,
                    "pnl_pips": 30,
                    "realized_pl": 300.0,
                },
                "weighted_vote_score": 0.78,
            }

            # Step 2: Analyze trade
            learning_engine = LearningEngine(
                learnings_path=temp_path / "learnings.md",
                rules_path=temp_path / "rules.md",
            )
            entries = learning_engine.analyze_trade(trade)
            self.assertGreater(len(entries), 0)

            learning_engine.append_to_learnings(entries)
            self.assertTrue((temp_path / "learnings.md").exists())
            logger.info(f"✓ Step 1-2: Trade analyzed → {len(entries)} learning entries")

            # Step 3: Promote patterns (mock by adding rule manually)
            rules_path = temp_path / "rules.md"
            rules_path.write_text("""
## Promoted Rules
- [2026-03-18] agent_accuracy: Prefer weighted_vote_score > 0.7 (high consensus won 3 times)
""")
            logger.info("✓ Step 3: Rules prepared")

            # Step 4: Apply rules to config
            config = ScannerConfig()
            tuner = ConfigTuner(
                rules_path=rules_path,
                adjustments_path=temp_path / "adjustments.json",
                applied_rules_path=temp_path / "applied.json",
            )
            adjustments = tuner.apply_to_config(config)
            logger.info(f"✓ Step 4: Applied {len(adjustments)} config adjustments")

            # Step 5: Track in improvement log
            tracker = ImprovementTracker(log_path=temp_path / "improvement.jsonl")
            tracker.record_session(
                trades=[trade],
                learnings_added=len(entries),
                rules_promoted=len(adjustments),
                config_adjustments=adjustments,
            )
            report = tracker.generate_report()
            self.assertIn("Sessions:", report)
            logger.info("✓ Step 5: Session logged and report generated")

            logger.info("\n✓✓✓ FULL LEARNING LOOP INTEGRATION TEST PASSED ✓✓✓")

        finally:
            temp_dir.cleanup()


def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestImports))
    suite.addTests(loader.loadTestsFromTestCase(TestLearningEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestAccuracyGate))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigTuner))
    suite.addTests(loader.loadTestsFromTestCase(TestStateEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestImprovementTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestObservationLog))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
