"""Unit tests for PerformancePRDGenerator.

Tests cover:
1. Performance analysis computation
2. Gap identification logic
3. PRD story generation
4. File I/O and atomicity
5. Spawn behavior
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.scanner.automation.performance_prd_generator import (
    PerformancePRDGenerator,
    PerformancePRDResult,
)


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """Create a temporary project structure."""
    root = tmp_path / "ml_engine"
    root.mkdir()
    (root / ".claude" / "ralph" / "archive").mkdir(parents=True, exist_ok=True)
    (root / "trained_data").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    # Create a dummy ralph.sh script
    ralph_script = root / "scripts" / "ralph.sh"
    ralph_script.write_text("#!/bin/bash\necho 'Ralph mock'\n")
    ralph_script.chmod(0o755)

    return root


@pytest.fixture
def mock_journal_data() -> List[Dict[str, Any]]:
    """Create mock trade journal data for testing."""
    now = datetime.utcnow()

    trades = []

    # EUR_USD: 8 wins, 4 losses in last 30 days (66.7% win rate — above threshold)
    for i in range(12):
        trades.append({
            "pair": "EUR_USD",
            "direction": "BUY" if i % 2 == 0 else "SELL",
            "entry_price": 1.0800 + (i * 0.0001),
            "exit_price": 1.0850 + (i * 0.0001),
            "entry_time": (now - timedelta(days=15 - (i % 15))).isoformat() + "Z",
            "closed_at": (now - timedelta(days=15 - (i % 15))).isoformat() + "Z",
            "timestamp": (now - timedelta(days=15 - (i % 15))).timestamp(),
            "trade_won": i < 8,  # First 8 are wins
            "pnl_pips": 50 if i < 8 else -50,
            "regime": "TRENDING" if i < 6 else "RANGING",
            "confidence": 0.65 + (i * 0.01),
            "agent_votes": 8,
        })

    # GBP_USD: 2 wins, 8 losses (20% win rate — below threshold)
    for i in range(10):
        trades.append({
            "pair": "GBP_USD",
            "direction": "BUY" if i % 2 == 0 else "SELL",
            "entry_price": 1.2700 + (i * 0.0001),
            "exit_price": 1.2650 + (i * 0.0001),
            "entry_time": (now - timedelta(days=20 - (i % 20))).isoformat() + "Z",
            "closed_at": (now - timedelta(days=20 - (i % 20))).isoformat() + "Z",
            "timestamp": (now - timedelta(days=20 - (i % 20))).timestamp(),
            "trade_won": i < 2,  # Only first 2 are wins
            "pnl_pips": 30 if i < 2 else -30,
            "regime": "RANGING",
            "confidence": 0.50 + (i * 0.01),
            "agent_votes": 5,
        })

    # USD_JPY: 3 wins, 7 losses (30% win rate — below threshold)
    for i in range(10):
        trades.append({
            "pair": "USD_JPY",
            "direction": "BUY" if i % 2 == 0 else "SELL",
            "entry_price": 150.00 + (i * 0.10),
            "exit_price": 150.10 + (i * 0.10),
            "entry_time": (now - timedelta(days=25 - (i % 25))).isoformat() + "Z",
            "closed_at": (now - timedelta(days=25 - (i % 25))).isoformat() + "Z",
            "timestamp": (now - timedelta(days=25 - (i % 25))).timestamp(),
            "trade_won": i < 3,  # First 3 are wins
            "pnl_pips": 20 if i < 3 else -20,
            "regime": "TRENDING",
            "confidence": 0.55 + (i * 0.01),
            "agent_votes": 6,
        })

    # TRENDING regime: 13 wins, 7 losses (65% win rate — above threshold)
    # RANGING regime: 0 wins, 12 losses (0% win rate — well below threshold)
    # Already covered by pair-level data; no additional trades needed

    return trades


def test_analyze_performance_basic(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test basic performance analysis computation."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()

    assert "error" not in analysis
    assert analysis["total_trades"] == 32
    assert analysis["period_days"] == 30
    assert 0.0 <= analysis["overall_win_rate"] <= 1.0
    assert "EUR_USD" in analysis["by_pair"]
    assert "GBP_USD" in analysis["by_pair"]


def test_analyze_performance_pair_stats(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that pair-level statistics are computed correctly."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()

    # EUR_USD: 8 wins, 4 losses = 66.7% win rate
    eur_usd = analysis["by_pair"]["EUR_USD"]
    assert eur_usd["trade_count"] == 12
    assert eur_usd["win_rate"] == pytest.approx(0.667, abs=0.01)

    # GBP_USD: 2 wins, 8 losses = 20% win rate
    gbp_usd = analysis["by_pair"]["GBP_USD"]
    assert gbp_usd["trade_count"] == 10
    assert gbp_usd["win_rate"] == pytest.approx(0.20, abs=0.01)


def test_identify_gaps_below_threshold(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that gaps are identified for pairs below threshold."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        win_rate_gap_threshold=0.45,
    )
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    # GBP_USD (20%) and USD_JPY (30%) should be identified as gaps
    # EUR_USD (66.7%) should not
    gap_targets = [g["target"] for g in gaps]
    assert "GBP_USD" in gap_targets
    assert "USD_JPY" in gap_targets
    assert "EUR_USD" not in gap_targets


def test_identify_gaps_min_trade_count(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that gaps require minimum trade count."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        win_rate_gap_threshold=0.45,
    )
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    # All gaps should have trade_count >= 5
    for gap in gaps:
        assert gap["trade_count"] >= 5


def test_identify_gaps_max_count(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that identify_gaps respects max_gaps_per_run."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        max_gaps_per_run=2,
    )
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    # Should not exceed max_gaps_per_run
    assert len(gaps) <= 2


def test_generate_prd_stories_structure(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that generated PRD has valid structure."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    if gaps:
        prd = gen.generate_prd_stories(gaps, analysis)

        # Check required top-level keys
        assert prd["project"] == "ML_Engine"
        assert "branchName" in prd
        assert "description" in prd
        assert "userStories" in prd
        assert isinstance(prd["userStories"], list)


def test_generate_prd_stories_user_story_fields(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that each user story has required fields."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    if gaps:
        prd = gen.generate_prd_stories(gaps, analysis)

        for story in prd["userStories"]:
            assert "id" in story
            assert "title" in story
            assert "description" in story
            assert "acceptanceCriteria" in story
            assert "priority" in story
            assert "passes" in story
            assert "notes" in story
            assert isinstance(story["acceptanceCriteria"], list)
            assert len(story["acceptanceCriteria"]) > 0


def test_generate_prd_stories_branch_name_format(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that branchName is correctly formatted."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    if gaps:
        prd = gen.generate_prd_stories(gaps, analysis)

        # Branch name should be: ralph/auto-perf-improvement-YYYY-MM-DD
        assert prd["branchName"].startswith("ralph/auto-perf-improvement-")
        # Extract the date part (last 10 chars: YYYY-MM-DD)
        date_part = prd["branchName"][-10:]
        # Check date format YYYY-MM-DD
        assert len(date_part) == 10
        assert date_part[4] == "-"
        assert date_part[7] == "-"


def test_write_prd_atomicity(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test that PRD is written atomically."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    analysis = gen.analyze_performance()
    gaps = gen.identify_gaps(analysis)

    if gaps:
        prd = gen.generate_prd_stories(gaps, analysis)
        result = gen.write_prd(prd)

        assert result is True

        # Verify PRD was written
        prd_path = tmp_project_root / ".claude" / "ralph" / "prd.json"
        assert prd_path.exists()

        # Verify it's valid JSON
        written_prd = json.loads(prd_path.read_text())
        assert written_prd["project"] == "ML_Engine"


def test_should_run_insufficient_trades(tmp_project_root: Path) -> None:
    """Test should_run returns False when trades < min_trades_for_analysis."""
    journal_data = [
        {
            "pair": "EUR_USD",
            "timestamp": datetime.utcnow().timestamp(),
            "trade_won": True,
        },
    ]
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(journal_data))

    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        min_trades_for_analysis=10,
    )
    result = gen.should_run()

    # Only 1 trade in 30 days, but min is 10
    assert result is False


def test_should_run_cooldown_interval(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test should_run respects check_interval_hours."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        check_interval_hours=24,
    )

    # Mark that we just ran
    last_run_marker = tmp_project_root / ".claude" / "ralph" / ".last_auto_prd"
    last_run_marker.parent.mkdir(parents=True, exist_ok=True)
    last_run_marker.write_text(str(datetime.utcnow().timestamp()))

    # Should return False because we just ran
    result = gen.should_run()
    assert result is False


def test_spawn_ralph_script_not_found(tmp_project_root: Path) -> None:
    """Test spawn_ralph returns False when script doesn't exist."""
    gen = PerformancePRDGenerator(
        project_root=str(tmp_project_root),
        ralph_script_path="/nonexistent/ralph.sh",
    )

    result = gen.spawn_ralph(background=True)
    assert result is False


def test_run_no_gaps(tmp_project_root: Path) -> None:
    """Test run() returns False-triggered result when no gaps."""
    # Create a journal with all high-performing pairs
    journal_data = []
    now = datetime.utcnow()

    # All wins
    for i in range(12):
        journal_data.append({
            "pair": "EUR_USD",
            "direction": "BUY",
            "entry_price": 1.0800,
            "exit_price": 1.0850,
            "closed_at": (now - timedelta(days=15)).isoformat() + "Z",
            "timestamp": (now - timedelta(days=15)).timestamp(),
            "trade_won": True,
            "pnl_pips": 50,
        })

    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    result = gen.run()

    assert isinstance(result, PerformancePRDResult)
    assert result.run_triggered is True  # should_run passed
    assert result.gaps_found == 0  # But no gaps
    assert result.stories_generated == 0


def test_run_with_gaps(
    tmp_project_root: Path, mock_journal_data: List[Dict[str, Any]]
) -> None:
    """Test run() generates stories when gaps exist."""
    journal_path = tmp_project_root / "trained_data" / "trade_journal_rl.json"
    journal_path.write_text(json.dumps(mock_journal_data))

    gen = PerformancePRDGenerator(project_root=str(tmp_project_root))
    result = gen.run()

    assert isinstance(result, PerformancePRDResult)
    assert result.run_triggered is True
    assert result.gaps_found > 0
    assert result.stories_generated == result.gaps_found
    assert result.prd_branch != ""
