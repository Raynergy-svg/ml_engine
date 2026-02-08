"""Weekly Action Plan Tracker for $100K → $1M Strategy.

Tracks progress against the aggressive scaling action plan:
- Week 1: Validation (5 lots, measure slippage, 20 trades)
- Week 2: Optimization (limit orders, circuit breakers, scale to 8 lots)
- Week 3-4: Scaling (intraday compounding, second broker)
- Month 2-3: Acceleration (full Kelly, multiple pairs)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class ActionItem:
    """Single action item to track."""
    id: str
    description: str
    week: int
    category: str
    status: ActionStatus = ActionStatus.NOT_STARTED
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    notes: str = ""
    completed_at: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        if self.target_value and self.current_value:
            return min(100, (self.current_value / self.target_value) * 100)
        if self.status == ActionStatus.COMPLETED:
            return 100
        return 0


@dataclass
class WeeklyStats:
    """Weekly performance statistics."""
    week_number: int
    start_date: str
    end_date: str
    starting_equity: float
    ending_equity: float
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_slippage_pips: float = 0.0
    slippage_measurements: List[float] = field(default_factory=list)
    max_daily_drawdown: float = 0.0
    best_daily_return: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def weekly_return(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity


# Default action plan
DEFAULT_ACTION_PLAN = [
    # Week 1: Validation
    ActionItem("w1_reduce_size", "Reduce size to 5 lots", 1, "position_sizing", target_value=5),
    ActionItem("w1_measure_slippage", "Measure slippage on each trade", 1, "execution", target_value=20),
    ActionItem("w1_track_execution", "Track expected vs actual price", 1, "execution"),
    ActionItem("w1_20_trades", "Complete 20 trades to confirm win rate", 1, "validation", target_value=20),
    ActionItem("w1_verify_78pct", "Verify 78%+ win rate holds", 1, "validation", target_value=78),

    # Week 2: Optimization
    ActionItem("w2_limit_orders", "Implement limit orders", 2, "execution"),
    ActionItem("w2_slippage_target", "Reduce slippage to <2 pips", 2, "execution", target_value=2),
    ActionItem("w2_circuit_breakers", "Add drawdown circuit breakers", 2, "risk"),
    ActionItem("w2_scale_8lots", "Scale to 8 lots if win rate >75%", 2, "position_sizing", target_value=8),

    # Week 3-4: Scaling
    ActionItem("w3_compounding", "Implement intraday compounding", 3, "position_sizing"),
    ActionItem("w3_recalc_size", "Recalculate size per trade", 3, "position_sizing"),
    ActionItem("w3_second_broker", "Open second broker account", 3, "execution"),
    ActionItem("w3_split_orders", "Implement split orders for 10+ lots", 3, "execution"),
    ActionItem("w4_target_200k", "Target $200K+ by end of month 1", 4, "growth", target_value=200000),

    # Month 2-3: Acceleration
    ActionItem("m2_full_kelly", "Full Kelly sizing (10-15 lots)", 5, "position_sizing", target_value=15),
    ActionItem("m2_multiple_pairs", "Trade multiple pairs (5 trained)", 5, "diversification", target_value=5),
    ActionItem("m2_automation", "Full automation of execution", 5, "execution"),
    ActionItem("m3_target_1m", "Target $1M by month 3", 9, "growth", target_value=1000000),
]


class WeeklyActionTracker:
    """
    Track weekly progress against the $100K → $1M action plan.

    Usage:
        tracker = WeeklyActionTracker()

        # Check current week's actions
        actions = tracker.get_current_week_actions()

        # Update an action
        tracker.update_action("w1_reduce_size", current_value=5)

        # Complete an action
        tracker.complete_action("w1_limit_orders", notes="Implemented in aggressive_scaling.py")

        # Log weekly stats
        tracker.log_weekly_stats(week=1, stats={...})
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("trained_data") / "scaling_action_plan.json"
        self.actions: List[ActionItem] = []
        self.weekly_stats: List[WeeklyStats] = []
        self.start_date: Optional[str] = None
        self._load()

    def _load(self) -> None:
        """Load action plan from disk."""
        if self.data_path.exists():
            try:
                data = json.loads(self.data_path.read_text())
                self.start_date = data.get("start_date")

                self.actions = []
                for a in data.get("actions", []):
                    item = ActionItem(
                        id=a["id"],
                        description=a["description"],
                        week=a["week"],
                        category=a["category"],
                        status=ActionStatus(a.get("status", "not_started")),
                        target_value=a.get("target_value"),
                        current_value=a.get("current_value"),
                        notes=a.get("notes", ""),
                        completed_at=a.get("completed_at"),
                    )
                    self.actions.append(item)

                self.weekly_stats = []
                for s in data.get("weekly_stats", []):
                    stat = WeeklyStats(
                        week_number=s["week_number"],
                        start_date=s["start_date"],
                        end_date=s["end_date"],
                        starting_equity=s["starting_equity"],
                        ending_equity=s["ending_equity"],
                        total_trades=s.get("total_trades", 0),
                        wins=s.get("wins", 0),
                        losses=s.get("losses", 0),
                        total_pnl=s.get("total_pnl", 0),
                        avg_slippage_pips=s.get("avg_slippage_pips", 0),
                        slippage_measurements=s.get("slippage_measurements", []),
                        max_daily_drawdown=s.get("max_daily_drawdown", 0),
                        best_daily_return=s.get("best_daily_return", 0),
                    )
                    self.weekly_stats.append(stat)

            except Exception as e:
                logger.warning(f"Failed to load action plan: {e}")
                self._init_default()
        else:
            self._init_default()

    def _init_default(self) -> None:
        """Initialize with default action plan."""
        self.start_date = datetime.now(timezone.utc).date().isoformat()
        self.actions = [ActionItem(**asdict(a)) for a in DEFAULT_ACTION_PLAN]
        self.weekly_stats = []
        self._save()

    def _save(self) -> None:
        """Save action plan to disk."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "start_date": self.start_date,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "actions": [
                {
                    "id": a.id,
                    "description": a.description,
                    "week": a.week,
                    "category": a.category,
                    "status": a.status.value,
                    "target_value": a.target_value,
                    "current_value": a.current_value,
                    "notes": a.notes,
                    "completed_at": a.completed_at,
                }
                for a in self.actions
            ],
            "weekly_stats": [
                {
                    "week_number": s.week_number,
                    "start_date": s.start_date,
                    "end_date": s.end_date,
                    "starting_equity": s.starting_equity,
                    "ending_equity": s.ending_equity,
                    "total_trades": s.total_trades,
                    "wins": s.wins,
                    "losses": s.losses,
                    "total_pnl": s.total_pnl,
                    "avg_slippage_pips": s.avg_slippage_pips,
                    "slippage_measurements": s.slippage_measurements,
                    "max_daily_drawdown": s.max_daily_drawdown,
                    "best_daily_return": s.best_daily_return,
                }
                for s in self.weekly_stats
            ]
        }
        self.data_path.write_text(json.dumps(data, indent=2))

    @property
    def current_week(self) -> int:
        """Calculate current week number since start."""
        if not self.start_date:
            return 1

        start = datetime.fromisoformat(self.start_date).date()
        today = datetime.now(timezone.utc).date()
        days_elapsed = (today - start).days
        return max(1, (days_elapsed // 7) + 1)

    def get_current_week_actions(self) -> List[ActionItem]:
        """Get actions for current week."""
        week = self.current_week
        return [a for a in self.actions if a.week == week]

    def get_actions_by_week(self, week: int) -> List[ActionItem]:
        """Get actions for a specific week."""
        return [a for a in self.actions if a.week == week]

    def get_actions_by_status(self, status: ActionStatus) -> List[ActionItem]:
        """Get actions by status."""
        return [a for a in self.actions if a.status == status]

    def get_action(self, action_id: str) -> Optional[ActionItem]:
        """Get a specific action by ID."""
        for a in self.actions:
            if a.id == action_id:
                return a
        return None

    def update_action(
        self,
        action_id: str,
        status: Optional[ActionStatus] = None,
        current_value: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Update an action's progress."""
        action = self.get_action(action_id)
        if not action:
            logger.warning(f"Action not found: {action_id}")
            return False

        if status:
            action.status = status
        if current_value is not None:
            action.current_value = current_value
            # Auto-complete if target reached
            if action.target_value and current_value >= action.target_value:
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.now(timezone.utc).isoformat()
        if notes:
            action.notes = notes

        self._save()
        return True

    def complete_action(self, action_id: str, notes: str = "") -> bool:
        """Mark an action as completed."""
        return self.update_action(
            action_id,
            status=ActionStatus.COMPLETED,
            notes=notes
        )

    def start_action(self, action_id: str) -> bool:
        """Mark an action as in progress."""
        return self.update_action(action_id, status=ActionStatus.IN_PROGRESS)

    def log_weekly_stats(
        self,
        week_number: int,
        starting_equity: float,
        ending_equity: float,
        total_trades: int,
        wins: int,
        losses: int,
        total_pnl: float,
        slippage_measurements: List[float],
        max_daily_drawdown: float = 0.0,
        best_daily_return: float = 0.0,
    ) -> WeeklyStats:
        """Log statistics for a completed week."""
        # Calculate week dates
        if self.start_date:
            start_dt = datetime.fromisoformat(self.start_date).date()
            week_start = start_dt + timedelta(days=(week_number - 1) * 7)
            week_end = week_start + timedelta(days=6)
        else:
            week_start = datetime.now(timezone.utc).date()
            week_end = week_start + timedelta(days=6)

        avg_slippage = sum(slippage_measurements) / len(slippage_measurements) if slippage_measurements else 0

        stats = WeeklyStats(
            week_number=week_number,
            start_date=week_start.isoformat(),
            end_date=week_end.isoformat(),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            avg_slippage_pips=avg_slippage,
            slippage_measurements=slippage_measurements,
            max_daily_drawdown=max_daily_drawdown,
            best_daily_return=best_daily_return,
        )

        # Replace existing or append
        existing_idx = next(
            (i for i, s in enumerate(self.weekly_stats) if s.week_number == week_number),
            None
        )
        if existing_idx is not None:
            self.weekly_stats[existing_idx] = stats
        else:
            self.weekly_stats.append(stats)

        self._save()

        # Auto-update related actions based on stats
        self._auto_update_actions(stats)

        return stats

    def _auto_update_actions(self, stats: WeeklyStats) -> None:
        """Auto-update action progress based on weekly stats."""
        # Update trade count
        total_trades_all_weeks = sum(s.total_trades for s in self.weekly_stats)
        self.update_action("w1_20_trades", current_value=total_trades_all_weeks)

        # Update win rate verification
        total_wins = sum(s.wins for s in self.weekly_stats)
        if total_trades_all_weeks > 0:
            win_rate = (total_wins / total_trades_all_weeks) * 100
            self.update_action("w1_verify_78pct", current_value=win_rate)

        # Update slippage measurement count
        total_slippage_measurements = sum(len(s.slippage_measurements) for s in self.weekly_stats)
        self.update_action("w1_measure_slippage", current_value=total_slippage_measurements)

        # Update slippage target (week 2)
        if stats.avg_slippage_pips > 0:
            self.update_action("w2_slippage_target", current_value=stats.avg_slippage_pips)

        # Update equity targets
        latest_equity = stats.ending_equity
        self.update_action("w4_target_200k", current_value=latest_equity)
        self.update_action("m3_target_1m", current_value=latest_equity)

    def get_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive progress report."""
        current_week = self.current_week

        # Action status summary
        total_actions = len(self.actions)
        completed = len([a for a in self.actions if a.status == ActionStatus.COMPLETED])
        in_progress = len([a for a in self.actions if a.status == ActionStatus.IN_PROGRESS])
        not_started = len([a for a in self.actions if a.status == ActionStatus.NOT_STARTED])

        # Calculate phase
        if current_week <= 2:
            phase = "Validation"
        elif current_week <= 4:
            phase = "Optimization"
        elif current_week <= 8:
            phase = "Scaling"
        else:
            phase = "Acceleration"

        # Weekly stats summary
        if self.weekly_stats:
            latest_stats = max(self.weekly_stats, key=lambda s: s.week_number)
            total_pnl = sum(s.total_pnl for s in self.weekly_stats)
            total_trades = sum(s.total_trades for s in self.weekly_stats)
            total_wins = sum(s.wins for s in self.weekly_stats)
            overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            avg_slippage = sum(s.avg_slippage_pips for s in self.weekly_stats) / len(self.weekly_stats)
        else:
            latest_stats = None
            total_pnl = 0
            total_trades = 0
            overall_win_rate = 0
            avg_slippage = 0

        # Current week actions
        current_actions = self.get_current_week_actions()
        current_week_progress = sum(a.progress_pct for a in current_actions) / len(current_actions) if current_actions else 0

        return {
            "current_week": current_week,
            "phase": phase,
            "start_date": self.start_date,
            "days_elapsed": (datetime.now(timezone.utc).date() - datetime.fromisoformat(self.start_date).date()).days if self.start_date else 0,
            "action_summary": {
                "total": total_actions,
                "completed": completed,
                "in_progress": in_progress,
                "not_started": not_started,
                "completion_pct": round(completed / total_actions * 100, 1),
            },
            "current_week_actions": [
                {
                    "id": a.id,
                    "description": a.description,
                    "status": a.status.value,
                    "progress_pct": round(a.progress_pct, 1),
                    "target": a.target_value,
                    "current": a.current_value,
                }
                for a in current_actions
            ],
            "current_week_progress_pct": round(current_week_progress, 1),
            "performance": {
                "total_trades": total_trades,
                "overall_win_rate": round(overall_win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_slippage_pips": round(avg_slippage, 2),
                "latest_equity": latest_stats.ending_equity if latest_stats else None,
            },
            "weekly_stats_summary": [
                {
                    "week": s.week_number,
                    "trades": s.total_trades,
                    "win_rate": round(s.win_rate * 100, 1),
                    "pnl": round(s.total_pnl, 2),
                    "return_pct": round(s.weekly_return * 100, 1),
                }
                for s in sorted(self.weekly_stats, key=lambda s: s.week_number)
            ],
        }

    def print_dashboard(self) -> None:
        """Print a formatted dashboard."""
        report = self.get_progress_report()

        print("\n" + "=" * 70)
        print("📊 AGGRESSIVE SCALING PROGRESS DASHBOARD")
        print("=" * 70)

        print(f"\n🗓️  Week {report['current_week']} | Phase: {report['phase']}")
        print(f"   Started: {report['start_date']} ({report['days_elapsed']} days)")

        print(f"\n📋 Actions: {report['action_summary']['completed']}/{report['action_summary']['total']} completed ({report['action_summary']['completion_pct']}%)")

        print("\n🎯 Current Week Actions:")
        for a in report['current_week_actions']:
            status_icon = "✅" if a['status'] == 'completed' else "🔄" if a['status'] == 'in_progress' else "⬜"
            progress = f" [{a['current']}/{a['target']}]" if a['target'] else ""
            print(f"   {status_icon} {a['description']}{progress}")

        perf = report['performance']
        print("\n📈 Performance:")
        print(f"   Total Trades: {perf['total_trades']}")
        print(f"   Win Rate: {perf['overall_win_rate']}%")
        print(f"   Total PnL: ${perf['total_pnl']:,.2f}")
        print(f"   Avg Slippage: {perf['avg_slippage_pips']} pips")
        if perf['latest_equity']:
            print(f"   Current Equity: ${perf['latest_equity']:,.2f}")

        if report['weekly_stats_summary']:
            print("\n📅 Weekly Returns:")
            for s in report['weekly_stats_summary']:
                print(f"   Week {s['week']}: {s['trades']} trades, {s['win_rate']}% win rate, ${s['pnl']:,.0f} ({s['return_pct']:+.1f}%)")

        print("\n" + "=" * 70)


# ============================================================================
# Integration with Trade Journal
# ============================================================================

def sync_from_trade_journal(tracker: WeeklyActionTracker, journal_path: Optional[Path] = None) -> None:
    """
    Sync tracker with trade journal data.

    Args:
        tracker: Action tracker to update
        journal_path: Path to trade journal JSON
    """
    journal_path = journal_path or Path("trained_data") / "trade_journal.json"

    if not journal_path.exists():
        logger.warning("Trade journal not found")
        return

    try:
        data = json.loads(journal_path.read_text())
        trades = data.get("trades", [])

        if not trades:
            return

        # Group trades by week
        from collections import defaultdict
        weekly_trades = defaultdict(list)

        for trade in trades:
            trade_date = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00")).date()
            if tracker.start_date:
                start = datetime.fromisoformat(tracker.start_date).date()
                week = ((trade_date - start).days // 7) + 1
            else:
                week = 1
            weekly_trades[week].append(trade)

        # Update weekly stats
        cumulative_equity = 100000  # Starting equity assumption

        for week_num in sorted(weekly_trades.keys()):
            week_trades = weekly_trades[week_num]

            wins = len([t for t in week_trades if t.get("status") == "win"])
            losses = len([t for t in week_trades if t.get("status") == "loss"])
            total_pnl = sum(t.get("pnl", 0) or 0 for t in week_trades)
            slippage_measurements = [
                abs(t.get("slippage_pips", 0))
                for t in week_trades
                if t.get("slippage_pips") is not None
            ]

            starting_equity = cumulative_equity
            cumulative_equity += total_pnl

            tracker.log_weekly_stats(
                week_number=week_num,
                starting_equity=starting_equity,
                ending_equity=cumulative_equity,
                total_trades=len(week_trades),
                wins=wins,
                losses=losses,
                total_pnl=total_pnl,
                slippage_measurements=slippage_measurements,
            )

        logger.info(f"Synced {len(trades)} trades from journal")

    except Exception as e:
        logger.error(f"Failed to sync from journal: {e}")


# ============================================================================
# CLI and Main
# ============================================================================

if __name__ == "__main__":
    import sys

    tracker = WeeklyActionTracker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "dashboard":
            tracker.print_dashboard()

        elif command == "sync":
            sync_from_trade_journal(tracker)
            print("✅ Synced from trade journal")
            tracker.print_dashboard()

        elif command == "complete" and len(sys.argv) > 2:
            action_id = sys.argv[2]
            notes = sys.argv[3] if len(sys.argv) > 3 else ""
            if tracker.complete_action(action_id, notes):
                print(f"✅ Completed: {action_id}")
            else:
                print(f"❌ Action not found: {action_id}")

        elif command == "start" and len(sys.argv) > 2:
            action_id = sys.argv[2]
            if tracker.start_action(action_id):
                print(f"🔄 Started: {action_id}")
            else:
                print(f"❌ Action not found: {action_id}")

        elif command == "update" and len(sys.argv) > 3:
            action_id = sys.argv[2]
            value = float(sys.argv[3])
            if tracker.update_action(action_id, current_value=value):
                print(f"📝 Updated: {action_id} = {value}")
            else:
                print(f"❌ Action not found: {action_id}")

        elif command == "reset":
            tracker._init_default()
            print("🔄 Reset to default action plan")

        else:
            print("Usage:")
            print("  python scaling_action_plan.py dashboard")
            print("  python scaling_action_plan.py sync")
            print("  python scaling_action_plan.py complete <action_id> [notes]")
            print("  python scaling_action_plan.py start <action_id>")
            print("  python scaling_action_plan.py update <action_id> <value>")
            print("  python scaling_action_plan.py reset")
    else:
        # Default: show dashboard
        tracker.print_dashboard()
