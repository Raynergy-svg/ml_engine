"""
CONFIG SCREEN -- Tier 7 Quant Control Room

Four-panel layout in a 2x3 grid:
  Row 0 (full width): Active Profile selector (RadioSet) + Apply Profile button
  Row 1, Col 0: Tier 7 scanner thresholds and execution controls
  Row 1, Col 1: Agent, learning, execution, and infrastructure toggles
  Row 2 (full width): Config proposal/approval history

Staged edits pattern: changes are highlighted amber until explicitly queued.
Drop-in replacement for PlaceholderContent in the Config TabPane.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

from rich.text import Text

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Rule,
    Switch,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (matches theme.tcss)
# ---------------------------------------------------------------------------
_BG = "#131320"
_BORDER = "#2a2a4a"
_PRIMARY = "#00ffcc"
_SECONDARY = "#ff00ff"
_POSITIVE = "#00ff41"
_NEGATIVE = "#ff1744"
_WARNING = "#ffab00"
_TEXT = "#e0e0ff"
_DIM = "#6666aa"
_DATA = "#7c4dff"


# ---------------------------------------------------------------------------
# Numeric field definitions: (field_name, label, default)
# Grouped by Tier 7 operating category for display.
# ---------------------------------------------------------------------------

_CONFIG_FIELD_GROUPS: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("Entry Quality Gates", [
        ("min_confidence", "Min Confidence", 42.0),
        ("min_momentum", "Momentum Floor", 0.06),
        ("min_tcn_probability", "TCN Probability", 0.60),
        ("final_score_threshold", "Final Score", 0.45),
        ("meta_labeler_threshold", "Meta Labeler", 0.52),
    ]),
    ("Weighted Vote Stack", [
        ("weighted_vote_threshold", "Vote Threshold", 0.45),
        ("min_agent_consensus_ratio", "Min Consensus", 0.25),
        ("sub_inference_min_confidence", "Sub-Agent Conf", 0.30),
        ("sub_inference_vote_threshold", "Sub-Agent Vote", 0.34),
        ("sub_inference_max_candidates", "Sub Candidates", 10),
    ]),
    ("Uncertainty & Drift", [
        ("max_uncertainty_score", "Max Uncertainty", 0.40),
        ("max_model_disagreement", "Max Disagreement", 0.65),
        ("disagreement_hard_floor", "Disagree Floor", 0.50),
        ("max_uncertainty_std", "Max Uncert Std", 0.15),
        ("staleness_uncertainty_threshold", "Stale Uncertainty", 0.35),
        ("staleness_age_threshold_days", "Stale Age Days", 7.0),
    ]),
    ("Risk & Trade Structure", [
        ("risk_per_trade_pct", "Base Risk %", 0.05),
        ("max_open_risk_pct", "Portfolio Risk", 0.30),
        ("max_drawdown_pct", "Max Drawdown", 0.025),
        ("min_risk_reward_ratio", "Min R:R", 1.5),
        ("min_atr_pips", "Min ATR Pips", 5.0),
        ("atr_sl_multiplier", "ATR SL Multi", 1.0),
        ("atr_tp_multiplier", "ATR TP Multi", 2.5),
        ("min_sl_pips", "Min SL Pips", 8.0),
        ("max_sl_pips", "Max SL Pips", 30.0),
        ("min_tp_pips", "Min TP Pips", 20.0),
        ("max_tp_pips", "Max TP Pips", 60.0),
    ]),
    ("Execution Controls", [
        ("daily_trade_limit", "Daily Limit", 30),
        ("max_concurrent_trades", "Max Concurrent", 10),
        ("max_correlated_exposure", "Max Correlated", 2),
        ("max_order_attempts", "Order Attempts", 2),
        ("max_data_age_seconds", "Max Data Age", 600.0),
        ("rejection_downsize_factor", "Reject Downsize", 0.5),
        ("max_spread_pips", "Max Spread", 3.0),
        ("max_slippage_pips", "Max Slippage", 2.0),
        ("min_liquidity_score", "Min Liquidity", 0.3),
    ]),
]

# Backward-compatible flattened catalog used by tests and event handlers.
_GATE_FIELDS: list[tuple[str, str, float]] = [
    field
    for _section, fields in _CONFIG_FIELD_GROUPS
    for field in fields
]

_TOGGLE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Scanner Agents", [
        ("enable_trend_agent", "Trend"),
        ("enable_mean_reversion_agent", "Mean Reversion"),
        ("enable_volatility_agent", "Volatility"),
        ("enable_risk_sentinel_agent", "Risk Sentinel"),
        ("enable_uncertainty_agent", "Uncertainty"),
        ("enable_execution_quality_agent", "Exec Quality"),
        ("enable_momentum_agent", "Momentum"),
        ("enable_news_risk_agent", "News Risk"),
        ("enable_multi_timeframe_agent", "Multi-TF"),
        ("enable_pair_performance_agent", "Pair Perf"),
        ("enable_session_timing_agent", "Session Time"),
        ("enable_support_resistance_agent", "Support/Resist"),
        ("enable_trader_readiness_agent", "Trader Ready"),
        ("enable_order_flow_agent", "Order Flow"),
        ("enable_devil_advocate_agent", "Devil Advocate"),
    ]),
    ("RL & Adaptive Intelligence", [
        ("use_rl_sizer", "RL Sizer"),
        ("use_rl_gates", "RL Gates"),
        ("use_rl_exits", "RL Exits"),
        ("enable_agent_trade_promotion", "Agent Promotion"),
        ("enable_threshold_optimizer", "Threshold Optimizer"),
        ("enable_confidence_calibration", "Confidence Cal"),
        ("enable_model_calibration", "Model Calibration"),
        ("enable_model_bandit", "Model Bandit"),
        ("enable_model_routing", "Model Routing"),
        ("enable_concept_drift_detection", "Drift Detection"),
        ("enable_ensemble_disagreement", "Disagreement Signal"),
        ("enable_trade_outcome_prediction", "Outcome Predictor"),
        ("enable_trade_explainability", "Explainability"),
        ("enable_observational_learning", "Observation Learning"),
    ]),
    ("Execution & Portfolio", [
        ("enable_execution", "Execution"),
        ("position_sizing_enabled", "Position Sizing"),
        ("enable_smart_execution", "Smart Execution"),
        ("enable_execution_routing", "Execution Routing"),
        ("enable_execution_quality_tracking", "TCA Tracking"),
        ("enable_execution_quality_optimizer", "TCA Optimizer"),
        ("enable_dynamic_sl_tp", "Dynamic SL/TP"),
        ("enable_live_position_management", "Live Management"),
        ("enable_position_timeout", "Position Timeout"),
        ("enable_dynamic_hedging", "Dynamic Hedging"),
        ("enable_kelly_sizing", "Kelly Sizing"),
        ("use_hrp", "HRP Portfolio"),
    ]),
    ("Platform & Safety", [
        ("enable_session_filter", "Session Filter"),
        ("enable_calendar_blackout", "Calendar Blackout"),
        ("soft_uncertainty_blocking", "Soft Uncertainty"),
        ("enable_circuit_breakers", "Circuit Breakers"),
        ("enable_memory_manager", "Memory Manager"),
        ("enable_health_registry", "Health Registry"),
        ("enable_module_activation", "Module Activation"),
        ("enable_agent_lifecycle", "Agent Lifecycle"),
        ("enable_observation_consumer", "Observation Consumer"),
        ("enable_replay_validator", "Replay Validator"),
        ("enable_agent_accuracy_matrix", "Agent Accuracy"),
        ("enable_pair_regime_agent_matrix", "Pair-Regime Matrix"),
        ("enable_signal_timing", "Signal Timing"),
    ]),
]

_AGENT_TOGGLES: list[tuple[str, str]] = [
    field
    for _section, fields in _TOGGLE_GROUPS
    for field in fields
]

# Available profiles
_PROFILES = ["conservative", "balanced", "aggressive", "smart"]


def _normalize_adjustment_row(entry: dict[str, Any], status: str) -> dict[str, Any]:
    """Normalize legacy, pending, and approved adjustment records for display."""
    field_name = entry.get("key", entry.get("field", "---"))
    old_value = entry.get(
        "old_value",
        entry.get("old", entry.get("current_value", "---")),
    )
    new_value = entry.get(
        "new_value",
        entry.get("new", entry.get("proposed_value", "---")),
    )
    return {
        "timestamp": entry.get("timestamp", "---"),
        "status": entry.get("status", status),
        "field": field_name,
        "old": old_value,
        "new": new_value,
        "source": entry.get("source", "---"),
    }


def _read_adjustment_entries(path: Path, status: str) -> list[dict[str, Any]]:
    """Read pending or approved adjustment records from a JSON store."""
    try:
        if not path.exists():
            return []
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return []

    if isinstance(parsed, dict):
        if "proposals" in parsed and isinstance(parsed["proposals"], list):
            entries = parsed["proposals"]
        elif "history" in parsed and isinstance(parsed["history"], list):
            entries = parsed["history"]
        else:
            entries = []
    elif isinstance(parsed, list):
        entries = parsed
    else:
        entries = []

    return [
        _normalize_adjustment_row(entry, status)
        for entry in entries
        if isinstance(entry, dict)
    ]


def _load_adjustment_rows(project_root: Path, limit: int = 40) -> list[dict[str, Any]]:
    """Return most recent pending/approved config rows, newest first."""
    root = project_root / ".claude"
    rows = (
        _read_adjustment_entries(root / "pending_adjustments.json", "pending")
        + _read_adjustment_entries(root / "config_adjustments.json", "approved")
    )
    rows.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
    return rows[:limit]


# ---------------------------------------------------------------------------
# ConfigScreen -- Main Container
# ---------------------------------------------------------------------------

class ConfigScreen(Container):
    """The Control Room -- F5 screen.

    Four-panel layout:
      - Profile selector (top, full width)
      - Gate thresholds (middle-left) + Agent toggles (middle-right)
      - Config change history DataTable (bottom, full width)

    Edits are staged (amber highlight) until queued via the approval pipeline.
    """

    BINDINGS = [
        Binding("s", "stage_profile", "Stage", show=True),
        Binding("q", "queue_for_approval", "Queue", show=True),
        Binding("r", "reset_form", "Reset", show=True),
    ]

    DEFAULT_CSS = """
    ConfigScreen {
        height: 1fr;
        layout: vertical;
    }
    """

    def __init__(self, project_root: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._project_root: Path = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self._staged_changes: dict[str, tuple[Any, Any]] = {}  # field -> (old, new)
        self._current_config: dict[str, Any] = {}
        self._active_profile: str = "smart"

    def compose(self) -> ComposeResult:
        # -- Row 0: Profile Selector --
        with Vertical(id="config-profile-panel", classes="panel"):
            yield Label("  ACTIVE PROFILE", classes="panel-title")
            with Horizontal(id="config-profile-row"):
                yield RadioSet(
                    RadioButton("Conservative", id="radio-conservative"),
                    RadioButton("Balanced", id="radio-balanced"),
                    RadioButton("Aggressive", id="radio-aggressive"),
                    RadioButton("Smart", id="radio-smart", value=True),
                    id="config-profile-radios",
                )
                yield Button("STAGE PROFILE", id="btn-apply-profile", variant="primary")
                yield Button("QUEUE FOR APPROVAL", id="btn-save-config", variant="success")
                yield Button("RESET FORM", id="btn-reset-config", variant="warning")
                yield Label("", id="config-status-label")

        # -- Row 1: Gate Thresholds (left) + Agent Toggles (right) --
        with Horizontal(id="config-middle-row"):
            with Vertical(id="config-gates-panel", classes="panel"):
                yield Label("  TIER 7 GATES / RISK / EXECUTION", classes="panel-title")
                with ScrollableContainer(id="config-gates-scroll"):
                    for idx, (section, fields) in enumerate(_CONFIG_FIELD_GROUPS):
                        if idx:
                            yield Rule()
                        yield Label(f"  {section}", id=f"gate-section-{idx}")
                        for field_name, label, default in fields:
                            with Horizontal(classes="config-field-row"):
                                yield Label(f"  {label}", classes="config-field-label")
                                yield Input(
                                    value=str(default),
                                    id=f"gate-{field_name}",
                                    placeholder=str(default),
                                    classes="config-field-input",
                                )

            with Vertical(id="config-agents-panel", classes="panel"):
                yield Label("  TIER 7 MODULES", classes="panel-title")
                with ScrollableContainer(id="config-agents-scroll"):
                    for idx, (section, toggles) in enumerate(_TOGGLE_GROUPS):
                        if idx:
                            yield Rule()
                        yield Label(f"  {section}", id=f"toggle-section-{idx}")
                        for field_name, display_name in toggles:
                            with Horizontal(classes="config-toggle-row"):
                                yield Label(f"  {display_name}", classes="config-toggle-label")
                                yield Switch(
                                    value=True,
                                    id=f"toggle-{field_name}",
                                    classes="config-toggle-switch",
                                )

        # -- Row 2: Config Change History --
        with Vertical(id="config-history-panel", classes="panel"):
            yield Label("  CONFIG PROPOSALS / APPROVAL HISTORY", classes="panel-title")
            yield DataTable(id="config-history-table", cursor_type="row")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Initialize tables and load live config data."""
        # Set up history table columns
        history_table = self.query_one("#config-history-table", DataTable)
        history_table.add_columns("Timestamp", "Status", "Field", "Old", "New", "Source")
        history_table.zebra_stripes = True

        # Load current config values into the inputs
        self._load_config()
        self._load_change_history()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        """Track staged changes when an input value changes."""
        input_id = event.input.id or ""
        if not input_id.startswith("gate-"):
            return

        field_name = input_id.removeprefix("gate-")
        new_value_str = event.value.strip()

        # Validate: must be a valid float
        try:
            new_value = float(new_value_str)
        except (ValueError, TypeError):
            return

        # Compare against current config
        old_value = self._current_config.get(field_name, 0.0)
        try:
            old_value = float(old_value)
        except (ValueError, TypeError):
            old_value = 0.0

        if abs(new_value - old_value) > 1e-9:
            self._staged_changes[field_name] = (old_value, new_value)
            # Highlight the input amber
            event.input.styles.border = ("solid", _WARNING)
        else:
            # Remove from staged if reverted to original
            self._staged_changes.pop(field_name, None)
            event.input.styles.border = ("solid", _BORDER)

        self._update_status_label()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Track staged agent toggle changes."""
        switch_id = event.switch.id or ""
        if not switch_id.startswith("toggle-"):
            return

        field_name = switch_id.removeprefix("toggle-")
        new_value = event.value
        old_value = self._current_config.get(field_name, True)

        if new_value != old_value:
            self._staged_changes[field_name] = (old_value, new_value)
        else:
            self._staged_changes.pop(field_name, None)

        self._update_status_label()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks — delegates to action methods (shared dispatch)."""
        button_id = event.button.id or ""

        if button_id == "btn-save-config":
            self.action_queue_for_approval()
        elif button_id == "btn-reset-config":
            self.action_reset_form()
        elif button_id == "btn-apply-profile":
            self.action_stage_profile()

    # ------------------------------------------------------------------
    # Actions (BINDINGS dispatch) — keyboard equivalents of the buttons
    # ------------------------------------------------------------------

    def action_stage_profile(self) -> None:
        """Stage the selected profile's differences for approval (key: s)."""
        self._apply_selected_profile()

    def action_queue_for_approval(self) -> None:
        """Queue all staged changes through the approval pipeline (key: q)."""
        self._save_staged_changes()

    def action_reset_form(self) -> None:
        """Reset all inputs to the selected profile's defaults (key: r)."""
        self._reset_to_profile()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Track selected profile from RadioSet."""
        if event.radio_set.id != "config-profile-radios":
            return
        # Map index to profile name
        idx = event.index
        if 0 <= idx < len(_PROFILES):
            self._active_profile = _PROFILES[idx]

    # ------------------------------------------------------------------
    # Public: update from snapshot
    # ------------------------------------------------------------------

    def update_from_snapshot(self, snap: Any) -> None:
        """Refresh config data. Called periodically by the main app.

        Live scanner config values arrive through DashboardSnapshot when the
        embedded scanner is running. Do not overwrite a human's staged edits.
        """
        config_values = getattr(snap, "config_values", None)
        if config_values and not self._staged_changes:
            self._active_profile = str(
                getattr(snap, "config_profile", self._active_profile)
                or self._active_profile
            )
            self._populate_from_values(dict(config_values), update_current=True)
        self._load_change_history()

    # ------------------------------------------------------------------
    # Private: Load config
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load current ScannerConfig values into the input fields.

        Mythos audit 2026-05-04 — display must reflect what the LIVE
        Scanner has, not a fresh ScannerConfig() with defaults.
        Without applying on-disk adjustments, the screen could show
        min_confidence=35.0 (smart profile default) while the running
        Scanner had min_confidence=42.0 (after meta-pipeline promoted
        a config_delta and EmbeddedScanner's ConfigAdjuster picked it
        up via the disk bridge). Operator looked at one number, the
        runtime acted on another.
        """
        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
            config.apply_profile("smart")
        except Exception as exc:
            logger.debug("Failed to load ScannerConfig: %s", exc)
            config = None

        if config is None:
            return

        # Apply persisted adjustments so the display matches the running
        # Scanner (whose own ConfigAdjuster reads the same disk file
        # every cycle via embedded_scanner.py:738-749).
        try:
            from src.scanner.automation.config_adjuster import ConfigAdjuster
            adjuster = ConfigAdjuster()
            adjuster.apply_adjustments(config, current_cycle=0)
        except Exception as adj_err:
            logger.debug(
                "ConfigScreen: applying on-disk adjustments for display "
                "failed (showing profile defaults): %s",
                adj_err,
            )

        self._active_profile = getattr(config, "profile", "smart")
        values = self._extract_config_values(config)
        self._populate_from_values(values, update_current=True)

        # Set profile radio button
        self._sync_profile_radio()

        # Clear any staged changes
        self._staged_changes.clear()
        self._update_status_label()

    def _extract_config_values(self, config: Any) -> dict[str, Any]:
        """Extract only the fields this screen owns from a ScannerConfig."""
        values: dict[str, Any] = {}
        for field_name, _label, default in _GATE_FIELDS:
            values[field_name] = getattr(config, field_name, default)
        for field_name, _display_name in _AGENT_TOGGLES:
            values[field_name] = getattr(config, field_name, True)
        return values

    def _populate_from_values(self, values: dict[str, Any], *, update_current: bool) -> None:
        """Populate widgets from a config value map."""
        for field_name, _label, default in _GATE_FIELDS:
            value = values.get(field_name, default)
            if update_current:
                self._current_config[field_name] = value

            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.value = str(value)
                inp.styles.border = ("solid", _BORDER)
            except Exception:
                pass

        for field_name, _display_name in _AGENT_TOGGLES:
            value = values.get(field_name, True)
            if update_current:
                self._current_config[field_name] = value

            try:
                switch = self.query_one(f"#toggle-{field_name}", Switch)
                switch.value = bool(value)
            except Exception:
                pass

    def _sync_profile_radio(self) -> None:
        """Reflect the active profile in the RadioSet."""
        try:
            profile_idx = _PROFILES.index(self._active_profile)
            radio_set = self.query_one("#config-profile-radios", RadioSet)
            # Textual RadioSet: pressing the button at the index
            buttons = list(radio_set.query(RadioButton))
            if 0 <= profile_idx < len(buttons):
                buttons[profile_idx].value = True
        except (ValueError, Exception):
            pass

    # ------------------------------------------------------------------
    # Private: Save staged changes
    # ------------------------------------------------------------------

    def _save_staged_changes(self) -> None:
        """Queue staged changes through the approval pipeline."""
        if not self._staged_changes:
            self._flash_status("No changes to queue", _DIM)
            return

        try:
            from src.scanner.automation.config_adjuster import ConfigAdjuster
        except Exception as exc:
            logger.error("Failed to load ConfigAdjuster for config queue: %s", exc)
            self._flash_status("ERROR: Approval queue unavailable", _NEGATIVE)
            return

        adjuster = ConfigAdjuster(
            persistence_path=self._project_root / ".claude" / "config_adjustments.json",
            pending_path=self._project_root / ".claude" / "pending_adjustments.json",
        )

        queued = 0
        invalid = 0
        staged_items = list(self._staged_changes.items())
        for field_name, (old_val, new_val) in self._staged_changes.items():
            try:
                coerced_val = self._coerce_value(field_name, old_val, new_val)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "ConfigScreen: invalid staged value for %s: %s",
                    field_name,
                    exc,
                )
                invalid += 1
                continue

            adjuster.collect_adjustment(
                source="tui_config",
                key=field_name,
                value=coerced_val,
                reason=(
                    f"Operator staged Tier 7 config change from Config tab "
                    f"({self._format_value(old_val)} -> {self._format_value(coerced_val)}, "
                    f"profile={self._active_profile})"
                ),
            )
            queued += 1

        self._staged_changes.clear()

        # Reset input borders to default
        for field_name, _, _ in _GATE_FIELDS:
            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.styles.border = ("solid", _BORDER)
            except Exception:
                pass

        # Reload history table
        self._load_change_history()
        if invalid:
            self._flash_status(f"Queued {queued}; {invalid} invalid", _WARNING)
        else:
            self._flash_status(f"Queued {queued} pending approval", _POSITIVE)
        self._update_status_label()

        logger.info(
            "ConfigScreen queued %d/%d staged Tier 7 config changes",
            queued,
            len(staged_items),
        )

    # ------------------------------------------------------------------
    # Private: Reset to profile
    # ------------------------------------------------------------------

    def _reset_to_profile(self) -> None:
        """Reset all inputs to the selected profile's defaults."""
        self._staged_changes.clear()

        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
            config.apply_profile(self._active_profile)
        except Exception as exc:
            logger.debug("Failed to reset to profile: %s", exc)
            self._flash_status("ERROR: Profile reset failed", _NEGATIVE)
            return

        self._populate_from_values(self._extract_config_values(config), update_current=True)

        self._flash_status(f"Reset to {self._active_profile} profile", _PRIMARY)
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: Apply selected profile
    # ------------------------------------------------------------------

    def _apply_selected_profile(self) -> None:
        """Stage the selected profile's differences for approval."""
        self._staged_changes.clear()

        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
            config.apply_profile(self._active_profile)
        except Exception as exc:
            logger.debug("Failed to apply profile '%s': %s", self._active_profile, exc)
            self._flash_status(f"ERROR: Unknown profile '{self._active_profile}'", _NEGATIVE)
            return

        profile_values = self._extract_config_values(config)
        self._populate_from_values(profile_values, update_current=False)

        for field_name, new_value in profile_values.items():
            old_value = self._current_config.get(field_name)
            if old_value != new_value:
                self._staged_changes[field_name] = (old_value, new_value)

        self._highlight_staged_inputs()
        self._flash_status(
            f"Staged {len(self._staged_changes)} from {self._active_profile.upper()}",
            _WARNING if self._staged_changes else _DIM,
        )
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: Load change history
    # ------------------------------------------------------------------

    def _load_change_history(self) -> None:
        """Load pending and approved config changes."""
        try:
            history_table = self.query_one("#config-history-table", DataTable)
        except Exception:
            return

        history_table.clear()

        recent = _load_adjustment_rows(self._project_root, limit=40)

        for entry in recent:
            ts = str(entry.get("timestamp", "---"))
            # Truncate to readable format
            if len(ts) > 19:
                ts = ts[:19]

            status = str(entry.get("status", "---")).upper()
            field_name = str(entry.get("field", "---"))
            old_val = entry.get("old", "---")
            new_val = entry.get("new", "---")
            source = str(entry.get("source", "---"))

            # Format values for display
            old_str = self._format_value(old_val)
            new_str = self._format_value(new_val)

            # Styled cells
            ts_text = Text(ts, style=_DIM)
            status_style = (
                _POSITIVE
                if status == "APPROVED"
                else _WARNING
                if status == "PENDING"
                else _NEGATIVE
            )
            status_text = Text(status, style=status_style)
            field_text = Text(field_name, style=_TEXT)
            old_text = Text(old_str, style=_NEGATIVE)
            new_text = Text(new_str, style=_POSITIVE)
            source_text = Text(source, style=_DATA)

            history_table.add_row(ts_text, status_text, field_text, old_text, new_text, source_text)

        # If no entries, show a placeholder row.
        # Must match the 6-column schema: Timestamp | Status | Field | Old | New | Source.
        if not recent:
            history_table.add_row(
                Text("---", style=_DIM),
                Text("---", style=_DIM),
                Text("No config proposals recorded", style=_DIM),
                Text("---", style=_DIM),
                Text("---", style=_DIM),
                Text("---", style=_DIM),
            )

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------

    def _coerce_value(self, field_name: str, old_value: Any, new_value: Any) -> Any:
        """Coerce TUI string/float values to the existing ScannerConfig type."""
        if (
            isinstance(old_value, bool)
            or field_name.startswith(("enable_", "use_"))
            or field_name.endswith("_enabled")
        ):
            if isinstance(new_value, str):
                return new_value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(new_value)

        if isinstance(old_value, int) and not isinstance(old_value, bool):
            return int(float(new_value))

        if isinstance(old_value, float):
            return float(new_value)

        return new_value

    def _highlight_staged_inputs(self) -> None:
        """Mark all staged widgets amber and reset non-staged fields."""
        staged = set(self._staged_changes)
        for field_name, _label, _default in _GATE_FIELDS:
            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.styles.border = ("solid", _WARNING if field_name in staged else _BORDER)
            except Exception:
                pass

    @staticmethod
    def _format_value(val: Any) -> str:
        """Format a config value for display in the history table."""
        if isinstance(val, bool):
            return "ON" if val else "OFF"
        if isinstance(val, float):
            if val == int(val) and abs(val) < 1000:
                return f"{val:.1f}"
            return f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
        return str(val)

    def _update_status_label(self) -> None:
        """Update the status label with staged change count."""
        try:
            label = self.query_one("#config-status-label", Label)
        except Exception:
            return

        count = len(self._staged_changes)
        if count > 0:
            label.update(
                Text.from_markup(
                    f"  [{_WARNING}]{count} staged change{'s' if count != 1 else ''}[/]"
                    f"  [{_DIM}]Queue for approval[/]"
                )
            )
        else:
            label.update(
                Text.from_markup(
                    f"  [{_DIM}]Profile: [{_PRIMARY}]{self._active_profile.upper()}[/][/]"
                )
            )

    def _flash_status(self, message: str, color: str) -> None:
        """Flash a status message in the status label."""
        try:
            label = self.query_one("#config-status-label", Label)
            label.update(
                Text.from_markup(f"  [{color}]{message}[/]")
            )
        except Exception:
            pass
