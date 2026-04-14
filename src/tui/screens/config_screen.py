"""
CONFIG SCREEN -- "The Control Room"

Four-panel layout in a 2x3 grid:
  Row 0 (full width): Active Profile selector (RadioSet) + Apply Profile button
  Row 1, Col 0: Gate Thresholds -- grouped Input fields for scanner config values
  Row 1, Col 1: Agent Toggles (Switch per agent) + Pairs checkboxes
  Row 2 (full width): Config Change History (DataTable)

Staged edits pattern: changes are highlighted amber until explicitly saved.
Drop-in replacement for PlaceholderContent in the Config TabPane.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Rule,
    Static,
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
# Gate threshold field definitions: (field_name, label, default, step, fmt)
# Grouped by category for display
# ---------------------------------------------------------------------------

_GATE_FIELDS: list[tuple[str, str, float]] = [
    # Confidence & voting
    ("min_confidence", "Min Confidence", 42.0),
    ("weighted_vote_threshold", "Vote Threshold", 0.45),
    ("min_agent_consensus_ratio", "Min Consensus", 0.25),
    ("min_momentum", "Min Momentum", 0.06),
    # Uncertainty & disagreement
    ("max_uncertainty_score", "Max Uncertainty", 0.43),
    ("max_model_disagreement", "Max Disagreement", 0.28),
    ("max_uncertainty_std", "Max Uncert Std", 0.15),
    # Risk
    ("max_open_risk_pct", "Max Portfolio Risk", 0.15),
    ("risk_per_trade_pct", "Base Risk %", 0.05),
    ("max_drawdown_pct", "Max Drawdown %", 0.025),
    # Trade structure
    ("min_risk_reward_ratio", "Min R:R Ratio", 1.5),
    ("atr_sl_multiplier", "ATR SL Multi", 1.0),
    ("atr_tp_multiplier", "ATR TP Multi", 2.5),
    # Spread & ATR
    ("min_atr_pips", "Min ATR Pips", 5.0),
    ("min_tcn_probability", "Min TCN Prob", 0.60),
    ("final_score_threshold", "Final Score", 0.45),
]

# Agent toggle definitions: (config_field, display_name)
_AGENT_TOGGLES: list[tuple[str, str]] = [
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
    ("enable_devil_advocate", "Devil Advocate"),
]

# Available profiles
_PROFILES = ["conservative", "balanced", "aggressive", "smart"]


# ---------------------------------------------------------------------------
# ConfigScreen -- Main Container
# ---------------------------------------------------------------------------

class ConfigScreen(Container):
    """The Control Room -- F5 screen.

    Four-panel layout:
      - Profile selector (top, full width)
      - Gate thresholds (middle-left) + Agent toggles (middle-right)
      - Config change history DataTable (bottom, full width)

    Edits are staged (amber highlight) until saved via the SAVE button.
    """

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
        self._active_profile: str = "balanced"

    def compose(self) -> ComposeResult:
        # -- Row 0: Profile Selector --
        with Vertical(id="config-profile-panel", classes="panel"):
            yield Label("  ACTIVE PROFILE", classes="panel-title")
            with Horizontal(id="config-profile-row"):
                yield RadioSet(
                    RadioButton("Conservative", id="radio-conservative"),
                    RadioButton("Balanced", id="radio-balanced", value=True),
                    RadioButton("Aggressive", id="radio-aggressive"),
                    RadioButton("Smart", id="radio-smart"),
                    id="config-profile-radios",
                )
                yield Button("APPLY PROFILE", id="btn-apply-profile", variant="primary")
                yield Button("SAVE", id="btn-save-config", variant="success")
                yield Button("RESET TO PROFILE", id="btn-reset-config", variant="warning")
                yield Label("", id="config-status-label")

        # -- Row 1: Gate Thresholds (left) + Agent Toggles (right) --
        with Horizontal(id="config-middle-row"):
            with Vertical(id="config-gates-panel", classes="panel"):
                yield Label("  GATE THRESHOLDS", classes="panel-title")
                with ScrollableContainer(id="config-gates-scroll"):
                    # Confidence & Voting section
                    yield Label("  Confidence & Voting", id="gate-section-conf")
                    for field_name, label, default in _GATE_FIELDS[:4]:
                        with Horizontal(classes="config-field-row"):
                            yield Label(f"  {label}", classes="config-field-label")
                            yield Input(
                                value=str(default),
                                id=f"gate-{field_name}",
                                placeholder=str(default),
                                classes="config-field-input",
                            )
                    yield Rule()
                    # Uncertainty & Disagreement section
                    yield Label("  Uncertainty & Disagreement", id="gate-section-uncert")
                    for field_name, label, default in _GATE_FIELDS[4:7]:
                        with Horizontal(classes="config-field-row"):
                            yield Label(f"  {label}", classes="config-field-label")
                            yield Input(
                                value=str(default),
                                id=f"gate-{field_name}",
                                placeholder=str(default),
                                classes="config-field-input",
                            )
                    yield Rule()
                    # Risk section
                    yield Label("  Risk Management", id="gate-section-risk")
                    for field_name, label, default in _GATE_FIELDS[7:10]:
                        with Horizontal(classes="config-field-row"):
                            yield Label(f"  {label}", classes="config-field-label")
                            yield Input(
                                value=str(default),
                                id=f"gate-{field_name}",
                                placeholder=str(default),
                                classes="config-field-input",
                            )
                    yield Rule()
                    # Trade Structure section
                    yield Label("  Trade Structure", id="gate-section-trade")
                    for field_name, label, default in _GATE_FIELDS[10:13]:
                        with Horizontal(classes="config-field-row"):
                            yield Label(f"  {label}", classes="config-field-label")
                            yield Input(
                                value=str(default),
                                id=f"gate-{field_name}",
                                placeholder=str(default),
                                classes="config-field-input",
                            )
                    yield Rule()
                    # Spread & ATR section
                    yield Label("  Filters & Scores", id="gate-section-filter")
                    for field_name, label, default in _GATE_FIELDS[13:]:
                        with Horizontal(classes="config-field-row"):
                            yield Label(f"  {label}", classes="config-field-label")
                            yield Input(
                                value=str(default),
                                id=f"gate-{field_name}",
                                placeholder=str(default),
                                classes="config-field-input",
                            )

            with Vertical(id="config-agents-panel", classes="panel"):
                yield Label("  AGENT TOGGLES", classes="panel-title")
                with ScrollableContainer(id="config-agents-scroll"):
                    for field_name, display_name in _AGENT_TOGGLES:
                        with Horizontal(classes="config-toggle-row"):
                            yield Label(f"  {display_name}", classes="config-toggle-label")
                            yield Switch(
                                value=True,
                                id=f"toggle-{field_name}",
                                classes="config-toggle-switch",
                            )

        # -- Row 2: Config Change History --
        with Vertical(id="config-history-panel", classes="panel"):
            yield Label("  CONFIG CHANGE HISTORY", classes="panel-title")
            yield DataTable(id="config-history-table", cursor_type="row")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Initialize tables and load live config data."""
        # Set up history table columns
        history_table = self.query_one("#config-history-table", DataTable)
        history_table.add_columns("Timestamp", "Field", "Old", "New", "Source")
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
        """Handle button clicks."""
        button_id = event.button.id or ""

        if button_id == "btn-save-config":
            self._save_staged_changes()
        elif button_id == "btn-reset-config":
            self._reset_to_profile()
        elif button_id == "btn-apply-profile":
            self._apply_selected_profile()

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

        For the config screen, we mainly care about reloading history
        since config values don't change mid-session via the snapshot.
        """
        self._load_change_history()

    # ------------------------------------------------------------------
    # Private: Load config
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load current ScannerConfig values into the input fields."""
        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
            config.apply_profile(config.profile)
        except Exception as exc:
            logger.debug("Failed to load ScannerConfig: %s", exc)
            config = None

        if config is None:
            return

        self._active_profile = getattr(config, "profile", "balanced")

        # Populate gate threshold inputs
        for field_name, _label, default in _GATE_FIELDS:
            value = getattr(config, field_name, default)
            self._current_config[field_name] = value

            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.value = str(value)
                inp.styles.border = ("solid", _BORDER)
            except Exception:
                pass

        # Populate agent toggle switches
        for field_name, _display_name in _AGENT_TOGGLES:
            value = getattr(config, field_name, True)
            self._current_config[field_name] = value

            try:
                switch = self.query_one(f"#toggle-{field_name}", Switch)
                switch.value = bool(value)
            except Exception:
                pass

        # Set profile radio button
        try:
            profile_idx = _PROFILES.index(self._active_profile)
            radio_set = self.query_one("#config-profile-radios", RadioSet)
            # Textual RadioSet: pressing the button at the index
            buttons = list(radio_set.query(RadioButton))
            if 0 <= profile_idx < len(buttons):
                buttons[profile_idx].value = True
        except (ValueError, Exception):
            pass

        # Clear any staged changes
        self._staged_changes.clear()
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: Save staged changes
    # ------------------------------------------------------------------

    def _save_staged_changes(self) -> None:
        """Apply staged changes to the ScannerConfig and persist to history."""
        if not self._staged_changes:
            self._flash_status("No changes to save", _DIM)
            return

        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
        except Exception as exc:
            logger.error("Failed to load ScannerConfig for save: %s", exc)
            self._flash_status("ERROR: Config load failed", _NEGATIVE)
            return

        # Apply each staged change to config
        history_entries: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        for field_name, (old_val, new_val) in self._staged_changes.items():
            # Type coerce based on field
            if isinstance(old_val, bool) or field_name.startswith("enable_"):
                coerced_val = bool(new_val)
            else:
                try:
                    coerced_val = float(new_val)
                except (ValueError, TypeError):
                    continue

            if hasattr(config, field_name):
                setattr(config, field_name, coerced_val)

            history_entries.append({
                "timestamp": now_iso,
                "field": field_name,
                "old": old_val,
                "new": coerced_val,
                "source": "TUI Config Screen",
            })

        # Persist history to config_adjustments.json
        self._append_change_history(history_entries)

        # Update current config cache and clear staging
        for field_name, (_, new_val) in self._staged_changes.items():
            self._current_config[field_name] = new_val

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
        self._flash_status(f"Saved {len(history_entries)} changes", _POSITIVE)
        self._update_status_label()

    def _append_change_history(self, entries: list[dict[str, Any]]) -> None:
        """Append change entries to .claude/config_adjustments.json."""
        adj_path = self._project_root / ".claude" / "config_adjustments.json"

        data: dict[str, Any] = {}
        try:
            if adj_path.exists():
                raw = adj_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data = parsed
                elif isinstance(parsed, list):
                    # Legacy format: bare list
                    data = {"version": 1, "history": parsed, "total_adjustments": len(parsed)}
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to read config_adjustments.json: %s", exc)
            data = {"version": 1, "history": [], "total_adjustments": 0}

        # Ensure structure
        if "history" not in data:
            data["history"] = []
        if "total_adjustments" not in data:
            data["total_adjustments"] = 0

        data["history"].extend(entries)
        data["total_adjustments"] = len(data["history"])
        data["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Atomic write: write to .tmp then rename
        tmp_path = adj_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.rename(adj_path)
        except OSError as exc:
            logger.error("Failed to write config_adjustments.json: %s", exc)

    # ------------------------------------------------------------------
    # Private: Reset to profile
    # ------------------------------------------------------------------

    def _reset_to_profile(self) -> None:
        """Reset all inputs to the selected profile's defaults."""
        self._staged_changes.clear()

        try:
            from src.scanner.config import ScannerConfig, SCAN_PROFILES
            config = ScannerConfig()
            config.apply_profile(self._active_profile)
        except Exception as exc:
            logger.debug("Failed to reset to profile: %s", exc)
            self._flash_status("ERROR: Profile reset failed", _NEGATIVE)
            return

        # Update all inputs from the fresh config
        for field_name, _label, default in _GATE_FIELDS:
            value = getattr(config, field_name, default)
            self._current_config[field_name] = value

            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.value = str(value)
                inp.styles.border = ("solid", _BORDER)
            except Exception:
                pass

        for field_name, _display_name in _AGENT_TOGGLES:
            value = getattr(config, field_name, True)
            self._current_config[field_name] = value

            try:
                switch = self.query_one(f"#toggle-{field_name}", Switch)
                switch.value = bool(value)
            except Exception:
                pass

        self._flash_status(f"Reset to {self._active_profile} profile", _PRIMARY)
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: Apply selected profile
    # ------------------------------------------------------------------

    def _apply_selected_profile(self) -> None:
        """Apply the selected profile via RadioSet and reload config values."""
        self._staged_changes.clear()

        try:
            from src.scanner.config import ScannerConfig
            config = ScannerConfig()
            config.apply_profile(self._active_profile)
        except Exception as exc:
            logger.debug("Failed to apply profile '%s': %s", self._active_profile, exc)
            self._flash_status(f"ERROR: Unknown profile '{self._active_profile}'", _NEGATIVE)
            return

        # Reload all inputs from the applied profile config
        for field_name, _label, default in _GATE_FIELDS:
            value = getattr(config, field_name, default)
            self._current_config[field_name] = value

            try:
                inp = self.query_one(f"#gate-{field_name}", Input)
                inp.value = str(value)
                inp.styles.border = ("solid", _BORDER)
            except Exception:
                pass

        for field_name, _display_name in _AGENT_TOGGLES:
            value = getattr(config, field_name, True)
            self._current_config[field_name] = value

            try:
                switch = self.query_one(f"#toggle-{field_name}", Switch)
                switch.value = bool(value)
            except Exception:
                pass

        # Record the profile switch in history
        self._append_change_history([{
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "field": "profile",
            "old": self._active_profile,
            "new": self._active_profile,
            "source": "TUI Profile Switch",
        }])

        self._load_change_history()
        self._flash_status(f"Applied profile: {self._active_profile.upper()}", _POSITIVE)
        self._update_status_label()

    # ------------------------------------------------------------------
    # Private: Load change history
    # ------------------------------------------------------------------

    def _load_change_history(self) -> None:
        """Load config change history from .claude/config_adjustments.json."""
        try:
            history_table = self.query_one("#config-history-table", DataTable)
        except Exception:
            return

        history_table.clear()

        adj_path = self._project_root / ".claude" / "config_adjustments.json"
        entries: list[dict[str, Any]] = []

        try:
            if adj_path.exists():
                raw = adj_path.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    entries = parsed.get("history", [])
                    if not isinstance(entries, list):
                        entries = []
                elif isinstance(parsed, list):
                    entries = parsed
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to read config_adjustments.json: %s", exc)
            entries = []

        # Show last 30, most recent first
        recent = list(reversed(entries[-30:]))

        for entry in recent:
            if not isinstance(entry, dict):
                continue

            ts = str(entry.get("timestamp", "---"))
            # Truncate to readable format
            if len(ts) > 19:
                ts = ts[:19]

            field_name = str(entry.get("field", "---"))
            old_val = entry.get("old", "---")
            new_val = entry.get("new", "---")
            source = str(entry.get("source", "---"))

            # Format values for display
            old_str = self._format_value(old_val)
            new_str = self._format_value(new_val)

            # Styled cells
            ts_text = Text(ts, style=_DIM)
            field_text = Text(field_name, style=_TEXT)
            old_text = Text(old_str, style=_NEGATIVE)
            new_text = Text(new_str, style=_POSITIVE)
            source_text = Text(source, style=_DATA)

            history_table.add_row(ts_text, field_text, old_text, new_text, source_text)

        # If no entries, show a placeholder row
        if not recent:
            history_table.add_row(
                Text("---", style=_DIM),
                Text("No config changes recorded", style=_DIM),
                Text("---", style=_DIM),
                Text("---", style=_DIM),
                Text("---", style=_DIM),
            )

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------

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
                    f"  [{_DIM}]Press SAVE to apply[/]"
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
