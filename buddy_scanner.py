"""BuddyScanner shim – wraps src.scanner.Scanner for CLI compatibility."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, List, Optional

from src.scanner import Scanner, ScannerConfig, ScanResult, PairAnalysis, ScannerDisplay


class BuddyScanner:
    """Thin wrapper around src.scanner.Scanner with the interface expected by cli/buddy_scanning.py."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        account_equity: Optional[float] = None,
        use_rl_sizer: bool = True,
    ):
        cfg = ScannerConfig()
        if config_path:
            from pathlib import Path
            cfg.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        if account_equity is not None:
            cfg.account_equity = account_equity
        cfg.use_rl_sizer = use_rl_sizer
        cfg.non_interactive = True  # CLI never prompts
        self._scanner = Scanner(config=cfg)
        self._display = ScannerDisplay()

    def _render_clean_output(
        self,
        analyses: List[PairAnalysis],
        *,
        model_type: str,
        granularity: str,
    ) -> None:
        """Render compact Codex-style scanner output with explicit reasoning."""
        c = self._display.console
        planner_orange = "color(215)"
        planner_sand = "color(223)"
        planner_cyan = "color(117)"
        planner_slate = "color(246)"
        c.print()
        c.print(
            f"[bold {planner_orange}]BUDDY PLANNER SCAN[/bold {planner_orange}]  "
            f"[{planner_cyan}]{model_type}[/{planner_cyan}] | [{planner_sand}]{granularity}[/{planner_sand}]"
        )
        c.print(f"[{planner_slate}]Pairs shown: {len(analyses)}[/{planner_slate}]")
        c.print()

        for idx, a in enumerate(analyses, start=1):
            pair = a.pair.replace("_", "/")
            direction = (a.direction or "HOLD").upper()
            session_blocked = bool(a.error and str(a.error).lower().startswith("outside trading session"))
            status = "TRADEABLE" if a.gates_passed else ("SESSION" if session_blocked else ("ERROR" if a.error else "WATCH"))
            conf = int(round(float(a.confidence) * 100))
            status_style = {
                "TRADEABLE": planner_cyan,
                "SESSION": planner_sand,
                "ERROR": "red",
                "WATCH": planner_slate,
            }.get(status, planner_slate)
            direction_style = planner_cyan if direction in {"LONG", "SHORT"} else planner_slate
            c.print(
                f"[{planner_slate}]{idx}.[/{planner_slate}] [bold {planner_sand}]{pair}[/bold {planner_sand}]  "
                f"[{direction_style}]{direction}[/{direction_style}]  "
                f"[{planner_orange}]{conf}%[/{planner_orange}]  "
                f"[{status_style}][{status}][/{status_style}]"
            )

            if a.error:
                c.print(f"   [{planner_slate}]why:[/{planner_slate}] [{planner_sand}]{a.error}[/{planner_sand}]")
                continue

            m_gate = "✓" if a.momentum_passed else "✗"
            a_gate = "✓" if a.confidence_passed else "✗"
            r_gate = "✓" if a.risk_passed else "✗"
            if a.agent_total > 0:
                agent_state = "confirmed" if a.agent_passed else "weak"
                agent_text = f"{agent_state} ({a.agent_votes}/{a.agent_total})"
            else:
                agent_text = "n/a"

            master = a.master_pair.replace("_", "/") if a.master_pair else pair
            c.print(
                f"   [{planner_slate}]why:[/{planner_slate}] gates M{m_gate} A{a_gate} R{r_gate} ({a.gate_summary}), "
                f"agent {agent_text}, master [{planner_sand}]{master}[/{planner_sand}]"
            )

            if a.gates_passed:
                promoted = " [agent-promoted]" if getattr(a, "agent_promoted", False) else ""
                c.print(
                    f"   [{planner_slate}]plan:[/{planner_slate}] "
                    f"[{planner_cyan}]SL {a.sl_pips:.0f} | TP {a.tp_pips:.0f}{promoted}[/{planner_cyan}]"
                )

        tradeable = [a.pair.replace("_", "/") for a in analyses if a.gates_passed]
        c.print()
        if tradeable:
            c.print(f"[{planner_cyan}]Tradeable:[/{planner_cyan}] [{planner_sand}]{', '.join(tradeable)}[/{planner_sand}]")
        else:
            c.print(f"[{planner_slate}]No tradeable setups.[/{planner_slate}]")

    # ------------------------------------------------------------------
    def scan(
        self,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
        top_n: int = 5,
        verbose: bool = True,
        prompt_train: bool = True,
        diversified: bool = False,
        force: bool = False,
        profile: str = "balanced",
        clean_output: bool = False,
        **kwargs: Any,
    ) -> List[PairAnalysis]:
        """Run scan and return a list of PairAnalysis objects."""
        # Apply per-call overrides expected by CLI flags.
        prev_granularity = self._scanner.config.granularity
        prev_session_filter = self._scanner.config.enable_session_filter
        profile_fields = (
            "profile",
            "min_confidence",
            "min_momentum",
            "max_drawdown_pct",
            "min_atr_pips",
            "min_volatility_regime",
            "use_tcn_volatility_filter",
        )
        prev_profile_values = {
            name: getattr(self._scanner.config, name)
            for name in profile_fields
        }
        prev_gate_min_regime = None
        if self._scanner._gate_evaluator is not None:
            prev_gate_min_regime = self._scanner._gate_evaluator.min_volatility_regime
        self._scanner.config.granularity = granularity
        self._scanner.config.apply_profile(profile)
        if self._scanner._gate_evaluator is not None:
            self._scanner._gate_evaluator.min_volatility_regime = self._scanner.config.min_volatility_regime
        if force:
            self._scanner.config.enable_session_filter = False

        try:
            result: ScanResult = self._scanner.scan(pairs=pairs)
        finally:
            # Restore mutable config to keep wrapper re-entrant for repeated calls.
            self._scanner.config.granularity = prev_granularity
            self._scanner.config.enable_session_filter = prev_session_filter
            for name, value in prev_profile_values.items():
                setattr(self._scanner.config, name, value)
            if self._scanner._gate_evaluator is not None and prev_gate_min_regime is not None:
                self._scanner._gate_evaluator.min_volatility_regime = prev_gate_min_regime

        # Sort by overall score and take top_n
        analyses = sorted(result.analyses, key=lambda a: a.overall_score, reverse=True)
        if top_n:
            analyses = analyses[:top_n]

        # Display results using the rich display
        if verbose:
            display_result = ScanResult(
                analyses=analyses,
                model_type=result.model_type,
                granularity=granularity,
            )
            if clean_output:
                self._render_clean_output(
                    analyses=analyses,
                    model_type=result.model_type,
                    granularity=granularity,
                )
            else:
                self._display.show_result(display_result)

        return analyses


@contextlib.contextmanager
def suppress_logging(level: int = logging.WARNING):
    """Context manager that temporarily raises the root log level."""
    root = logging.getLogger()
    prev = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(prev)
