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
        cfg.block_when_market_closed = True
        self._scanner = Scanner(config=cfg)
        self._display = ScannerDisplay()

    @staticmethod
    def _approval_holdback_reason(analysis: PairAnalysis) -> Optional[str]:
        """Summarize why a directional setup did not clear full approval."""
        if analysis.error:
            return None
        if analysis.is_tradeable:
            return None

        why_no_trade = list(getattr(analysis, "why_no_trade", []) or [])
        if why_no_trade:
            return str(why_no_trade[0])

        if bool(getattr(analysis, "blocked_by_circuit_breaker", False)):
            breakers = list(getattr(analysis, "circuit_breakers_triggered", []) or [])
            if breakers:
                return f"circuit breaker ({breakers[0]})"
            return "circuit breaker"

        if not bool(getattr(analysis, "volatility_gate_passed", True)):
            regime = str(getattr(analysis, "volatility_regime", "UNKNOWN") or "UNKNOWN").lower()
            return f"volatility regime ({regime})"

        if not bool(getattr(analysis, "execution_quality_passed", True)):
            return "execution quality"

        if analysis.momentum_passed and analysis.confidence_passed and analysis.risk_passed:
            return "advanced approval filter"

        return None

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
            market_closed = bool(a.error and str(a.error).lower().startswith("fx market closed"))
            status = "TRADEABLE" if a.is_tradeable else (
                "CLOSED" if market_closed else ("SESSION" if session_blocked else ("ERROR" if a.error else "WATCH"))
            )
            conf = int(round(float(a.confidence) * 100))
            status_style = {
                "TRADEABLE": planner_cyan,
                "CLOSED": planner_sand,
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
                agent_text = "not-run"

            # MTF confluence tag
            mtf_reason = next((ar for ar in (a.agent_reasons or []) if ar.get("name") == "multi_timeframe"), None)
            mtf_tag = ""
            if mtf_reason:
                mtf_count = mtf_reason.get("metadata", {}).get("confluence_count", 0)
                mtf_tag = f", MTF {mtf_count}/3"

            master = a.master_pair.replace("_", "/") if a.master_pair else pair
            c.print(
                f"   [{planner_slate}]why:[/{planner_slate}] core gates M{m_gate} A{a_gate} R{r_gate} ({a.gate_summary}), "
                f"agent {agent_text}{mtf_tag}, master [{planner_sand}]{master}[/{planner_sand}]"
            )

            if a.is_tradeable:
                promoted = " [agent-promoted]" if getattr(a, "agent_promoted", False) else ""
                soft_tag = ""
                for reason in (a.agent_reasons or []):
                    if reason.get("reason_code") == "uncertainty_soft_penalty":
                        soft_tag = " [soft-penalized]"
                        break
                c.print(
                    f"   [{planner_slate}]plan:[/{planner_slate}] "
                    f"[{planner_cyan}]SL {a.sl_pips:.0f} | TP {a.tp_pips:.0f}{promoted}{soft_tag}[/{planner_cyan}]"
                )
            else:
                holdback_reason = self._approval_holdback_reason(a)
                if holdback_reason:
                    c.print(
                        f"   [{planner_slate}]holdback:[/{planner_slate}] "
                        f"[{planner_sand}]{holdback_reason}[/{planner_sand}]"
                    )

        tradeable = [a.pair.replace("_", "/") for a in analyses if a.is_tradeable]
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
        run_agent_confirmation: bool = False,
        session_filter_enabled_override: Optional[bool] = None,
        session_window_utc_override: Optional[tuple[int, int]] = None,
        clean_output: bool = False,
        **kwargs: Any,
    ) -> List[PairAnalysis]:
        """Run scan and return a list of PairAnalysis objects."""
        # Apply per-call overrides expected by CLI flags.
        prev_granularity = self._scanner.config.granularity
        prev_session_filter = self._scanner.config.enable_session_filter
        profile_fields = (
            "profile",
            "min_tcn_probability",
            "min_confidence",
            "min_momentum",
            "max_drawdown_pct",
            "final_score_threshold",
            "max_uncertainty_std",
            "min_atr_pips",
            "min_volatility_regime",
            "use_tcn_volatility_filter",
            "weighted_vote_threshold",
            "sub_inference_min_confidence",
            "sub_inference_vote_threshold",
            "sub_inference_max_candidates",
            "agent_promotion_min_confidence",
            "max_uncertainty_score",
            "max_model_disagreement",
            "soft_uncertainty_blocking",
            "use_rl_sizer",
            "use_rl_gates",
            "use_rl_exits",
            "enable_agent_trade_promotion",
            "atr_sl_multiplier",
            "atr_tp_multiplier",
            "min_sl_pips",
            "max_sl_pips",
            "min_tp_pips",
            "max_tp_pips",
            "high_prob_threshold",
            "high_prob_tp_bonus",
            "enable_multi_timeframe_agent",
            "enable_pair_performance_agent",
            "min_risk_reward_ratio",
        )
        prev_profile_values = {
            name: getattr(self._scanner.config, name)
            for name in profile_fields
        }
        prev_gate_min_regime = None
        if self._scanner._gate_evaluator is not None:
            prev_gate_min_regime = self._scanner._gate_evaluator.min_volatility_regime
        prev_sub_inference_tradeable_only = self._scanner.config.sub_inference_tradeable_only
        prev_enable_sub_inference_agents = self._scanner.config.enable_sub_inference_agents
        prev_session_start_utc = self._scanner.config.session_start_utc
        prev_session_end_utc = self._scanner.config.session_end_utc
        self._scanner.config.granularity = granularity
        self._scanner.config.apply_profile(profile)
        if self._scanner._gate_evaluator is not None:
            self._scanner._gate_evaluator.min_volatility_regime = self._scanner.config.min_volatility_regime
        if run_agent_confirmation:
            # Allow confirmation agents to review directional watchlist setups.
            self._scanner.config.enable_sub_inference_agents = True
            self._scanner.config.sub_inference_tradeable_only = False
        if session_filter_enabled_override is not None:
            self._scanner.config.enable_session_filter = bool(session_filter_enabled_override)
        if session_window_utc_override is not None:
            self._scanner.config.session_start_utc = int(session_window_utc_override[0])
            self._scanner.config.session_end_utc = int(session_window_utc_override[1])
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
            self._scanner.config.sub_inference_tradeable_only = prev_sub_inference_tradeable_only
            self._scanner.config.enable_sub_inference_agents = prev_enable_sub_inference_agents
            self._scanner.config.session_start_utc = prev_session_start_utc
            self._scanner.config.session_end_utc = prev_session_end_utc
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


# ---------------------------------------------------------------------------
# CLI: subcommand dispatcher (Phase 96 — homework subcommand bootstrap).
#
# `buddy_scanner.py` historically functioned only as a library shim. The
# homework subsystem (Phase 96) needs a small operator-facing CLI so we can
# bootstrap the inbox from the existing trade journal without dragging in the
# full main.py argparse tree. We register only the homework subcommand here;
# all other CLI flows continue to live in main.py.
# ---------------------------------------------------------------------------


def _normalize_trade(trade: dict) -> tuple[dict, dict] | None:
    """Return a (trade, outcome) pair in the shape the HomeworkGenerator expects.

    Two journal shapes exist in the wild:

    1. **Nested-shape** (test fixtures, future schema): `outcome` is a dict
       with keys like `close_reason`, `close_time`, `realized_pl`, …; `agents`
       is a list of `{name, passed, score, weight, …}` dicts.

    2. **Flat-shape** (legacy production journal at trained_data/...): `outcome`
       is a string ("win"/"loss"); `close_reason`, `close_time`, `realized_pl`
       are promoted to the trade's top level; `agents` is a dict containing
       `agent_reasons` (the actual list of per-agent verdicts).

    Returns None if the trade is not closed (no `close_reason`).
    """
    if not isinstance(trade, dict):
        return None

    outcome_raw = trade.get("outcome")

    # Shape 1: nested outcome dict.
    if isinstance(outcome_raw, dict):
        if not outcome_raw.get("close_reason"):
            return None
        return trade, outcome_raw

    # Shape 2: flat journal — synthesize nested outcome and re-shape agents.
    close_reason = trade.get("close_reason")
    if not close_reason:
        return None

    outcome = {
        "close_time": trade.get("close_time", ""),
        "close_price": float(trade.get("close_price", trade.get("fill_price", 0.0)) or 0.0),
        "realized_pl": float(trade.get("realized_pl", 0.0) or 0.0),
        "close_reason": close_reason,
        "duration_minutes": int(trade.get("duration_minutes", 0) or 0),
        "mfe_pips": float(trade.get("mfe_pips", 0.0) or 0.0),
        "mae_pips": float(trade.get("mae_pips", 0.0) or 0.0),
    }

    # Build a shallow-copied trade with `agents` reshaped to list-of-dicts and
    # weighted_vote_score promoted to top-level (the generator reads it from
    # the trade record, not from the agents block).
    agents_raw = trade.get("agents")
    agents_list: list[dict] = []
    weighted_vote_score = float(trade.get("weighted_vote_score", 0.0) or 0.0)
    if isinstance(agents_raw, dict):
        agents_list = list(agents_raw.get("agent_reasons", []) or [])
        if "weighted_vote_score" in agents_raw:
            weighted_vote_score = float(agents_raw.get("weighted_vote_score") or 0.0)
    elif isinstance(agents_raw, list):
        agents_list = agents_raw

    # Regime field in the production journal is a dict; the generator stores
    # it as a string. Reduce to volatility_regime label.
    regime_raw = trade.get("regime")
    if isinstance(regime_raw, dict):
        regime_label = str(regime_raw.get("volatility_regime", "UNKNOWN"))
    else:
        regime_label = str(regime_raw or "UNKNOWN")

    normalized_trade = dict(trade)
    normalized_trade["agents"] = agents_list
    normalized_trade["weighted_vote_score"] = weighted_vote_score
    normalized_trade["regime"] = regime_label

    return normalized_trade, outcome


def _cmd_homework(args) -> int:
    """Handle the 'homework' subcommand."""
    import json
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    from src.scanner.automation.homework import (
        HomeworkGenerator,
        HomeworkStore,
    )

    if not args.generate_batch:
        print("Use --generate-batch to produce homework entries.")
        return 1

    journal_path = Path(
        os.environ.get(
            "BUDDY_TRADE_JOURNAL_PATH",
            "trained_data/trade_journal_rl.json",
        )
    )
    pending_path = (
        Path(os.environ["BUDDY_HOMEWORK_PENDING_PATH"])
        if os.environ.get("BUDDY_HOMEWORK_PENDING_PATH")
        else None
    )
    history_path = (
        Path(os.environ["BUDDY_HOMEWORK_HISTORY_PATH"])
        if os.environ.get("BUDDY_HOMEWORK_HISTORY_PATH")
        else None
    )

    if not journal_path.exists():
        print(f"Trade journal not found: {journal_path}", file=sys.stderr)
        return 2

    try:
        journal = json.loads(journal_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"Trade journal is not valid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(journal, list):
        print("Journal must be a JSON list of trade entries.", file=sys.stderr)
        return 2

    store = HomeworkStore(pending_path=pending_path, history_path=history_path)
    already_graded_ids = {e.trade_id for e in store.list_history()}
    already_pending_ids = {e.trade_id for e in store.list_pending()}

    candidates: list[tuple[dict, dict]] = []
    for raw_trade in journal:
        normalized = _normalize_trade(raw_trade)
        if normalized is None:
            continue
        trade, outcome = normalized
        tid = str(trade.get("trade_id", ""))
        if not args.include_graded and tid in already_graded_ids:
            continue
        if tid in already_pending_ids:
            continue
        if args.since:
            try:
                close_dt = datetime.fromisoformat(
                    str(outcome.get("close_time", "")).replace("Z", "+00:00")
                )
                since_dt = datetime.fromisoformat(args.since)
                if close_dt < since_dt:
                    continue
            except Exception:
                # Malformed timestamp shouldn't kill the whole batch.
                pass
        candidates.append((trade, outcome))  # noqa: PERF401 — explicit pair

    if args.last is not None:
        candidates = candidates[-int(args.last):]

    gen = HomeworkGenerator()
    generated = 0
    for trade, outcome in candidates:
        try:
            entry = gen.generate(trade, outcome)
            store.add(entry)
            generated += 1
            preview = (entry.proposed_lesson or "").strip().replace("\n", " ")[:60]
            print(
                f"  ✓ #{trade.get('trade_id')} {trade.get('pair')} "
                f"{trade.get('direction')} → {outcome.get('close_reason')}  ({preview})"
            )
        except Exception as exc:
            print(
                f"  ✗ #{trade.get('trade_id', '?')}: generation failed — {exc}",
                file=sys.stderr,
            )

    print(f"\nGenerated {generated} homework entries from {len(candidates)} candidates.")
    print(f"Pending file: {store.pending_path}")
    print("Open the F2 Inbox in the TUI to review.")
    return 0


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="buddy_scanner.py",
        description="Buddy scanner CLI — operator entrypoints for the homework subsystem.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    homework_p = subparsers.add_parser(
        "homework",
        help="Generate / manage trade homework entries (Phase 96)",
    )
    homework_p.add_argument(
        "--generate-batch",
        action="store_true",
        help="Generate homework for closed trades from the journal",
    )
    homework_p.add_argument(
        "--last",
        type=int,
        default=None,
        help="Only the last N closed trades (default: all unstudied)",
    )
    homework_p.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only trades closed since YYYY-MM-DD",
    )
    homework_p.add_argument(
        "--include-graded",
        action="store_true",
        help="Include trades that already have homework history",
    )

    return parser


def _main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "homework":
        return _cmd_homework(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(_main())
