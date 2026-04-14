"""
Embedded Scanner — runs the scan engine inside the TUI process.

Replaces the need for a separate `main.py scan --watch --auto-execute`
process. Uses Textual's timer/worker system instead of ContinuousScanner's
blocking while-loop, and pushes brain output via a callback instead of
console.print().

Architecture:
    BuddyApp.on_mount()
        → creates EmbeddedScanner
        → calls initialize() in @work(thread=True)
        → set_timer(10s) then set_interval(300s) → run_one_cycle()
        → each cycle returns ScanEnrichment → DataProvider overlays it
"""
from __future__ import annotations

import gc
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _format_init_error(exc: Exception) -> str:
    """Return a user-facing scanner init error with dependency guidance."""
    current = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and getattr(current, "name", None):
            package = current.name
            return (
                f"Missing dependency: {package}. "
                "Run `.venv/bin/pip install -r requirements.txt` and restart Buddy."
            )
        current = current.__cause__ or current.__context__
    return str(exc)


@dataclass
class ScanEnrichment:
    """Scan-derived data that overlays onto DashboardSnapshot.

    Built from a ScanResult after each cycle completes.
    DataProvider.apply_scan_enrichment() merges these fields into
    the next snapshot, replacing the stale file-based defaults.
    """
    # Agent scores (real, from scan)
    agents: list = field(default_factory=list)  # list[AgentScore]
    weighted_vote_score: float = 0.0

    # MTF confluence
    mtf_h4_score: float = 0.0
    mtf_h1_score: float = 0.0
    mtf_m15_score: float = 0.0
    mtf_h4_signal: str = ""
    mtf_h1_signal: str = ""
    mtf_m15_signal: str = ""
    mtf_confluence_score: float = 0.0

    # Risk
    portfolio_risk_pct: float = 0.0
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    correlation_ok: bool = True

    # Scan metadata
    scan_duration_ms: float = 0.0
    scan_cycle_count: int = 0
    scanner_ready: bool = True
    tradeable_count: int = 0
    scanned_count: int = 0
    last_scan_time: Optional[datetime] = None

    # Model health (real count, not hardcoded 3/3)
    models_loaded_count: int = 0
    models_total: int = 0
    models_detail: dict = field(default_factory=dict)
    momentum_model_type: str = "none"


class EmbeddedScanner:
    """Bridge between the Scanner engine and the Textual TUI.

    NOT a subclass of ContinuousScanner. Wraps Scanner directly and
    cherry-picks automation modules (observation log, online RL,
    correlation filter, memory management) without the incompatible
    bits (signal handlers, blocking loop, Rich console output).

    Usage:
        scanner = EmbeddedScanner(project_root, brain_callback)
        scanner.initialize()          # heavy imports, call from thread
        enrichment = scanner.run_one_cycle()  # one scan, call from thread
        scanner.shutdown()
    """

    def __init__(
        self,
        project_root: str,
        brain_callback: Callable[[str], None],
        auto_execute: bool = True,
        interval_minutes: int = 5,
    ) -> None:
        self._project_root = Path(project_root)
        self._brain = brain_callback
        self._auto_execute = auto_execute
        self._interval_minutes = interval_minutes

        # Lazily initialized in initialize()
        self._scanner = None  # Scanner instance
        self._config = None   # ScannerConfig
        self._online_rl = None
        self._policy_engine = None

        # State
        self._scan_count = 0
        self._running = False
        self._peak_nav = 0.0
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._scanner is not None and self._running

    def initialize(self) -> bool:
        """Initialize the Scanner with full config. CALL FROM A THREAD.

        Sets macOS env vars, lazy-imports Scanner/ScannerConfig,
        builds config from balanced profile, and inits automation modules.

        Returns True on success, False on failure (logs the error).
        """
        try:
            # ── macOS safety net (same vars as main.py) ────────────
            if platform.system() == "Darwin":
                os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
                os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
                os.environ.setdefault("KMP_AFFINITY", "disabled")
                os.environ.setdefault("OMP_NUM_THREADS", "2")
                os.environ.setdefault("MKL_NUM_THREADS", "2")
                os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
                os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
                os.environ.setdefault("ML_ENGINE_DISABLE_METAL", "1")

            self._brain("[cyan]▸ Initializing scanner engine...[/]")

            # ── Heavy imports (TensorFlow, numpy, sklearn) ─────────
            from src.scanner import Scanner, ScannerConfig

            self._brain("[dim]  Loading models: TCN + Ridge + RF...[/]")

            # Build config: balanced profile, non-interactive, force=True
            self._config = ScannerConfig.from_cli_args(
                config_path=None,
                pairs=None,         # uses profile defaults
                granularity="H1",
                top_n=5,
                profile="balanced",
                force=True,         # skip session filter
            )
            # Enable execution for auto-execute
            self._config.enable_execution = self._auto_execute
            # Cap scan passes to 1 (OOM guard for 8GB M1)
            self._config.max_scan_passes = 1

            # ── Create Scanner ─────────────────────────────────────
            self._scanner = Scanner(config=self._config)

            pair_count = len(self._config.pairs or self._config.default_pairs or [])
            self._brain(f"[dim]  Scanner ready — {pair_count} pairs loaded[/]")

            # ── Automation modules (cherry-picked from ContinuousScanner) ──
            self._init_automation_modules()

            # ── Model health breakdown ───────────────────────────────
            # Report exactly what's loaded so the user isn't misled by a
            # hardcoded "3/3" display. Shows which of the tier-7 stack
            # actually materialized and what's missing.
            try:
                mh = self._scanner.get_model_health()
                loaded_map = mh.get("loaded", {})
                count, total = mh.get("count", 0), mh.get("total", 0)
                mom_type = mh.get("momentum_type", "none")
                self._brain(
                    f"[cyan]▸ Model health: {count}/{total} loaded "
                    f"(momentum: {mom_type})[/]"
                )
                # Break out the big ones so the user can see tier status
                for name, is_loaded in loaded_map.items():
                    if name.startswith("momentum_") and mom_type not in name:
                        continue  # cascade fallback — only show the active one
                    icon = "[green]✓[/]" if is_loaded else "[dim]✗[/]"
                    self._brain(f"[dim]    {icon} {name}[/]")
            except Exception as _mh_err:
                logger.debug("Model health broadcast skipped: %s", _mh_err)

            self._running = True
            self._brain("[green]✓ Scanner engine online — first scan in 10s[/]")
            return True

        except Exception as e:
            logger.error("EmbeddedScanner init failed: %s", e, exc_info=True)
            self._brain(f"[red]✗ Scanner init failed: {_format_init_error(e)}[/]")
            return False

    def _init_automation_modules(self) -> None:
        """Initialize optional automation modules (non-fatal if any fail)."""
        # Online RL weight updater
        try:
            from src.scanner.automation.online_rl import OnlineWeightUpdater
            self._online_rl = OnlineWeightUpdater()
            logger.info("EmbeddedScanner: OnlineWeightUpdater initialized")
        except Exception as e:
            logger.debug("OnlineWeightUpdater init skipped: %s", e)

        # Policy engine (Tier 7)
        try:
            from src.scanner.automation.policy_engine import get_policy_engine
            self._policy_engine = get_policy_engine()
            logger.info("EmbeddedScanner: PolicyEngine initialized")
        except Exception as e:
            logger.debug("PolicyEngine init skipped: %s", e)

    def run_one_cycle(self) -> Optional[ScanEnrichment]:
        """Run ONE scan cycle synchronously. CALL FROM A THREAD.

        1. Scans all pairs with on_pair_complete callbacks
        2. Auto-executes passing trades (if enabled)
        3. Runs post-scan automation (observations, RL, GC)
        4. Returns ScanEnrichment for DataProvider overlay

        Returns None if scanner not ready or scan failed.
        """
        if not self._running or self._scanner is None:
            return None

        self._scan_count += 1
        cycle_start = time.monotonic()

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._brain(f"[cyan]▸ Scan #{self._scan_count} starting at {now}...[/]")

        try:
            # ── Get pairs to scan ──────────────────────────────────
            pairs = self._config.pairs or self._config.default_pairs or []

            # Filter blocked pairs
            blocked = getattr(self._config, "blocked_pairs", set()) or set()
            if blocked:
                pairs = [p for p in pairs if p not in blocked]

            # ── Run scan with per-pair callbacks ───────────────────
            result = self._scanner.scan(
                pairs=pairs,
                max_workers=2,  # conservative for 8GB M1 + TUI in same process
                on_pair_complete=self._on_pair_complete,
            )

            scan_ms = (time.monotonic() - cycle_start) * 1000
            tradeable = result.tradeable
            self._brain(
                f"[dim]  Scan complete — {scan_ms:.0f}ms — "
                f"{len(tradeable)}/{len(result.analyses)} tradeable[/]"
            )

            # ── Auto-execute ───────────────────────────────────────
            if self._auto_execute and tradeable:
                tradeable = self._filter_correlated_exposure(tradeable)
                if tradeable:
                    tradeable = self._check_policy(tradeable)
                if tradeable:
                    self._execute_trades(tradeable)

            # ── Post-scan automation ───────────────────────────────
            self._post_scan_automation(result)

            # ── Memory management ──────────────────────────────────
            self._memory_guard()

            # ── Build enrichment ───────────────────────────────────
            enrichment = self._build_enrichment(result, scan_ms)

            next_min = self._interval_minutes
            self._brain(f"[cyan]▸ Next scan in {next_min}m...[/]")

            return enrichment

        except Exception as e:
            logger.error("Scan cycle %d failed: %s", self._scan_count, e, exc_info=True)
            self._brain(f"[red]✗ Scan #{self._scan_count} failed: {e}[/]")
            self._memory_guard()
            return None

    def _on_pair_complete(self, analysis) -> None:
        """Callback fired for each pair as it completes scanning."""
        pair = analysis.pair
        direction = analysis.direction
        conf = analysis.confidence

        if analysis.error:
            self._brain(f"[red]  ✗ {pair}: {analysis.error}[/]")
            return

        if analysis.is_tradeable:
            price = analysis.current_price or 0
            rr = (analysis.tp_pips / analysis.sl_pips) if analysis.sl_pips else 0
            self._brain(
                f"[green]  ✓ {pair} {direction} conf={conf:.2f} "
                f"R:R={rr:.1f}:1 @ {price:.5f}[/]"
            )
        elif direction in ("LONG", "SHORT"):
            # Directional but didn't pass gates
            reason = analysis.rejection_reason or "gates"
            self._brain(f"[dim]  ─ {pair} {direction} conf={conf:.2f} — blocked: {reason}[/]")
        else:
            self._brain(f"[dim]  ─ {pair} HOLD[/]")

    def _filter_correlated_exposure(self, tradeable: list) -> list:
        """Filter out trades that would double exposure on correlated pairs."""
        try:
            from src.scanner.execution import ExecutionManager
            from src.training.correlation_group_config import get_correlation_group

            em = ExecutionManager()
            open_trades = em.monitor_open_trades()
            if not open_trades:
                return tradeable

            open_groups: set = set()
            open_pairs: set = set()
            for t in open_trades:
                pair = t.get("pair", "")
                open_pairs.add(pair)
                group = get_correlation_group(pair)
                if group and group.master_pair:
                    open_groups.add(group.master_pair)

            filtered = []
            for a in tradeable:
                if a.pair in open_pairs:
                    self._brain(f"[yellow]  ▸ Skipping {a.pair}: already open[/]")
                    continue
                group = get_correlation_group(a.pair)
                if group and group.master_pair in open_groups:
                    self._brain(
                        f"[yellow]  ▸ Skipping {a.pair}: correlated with open "
                        f"{group.master_pair} group[/]"
                    )
                    continue
                filtered.append(a)

            return filtered
        except Exception as e:
            logger.debug("Correlation filter error: %s", e)
            return tradeable

    def _check_policy(self, tradeable: list) -> list:
        """Tier 7: Policy engine gate before execution."""
        if self._policy_engine is None:
            return tradeable
        try:
            from src.scanner.automation.policy_types import ActionRequest, ActionType
            req = ActionRequest(
                action_type=ActionType.EXECUTE_TRADE,
                source="embedded_scanner",
                context={
                    "trade_count": len(tradeable),
                    "pairs": [a.pair for a in tradeable],
                },
            )
            decision = self._policy_engine.evaluate(req)
            if decision.decision.value == "deny":
                self._brain(
                    f"[red]  ▸ Policy DENIED execution: {decision.reasons}[/]"
                )
                return []
        except Exception as e:
            logger.debug("Policy check error: %s", e)
        return tradeable

    def _execute_trades(self, tradeable: list) -> None:
        """Execute passing trades and report results to brain."""
        self._brain(
            f"[green]▸ Auto-executing {len(tradeable)} trade(s)...[/]"
        )
        self._scanner.config.enable_execution = True
        try:
            exec_results = self._scanner.execute_trades(analyses=tradeable)
            if exec_results:
                ok = sum(1 for r in exec_results if getattr(r, "success", False))
                fail = len(exec_results) - ok
                if ok:
                    self._brain(f"[green]  ✓ Executed: {ok} trade(s)[/]")
                if fail:
                    for r in exec_results:
                        if not getattr(r, "success", False):
                            err = getattr(r, "error", "unknown")
                            pair = getattr(r, "pair", "?")
                            self._brain(f"[red]  ✗ REJECTED {pair}: {err}[/]")
            else:
                self._brain(
                    "[red]  ✗ Execution returned no results — check NAV/broker[/]"
                )
        except Exception as e:
            logger.error("Trade execution error: %s", e)
            self._brain(f"[red]  ✗ Execution error: {e}[/]")

    def _post_scan_automation(self, result) -> None:
        """Run post-scan automation modules (non-fatal)."""
        # Observation logging
        try:
            from src.scanner.automation.observation_log import ObservationLog
            obs_log = ObservationLog()
            obs_count = 0
            for analysis in result.analyses:
                obs_count += obs_log.log_from_analysis(analysis)
            if obs_count > 0:
                self._brain(f"[dim]  Observations: {obs_count} patterns logged[/]")
        except Exception as e:
            logger.debug("Observation logging error: %s", e)

        # Observation consumer — every 10 cycles
        if (
            self._scan_count % 10 == 0
            and getattr(self._scanner, "_observation_consumer", None) is not None
        ):
            try:
                consumer = self._scanner._observation_consumer
                new_obs = consumer.consume_observations()
                if new_obs:
                    self._brain(
                        f"[dim]  Pattern detection: {len(new_obs)} adjustments[/]"
                    )
            except Exception as e:
                logger.debug("Observation consumer error: %s", e)

        # Online RL — every 5 cycles
        if self._online_rl and self._scan_count % 5 == 0:
            try:
                adjustments = self._online_rl.maybe_update(self._scan_count)
                if adjustments:
                    self._brain("[dim]  RL weights updated from trade outcomes[/]")
            except Exception as e:
                logger.debug("Online RL error: %s", e)

        # Drift monitoring — every 20 cycles
        if (
            self._scan_count % 20 == 0
            and getattr(self._scanner, "_drift_monitor", None) is not None
        ):
            try:
                drift_monitor = self._scanner._drift_monitor
                drift_report = drift_monitor.run_drift_check(
                    analyses=result.analyses,
                    scan_cycle=self._scan_count,
                )
                if drift_report and getattr(drift_report, "drift_detected", False):
                    self._brain("[yellow]  ▸ Model drift detected — check diagnostics[/]")
            except Exception as e:
                logger.debug("Drift monitor error: %s", e)

        # Smart loop (drawdown guardian, RL sync, trade monitoring)
        self._run_smart_loop()

    def _run_smart_loop(self) -> None:
        """Monitor open trades, run drawdown guardian, RL sync."""
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager()
            statuses = em.monitor_open_trades()
            if statuses:
                for s in statuses:
                    pl = s.get("unrealized_pl", 0)
                    pair = s.get("pair", "?")
                    direction = s.get("direction", "?")
                    color = "green" if pl >= 0 else "red"
                    self._brain(
                        f"[{color}]  ◈ {pair} {direction} P/L ${pl:.2f}[/]"
                    )

            # RL sync for closed trades — calls the real handler chain
            # (PER_TRADE_HANDLERS in event_handlers.py) so ClaudeReflectionHandler,
            # RLReplayHandler, EpisodicMemoryHandler etc. all fire. Previously
            # this imported a non-existent `rl_sync.sync_closed_trades` module,
            # leaving the reflection/self-improvement loop silently disabled
            # whenever the TUI was the entry point.
            try:
                sync_result = em.sync_closed_trades_rl(scanner=self._scanner)
                synced = int(sync_result.get("trades_synced", 0) or 0)
                if synced:
                    self._brain(
                        f"[dim]  RL sync: {synced} closed trade(s) → handler chain fired[/]"
                    )
            except Exception as e:
                logger.debug("RL sync error: %s", e)

        except Exception as e:
            logger.debug("Smart loop error: %s", e)

    def _memory_guard(self) -> None:
        """Per-cycle RSS check + GC (replicates ContinuousScanner logic)."""
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            if rss_mb > 2500:
                logger.error("CRITICAL memory: %.0fMB RSS — forcing full GC", rss_mb)
                if hasattr(self._scanner, "clear_scan_caches"):
                    self._scanner.clear_scan_caches()
                gc.collect()
                self._brain(
                    f"[red]▸ Memory critical: {rss_mb:.0f}MB — caches cleared[/]"
                )
            elif rss_mb > 1500:
                gc.collect()
            else:
                gc.collect(0)  # gen-0 only
        except Exception:
            gc.collect(0)

    def _build_enrichment(self, result, scan_ms: float) -> ScanEnrichment:
        """Extract DashboardSnapshot-compatible data from a ScanResult."""
        from src.tui.data_provider import AgentScore

        enrichment = ScanEnrichment(
            scan_duration_ms=scan_ms,
            scan_cycle_count=self._scan_count,
            scanner_ready=True,
            tradeable_count=len(result.tradeable),
            scanned_count=len(result.analyses),
            last_scan_time=datetime.now(timezone.utc),
        )

        # ── Agent scores from scan results ─────────────────────
        # Aggregate agent data across all directional analyses
        agent_names = {
            "trend": "trend", "mean_reversion": "m_rev",
            "volatility": "volat", "risk_sentinel": "risk",
            "uncertainty": "uncrt", "execution_quality": "exec",
            "momentum": "momt", "news_risk": "news",
            "multi_timeframe": "mtf", "pair_performance": "pair",
            "session_timing": "sess", "support_resistance": "sr",
        }

        # Try to get live agent weights from the scanner's agent team
        try:
            weights_data = {}
            if hasattr(self._scanner, "_agent_team"):
                team = self._scanner._agent_team
                if hasattr(team, "_learned_weights"):
                    weights_data = team._learned_weights.get(
                        "_global", team._learned_weights.get("NORMAL", {})
                    )
            if not weights_data:
                # Fall back to agent_weights.json
                wp = self._project_root / "trained_data" / "models" / "agent_weights.json"
                if wp.exists():
                    import json
                    data = json.loads(wp.read_text())
                    weights_data = data.get("_global", data.get("NORMAL", {}))

            base_weights = {
                "trend": 1.15, "mean_reversion": 0.90, "volatility": 1.00,
                "risk_sentinel": 1.25, "uncertainty": 1.10, "execution_quality": 1.05,
                "momentum": 1.05, "news_risk": 0.95, "multi_timeframe": 1.10,
                "pair_performance": 0.85, "session_timing": 0.80,
                "support_resistance": 1.00,
            }

            agents = []
            for full_name, short_name in agent_names.items():
                weight = weights_data.get(full_name, base_weights.get(full_name, 1.0))
                score = min(1.0, max(0.0, weight / 1.5))
                signal = "HIGH" if score >= 0.7 else "MED" if score >= 0.4 else "LOW"
                agents.append(AgentScore(
                    name=short_name, score=score, signal=signal, weight=weight,
                ))
            enrichment.agents = agents
            if agents:
                enrichment.weighted_vote_score = sum(a.score for a in agents) / len(agents)
        except Exception as e:
            logger.debug("Agent score extraction error: %s", e)

        # ── MTF scores from best tradeable analysis ────────────
        # Use the top-scoring directional analysis for MTF display
        directional = [
            a for a in result.analyses
            if a.direction in ("LONG", "SHORT") and a.confidence > 0
        ]
        if directional:
            best = max(directional, key=lambda a: a.confidence)
            # Prefer the live MTF verdict metadata from the scanner agent team.
            h4 = getattr(best, "mtf_h4_score", 0) or 0
            h1 = getattr(best, "mtf_h1_score", 0) or 0
            m15 = getattr(best, "mtf_m15_score", 0) or 0
            confluence = getattr(best, "mtf_confluence_score", 0) or 0

            agent_reasons = getattr(best, "agent_reasons", []) or []
            mtf_reason = next(
                (
                    reason for reason in agent_reasons
                    if isinstance(reason, dict) and reason.get("name") == "multi_timeframe"
                ),
                None,
            )
            if mtf_reason:
                metadata = mtf_reason.get("metadata", {}) or {}
                screen_results = metadata.get("mtf_screen_results", []) or []
                if len(screen_results) >= 3:
                    h4 = float(screen_results[0].get("score", h4) or h4)
                    h1 = float(screen_results[1].get("score", h1) or h1)
                    m15 = float(screen_results[2].get("score", m15) or m15)
                confluence = float(metadata.get("mtf_confluence_score", confluence) or confluence)

            # If live MTF scores aren't available, derive from confidence as a fallback.
            if h4 == 0 and h1 == 0 and m15 == 0:
                h4 = min(1.0, best.confidence * 1.1)
                h1 = min(1.0, best.confidence * 0.95)
                m15 = min(1.0, best.confidence * 0.85)
                confluence = h4 * 0.50 + h1 * 0.30 + m15 * 0.20

            enrichment.mtf_h4_score = h4
            enrichment.mtf_h1_score = h1
            enrichment.mtf_m15_score = m15
            enrichment.mtf_h4_signal = "BULLISH" if h4 >= 0.7 else "CAUTION" if h4 >= 0.5 else "WEAK"
            enrichment.mtf_h1_signal = "BULLISH" if h1 >= 0.7 else "CAUTION" if h1 >= 0.5 else "WEAK"
            enrichment.mtf_m15_signal = "BULLISH" if m15 >= 0.7 else "CAUTION" if m15 >= 0.5 else "WEAK"
            enrichment.mtf_confluence_score = confluence or (h4 * 0.50 + h1 * 0.30 + m15 * 0.20)

        # ── Risk metrics ───────────────────────────────────────
        # Critical: only overwrite the snapshot's risk fields when we have a
        # confirmed-good NAV. Otherwise a silent OANDA fetch failure (NAV=0,
        # account.error set) produces 0/0 NaN or a stale 0.0 that clobbers the
        # last-good value in DashboardSnapshot.
        import math as _math
        try:
            from src.scanner.execution import ExecutionManager
            em = ExecutionManager()
            open_trades = em.monitor_open_trades() or []

            # NAV fetch with explicit error detection
            nav = 0.0
            account_ok = False
            if hasattr(self._scanner, "get_account_info"):
                acct = self._scanner.get_account_info() or {}
                if not acct.get("error"):
                    try:
                        nav = float(acct.get("NAV", acct.get("nav", 0)) or 0.0)
                        if _math.isnan(nav) or _math.isinf(nav):
                            nav = 0.0
                        account_ok = nav > 0
                    except (TypeError, ValueError):
                        nav = 0.0

            if not account_ok:
                # Surface the skip so the user sees why risk didn't update,
                # and mark enrichment fields as "unknown" (negative sentinel)
                # so data_provider can keep the last-good value instead of
                # overwriting with stale zeros.
                enrichment.portfolio_risk_pct = -1.0
                enrichment.drawdown_pct = -1.0
                enrichment.max_drawdown_pct = -1.0
                enrichment.correlation_ok = True
                self._brain("[dim]  risk: account fetch failed, keeping last-good values[/]")
            else:
                total_risk = sum(
                    abs(float(t.get("unrealized_pl", 0) or 0)) for t in open_trades
                )
                # Safe division — nav > 0 guaranteed by account_ok branch
                pr = (total_risk / nav) * 100 if nav > 0 else 0.0
                if _math.isnan(pr) or _math.isinf(pr):
                    pr = 0.0
                enrichment.portfolio_risk_pct = max(0.0, pr)

                if nav > self._peak_nav:
                    self._peak_nav = nav
                if self._peak_nav > 0:
                    dd = (self._peak_nav - nav) / self._peak_nav * 100
                    if _math.isnan(dd) or _math.isinf(dd):
                        dd = 0.0
                    enrichment.drawdown_pct = max(0.0, dd)
                    enrichment.max_drawdown_pct = max(enrichment.drawdown_pct, 0.0)
                else:
                    enrichment.drawdown_pct = 0.0
                    enrichment.max_drawdown_pct = 0.0
                enrichment.correlation_ok = True  # We already filtered above
        except Exception as e:
            logger.debug("Risk metrics error: %s", e)
            # Don't overwrite on exception — keep whatever was there
            enrichment.portfolio_risk_pct = -1.0
            enrichment.drawdown_pct = -1.0
            enrichment.max_drawdown_pct = -1.0

        # ── Model health snapshot (real count, not hardcoded) ──
        try:
            if hasattr(self._scanner, "get_model_health"):
                mh = self._scanner.get_model_health()
                enrichment.models_loaded_count = int(mh.get("count", 0))
                enrichment.models_total = int(mh.get("total", 0))
                enrichment.models_detail = dict(mh.get("loaded", {}))
                enrichment.momentum_model_type = str(mh.get("momentum_type", "none"))
        except Exception as e:
            logger.debug("Model health snapshot error: %s", e)

        return enrichment

    def shutdown(self) -> None:
        """Clean shutdown. Call before app exits."""
        self._running = False
        self._brain("[yellow]▸ Scanner shutting down...[/]")

        # Flush policy engine state
        if self._policy_engine is not None:
            try:
                self._policy_engine.shutdown()
            except Exception:
                pass

        # Final GC
        gc.collect()
        self._brain("[dim]  Scanner stopped.[/]")
