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
import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only (PEP 563 deferred): used by the `counters` kwarg below.
    from src.tui.widgets.stats_bar import ScanCounters

logger = logging.getLogger(__name__)


def _consume_config_dirty_flag(state_path: Path) -> bool:
    """Read `state.json:config_dirty`; if true, clear atomically and return True.

    Tier 2 T8: paired with `AdjustmentApprover._mark_config_dirty`. Called at
    the top of every `EmbeddedScanner.run_one_cycle` before any scan work
    — if true, the scanner reloads the `ConfigAdjuster` cache so newly
    approved adjustments take effect this cycle, not next-tick.

    Returns True iff the flag was true on entry. Best-effort: any read/write
    failure returns the safest answer (False = "don't trigger a reload this
    cycle"). The next ConfigAdjuster apply tick is the fallback consumer.

    Atomicity: clear-write uses tmp+rename so a concurrent reader cannot
    observe a half-written state file. If the clear-write itself fails after
    a successful read, we still return True so the caller reloads — better
    to reload twice than to miss the new approval entirely.
    """
    try:
        if not state_path.exists():
            return False
        data = json.loads(state_path.read_text())
        if not isinstance(data, dict):
            return False
    except (OSError, json.JSONDecodeError):
        return False

    if not data.get("config_dirty"):
        return False

    data["config_dirty"] = False
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(state_path)
    except OSError as e:
        logger.warning(
            "_consume_config_dirty_flag: clear-write failed path=%s err=%r — "
            "returning True anyway (reload once more is safer than missing "
            "the new approval)", state_path, e,
        )
    return True


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

    # US-513: Max age (days) over loaded ensemble component trained_at deltas.
    # 0.0 = unknown / fresh; > 7 triggers Overview red banner + uncertainty
    # hard-block tightening to 0.35.
    max_component_age_days: float = 0.0

    # Running ScannerConfig scalar snapshot for the Config tab.
    config_profile: str = "smart"
    config_values: dict = field(default_factory=dict)

    # US-002: Per-instrument ATR (pips) from the last scan cycle. Keyed by
    # OANDA instrument symbol (e.g. ``"EUR_USD"``). DataProvider.refresh()
    # pipes these into TradeRow.live_atr_pips during _refresh_trades so the
    # drill-down can show real model-driven ATR instead of the previous
    # back-derived ``sl_dist * 10000`` fake.
    atr_value: dict = field(default_factory=dict)


class EmbeddedScanner:
    """Bridge between the Scanner engine and the Textual TUI.

    NOT a subclass of ContinuousScanner. Wraps Scanner directly and
    cherry-picks automation modules (observation log, online RL,
    correlation filter, memory management) without the incompatible
    bits (signal handlers, blocking loop, Rich console output).

    Usage:
        scanner = EmbeddedScanner(brain_callback, project_root=root)
        scanner.initialize()          # heavy imports, call from thread
        enrichment = scanner.run_one_cycle()  # one scan, call from thread
        scanner.shutdown()
    """

    def __init__(
        self,
        brain_callback: Callable[[str], None],
        project_root: Optional[str] = None,
        auto_execute: bool = True,
        interval_minutes: int = 5,
        counters: Optional["ScanCounters"] = None,
    ) -> None:
        # Tier 1 T6: `project_root` is optional so test harnesses can
        # construct the scanner without wiring a real workspace path.
        # Production callers in src/tui/app.py still pass
        # `project_root=str(PROJECT_ROOT)` as a kwarg.
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._brain = brain_callback
        self._auto_execute = auto_execute
        self._interval_minutes = interval_minutes

        # Tier 1 T6: inline error banner. Non-gate exception paths (genuine
        # errors like FileNotFoundError, model load failures, OANDA
        # timeouts, generic Exception fallbacks) set this to a one-line
        # message. The TUI app.py watches the field on each periodic
        # refresh and surfaces it via Textual `notify(...)` with a
        # "Ctrl+L to view full log" hint. Gate-rejection paths leave
        # this untouched (gate failures are expected outcomes).
        self.error_banner: Optional[str] = None

        # Tier 1 T3: lifetime work-unit counters (cycles, pairs, gates, trades).
        # Shared ref handed to the TUI StatsBar; bumped inline in run_one_cycle
        # at the natural phase boundaries. The caller (BuddyApp) may inject a
        # pre-built ScanCounters so the same instance backs the F1 widget; if
        # omitted (tests, headless harnesses), a fresh one is created.
        from src.tui.widgets.stats_bar import ScanCounters
        self.counters = counters if counters is not None else ScanCounters()

        # Lazily initialized in initialize()
        self._scanner = None  # Scanner instance
        self._config = None   # ScannerConfig
        self._online_rl = None
        self._policy_engine = None
        self._config_adjuster = None
        self._reconciler = None
        self._reconciler_task = None
        self._reconciler_thread = None
        # The dedicated asyncio event loop the reconciler runs on. Captured so
        # shutdown() can stop it cross-thread (call_soon_threadsafe) and join
        # the daemon thread — otherwise the loop + thread leak past app exit.
        self._reconciler_loop = None
        self._maintenance = None  # IdleMaintenance — journal sync + retrain
        self._freshness_check_interval = 10  # every N cycles
        self._maintenance_interval = 5  # run IdleMaintenance.run_if_needed every N cycles

        # State
        self._scan_count = 0
        self._running = False
        self._peak_nav = 0.0
        self._lock = threading.Lock()

        # Tier 1 T4: transient phase indicator shared with the F1 TUI
        # widget. Scanner thread writes via .set / .clear at phase
        # boundaries in run_one_cycle; PhaseIndicator widget polls
        # .format() every 0.5s. Plain attribute writes are atomic in
        # CPython — no lock needed for str reads.
        from src.tui.widgets.phase_indicator import PhaseState
        self.phase_state = PhaseState()

        # Cycle-level autonomy triggers (periodic / self-heal / rejection).
        # Fires Claude on schedule regardless of whether trades closed,
        # so buddy isn't dormant just because OANDA is quiet.
        try:
            from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers
            self._autonomy = CycleAutonomyTriggers(brain_callback=self._brain)
        except ImportError:
            self._autonomy = None

        # Deterministic briefing snapshot writer (Angle 1'). Emits
        # .claude/brain/snapshot.md every N cycles + at boot. Pure
        # pull-from-disk, no LLM, NEVER touches briefing.md. Honors
        # ScannerConfig.disable_briefing_snapshot AND env
        # BUDDY_DISABLE_BRIEFING_SNAPSHOT=1.
        self._snapshot_writer = None
        try:
            from src.scanner.automation.briefing_snapshot import BriefingSnapshotWriter
            self._snapshot_writer = BriefingSnapshotWriter(project_root=self._project_root)
            # Boot-write fires immediately so the operator doesn't have to
            # wait one cycle for the first snapshot.
            if not self._briefing_snapshot_disabled():
                try:
                    self._snapshot_writer.write_now(trigger="boot")
                except Exception as _bs_err:
                    logger.debug("briefing_snapshot boot write error: %s", _bs_err)
        except ImportError as _bs_err:
            logger.debug("briefing_snapshot import skipped: %s", _bs_err)

    def _briefing_snapshot_disabled(self) -> bool:
        """True if snapshot writing should be skipped this cycle.

        Two kill-switches: env var (operator override at process start) AND
        ScannerConfig.disable_briefing_snapshot (per-profile/runtime toggle).
        Either flips it off.
        """
        if os.environ.get("BUDDY_DISABLE_BRIEFING_SNAPSHOT") == "1":
            return True
        cfg = self._config
        if cfg is not None and bool(getattr(cfg, "disable_briefing_snapshot", False)):
            return True
        return False

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

            # Build config: smart profile — tighter gates, hard uncertainty blocking,
            # asymmetric R:R (2.3:1), tuned from 2026-04-15 loss-streak learnings.
            # Changed from "balanced" on 2026-04-16 to match hedge-fund-grade selectivity.
            self._config = ScannerConfig.from_cli_args(
                config_path=None,
                # 2026-05-10: 2-pair constraint LIFTED. Per Phase 5.D + Phase 7,
                # USD_JPY/EUR_USD/EUR_JPY/USD_CAD/AUD_USD/GBP_USD/USD_CHF/NZD_USD
                # all M15-retrained with valid scalers. Falling through to
                # ScannerConfig default (15 majors+crosses). Use
                # blocked_pairs in the profile to opt out of any pair that
                # still has issues.
                # Switched H1 → M15 on 2026-05-07: M15-trained model holdout
                # 70.0% validated over 22 months of data (vs ~50% on H1 across
                # all architectures incl. Chronos foundation models). M15
                # 24-bar-forward direction is the operative trading horizon.
                granularity="M15",
                top_n=5,
                profile="smart",
                force=True,         # skip session filter
            )
            self._sync_meta_env_from_config(self._config)
            # Enable execution for auto-execute
            self._config.enable_execution = self._auto_execute
            # Cap scan passes to 1 (OOM guard for 8GB M1)
            self._config.max_scan_passes = 1

            # ── Create Scanner ─────────────────────────────────────
            self._scanner = Scanner(config=self._config)
            self._scan_count = self._load_persisted_scan_count()

            pair_count = len(self._resolve_scan_pairs())
            self._brain(f"[dim]  Scanner ready — {pair_count} pairs loaded[/]")

            # ── Automation modules (cherry-picked from ContinuousScanner) ──
            self._init_automation_modules()

            # Force gate evaluator init so model health report is accurate
            if hasattr(self._scanner, '_init_gate_evaluator'):
                try:
                    self._scanner._init_gate_evaluator()
                except Exception:
                    pass

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

                # Model training freshness — surface stale models loudly
                freshness = mh.get("freshness", {}) or {}
                f_status = freshness.get("status", "UNKNOWN")
                f_oldest = freshness.get("oldest_age_days")
                f_stale = freshness.get("stale_models") or []
                if f_status == "CRITICAL":
                    self._brain(
                        f"[red]▸ MODELS CRITICAL — oldest {f_oldest:.0f}d, "
                        f"retrain immediately: {', '.join(f_stale)}[/]"
                    )
                elif f_status == "STALE":
                    self._brain(
                        f"[yellow]▸ Models stale — oldest {f_oldest:.0f}d: "
                        f"{', '.join(f_stale)}[/]"
                    )
                elif f_status == "AGING":
                    self._brain(
                        f"[dim]▸ Models aging — oldest {f_oldest:.0f}d "
                        f"(retrain within 7 days)[/]"
                    )
                elif f_status == "FRESH" and f_oldest is not None:
                    self._brain(f"[dim]▸ Models fresh — oldest {f_oldest:.0f}d[/]")
            except Exception as _mh_err:
                logger.debug("Model health broadcast skipped: %s", _mh_err)

            self._running = True
            self._brain("[green]✓ Scanner engine online — first scan in 10s[/]")
            return True

        except Exception as e:
            msg = f"Scanner init failed: {_format_init_error(e)}"
            logger.error("EmbeddedScanner init failed: %s", e, exc_info=True)
            self._brain(f"[red]✗ {msg}[/]")
            # Tier 1 T6: surface init failure as inline banner so the
            # operator sees the dependency / model-load error in the
            # TUI without tailing logs.
            self.error_banner = msg
            return False

    def get_config(self) -> Any:
        """Return the live ScannerConfig object used by the embedded scanner."""
        return self._config

    def get_model_health(self) -> dict:
        """Return Scanner.get_model_health() when the engine is initialized."""
        if self._scanner is None or not hasattr(self._scanner, "get_model_health"):
            return {}
        return self._scanner.get_model_health() or {}

    def _load_persisted_scan_count(self) -> int:
        """Seed local cycle count from .claude/state.json after restarts."""
        try:
            from src.scanner.automation.state_engine import StateEngine
            state = StateEngine().load_state()
            return int(state.get("scan_cycle_count", state.get("scan_cycles", 0)) or 0)
        except Exception:
            return 0

    def _persist_next_scan_count(self) -> int:
        """Increment the shared scan counter and mirror it locally."""
        try:
            from src.scanner.automation.state_engine import StateEngine
            count = int(StateEngine().increment_scan_cycle() or 0)
            if count > 0:
                self._scan_count = count
                return count
        except Exception as exc:
            logger.debug("scan count persistence failed: %s", exc)
        self._scan_count += 1
        return self._scan_count

    def _resolve_scan_pairs(self) -> list[str]:
        """Resolve the current instrument list from ScannerConfig.

        The TUI can switch asset_class at runtime. ScannerConfig.active_instruments
        is the canonical selector for FX/Futures/Hybrid, while pairs remains an
        optional CLI override.
        """
        if self._config is None:
            return []
        explicit_pairs = list(getattr(self._config, "pairs", None) or [])
        if explicit_pairs:
            pairs = explicit_pairs
        else:
            active = getattr(self._config, "active_instruments", None)
            pairs = list(active or getattr(self._config, "default_pairs", []) or [])
        blocked = set(getattr(self._config, "blocked_pairs", set()) or set())
        if blocked:
            pairs = [p for p in pairs if p not in blocked]
        return pairs

    def _sync_meta_env_from_config(self, config: Any) -> None:
        """Bridge ScannerConfig flags to meta_manager.is_enabled().

        The TUI embeds Scanner directly and does not instantiate Orchestrator,
        so Orchestrator's env export path never runs here. Without this bridge,
        smart profile can have enable_meta_manager=True while
        meta_manager.is_enabled() still reads the bootstrap default
        BUDDY_META_MANAGER_ENABLED=0 and silently skips per-cycle routing.
        """
        enabled = bool(getattr(config, "enable_meta_manager", False))
        use_llm = bool(getattr(config, "meta_manager_use_llm", False))
        os.environ["BUDDY_META_MANAGER_ENABLED"] = "1" if enabled else "0"
        os.environ["BUDDY_META_USE_LLM"] = "1" if use_llm else "0"
        logger.info(
            "EmbeddedScanner: meta env synced enabled=%s use_llm=%s",
            enabled,
            use_llm,
        )

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

        # US-602: OANDA state reconciliation watchdog (60s auto-correct)
        try:
            from src.scanner.automation.state_reconciler import StateReconciler
            self._reconciler = StateReconciler(project_root=self._project_root)
            self._start_reconciler_loop()
            logger.info("EmbeddedScanner: StateReconciler initialized (60s)")
        except Exception as e:
            logger.debug("StateReconciler init skipped: %s", e)

        # US-605: Outcome backfill — catch-up pass against OANDA tx stream.
        # Best-effort: never allowed to block or crash startup.
        try:
            from src.scanner.automation.outcome_backfill import OutcomeBackfill
            backfill = OutcomeBackfill(project_root=self._project_root)
            bf_result = backfill.run_once()
            if bf_result.error:
                logger.info(
                    "EmbeddedScanner: OutcomeBackfill skipped (%s)", bf_result.error
                )
            else:
                logger.info(
                    "EmbeddedScanner: OutcomeBackfill ran (matched=%d, unmatched=%d, cursor=%s)",
                    bf_result.matched,
                    len(bf_result.unmatched_trade_ids),
                    bf_result.last_tx_id,
                )
        except Exception as e:  # noqa: BLE001 — never fail boot
            logger.debug("OutcomeBackfill init skipped: %s", e)

        # ConfigAdjuster — consumes pending config adjustments each cycle
        try:
            from src.scanner.automation.config_adjuster import ConfigAdjuster
            self._config_adjuster = ConfigAdjuster()
            pending = len(getattr(self._config_adjuster, "_pending", {}))
            if pending:
                logger.info("EmbeddedScanner: ConfigAdjuster loaded with %d pending adjustments", pending)
            else:
                logger.info("EmbeddedScanner: ConfigAdjuster initialized (no pending)")
        except Exception as e:
            logger.debug("ConfigAdjuster init skipped: %s", e)

        # IdleMaintenance — journal sync (every cycle invocation) + gate
        # retrain check (24h cooldown enforced inside the helper). Was
        # only wired into ContinuousScanner (the headless CLI path) per
        # the 2026-05-19 audit; live TUI path was therefore never running
        # OANDA journal sync or background retrains. Throttled to every
        # self._maintenance_interval cycles (default 5) in
        # _run_post_scan_automation to keep OANDA listClosedTrades calls
        # off the per-cycle hot path.
        try:
            from src.scanner.automation.maintenance import IdleMaintenance
            self._maintenance = IdleMaintenance()
            logger.info("EmbeddedScanner: IdleMaintenance initialized")
        except Exception as e:
            logger.debug("IdleMaintenance init skipped: %s", e)

    def _start_reconciler_loop(self) -> None:
        """Run StateReconciler.run_forever on a dedicated asyncio thread.

        Runs in its own event loop so the watchdog ticks every 60s even while
        the Textual TUI loop is busy rendering or the scan thread is blocking
        on OANDA. Swallows all errors — never allowed to crash the scanner.
        """
        import asyncio as _asyncio

        def _runner() -> None:
            loop = None
            try:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                self._reconciler_loop = loop
                self._reconciler_task = loop.create_task(self._reconciler.run_forever())
                loop.run_forever()
            except Exception as exc:  # noqa: BLE001
                logger.error("StateReconciler thread crashed: %s", exc)
            finally:
                # run_forever() has returned (loop.stop() was called from
                # shutdown). Drain + close so the loop releases its resources
                # rather than leaking a half-open selector.
                if loop is not None:
                    try:
                        if self._reconciler_task is not None:
                            self._reconciler_task.cancel()
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    except Exception:  # noqa: BLE001
                        pass
                    finally:
                        loop.close()

        t = threading.Thread(target=_runner, name="state-reconciler", daemon=True)
        t.start()
        self._reconciler_thread = t

    def _init_scanner(self) -> None:
        """Lazy-ensure seam called at the top of run_one_cycle (Tier 1 T6).

        The heavy init still lives in `initialize()` — callers are
        expected to run that from a worker thread before cycling. This
        method exists as a stable hook so genuine init errors
        (FileNotFoundError, ModuleNotFoundError, model load failures)
        can be raised from a single, monkeypatch-friendly entry point
        and caught by `run_one_cycle` to populate `error_banner`.
        Default behavior is a no-op.
        """
        return None

    def dismiss_error_banner(self) -> None:
        """Clear the inline error banner (Tier 1 T6).

        Called by the TUI after the operator dismisses the toast or
        opens the log viewer via Ctrl+L. The next non-gate exception
        re-arms the banner with a fresh message.
        """
        self.error_banner = None

    def run_one_cycle(self) -> Optional[ScanEnrichment]:
        """Run ONE scan cycle synchronously. CALL FROM A THREAD.

        1. Scans all pairs with on_pair_complete callbacks
        2. Auto-executes passing trades (if enabled)
        3. Runs post-scan automation (observations, RL, GC)
        4. Returns ScanEnrichment for DataProvider overlay

        Returns None if scanner not ready or scan failed.
        """
        # Tier 1 T6: surface init-time errors via `error_banner`. Re-raises
        # so callers and tests still observe the original failure type.
        try:
            self._init_scanner()
        except Exception as e:
            msg = f"Scanner init failed: {e}"
            self._brain(f"[red]✗ {msg}[/]")
            self.error_banner = msg
            logger.error("Scanner init failed: %s", e, exc_info=True)
            raise

        if not self._running or self._scanner is None:
            return None

        # Tier 2 T8: top-of-cycle config-dirty consume. Runs BEFORE any
        # scan/gate/execute work so the reload is atomic with respect to
        # the rest of the cycle — we never trade on half-applied config.
        # Pair with `AdjustmentApprover._mark_config_dirty` which sets the
        # flag whenever a new approved adjustment lands on disk.
        try:
            state_path = self._project_root / ".claude" / "state.json"
            if _consume_config_dirty_flag(state_path):
                self._reload_config_now()
                self._brain("[cyan]▸ config reloaded mid-session[/]")
        except Exception as _reload_err:
            # Reload is best-effort. A failure here must NOT block the
            # cycle — the next ConfigAdjuster apply tick will still pick
            # up the new approved entries. Log so the gap is observable.
            logger.warning(
                "T8 config-dirty reload error (non-blocking): %s", _reload_err,
            )

        # Keep the meta-cybernetic pipeline owned by the TUI runtime too.
        # This must run before the trading halt gate: an auto-halt should
        # stop scans/execution, not leave approved packages or post-deploy
        # reviews stranded until an external script drains MetaManager.
        self._maybe_drain_meta_pipeline()

        # Mythos audit 2026-04-30 — halt-aware scan skip. The auto-halt
        # in commit eacb617 sets StateEngine.halted=True when consecutive
        # losses ≥ threshold, but neither EmbeddedScanner nor Scanner
        # itself was checking that flag — only continuous.py:_run_smart_loop
        # (CLI path) did. Operator caught it: bot kept scanning at 14:33
        # despite halt set at 14:30. Closes the loop: when halted, skip
        # the cycle entirely, emit ONE brain message, and let the heartbeat
        # tick keep updating so the TUI stays alive for operator un-halt.
        try:
            from src.scanner.automation.state_engine import StateEngine
            if StateEngine().get_halted():
                if not getattr(self, "_halt_message_emitted", False):
                    self._brain(
                        "[bold red]◈ SCANNER HALTED — auto-halt active. "
                        "Toggle halted=false in state.json or via TUI 'u' "
                        "key to resume.[/]"
                    )
                    self._halt_message_emitted = True
                return None
            # Reset the latch so a re-halt fires a fresh brain message.
            self._halt_message_emitted = False
        except Exception as _halt_err:
            logger.debug("Halted check error (non-blocking): %s", _halt_err)

        self._persist_next_scan_count()
        # T3: cycle boundary — increment cumulative cycles counter.
        self.counters.bump_cycle()
        cycle_start = time.monotonic()

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._brain(f"[cyan]▸ Scan #{self._scan_count} starting at {now}...[/]")
        # Tier 1 T4: phase=scanning at cycle start.
        self.phase_state.set("scanning", f"cycle #{self._scan_count}")

        try:
            # ── Get instruments to scan ─────────────────────────────
            pairs = self._resolve_scan_pairs()
            if not pairs:
                self._brain("[yellow]  ▸ No active instruments configured; scan skipped[/]")
                return None

            # ── Run scan with per-pair callbacks ───────────────────
            result = self._scanner.scan(
                pairs=pairs,
                max_workers=2,  # conservative for 8GB M1 + TUI in same process
                on_pair_complete=self._on_pair_complete,
            )

            scan_ms = (time.monotonic() - cycle_start) * 1000
            tradeable = result.tradeable
            # T3: per-pair / per-gate phase boundary — every analysis went
            # through the scan pipeline and the gate evaluator.
            analyses_count = len(result.analyses)
            self.counters.bump_pair(analyses_count)
            self.counters.bump_gates_checked(analyses_count)
            self._brain(
                f"[dim]  Scan complete — {scan_ms:.0f}ms — "
                f"{len(tradeable)}/{len(result.analyses)} tradeable[/]"
            )
            # Tier 1 T4: phase=gate-check after per-pair scan completes.
            self.phase_state.set(
                "gate-check",
                f"{len(tradeable)}/{len(result.analyses)} pairs",
            )

            # ── Auto-execute ───────────────────────────────────────
            trades_executed = 0
            tradeable_before_filters = len(tradeable)
            if self._auto_execute and tradeable:
                # Tier 1 T4: phase=executing once we know we're going to
                # try execution. Detail updates again after trades close.
                self.phase_state.set("executing", f"{len(tradeable)} candidates")
                tradeable = self._filter_correlated_exposure(tradeable)
                if tradeable:
                    tradeable = self._check_policy(tradeable)
                if tradeable:
                    trades_executed = self._execute_trades(tradeable)
                    self.phase_state.set("executing", f"{trades_executed} trades")
            # T3: execution boundary — record cumulative successful trades.
            if trades_executed:
                self.counters.bump_trade(int(trades_executed))

            # ── Post-scan automation ───────────────────────────────
            self._post_scan_automation(result)

            # ── Memory management ──────────────────────────────────
            self._memory_guard()

            # ── Cycle-level autonomy (periodic / self-heal / rejection) ──
            # Fires Claude on schedule regardless of whether any trade
            # closed this cycle. Without this, buddy sits silent whenever
            # OANDA is quiet — which the user experiences as "brainless".
            if self._autonomy is not None:
                try:
                    self._autonomy.on_cycle_complete(
                        scan_count=self._scan_count,
                        scan_result=result,
                        trades_executed=int(trades_executed or 0),
                        tradeable_count=int(tradeable_before_filters),
                    )
                except Exception as _aut_err:
                    logger.debug("cycle_autonomy trigger error: %s", _aut_err)

            # ── Per-cycle meta-pipeline routing ────────────────────
            # Mythos audit 2026-04-30 — methodology fix for commit
            # f070d39. That commit added the same routing to
            # Orchestrator._post_trade_diagnostics_dispatch, which is
            # NEVER called by the TUI (EmbeddedScanner has its own
            # scan loop and doesn't instantiate Orchestrator). The
            # routing we shipped was dead code in the live path.
            #
            # Re-wired here so the meta-pipeline gets the per-cycle
            # diagnostic signal it was supposed to get. Throttled via
            # signature dedup so identical (status, recommended_actions)
            # tuples on consecutive cycles don't create duplicate
            # ChangePackages.
            self._maybe_route_to_meta_per_cycle()

            # ── Deterministic briefing snapshot (Angle 1') ────────
            # Cadence-gated mirror of runtime state to
            # .claude/brain/snapshot.md. Pure pull-from-disk, no LLM,
            # diff-and-write so identical state doesn't churn the file.
            # NEVER touches the human-curated briefing.md. Wired AFTER
            # _maybe_route_to_meta_per_cycle so meta-pipeline state is
            # captured in the snapshot.
            if (
                self._snapshot_writer is not None
                and not self._briefing_snapshot_disabled()
            ):
                try:
                    every_n = int(getattr(
                        self._config,
                        "briefing_snapshot_every_n_cycles",
                        12,
                    ) or 12)
                    self._snapshot_writer.maybe_write(
                        cycle_count=int(self._scan_count or 0),
                        every_n=every_n,
                    )
                except Exception as _bs_err:
                    logger.debug("briefing_snapshot per-cycle error: %s", _bs_err)

            # ── Build enrichment ───────────────────────────────────
            enrichment = self._build_enrichment(result, scan_ms)

            next_min = self._interval_minutes
            self._brain(f"[cyan]▸ Next scan in {next_min}m...[/]")
            # Tier 1 T4: clear back to idle at cycle end.
            self.phase_state.clear()

            return enrichment

        except Exception as e:
            msg = f"Scan #{self._scan_count} failed: {e}"
            logger.error("Scan cycle %d failed: %s", self._scan_count, e, exc_info=True)
            self._brain(f"[red]✗ {msg}[/]")
            # Tier 1 T6: surface non-gate cycle failures as inline banner.
            self.error_banner = msg
            self._memory_guard()
            # Tier 1 T4: clear on failure so the indicator doesn't get
            # stuck mid-phase forever.
            self.phase_state.clear()
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
            # Directional but didn't pass gates.
            #
            # 2026-05-19 (observability fix): enrich the line with the names of
            # any vetoing agents (passed=False OR block_trade=True). Operator
            # symptom that motivated this: "0/15 tradeable, no idea WHICH
            # agent is doing it." Reads from `analysis.agent_reasons` (the
            # real top-level field) — `analysis.agents` does NOT exist on
            # PairAnalysis. Keeps the existing `blocked: <reason>` token
            # intact for downstream parsers that look for it; appends a
            # `[veto: a,b,c]` token only when at least one vetoer is found.
            reason = analysis.rejection_reason or "gates"
            _vetoers: list = []
            _seen: set = set()
            for _ar in (getattr(analysis, "agent_reasons", None) or []):
                if not isinstance(_ar, dict):
                    continue
                _name = _ar.get("name")
                if not _name or _name in _seen:
                    continue
                if _ar.get("block_trade") is True or _ar.get("passed") is False:
                    _seen.add(_name)
                    _vetoers.append(str(_name))
            if _vetoers:
                _veto_token = f" [veto: {', '.join(_vetoers)}]"
            else:
                _veto_token = ""
            self._brain(
                f"[dim]  ─ {pair} {direction} conf={conf:.2f} — "
                f"blocked: {reason}{_veto_token}[/]"
            )
        else:
            self._brain(f"[dim]  ─ {pair} HOLD[/]")

    def _maybe_route_to_meta_per_cycle(self) -> None:
        """Run PostTradeDiagnostics every cycle and route non-HEALTHY
        results into the meta-pipeline.

        Mythos audit 2026-04-30. The orchestrator's
        _post_trade_diagnostics_dispatch implements this same flow but
        is NEVER reached from the TUI path (EmbeddedScanner does NOT
        instantiate Orchestrator). This is the live equivalent.

        Throttling layers (defense in depth):
          1. is_enabled() flag check — no routing when meta is opt-out
          2. status==HEALTHY filter — no routing on a clean cycle
          3. empty actions filter — no routing when there's nothing
             actionable
          4. signature dedup on (status, sorted recommended_actions) —
             identical cycles don't create fresh packages
          5. meta-pipeline G8 dedup at intake — defense in depth, drops
             in-flight hash matches even if signature dedup has a bug
        """
        try:
            from src.scanner.feedback.diagnostics import PostTradeDiagnostics
        except ImportError:
            return
        try:
            from src.scanner.automation.meta_manager import (
                is_enabled as _meta_enabled,
                route_incident,
            )
        except ImportError:
            return
        if not _meta_enabled():
            return

        try:
            diag = PostTradeDiagnostics().run()
        except Exception as e:
            logger.debug("EmbeddedScanner: PostTradeDiagnostics failed err=%r", e)
            return
        if not isinstance(diag, dict):
            return
        status = str(diag.get("status", "")).upper()
        if status in ("HEALTHY", ""):
            return
        actions = diag.get("recommended_actions") or []
        if not isinstance(actions, list) or not actions:
            return

        try:
            sig = (
                status,
                tuple(sorted(str(a) for a in actions if isinstance(a, str))),
            )
        except Exception:
            return
        last_sig = getattr(self, "_last_meta_route_sig", None)
        if sig == last_sig:
            return
        self._last_meta_route_sig = sig

        try:
            routed = route_incident({
                "kind": "self_heal",
                "source": "tui_embedded_scanner_per_cycle",
                "diag": diag,
            })
            logger.info(
                "embedded_scanner.meta_route status=%s actions=%d routed=%s",
                status, len(actions), routed,
            )
        except Exception as e:
            logger.debug("embedded_scanner.meta_route_failed err=%r", e)

    def _reload_config_now(self) -> None:
        """Invalidate the ConfigAdjuster cache and re-apply pending adjustments.

        Tier 2 T8: triggered when `_consume_config_dirty_flag` returns True
        at the top of `run_one_cycle`. Pair with `AdjustmentApprover._save_approved`
        which sets the flag on every successful approval write.

        Uses Tier 1 T5's `ConfigAdjuster._invalidate_cache` so the next
        `apply_adjustments` call re-reads `config_adjustments.json` from disk
        instead of serving stale in-memory state. Best-effort — adjuster /
        config may be None during tests or early init.
        """
        if self._config_adjuster is None or self._config is None:
            return
        try:
            self._config_adjuster._invalidate_cache()
            applied = self._config_adjuster.apply_adjustments(
                self._config, current_cycle=self._scan_count,
            )
            if applied:
                names = [a.get("key", "?") for a in applied]
                logger.info(
                    "T8 reload applied %d adjustment(s): %s",
                    len(applied), ", ".join(names),
                )
        except Exception as e:
            logger.warning("T8 _reload_config_now apply error: %s", e)

    def _maybe_drain_meta_pipeline(self) -> None:
        """Advance approved/deployed meta ChangePackages from the TUI path.

        Orchestrator owns this in the CLI smart-loop path, but the TUI embeds
        Scanner directly and never instantiates Orchestrator. Without this,
        approved meta packages can sit idle unless the operator happens to
        use the Inbox approve-all inline drain or a standalone script.
        """
        try:
            from src.scanner.automation import meta_manager as _mm
        except ImportError:
            return
        try:
            if not _mm.is_enabled():
                return
            if _mm._PRODUCTION_MGR is None:
                _mm._PRODUCTION_MGR = _mm._build_production_manager()
            counters = _mm._PRODUCTION_MGR.drain(current_cycle=self._scan_count)
            if any(int(v or 0) for v in counters.values()):
                logger.info("embedded_scanner.meta_drain %s", counters)
        except Exception as e:
            logger.debug("embedded_scanner.meta_drain_failed err=%r", e)

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

    def _execute_trades(self, tradeable: list) -> int:
        """Execute passing trades and report results to brain.

        Returns the number of trades that successfully executed (0 on error
        or total-reject). Used by cycle_autonomy to detect rejection streaks.
        """
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
                return int(ok)
            else:
                self._brain(
                    "[red]  ✗ Execution returned no results — check NAV/broker[/]"
                )
                return 0
        except Exception as e:
            logger.error("Trade execution error: %s", e)
            self._brain(f"[red]  ✗ Execution error: {e}[/]")
            return 0

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

        # Dry-run validation telemetry (US-604) — emit one JSONL row per
        # directional candidate so analyze_dry_run.py can report gate firing
        # distributions. Write-only; no execution side effects. Mirrors
        # continuous.py:529-540. Closes the TUI wiring gap flagged in the
        # 2026-04-25 phase95_evidence.md US-606 close-out review.
        try:
            from src.scanner.automation.validation_stats import (
                ScanDistributionStats,
            )
            if not hasattr(self, "_validation_stats"):
                self._validation_stats = ScanDistributionStats()
            self._validation_stats.record_cycle(result.analyses or [])
        except Exception as _vs_err:
            logger.debug("validation_stats record error: %s", _vs_err)

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

        # IdleMaintenance — journal sync + retrain check, every N cycles.
        # The 24h retrain cooldown is enforced inside run_if_needed; the
        # outer throttle keeps OANDA listClosedTrades calls bounded on
        # short scan intervals. Pre-2026-05-19 this was only wired into
        # ContinuousScanner (the headless CLI), so the live TUI runtime
        # was never running journal sync or background retrains.
        if (
            self._maintenance is not None
            and self._maintenance_interval > 0
            and self._scan_count % self._maintenance_interval == 0
        ):
            try:
                self._maintenance.run_if_needed()
            except Exception as e:
                logger.debug("IdleMaintenance run_if_needed error: %s", e)

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

        # ConfigAdjuster — apply pending config adjustments every cycle
        if self._config_adjuster is not None and self._config is not None:
            try:
                applied = self._config_adjuster.apply_adjustments(
                    self._config, self._scan_count,
                )
                if applied:
                    names = [a["key"] for a in applied]
                    self._brain(
                        f"[yellow]  ▸ Config adjusted: {', '.join(names)}[/]"
                    )
                self._config_adjuster.save_state()
            except Exception as e:
                logger.debug("ConfigAdjuster apply error: %s", e)

        # Model freshness — periodic check every N cycles
        if self._scan_count % self._freshness_check_interval == 0:
            try:
                from src.scanner.automation.model_freshness import get_model_freshness
                freshness = get_model_freshness()
                f_status = freshness.get("status", "UNKNOWN")
                f_oldest = freshness.get("oldest_age_days")
                if f_status == "CRITICAL":
                    self._brain(
                        f"[red]  ▸ MODELS CRITICAL — oldest {f_oldest:.0f}d, retrain needed[/]"
                    )
                elif f_status == "STALE":
                    self._brain(
                        f"[yellow]  ▸ Models stale — oldest {f_oldest:.0f}d[/]"
                    )
            except Exception as e:
                logger.debug("Model freshness check error: %s", e)

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
        enrichment.config_profile = str(getattr(self._config, "profile", "smart") or "smart")
        enrichment.config_values = self._snapshot_config_values()

        # US-002: per-instrument ATR (pips) — feeds TradeRow.live_atr_pips in
        # DataProvider.refresh(). Skip rows where atr_pips is missing or
        # non-positive (cold start / failed analysis) so the drill-down
        # renders the "—" placeholder rather than a misleading 0.0.
        atr_by_instrument: dict[str, float] = {}
        for pa in result.analyses or []:
            pair = getattr(pa, "pair", "") or ""
            atr = float(getattr(pa, "atr_pips", 0.0) or 0.0)
            if pair and atr > 0.0:
                atr_by_instrument[pair] = atr
        enrichment.atr_value = atr_by_instrument

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

        # US-513: Model staleness — max ensemble component age in days
        try:
            if hasattr(self._scanner, "get_model_staleness_days"):
                enrichment.max_component_age_days = float(
                    self._scanner.get_model_staleness_days() or 0.0
                )
        except Exception as e:
            logger.debug("Model staleness snapshot error: %s", e)

        return enrichment

    def _snapshot_config_values(self) -> dict:
        """Return scalar live ScannerConfig values safe for the TUI snapshot."""
        if self._config is None:
            return {}
        try:
            import dataclasses
            values = {}
            for field_info in dataclasses.fields(self._config):
                value = getattr(self._config, field_info.name, None)
                if isinstance(value, (bool, int, float, str)):
                    values[field_info.name] = value
            return values
        except Exception as exc:
            logger.debug("Config snapshot error: %s", exc)
            return {}

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

        # IdleMaintenance — signal cooperative stop. The helper's stop()
        # sets _stop_requested, terminates any in-flight retrain subprocess
        # within ~2s, and joins the daemon thread. Best-effort — never
        # raise during shutdown.
        if self._maintenance is not None:
            try:
                self._maintenance.stop()
            except Exception:
                pass

        # StateReconciler — stop the dedicated asyncio loop cross-thread, then
        # join the daemon thread so it doesn't outlive the app. Previously the
        # loop ran forever and the thread was never joined (leak on every
        # shutdown). stop() makes run_forever() return; the _runner finally
        # block closes the loop; join(timeout=2) reaps the thread.
        loop = self._reconciler_loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        thread = self._reconciler_thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
            if thread.is_alive():
                logger.warning(
                    "StateReconciler thread did not stop within 2s of shutdown"
                )
        self._reconciler_loop = None
        self._reconciler_thread = None
        self._reconciler_task = None

        # Final GC
        gc.collect()
        self._brain("[dim]  Scanner stopped.[/]")
