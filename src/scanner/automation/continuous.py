"""
Continuous Scanner - Run scans in a loop with configurable intervals.

Provides watch mode functionality for real-time market monitoring.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from src.scanner.engine import Scanner
    from src.scanner.results import ScanResult

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

logger = logging.getLogger(__name__)


@dataclass
class ContinuousConfig:
    """Configuration for continuous scanning."""

    interval_minutes: int = 5
    auto_execute: bool = False
    top_n: int = 5
    pairs: Optional[List[str]] = None
    granularity: str = "H1"
    enable_maintenance: bool = True
    max_scans: Optional[int] = None  # None = unlimited


class ContinuousScanner:
    """
    Run scans in a loop with configurable intervals.

    Features:
    - Graceful shutdown on Ctrl+C
    - Idle-time maintenance (gate retraining, journal sync)
    - Auto-execution of passing trades
    - Progress indication between scans

    Example:
        >>> scanner = Scanner(config)
        >>> continuous = ContinuousScanner(scanner)
        >>> continuous.run(interval_minutes=5, auto_execute=True)
    """

    def __init__(
        self,
        scanner: "Scanner",
        config: Optional[ContinuousConfig] = None,
    ):
        self.scanner = scanner
        self.config = config or ContinuousConfig()
        self._running = False
        self._scan_count = 0
        self._maintenance = None
        self._portfolio_optimizer = None

        # Journal cache: avoid re-reading trade_journal_rl.json on every cycle.
        # Invalidated by file mtime — only re-reads when a trade actually closes.
        self._journal_cache: list = []
        self._journal_mtime: float = 0.0

        # Phase 56 (US-350): Scan Diagnostics Reporter
        self._scan_diagnostics = None
        try:
            from src.scanner.scan_diagnostics import ScanDiagnosticsReporter
            self._scan_diagnostics = ScanDiagnosticsReporter()
            logger.info("Phase 56 (US-350): Scan diagnostics reporter initialized")
        except Exception as _sdr_err:
            logger.debug(f"Phase 56: Scan diagnostics reporter init deferred: {_sdr_err}")

        # Phase 57 (US-356): Penalty Auditor — surfaces recommendation when stalled
        self._penalty_auditor = None
        try:
            from src.scanner.penalty_audit import PenaltyAuditor
            self._penalty_auditor = PenaltyAuditor()
            logger.info("Phase 57 (US-356): Penalty auditor initialized")
        except Exception as _pa_err:
            logger.debug(f"Phase 57: Penalty auditor init deferred: {_pa_err}")

        # Phase 58 (US-360): Gate Pass Predictor — simulate pass rate under penalty-free conditions
        self._gate_pass_predictor = None
        try:
            from src.scanner.gate_pass_predictor import GatePassPredictor
            self._gate_pass_predictor = GatePassPredictor()
            logger.info("Phase 58 (US-360): Gate pass predictor initialized")
        except Exception as _gpp_err:
            logger.debug(f"Phase 58: Gate pass predictor init deferred: {_gpp_err}")

        # Phase 58 (US-361): Adaptive Confidence Floor — conservative threshold auto-reduction
        self._adaptive_floor = None
        try:
            from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
            _initial_threshold = getattr(self.scanner.config, "min_confidence", 58.0)
            self._adaptive_floor = AdaptiveConfidenceFloor(
                initial_threshold=float(_initial_threshold),
            )
            logger.info("Phase 58 (US-361): Adaptive confidence floor initialized")
        except Exception as _af_err:
            logger.debug(f"Phase 58: Adaptive confidence floor init deferred: {_af_err}")

        # Phase 59 (US-363): Signal Funnel Tracker — per-scan gate attrition visibility
        self._signal_funnel = None
        try:
            from src.scanner.signal_funnel_tracker import SignalFunnelTracker
            self._signal_funnel = SignalFunnelTracker()
            logger.info("Phase 59 (US-363): Signal funnel tracker initialized")
        except Exception as _sft_err:
            logger.debug(f"Phase 59: Signal funnel tracker init deferred: {_sft_err}")

        # Phase 60 (US-369) + Phase 65 (US-386): Scan Health Synthesizer — unified health report
        self._scan_health = None
        try:
            from src.scanner.scan_health_synthesizer import ScanHealthSynthesizer
            self._scan_health = ScanHealthSynthesizer(
                signal_funnel=self._signal_funnel,
                gate_proximity=getattr(self.scanner, "_gate_proximity", None),
                ensemble_divergence=getattr(self.scanner, "_ensemble_divergence", None),
                raw_conf_recorder=getattr(self.scanner, "_raw_conf_recorder", None),
                scan_diagnostics=self._scan_diagnostics,
                # Phase 65 (US-386): wire momentum + risk recorders for enriched diagnostics
                momentum_recorder=getattr(self.scanner, "_momentum_recorder", None),
                risk_recorder=getattr(self.scanner, "_risk_recorder", None),
                momentum_max_threshold=getattr(
                    getattr(self.scanner, "config", None), "min_momentum", 0.45
                ),
                risk_max_drawdown=getattr(
                    getattr(self.scanner, "config", None), "max_drawdown_pct", 0.025
                ),
                # Phase 66 (US-388): wire vote recorder for agent consensus diagnostics
                vote_recorder=getattr(self.scanner, "_vote_recorder", None),
                weighted_vote_threshold=getattr(
                    getattr(self.scanner, "config", None), "weighted_vote_threshold", 0.55
                ),
            )
            logger.info("Phase 60+65+66 (US-369/US-386/US-388): Scan health synthesizer initialized with momentum+risk+vote recorders")
        except Exception as _shs_err:
            logger.debug(f"Phase 60: Scan health synthesizer init deferred: {_shs_err}")

        # Initialize maintenance if enabled
        if self.config.enable_maintenance:
            from src.scanner.automation.maintenance import IdleMaintenance
            self._maintenance = IdleMaintenance()

        # Initialize portfolio optimizer for dynamic pair rotation
        try:
            from src.scanner.automation.portfolio_optimizer import PortfolioOptimizer
            self._portfolio_optimizer = PortfolioOptimizer()
        except Exception as e:
            logger.debug(f"Portfolio optimizer initialization error: {e}")

        # Initialize online RL weight updater (US-060)
        self._online_rl = None
        try:
            from src.scanner.automation.online_rl import OnlineWeightUpdater
            self._online_rl = OnlineWeightUpdater()
        except Exception as e:
            logger.debug(f"Online RL updater initialization error: {e}")

        # Phase 20 (US-126): Wire memory manager — register caches and log files
        if getattr(scanner, "_memory_manager", None) is not None:
            try:
                mm = scanner._memory_manager
                # Register scanner caches for LRU eviction
                for cache_name, cache_attr in [
                    ("cached_results", "_cached_results"),
                    ("last_prices", "_last_prices"),
                    ("feature_snapshots", "_feature_snapshots"),
                    ("raw_snapshots", "_raw_snapshots"),
                    ("pair_returns", "_pair_returns"),
                ]:
                    cache_dict = getattr(scanner, cache_attr, None)
                    if isinstance(cache_dict, dict):
                        mm.register_cache(cache_name, cache_dict, max_size=200)

                # Register major log files for rotation
                from pathlib import Path as _Path
                _root = _Path(__file__).resolve().parent.parent.parent.parent
                for log_name in [
                    "trained_data/observations.jsonl",
                    "trained_data/trade_journal_rl.json",
                    "trained_data/scan_cycle_log.json",
                ]:
                    log_path = _root / log_name
                    if log_path.exists():
                        mm.register_log(str(log_path), max_lines=10000, keep_lines=5000)
                logger.info("Memory manager: caches and logs registered")
            except Exception as mm_err:
                logger.debug(f"Memory manager registration error: {mm_err}")

        # Phase 20 (US-124): Module activation dispatcher
        self._module_dispatcher = None
        if getattr(scanner.config, "enable_module_activation", False):
            try:
                from src.scanner.automation.module_dispatcher import ModuleDispatcher
                self._module_dispatcher = ModuleDispatcher(scanner=scanner)
                registered = self._module_dispatcher.register_all_modules()
                logger.info("Module dispatcher initialized (%d modules)", registered)
            except Exception as e:
                logger.debug(f"Module dispatcher initialization error: {e}")

    def run(
        self,
        pairs: Optional[List[str]] = None,
        granularity: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        auto_execute: bool = False,
        top_n: Optional[int] = None,
        on_scan_complete: Optional[Callable[["ScanResult"], None]] = None,
    ) -> int:
        """
        Run continuous scanning loop.

        Args:
            pairs: List of pairs to scan (overrides config)
            granularity: Timeframe (overrides config)
            interval_minutes: Minutes between scans (overrides config)
            auto_execute: Auto-execute passing trades (overrides config)
            top_n: Number of top results (overrides config)
            on_scan_complete: Callback after each scan

        Returns:
            Number of scans completed
        """
        # Apply overrides
        pairs = pairs or self.config.pairs
        granularity = granularity or self.config.granularity
        interval_minutes = interval_minutes or self.config.interval_minutes
        top_n = top_n or self.config.top_n

        # Setup signal handler for graceful shutdown
        self._running = True
        self._setup_signal_handler()

        self._scan_count = 0

        # Backup critical state files on watch mode start
        try:
            from scripts.backup_state import create_backup
            create_backup()
            logger.info("Pre-session backup created")
        except Exception as backup_err:
            logger.debug(f"Pre-session backup skipped: {backup_err}")

        if console:
            console.print("\n[bold cyan]🔄 CONTINUOUS SCAN MODE[/bold cyan]")
            console.print(f"[dim]Scanning every {interval_minutes} minutes. Press Ctrl+C to stop.[/dim]")
            console.print("=" * 70)

        while self._running:
            self._scan_count += 1

            # Check max scans limit
            if self.config.max_scans and self._scan_count > self.config.max_scans:
                if console:
                    console.print(f"\n[yellow]Max scans ({self.config.max_scans}) reached[/yellow]")
                break

            try:
                if console:
                    now = datetime.now().strftime("%H:%M:%S")
                    console.print(f"\n[bold]── Scan #{self._scan_count} at {now} ──[/bold]")

                # Filter out blocked pairs before scanning
                scan_pairs = pairs
                if scan_pairs and self.scanner.config.blocked_pairs:
                    filtered = [p for p in scan_pairs if p not in self.scanner.config.blocked_pairs]
                    if len(filtered) < len(scan_pairs):
                        skipped = [p for p in scan_pairs if p in self.scanner.config.blocked_pairs]
                        if console:
                            console.print(f"[dim]Skipping blocked pairs: {', '.join(skipped)}[/dim]")
                        scan_pairs = filtered

                # Apply dynamic pair rotation every rotation_interval cycles
                self._apply_pair_rotation(scan_pairs)

                # Phase 23 (US-143): Apply immediate gate adjustments from observations
                if getattr(self.scanner, "_observation_consumer", None) is not None:
                    try:
                        _imm = self.scanner._observation_consumer.get_immediate_adjustments(
                            current_cycle=self._scan_count
                        )
                        if _imm:
                            for _key, _val in _imm.items():
                                if _key == "min_confidence_offset":
                                    # Temporary offset, not absolute override
                                    _curr = getattr(self.scanner.config, "min_confidence", 55)
                                    setattr(self.scanner.config, "min_confidence", max(40, _curr + _val))
                                elif _key == "regime_consensus_map_normal":
                                    _rcm = getattr(self.scanner.config, "regime_consensus_map", {})
                                    if isinstance(_rcm, dict):
                                        _rcm["NORMAL"] = _val
                                elif hasattr(self.scanner.config, _key):
                                    setattr(self.scanner.config, _key, _val)
                            if console:
                                console.print(
                                    f"  [cyan]US-143 immediate adjustments: "
                                    f"{', '.join(f'{k}={v}' for k, v in _imm.items())}[/cyan]"
                                )
                    except Exception as _imm_err:
                        logger.debug(f"Immediate adjustments skipped: {_imm_err}")

                # Run scan
                result = self.scanner.scan(
                    pairs=scan_pairs,
                    max_workers=4,
                )

                # Display results
                from src.scanner.display import ScannerDisplay
                display = ScannerDisplay()
                account_info = self.scanner.get_account_info()
                display.show_result(result, account_info=account_info)

                # Phase 23 (US-144): Execution summary dashboard
                if console and result and result.analyses:
                    try:
                        _analyses = result.analyses
                        _total = len(_analyses)
                        _directional = [a for a in _analyses if a.direction in {"LONG", "SHORT"}]
                        _gates_ok = [a for a in _directional if a.gates_passed]
                        _agent_ok = [a for a in _directional if getattr(a, "agent_passed", False)]
                        _tradeable = [a for a in _analyses if a.is_tradeable]
                        # Near-misses: confidence within 5% of threshold
                        _min_conf = float(getattr(self.scanner.config, "min_confidence", 55)) / 100.0
                        _near_miss = [
                            a for a in _directional
                            if not a.is_tradeable and abs(a.confidence - _min_conf) <= 0.05
                        ]
                        # Gate failure breakdown
                        _block_conf = sum(1 for a in _directional if not a.confidence_passed)
                        _block_mom = sum(1 for a in _directional if not a.momentum_passed)
                        _block_risk = sum(1 for a in _directional if not a.risk_passed)
                        # Regime distribution
                        from collections import Counter as _Ctr
                        _regimes = _Ctr(
                            str(getattr(a, "volatility_regime", "?")).upper()
                            for a in _directional
                        )
                        _regime_str = ", ".join(f"{r}:{c}" for r, c in _regimes.most_common(4))
                        # Top blocking gate
                        _blocks = {"confidence": _block_conf, "momentum": _block_mom, "risk": _block_risk}
                        _top_block = max(_blocks, key=_blocks.get) if any(_blocks.values()) else "none"

                        console.print(
                            f"\n  [bold]Execution Summary[/bold]: "
                            f"{_total} scanned → {len(_directional)} directional → "
                            f"{len(_gates_ok)} gates✓ → {len(_agent_ok)} agents✓ → "
                            f"[{'green' if _tradeable else 'yellow'}]{len(_tradeable)} tradeable[/{'green' if _tradeable else 'yellow'}]"
                        )
                        console.print(
                            f"  [dim]Blocks: conf={_block_conf} mom={_block_mom} risk={_block_risk} "
                            f"| Top blocker: {_top_block} | Near-misses: {len(_near_miss)} "
                            f"| Regimes: {_regime_str}[/dim]"
                        )
                    except Exception as _dash_err:
                        logger.debug("US-189: Dashboard formatting skipped: %s", _dash_err)

                # Callback if provided
                if on_scan_complete:
                    on_scan_complete(result)

                # Filter out observe-only pairs from tradeable list
                # Cold-start bypass: skip observe filter when insufficient trade history
                # (need 5 trades per pair for active status — impossible on fresh system)
                if self._portfolio_optimizer:
                    try:
                        observe_pairs = self._portfolio_optimizer.get_observe_only_pairs(
                            all_pairs=scan_pairs
                        )
                        tradeable = [a for a in result.analyses if a.is_tradeable]

                        # Check cold-start: if insufficient closed trade history, bypass filter
                        # get_active_pairs() pads with observe candidates so can't rely on its count
                        _rankings = self._portfolio_optimizer.rank_pairs()
                        _truly_active = [r for r in _rankings if getattr(r, "status", "") == "active"]
                        if len(_truly_active) < 2:
                            logger.info(
                                f"Cold-start mode: {len(_truly_active)} truly active pairs "
                                f"(need 5 closed trades/pair to graduate). Bypassing observe filter."
                            )
                            # Skip filtering — let gates be the gatekeeper
                        else:
                            # Remove trades on observe-only pairs
                            original_count = len(tradeable)
                            tradeable = [a for a in tradeable if a.pair not in observe_pairs]
                            if len(tradeable) < original_count:
                                filtered_out = original_count - len(tradeable)
                                if console:
                                    console.print(
                                        f"[dim]Filtered {filtered_out} observe-only pair(s) from execution[/dim]"
                                    )
                    except Exception as po_err:
                        logger.debug(f"Pair rotation filter error: {po_err}")
                        tradeable = [a for a in result.analyses if a.is_tradeable]
                else:
                    tradeable = [a for a in result.analyses if a.is_tradeable]

                # Auto-execute if enabled (use is_tradeable not gates_passed)
                if auto_execute:
                    tradeable = self._filter_correlated_exposure(tradeable)
                    if tradeable:
                        if console:
                            console.print(f"\n[green]Auto-executing {len(tradeable)} trade(s)...[/green]")
                        # Ensure execution is enabled — continuous.py was missing this
                        # (buddy_scanning.py sets it correctly, but continuous.py didn't)
                        self.scanner.config.enable_execution = True
                        self.scanner.execute_trades(
                            analyses=tradeable,
                        )

                # Log scan cycle for analytics
                self._log_scan_cycle(result, auto_execute)

                # Log observations from scan results (US-008)
                try:
                    from src.scanner.automation.observation_log import ObservationLog
                    obs_log = ObservationLog()
                    obs_count = 0
                    for analysis in result.analyses:
                        obs_count += obs_log.log_from_analysis(analysis)
                    if obs_count > 0 and console:
                        console.print(f"  [dim]Observations: {obs_count} patterns logged[/dim]")
                except Exception as obs_err:
                    logger.debug(f"Observation logging error: {obs_err}")

                # Phase 20 (US-123): Consume observations every 10 cycles
                if (
                    self._scan_count % 10 == 0
                    and getattr(self.scanner, "_observation_consumer", None) is not None
                ):
                    try:
                        consumer = self.scanner._observation_consumer
                        new_obs = consumer.consume_observations()
                        if new_obs > 0:
                            patterns = consumer.detect_patterns()
                            adjustments = consumer.recommend_adjustments()
                            alerts = consumer.check_alerts(window_hours=6)

                            if console:
                                console.print(
                                    f"  [dim]Observations consumed: {new_obs} new, "
                                    f"{len(patterns)} patterns, {len(adjustments)} recommendations[/dim]"
                                )

                            # Feed recommendations to config_tuner if available
                            if adjustments:
                                try:
                                    from src.scanner.automation.config_tuner import ConfigTuner
                                    tuner = ConfigTuner()
                                    for adj in adjustments[:3]:  # Top 3 recommendations
                                        tuner.propose_adjustment(
                                            param=adj.get("action", ""),
                                            delta=adj.get("delta", 0),
                                            reason=adj.get("reason", "observation_pattern"),
                                        )
                                except Exception as tune_err:
                                    logger.debug(f"Config tuner feed error: {tune_err}")

                            # Log spike alerts
                            for alert in alerts:
                                logger.warning(
                                    "Observation spike: %s (%d in %dh, severity=%s)",
                                    alert.get("type"), alert.get("count"),
                                    alert.get("window_hours"), alert.get("severity"),
                                )

                            consumer.save_state()
                    except Exception as oc_err:
                        logger.debug(f"Observation consumer error: {oc_err}")

                # Model A/B testing: shadow test candidate vs production (every scan)
                try:
                    self._run_shadow_tests(result)
                    # Check for promotion every 50th scan
                    if self._scan_count % 50 == 0:
                        self._check_model_promotion()
                except Exception as shadow_err:
                    logger.debug(f"Shadow test error: {shadow_err}")

                # Phase 20 (US-125): Periodic drift detection every 20 cycles
                if (
                    self._scan_count % 20 == 0
                    and getattr(self.scanner, "_drift_monitor", None) is not None
                    and result.analyses
                ):
                    try:
                        drift_monitor = self.scanner._drift_monitor
                        # Run drift check on each pair using cached candle data
                        drift_alerts = []
                        for analysis in result.analyses[:5]:  # Top 5 pairs
                            candle_cache = getattr(self.scanner, "_cached_candles", {})
                            candles = candle_cache.get(analysis.pair, [])
                            if len(candles) >= 100:
                                drift_report = drift_monitor.run_drift_check(
                                    pair=analysis.pair,
                                    historical_candles=candles,
                                )
                                if drift_report.get("severity") in ("mild", "critical"):
                                    drift_alerts.append(
                                        f"{analysis.pair}={drift_report['severity']}"
                                        f"(score={drift_report['drift_score']:.3f})"
                                    )
                        if drift_alerts and console:
                            console.print(
                                f"  [yellow]Drift detected: {', '.join(drift_alerts)}[/yellow]"
                            )
                        drift_monitor.save_state()
                    except Exception as dm_err:
                        logger.debug(f"Drift monitor error: {dm_err}")

                # Online RL micro-updates (US-060): every 5 cycles
                if self._online_rl:
                    try:
                        rl_adjustments = self._online_rl.maybe_update(self._scan_count)
                        if rl_adjustments:
                            logger.info(
                                "Online RL: %d micro-adjustments applied (cycle %d)",
                                len(rl_adjustments), self._scan_count,
                            )
                    except Exception as rl_err:
                        logger.debug(f"Online RL update error: {rl_err}")

                # Smart trading loop: monitor, drawdown guardian, RL sync, learning
                self._run_smart_loop()

                # Phase 20 (US-126): Memory management — cleanup every cycle, report every 10
                if getattr(self.scanner, "_memory_manager", None) is not None:
                    try:
                        mm = self.scanner._memory_manager
                        cleanup = mm.run_cleanup()

                        evictions = cleanup.get("evictions", {})
                        rotations = cleanup.get("rotations", {})
                        if (evictions or rotations) and console:
                            parts = []
                            if evictions:
                                total_evicted = sum(evictions.values())
                                parts.append(f"{total_evicted} cache entries evicted")
                            if rotations:
                                parts.append(f"{len(rotations)} logs rotated")
                            console.print(f"  [dim]Memory: {', '.join(parts)}[/dim]")

                        # RSS monitoring every 10 cycles
                        if self._scan_count % 10 == 0:
                            mem_report = cleanup.get("memory", {})
                            rss_mb = mem_report.get("rss_mb", 0)
                            is_critical = mem_report.get("rss_critical", False)
                            is_warning = mem_report.get("rss_warning", False)
                            if is_critical and console:
                                console.print(
                                    f"  [bold red]Memory CRITICAL: RSS={rss_mb:.0f}MB — throttling[/bold red]"
                                )
                                # Emergency cache clear on critical
                                for cache in mm._caches.values():
                                    cache._cache.clear()
                                    cache._access_order.clear()
                                logger.warning("Memory CRITICAL: emergency cache clear triggered")
                            elif is_warning and console:
                                console.print(
                                    f"  [yellow]Memory WARNING: RSS={rss_mb:.0f}MB[/yellow]"
                                )

                            # Pressure-driven log rotation: if RSS is warning+,
                            # force aggressive log rotation by temporarily halving
                            # max_lines thresholds, then running cleanup again.
                            if is_warning or is_critical:
                                logger.warning(
                                    "Memory pressure high (RSS=%.0fMB, %s) — "
                                    "forcing aggressive log rotation",
                                    rss_mb,
                                    "CRITICAL" if is_critical else "WARNING",
                                )
                                saved_thresholds = {}
                                try:
                                    for path, cfg in mm._log_files.items():
                                        saved_thresholds[path] = cfg["max_lines"]
                                        cfg["max_lines"] = cfg["max_lines"] // 2
                                    pressure_cleanup = mm.run_cleanup()
                                    pressure_rotations = pressure_cleanup.get("rotations", {})
                                    if pressure_rotations and console:
                                        console.print(
                                            f"  [yellow]Pressure rotation: "
                                            f"{len(pressure_rotations)} logs force-rotated[/yellow]"
                                        )
                                finally:
                                    for path, orig in saved_thresholds.items():
                                        if path in mm._log_files:
                                            mm._log_files[path]["max_lines"] = orig

                        if self._scan_count % 10 == 0:
                            mm.save_state()
                    except Exception as mm_err:
                        logger.debug(f"Memory cleanup error: {mm_err}")

                # Phase 20 (US-124): Module activation dispatch
                if self._module_dispatcher is not None:
                    try:
                        dispatch_results = self._module_dispatcher.execute_cycle(self._scan_count)
                        failed = [n for n, ok in dispatch_results.items() if not ok]
                        if failed and console:
                            console.print(f"  [dim red]Module dispatch failures: {', '.join(failed)}[/dim red]")
                        if self._scan_count % 10 == 0:
                            self._module_dispatcher.save_state()
                    except Exception as md_err:
                        logger.debug(f"Module dispatch error: {md_err}")

                # Phase 50: Crisis scenario generation every 50 cycles
                if (
                    self._scan_count % 50 == 0
                    and self._scan_count > 0
                    and getattr(self.scanner, "_crisis_generator", None) is not None
                ):
                    try:
                        crisis_gen = self.scanner._crisis_generator
                        candle_cache = getattr(self.scanner, "_cached_candles", {})
                        # Pick the first pair with sufficient candle history
                        _crisis_candles = None
                        for _pair, _candles in candle_cache.items():
                            if hasattr(_candles, "__len__") and len(_candles) >= 50:
                                import numpy as np
                                _crisis_candles = np.array(_candles, dtype=np.float64)
                                break
                        if _crisis_candles is not None:
                            scenarios = crisis_gen.generate_scenario_batch(
                                base_candles=_crisis_candles,
                                n_scenarios=5,
                            )
                            if scenarios and console:
                                console.print(
                                    f"  [dim]Crisis generator: {len(scenarios)} synthetic "
                                    f"scenarios generated (cycle {self._scan_count})[/dim]"
                                )
                            logger.info(
                                "CrisisGenerator: %d scenarios generated (cycle %d)",
                                len(scenarios) if scenarios else 0,
                                self._scan_count,
                            )
                        else:
                            logger.debug(
                                "CrisisGenerator: skipped — no pair with >= 50 cached candles"
                            )
                    except Exception as cg_err:
                        logger.debug(f"Crisis generator error: {cg_err}")

                # US-005: Strategy invention — run GA evolution every 100 cycles
                if (
                    self._scan_count % 100 == 0
                    and self._scan_count > 0
                    and getattr(self.scanner.config, "enable_strategy_invention", False)
                ):
                    try:
                        from src.strategy_invention.evolution_controller import EvolutionController

                        _evo = EvolutionController()
                        _evo.load()
                        # Simple fitness: count valid strategy blocks (no backtest needed at this stage)
                        _top = _evo.run_evolution_cycle(
                            fitness_fn=lambda s: getattr(s, "fitness", 0.0),
                            generations=getattr(self.scanner.config, "strategy_evolution_generations", 5),
                            population_size=getattr(self.scanner.config, "strategy_population_size", 30),
                        )
                        _evo.save()
                        _evo.retire_underperforming()
                        _summary = _evo.get_status_summary()
                        if console:
                            console.print(
                                f"  [cyan]Strategy evolution: {_summary.get('total_strategies', 0)} "
                                f"strategies (cycle {self._scan_count})[/cyan]"
                            )
                        logger.info("EvolutionController: cycle complete | %s", _summary)
                    except Exception as evo_err:
                        logger.debug(f"Strategy evolution error: {evo_err}")

                # Phase 21 (US-131): Apply pending config adjustments every 10 cycles
                if (
                    self._scan_count % 10 == 0
                    and getattr(self.scanner, "_config_adjuster", None) is not None
                ):
                    try:
                        adjuster = self.scanner._config_adjuster
                        applied = adjuster.apply_adjustments(self.scanner.config, self._scan_count)
                        if applied and console:
                            console.print(
                                f"  [cyan]Config adjusted: {len(applied)} changes "
                                f"({', '.join(a['key'] for a in applied)})[/cyan]"
                            )
                        adjuster.save_state()
                    except Exception as ca_err:
                        logger.debug(f"Config adjuster error: {ca_err}")

                # Idle maintenance
                if self._maintenance:
                    self._maintenance.run_if_needed()

            except Exception as e:
                error_class = type(e).__name__
                logger.error(f"Scan cycle #{self._scan_count} failed ({error_class}): {e}")
                if console:
                    console.print(f"\n[bold red]Scan #{self._scan_count} failed[/bold red]")
                    console.print(f"  [red]{error_class}: {e}[/red]")
                    # Actionable hints
                    err_str = str(e).lower()
                    if "connection" in err_str or "timeout" in err_str:
                        console.print("  [dim]Check: network connection, OANDA API status[/dim]")
                    elif "model" in err_str or "shape" in err_str or "not supported between" in err_str:
                        console.print("  [dim]Check: model compatibility — may need retraining[/dim]")
                    elif "import" in err_str or "module" in err_str:
                        console.print("  [dim]Check: conda environment activated with all dependencies[/dim]")
                    else:
                        console.print("  [dim]Check logs for full traceback[/dim]")

            if self._running:
                self._sleep_with_progress(interval_minutes)

        if console:
            console.print(f"\n[green]✓ Continuous scan stopped after {self._scan_count} scans[/green]")

        # Phase 22 (US-138) + Phase 29 (US-175): Persist all module states on graceful shutdown
        _shutdown_modules = [
            ("config_adjuster", "_config_adjuster"),
            ("threshold_optimizer", "_threshold_optimizer"),
            ("agent_lifecycle", "_agent_lifecycle"),
            ("observation_consumer", "_observation_consumer"),
            ("memory_manager", "_memory_manager"),
            # Phase 29 (US-175): Additional module state persistence
            ("health_registry", "_health_registry"),
            ("agent_accuracy_matrix", "_agent_accuracy_matrix"),
            ("pair_regime_agent_matrix", "_pair_regime_agent_matrix"),
            ("signal_timing", "_signal_timing"),
            ("model_calibration", "_model_calibration"),
            ("dynamic_risk_allocator", "_dynamic_risk_allocator"),
            ("execution_quality_optimizer", "_execution_quality_optimizer"),
            ("drift_monitor", "_drift_monitor"),
            ("module_dispatcher", "_module_dispatcher"),
            ("replay_validator", "_replay_validator"),
            ("module_activation", "_module_activation"),
            # Phase 33 (US-213): Additional modules with save_state()
            ("affinity_portfolio", "_affinity_portfolio"),
            ("attention_feedback", "_attention_feedback"),
            ("causal_filter", "_causal_filter"),
            ("execution_quality_tracker", "_exec_quality_tracker"),
            ("execution_router", "_execution_router"),
            ("model_bandit", "_model_bandit"),
            ("model_router", "_model_router"),
            ("pair_transfer", "_pair_transfer"),
            ("position_manager", "_position_manager"),
            ("regime_broadcaster", "_regime_broadcaster"),
            ("temporal_attention", "_temporal_attention"),
            ("training_augmenter", "_training_augmenter"),
            # Phase 50: Crisis generator uses save_log() instead of save_state()
            ("crisis_generator", "_crisis_generator"),
            # US-006–US-019: Newly registered modules
            ("counterfactual_learner", "_counterfactual_learner"),
            ("chain_memory", "_chain_memory"),
            ("drift_projector", "_drift_projector"),
            ("episodic_memory", "_episodic_memory"),
            ("self_model_state", "_self_model_state"),
            ("causal_counterfactual", "_causal_counterfactual"),
            # Phase 78: Missing save_log() modules discovered by gap analysis
            ("adaptive_lr", "_adaptive_lr"),
            ("entropy_sizer", "_entropy_sizer"),
            ("market_impact", "_market_impact"),
            ("regime_reward", "_regime_reward"),
            ("trade_explainer", "_trade_explainer"),
        ]
        for mod_label, mod_attr in _shutdown_modules:
            try:
                mod = getattr(self.scanner, mod_attr, None) if hasattr(self, "scanner") else None
                if mod is None:
                    mod = getattr(self, mod_attr, None)
                if mod is not None:
                    _saved = False
                    for _method in ("save_state", "save_log", "save", "save_projections", "save_analysis"):
                        if hasattr(mod, _method):
                            getattr(mod, _method)()
                            logger.info(f"Shutdown: {mod_label} persisted via {_method}()")
                            _saved = True
                            break
                    if not _saved:
                        logger.debug(f"Shutdown: {mod_label} has no persistence method")
            except Exception as _mod_err:
                logger.debug(f"Shutdown: {mod_label} save failed: {_mod_err}")

        # Phase 74: Persist scanner mutable state (EWMA, prices, regimes, cycle count)
        try:
            if hasattr(self, "scanner") and hasattr(self.scanner, "save_scan_state"):
                self.scanner.save_scan_state()
                logger.info("Shutdown: scanner scan state persisted")
        except Exception as _scan_state_err:
            logger.debug(f"Shutdown: scanner state save failed: {_scan_state_err}")

        # Gate attribution engine state - save from scanner actual EM if available
        try:
            _exec_mgr = getattr(self.scanner, "_execution_manager", None)
            if _exec_mgr is not None and getattr(_exec_mgr, "_gate_attribution", None) is not None:
                _exec_mgr._gate_attribution.save_state()
                logger.info("Shutdown: gate_attribution state saved (from scanner EM)")
        except Exception as _ga_err:
            logger.debug(f"Shutdown: gate_attribution save failed: {_ga_err}")

        # StateEngine has a multi-arg save_state() — call separately from the generic loop
        try:
            from src.scanner.automation.state_engine import StateEngine
            se = StateEngine()
            se.save_state(
                goal="continuous_scan",
                status="shutdown",
                done=[f"completed_{self._scan_count}_scans"],
                next_action="restart_scan_loop",
            )
            logger.info("Shutdown: state_engine state saved")
        except Exception as _se_err:
            logger.debug(f"Shutdown: state_engine save failed: {_se_err}")

        # Save session snapshot on clean exit (Phase 8: cross-session learning)
        try:
            from src.scanner.automation.session_snapshot import SessionSnapshotManager
            from src.scanner.agents._team import ScannerAgentTeam

            snapshot_mgr = SessionSnapshotManager()
            # Load current agent weights
            agent_team = ScannerAgentTeam(self.scanner.config)  # Phase 31 (US-187): was self.config (ContinuousConfig, not ScannerConfig)
            weights = agent_team.get_learned_weights()

            # Load pair accuracies
            pair_acc: Dict[str, float] = {}
            try:
                from src.scanner.automation.safe_json import safe_json_read
                from pathlib import Path
                acc_path = Path("trained_data/models/pair_accuracy.json")
                pair_acc = safe_json_read(acc_path, default={}) or {}
            except Exception as e:
                logger.debug(f"Session snapshot pair accuracy load skipped: {e}")

            snapshot_mgr.save_snapshot(
                agent_weights=weights,
                pair_accuracies=pair_acc,
                scan_cycles=self._scan_count,
                profile=getattr(self.config, "profile", "balanced"),
            )
        except Exception as snap_err:
            logger.debug(f"Session snapshot save skipped: {snap_err}")

        return self._scan_count

    def stop(self):
        """Stop the continuous scan loop."""
        self._running = False
        if console:
            console.print("\n[yellow]Stopping continuous scan...[/yellow]")

    def _setup_signal_handler(self):
        """Setup Ctrl+C handler for graceful shutdown."""
        def handler(sig, frame):
            self.stop()

        signal.signal(signal.SIGINT, handler)

    def _apply_pair_rotation(self, available_pairs: Optional[List[str]] = None) -> None:
        """Apply dynamic pair rotation based on rolling Sharpe ratio.

        Runs every rotation_interval cycles. Ranks pairs by Sharpe and updates
        active/observe status. Observe-only pairs are still scanned but not traded.

        Args:
            available_pairs: List of pairs available to scan (for filtering)
        """
        if not self._portfolio_optimizer:
            return

        try:
            # Check if it's time to rotate
            if not self._portfolio_optimizer.should_rotate(self._scan_count):
                return

            # Rank all pairs and get active set
            rankings = self._portfolio_optimizer.rank_pairs()
            active_pairs = self._portfolio_optimizer.get_active_pairs()
            observe_pairs = self._portfolio_optimizer.get_observe_only_pairs(all_pairs=available_pairs)

            if not rankings:
                logger.debug("No pair rankings generated in rotation check")
                return

            # Save rankings to disk
            self._portfolio_optimizer.save_rankings(rankings)

            # Log rotation decision
            self._portfolio_optimizer.log_rotation_decision(
                scan_cycle=self._scan_count,
                active_pairs=active_pairs,
                observe_pairs=observe_pairs,
            )

            # Display rotation summary if console available
            if console and (active_pairs or observe_pairs):
                console.print(f"\n[cyan]📊 PAIR ROTATION (Cycle {self._scan_count})[/cyan]")
                if active_pairs:
                    console.print(f"  [green]Active ({len(active_pairs)}): {', '.join(active_pairs)}")
                if observe_pairs:
                    console.print(f"  [yellow]Observe-only ({len(observe_pairs)}): {', '.join(observe_pairs[:5])}", end="")
                    if len(observe_pairs) > 5:
                        console.print(f" +{len(observe_pairs) - 5} more[/yellow]")
                    else:
                        console.print("[/yellow]")

        except Exception as e:
            logger.debug(f"Pair rotation error: {e}")

    def _log_scan_cycle(
        self,
        result: "ScanResult",
        auto_execute: bool,
    ) -> None:
        """Append a lightweight record of this scan cycle for post-hoc analytics."""
        import json
        from pathlib import Path

        log_path = Path("trained_data/scan_cycle_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        tradeable = [a for a in result.analyses if a.is_tradeable]
        record = {
            "timestamp": datetime.now().isoformat(),
            "scan_number": self._scan_count,
            "pairs_scanned": len(result.analyses),
            "tradeable_count": len(tradeable),
            "tradeable_pairs": [a.pair for a in tradeable],
            "auto_execute": auto_execute,
            "top_score": round(max((a.overall_score for a in result.analyses), default=0), 4),
            "model_type": result.model_type,
        }

        try:
            from src.scanner.automation.safe_json import safe_jsonl_append
            safe_jsonl_append(log_path, record)
        except Exception as e:
            logger.debug(f"Scan cycle log error: {e}")

        # Phase 56 (US-350): Feed scan result to diagnostics reporter
        if self._scan_diagnostics is not None:
            try:
                _diag_pairs = [
                    {
                        "confidence": float(a.confidence),
                        "is_tradeable": bool(a.is_tradeable),
                        "direction": str(getattr(a, "direction", "HOLD")),
                    }
                    for a in result.analyses
                ]
                self._scan_diagnostics.record_scan(_diag_pairs)
                # Phase 58 (US-361): Advance adaptive floor cooldown each scan
                if self._adaptive_floor is not None:
                    try:
                        self._adaptive_floor.tick()
                    except Exception as _af_tick_err:
                        logger.debug(f"Phase 58: adaptive_floor.tick() failed: {_af_tick_err}")
                if self._scan_diagnostics.should_report():
                    _diag_report = self._scan_diagnostics.generate_report()
                    self._scan_diagnostics.save_report(_diag_report)
                    if _diag_report.get("stalled"):
                        logger.warning(
                            "Phase 56 (US-350): STALLED — %d scans with no tradeable pair. %s",
                            _diag_report["scans_since_last_trade"],
                            _diag_report["diagnostic_summary"],
                        )
                        # Phase 57 (US-356): Run penalty audit when stalled
                        if self._penalty_auditor is not None:
                            try:
                                _audit = self._penalty_auditor.run_audit(
                                    scan_diagnostics_report=_diag_report,
                                )
                                self._penalty_auditor.save_audit(_audit)
                            except Exception as _pa_err:
                                logger.debug(f"Phase 57: Penalty audit failed: {_pa_err}")
                        # Phase 58 (US-360): Gate pass predictor — surface signal diagnostics
                        if self._gate_pass_predictor is not None:
                            try:
                                _recorder = getattr(self.scanner, "_raw_conf_recorder", None)
                                if _recorder is not None:
                                    _min_conf = getattr(self.scanner.config, "min_confidence", 58.0)
                                    _gpp_report = self._gate_pass_predictor.generate_report(
                                        _recorder,
                                        min_confidence=float(_min_conf),
                                        stall_info={"scans_since_last_trade": _diag_report.get("scans_since_last_trade", 0)},
                                    )
                                    # Phase 58 (US-361): Adaptive floor — lower threshold if warranted
                                    if self._adaptive_floor is not None:
                                        try:
                                            from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
                                            _gap_analyzer = ThresholdGapAnalyzer()
                                            _gap_report = _gap_analyzer.analyze(_recorder, float(_min_conf))
                                            _scans_stalled = _diag_report.get("scans_since_last_trade", 0)
                                            if self._adaptive_floor.should_adapt(_gap_report, _gpp_report, _scans_stalled):
                                                _delta = self._adaptive_floor.compute_adjustment(_gap_report, _gpp_report)
                                                if _delta != 0.0:
                                                    self._adaptive_floor.apply_adjustment(self.scanner.config, _delta)
                                        except Exception as _af_err:
                                            logger.debug(f"Phase 58: Adaptive floor failed: {_af_err}")
                            except Exception as _gpp_err:
                                logger.debug(f"Phase 58: Gate pass predictor failed: {_gpp_err}")
                    else:
                        logger.info(
                            "Phase 56 (US-350): Diagnostics report — gate_pass_rate=%.1f%% avg_conf=%.3f stalled=%s",
                            _diag_report["gate_pass_rate_pct"],
                            _diag_report["avg_confidence"],
                            _diag_report["stalled"],
                        )
                self._scan_diagnostics.save_state()
            except Exception as _sdr_err:
                logger.debug(f"Phase 56: Scan diagnostics record error: {_sdr_err}")

        # Phase 59 (US-363): Record signal funnel attrition for this scan
        if self._signal_funnel is not None and result.analyses:
            try:
                _funnel = self._signal_funnel.record_scan(result.analyses)
                _kill = _funnel.get("dominant_kill_stage", "none")
                if _kill not in ("none", ""):
                    logger.info(
                        "Phase 59 (US-363): Funnel — evaluated=%d tradeable=%d (%.1f%%) dominant_kill=%s",
                        _funnel.get("total_evaluated", 0),
                        _funnel.get("is_tradeable", 0),
                        _funnel.get("tradeable_pct", 0.0),
                        _kill,
                    )
                self._signal_funnel.save_state()
            except Exception as _sft_err:
                logger.debug(f"Phase 59: signal_funnel.record_scan failed: {_sft_err}")

        # Phase 60 (US-369): Unified health synthesis — every 10 scan cycles
        if self._scan_health is not None and self._scan_count % 10 == 0:
            try:
                _health = self._scan_health.synthesize()
                _status = _health.get("health_status", "HEALTHY")
                _summary = _health.get("summary", "")
                _top_action = _health.get("top_action", "")
                _bottlenecks = _health.get("bottlenecks", [])
                if _status == "CRITICAL":
                    logger.error(
                        "Phase 60 (US-369): HEALTH=%s | %s | top_action=%s | bottlenecks=%s",
                        _status, _summary, _top_action,
                        [b["name"] for b in _bottlenecks],
                    )
                elif _status == "DEGRADED":
                    _top_bn = _bottlenecks[0]["name"] if _bottlenecks else "none"
                    logger.warning(
                        "Phase 60 (US-369): HEALTH=%s | top_bottleneck=%s | action=%s",
                        _status, _top_bn, _top_action,
                    )
                else:
                    logger.info(
                        "Phase 60 (US-369): HEALTH=%s | %s",
                        _status, _summary,
                    )
            except Exception as _shs_err:
                logger.debug(f"Phase 60: scan_health.synthesize() failed: {_shs_err}")

    def _filter_correlated_exposure(self, tradeable: list) -> list:
        """Filter out trades that would double exposure on correlated pairs already open."""
        try:
            from src.scanner.execution import ExecutionManager
            from src.training.correlation_group_config import get_correlation_group

            em = ExecutionManager()
            open_trades = em.monitor_open_trades()
            if not open_trades:
                return tradeable

            # Build set of correlation groups currently exposed
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
                    if console:
                        console.print(f"  [dim]Skipping {a.pair}: already open[/dim]")
                    continue
                group = get_correlation_group(a.pair)
                if group and group.master_pair in open_groups:
                    if console:
                        console.print(
                            f"  [dim]Skipping {a.pair}: correlated with open "
                            f"{group.master_pair} group[/dim]"
                        )
                    continue
                filtered.append(a)

            return filtered

        except Exception as e:
            logger.debug(f"Correlation filter error: {e}")
            return tradeable

    def _run_smart_loop(self) -> None:
        """Run the smart trading loop: monitor, guardian, RL sync, learning, adaptation."""
        try:
            from src.scanner.execution import ExecutionManager

            em = ExecutionManager()

            # 1. Monitor open trades
            statuses = em.monitor_open_trades()
            if statuses and console:
                console.print(f"\n[dim]Open trades: {len(statuses)}[/dim]")
                for s in statuses:
                    pl_color = "green" if s["unrealized_pl"] >= 0 else "red"
                    console.print(
                        f"  [{pl_color}]{s['pair']} {s['direction']} "
                        f"P/L ${s['unrealized_pl']:.2f} "
                        f"(SL {s['sl_dist_pips']:.0f}p / TP {s['tp_dist_pips']:.0f}p, "
                        f"{s['time_in_minutes']}m)[/{pl_color}]"
                    )

            # 2. Drawdown guardian
            modifications = em.apply_drawdown_guardian()
            if modifications and console:
                for mod in modifications:
                    console.print(f"  [yellow]Guardian: {mod}[/yellow]")

            # 2b. Trailing stop updates
            try:
                trailing_updates = em.update_trailing_stops()
                if trailing_updates and console:
                    for upd in trailing_updates:
                        console.print(f"  [magenta]Trailing SL: {upd}[/magenta]")
            except Exception as e:
                if console:
                    console.print(f"  [dim red]Trailing stop error: {e}[/dim red]")

            # 2c. Phase 24 (US-147): Live position management via win_prob predictor
            _pm_interval = int(getattr(self.scanner.config, "position_management_interval", 3))
            if (
                getattr(self.scanner.config, "enable_position_management", True)
                and self._scan_count % _pm_interval == 0
                and statuses  # Only if there are open trades
            ):
                try:
                    from src.scanner.automation.position_manager import (
                        LivePositionManager,
                        PositionContext,
                    )
                    from src.scanner.automation.observation_log import ObservationLog

                    _pm = LivePositionManager()
                    _obs_log = ObservationLog()
                    _pm_actions = []

                    for s in statuses:
                        # Build PositionContext from OANDA trade status
                        _ctx = PositionContext(
                            trade_id=s.get("trade_id", ""),
                            pair=s.get("pair", ""),
                            direction="BUY" if s.get("direction") == "LONG" else "SELL",
                            entry_price=float(s.get("entry", 0)),
                            current_price=float(s.get("entry", 0)),  # Approximate
                            sl_price=float(s.get("sl_price", 0)),
                            tp_price=float(s.get("tp_price", 0)),
                            lot_size=abs(int(s.get("units", 0))),
                            bars_in_trade=max(1, s.get("time_in_minutes", 0) // 60),
                            mae_pips=0.0,  # Not tracked in monitor_open_trades
                            mfe_pips=max(0.0, s.get("tp_dist_pips", 0) * 0.3),  # Estimate
                            entry_atr=0.0,
                            current_atr=0.0,
                            momentum_since_entry=0.0,
                            regime=str(
                                getattr(self.scanner.config, "volatility_regime_default", "NORMAL")
                            ).upper(),
                        )

                        # Evaluate position
                        action = _pm.evaluate_position(_ctx)

                        if action.action == "HOLD":
                            continue  # No action needed

                        _pm_actions.append((s, action))

                        # Log to observation_log
                        _obs_log.log_observation(
                            pair=_ctx.pair,
                            category="position_management",
                            description=(
                                f"US-147: {action.action} for #{_ctx.trade_id} "
                                f"(win_prob={action.win_probability:.2f}, {action.reason})"
                            ),
                            metadata={
                                "trade_id": _ctx.trade_id,
                                "action": action.action,
                                "win_probability": action.win_probability,
                                "confidence": action.confidence,
                            },
                        )

                        # Execute the recommended action via OANDA
                        if action.action == "TIGHTEN_SL" and action.new_sl_price is not None:
                            success = em.modify_sl_oanda(
                                trade_id=_ctx.trade_id,
                                new_sl=action.new_sl_price,
                            )
                            if success and console:
                                console.print(
                                    f"  [yellow]PM: Tightened SL for {_ctx.pair} "
                                    f"#{_ctx.trade_id} → {action.new_sl_price:.5f} "
                                    f"(win_prob={action.win_probability:.2f})[/yellow]"
                                )
                        elif action.action == "EXIT_EARLY":
                            success = em.close_trade_oanda(trade_id=_ctx.trade_id)
                            if success and console:
                                console.print(
                                    f"  [red]PM: Early exit {_ctx.pair} "
                                    f"#{_ctx.trade_id} "
                                    f"(win_prob={action.win_probability:.2f})[/red]"
                                )
                            # Cleanup tracking
                            _pm.cleanup_trade(_ctx.trade_id)
                        elif action.action == "TAKE_PARTIAL":
                            # Log partial take — full partial close would need
                            # units-based close which is more complex; log for now
                            if console:
                                console.print(
                                    f"  [cyan]PM: Partial take recommended for {_ctx.pair} "
                                    f"#{_ctx.trade_id} "
                                    f"({action.close_ratio:.0%}, win_prob={action.win_probability:.2f})[/cyan]"
                                )

                    if _pm_actions and console:
                        console.print(
                            f"  [dim]Position manager: {len(_pm_actions)} actions "
                            f"across {len(statuses)} open trades[/dim]"
                        )

                    _pm.save_state()
                except Exception as pm_err:
                    logger.debug(f"Position manager error: {pm_err}")

            # 3. RL feedback sync
            rl_result = em.sync_closed_trades_rl(scanner=self.scanner)
            trades_synced = rl_result.get("trades_synced", 0)
            if trades_synced > 0 and console:
                console.print(
                    f"  [cyan]RL sync: {rl_result['detail']}[/cyan]"
                )
                if rl_result.get("new_weights"):
                    for agent, w in rl_result["new_weights"].items():
                        console.print(f"    {agent}: {w:.3f}")

            # 3b. Phase 20 (US-122): Update agent lifecycle fitness from closed trades
            if trades_synced > 0 and getattr(self.scanner, "_agent_lifecycle", None) is not None:
                try:
                    lifecycle = self.scanner._agent_lifecycle
                    # Read recently synced trades from journal for agent-level feedback
                    from pathlib import Path
                    from src.scanner.automation.safe_json import safe_json_read
                    journal_path = Path("trained_data/trade_journal_rl.json")
                    journal = safe_json_read(journal_path, default=[])
                    if isinstance(journal, list) and journal:
                        recent = journal[-trades_synced:]
                        for trade in recent:
                            won = trade.get("pnl_pips", 0) > 0 or trade.get("pnl", 0) > 0
                            regime = trade.get("volatility_regime", "NORMAL")
                            # Update each agent that voted on this trade
                            agent_votes = trade.get("analysis_context", {}).get("agent_reasons", [])
                            if agent_votes:
                                for av in agent_votes:
                                    agent_name = av.get("name", av.get("agent", ""))
                                    if agent_name:
                                        lifecycle.update_fitness(agent_name, won, regime)
                            else:
                                # Fallback: update all known agents
                                for agent_name in ["trend", "mean_reversion", "volatility",
                                                   "risk_sentinel", "uncertainty", "momentum"]:
                                    lifecycle.update_fitness(agent_name, won, regime)
                        lifecycle.save_state()
                        transitions = []
                        for agent_name in lifecycle._agents:
                            status = lifecycle.get_agent_status(agent_name)
                            if status != "ACTIVE":
                                transitions.append(f"{agent_name}={status}")
                        if transitions:
                            if console:
                                console.print(f"  [yellow]Agent lifecycle: {', '.join(transitions)}[/yellow]")
                            # US-122: Log status transitions to observation_log
                            try:
                                from src.scanner.automation.observation_log import ObservationLog
                                obs_log = ObservationLog()
                                for t in transitions:
                                    agent_name, agent_status = t.split("=", 1)
                                    obs_log.log_observation(
                                        pair="SYSTEM",
                                        category="agent_lifecycle",
                                        description=f"Agent '{agent_name}' status: {agent_status}",
                                        metadata={"agent": agent_name, "status": agent_status,
                                                  "regime": regime, "trades_synced": trades_synced},
                                    )
                            except Exception as _obs_err:
                                logger.debug("US-189: Agent lifecycle observation log skipped: %s", _obs_err)
                except Exception as lc_err:
                    logger.debug(f"Agent lifecycle update error: {lc_err}")

            # 3b-ii. Phase 26 (US-158): Agent degradation early-warning (every 5 cycles)
            if (
                self._scan_count % 5 == 0  # Phase 31 (US-186): was cycle_count (NameError)
                and getattr(self.scanner, "_agent_lifecycle", None) is not None
            ):
                try:
                    _degradation_warnings = self.scanner._agent_lifecycle.check_degradation()
                    if _degradation_warnings:
                        logger.info(
                            "US-158: %d agent(s) showing degradation",
                            len(_degradation_warnings),
                        )
                        try:
                            from src.scanner.automation.observation_log import ObservationLog
                            _obs = ObservationLog()
                            for _dw in _degradation_warnings:
                                _obs.log_observation(
                                    pair="SYSTEM",
                                    category="agent_degradation_warning",
                                    description=(
                                        f"US-158: Agent '{_dw['agent_name']}' degrading — "
                                        f"z={_dw['z_score']:.2f}, "
                                        f"baseline={_dw['baseline_rate']:.1%} → "
                                        f"recent={_dw['recent_rate']:.1%}"
                                    ),
                                    metadata=_dw,
                                )
                        except Exception as e:
                            logger.debug(f"Degradation observation logging skipped: {e}")
                except Exception as _deg_err:
                    logger.debug(f"US-158: Degradation check error: {_deg_err}")

            # 3c. Phase 25 (US-153): Feed trade outcomes to threshold_optimizer
            if (
                trades_synced > 0
                and getattr(self.scanner, "_threshold_optimizer", None) is not None
            ):
                try:
                    _to = self.scanner._threshold_optimizer
                    from pathlib import Path as _P25Path
                    from src.scanner.automation.safe_json import safe_json_read as _p25_read
                    _j25 = _p25_read(_P25Path("trained_data/trade_journal_rl.json"), default=[])
                    if isinstance(_j25, list) and _j25:
                        _recent = _j25[-trades_synced:]
                        _fed = 0
                        for _t in _recent:
                            _outcome = _t.get("outcome")
                            if _outcome is None:
                                continue
                            _regime = _t.get("regime", {})
                            _regime_name = (
                                _regime.get("volatility_regime", "NORMAL")
                                if isinstance(_regime, dict)
                                else str(_regime)
                            )
                            _gate_vals = {
                                "confidence": float(_t.get("confidence", 0)),
                                "momentum": float(
                                    _t.get("agents", {}).get("weighted_vote_score", 0)
                                ),
                                "rr_ratio": (
                                    float(_t.get("tp_pips", 0))
                                    / max(float(_t.get("sl_pips", 1)), 0.1)
                                ),
                            }
                            _won = _outcome.get("trade_won", False)
                            _to.update_outcome(_regime_name, _gate_vals, _won)
                            _fed += 1
                        if _fed > 0 and console:
                            console.print(
                                f"  [dim]ThresholdOptimizer: fed {_fed} outcomes "
                                f"(total: {_to._total_outcomes})[/dim]"
                            )
                        # Apply optimized thresholds every 10 cycles
                        if self._scan_count % 10 == 0 and _to._total_outcomes >= 10:
                            for _rn in ["NORMAL", "HIGH", "EXTREME"]:
                                _new_th = _to.optimize_thresholds(_rn)
                                if _new_th:
                                    logger.info(
                                        f"US-153: Optimized thresholds for {_rn}: {_new_th}"
                                    )
                        _to.save_state()
                except Exception as _to_err:
                    logger.debug(f"ThresholdOptimizer feed error: {_to_err}")

            # 3d. Phase 27 (US-163): Observation consumer feedback loop (every 10 cycles)
            if (
                self._scan_count % 10 == 0  # Phase 31 (US-186): was cycle_count (NameError)
                and getattr(self.scanner, "_observation_consumer", None) is not None
            ):
                try:
                    _oc = self.scanner._observation_consumer
                    _new_obs = _oc.consume_observations()
                    if _new_obs > 0:
                        _patterns = _oc.detect_patterns()
                        _recommendations = _oc.recommend_adjustments()
                        _alerts = _oc.check_alerts(window_hours=6)
                        _oc.save_state()
                        if _patterns or _recommendations or _alerts:
                            logger.info(
                                "US-163: ObservationConsumer — %d new obs, "
                                "%d patterns, %d recommendations, %d alerts",
                                _new_obs, len(_patterns),
                                len(_recommendations), len(_alerts),
                            )
                        if _alerts:
                            try:
                                from src.scanner.automation.observation_log import ObservationLog
                                _obs_log = ObservationLog()
                                for _alert in _alerts[:5]:  # Cap at 5 alerts per cycle
                                    _obs_log.log_observation(
                                        pair=_alert.get("pair", "SYSTEM"),
                                        category="observation_consumer_alert",
                                        description=(
                                            f"US-163: Alert — {_alert.get('type', 'unknown')}: "
                                            f"{_alert.get('message', '')}"
                                        ),
                                        metadata=_alert,
                                    )
                            except Exception as e:
                                logger.debug(f"Observation consumer alert logging skipped: {e}")
                except Exception as _oc_err:
                    logger.debug(f"US-163: ObservationConsumer error: {_oc_err}")

            # 4. Agent weight decay toward baseline
            try:
                from src.scanner.agents import ScannerAgentTeam
                team = ScannerAgentTeam(self.scanner.config)
                decayed = team.apply_weight_decay(decay_rate=0.02)
                if decayed and console:
                    console.print(f"  [dim]Weight decay applied ({len(decayed)} agents)[/dim]")
            except Exception as decay_err:
                logger.debug(f"Weight decay error: {decay_err}")

            # 5. Learning loop (US-012)
            self._run_learning_loop(em, trades_synced)

        except Exception as e:
            # Phase 32 (US-194): Escalated from debug to warning — this hides trading loop errors
            logger.warning(f"Smart loop error ({type(e).__name__}): {e}", exc_info=True)

    def _run_learning_loop(self, em: object, trades_synced: int) -> None:
        """Step 5: Learning engine analysis, promotion, config tuning, metrics.

        Wraps all learning components in try/except to never crash the scan loop.
        """
        try:
            import json
            from pathlib import Path
            from src.scanner.automation.learning_engine import LearningEngine
            from src.scanner.automation.config_tuner import ConfigTuner
            from src.scanner.automation.improvement_tracker import ImprovementTracker
            from src.scanner.automation.state_engine import StateEngine

            le = LearningEngine()
            learnings_added = 0
            rules_promoted = 0
            config_adjustments = []

            # 5a. Analyze each newly synced trade
            if trades_synced > 0:
                journal_path = Path("trained_data/trade_journal_rl.json")
                if journal_path.exists():
                    # Phase 32 (US-195): Guard JSON parse — use mtime cache
                    try:
                        entries = self._load_journal_cached(journal_path)
                    except (json.JSONDecodeError, OSError) as _je:
                        logger.warning(f"Learning loop journal parse failed: {_je}")
                        entries = []
                    if not isinstance(entries, list):
                        entries = []
                    closed = [e for e in entries if e.get("outcome") is not None]
                    # Phase 32 (US-195): Only process last trades_synced entries
                    closed = closed[-trades_synced:] if trades_synced < len(closed) else closed

                    for entry in closed:
                        insights = le.analyze_trade(entry)
                        if insights:
                            learnings_added += le.append_to_learnings(insights)

                        # Per-pair SL/TP adaptation (US-006)
                        le.update_pair_sl_tp(entry)

                        # Model A/B scoring: record trade outcome as shadow test result (US-015)
                        try:
                            from src.scanner.automation.model_manager import ModelManager
                            mm = ModelManager()
                            mm.score_from_trade_outcome(entry)
                        except Exception as mm_err:
                            logger.debug(f"ModelManager scoring error: {mm_err}")

                        # Adaptive risk scaling: feed outcome to scaler (US-028)
                        try:
                            from src.risk.adaptive_scaler import AdaptiveRiskScaler
                            scaler = AdaptiveRiskScaler()
                            outcome_data = entry.get("outcome", {})
                            won = float(outcome_data.get("realized_pl", 0)) > 0
                            scaler.record_outcome(won)
                        except Exception as scaler_err:
                            logger.debug(f"Risk scaler error: {scaler_err}")

                        # US-020: Counterfactual learner — analyse closed trades
                        try:
                            _cf_learner = getattr(self.scanner, "_counterfactual_learner", None)
                            if _cf_learner is not None:
                                _cf_insight = _cf_learner.analyze_closed_trade(entry)
                                if _cf_insight:
                                    learnings_added += 1
                                    logger.info("Counterfactual insight for trade %s", entry.get("trade_id", "?"))
                        except Exception as cf_err:
                            logger.debug(f"Counterfactual learner error: {cf_err}")

                        # LLM deep analysis for significant losses (US-009)
                        if getattr(self.scanner.config, "enable_llm_trade_analysis", False):
                            outcome = entry.get("outcome", {})
                            if outcome.get("realized_pl", 0) < -100:
                                pair_entries = [e for e in entries if e.get("pair") == entry.get("pair")]
                                llm_insights = le.deep_analyze_loss(entry, pair_entries)
                                if llm_insights:
                                    learnings_added += le.append_to_learnings(llm_insights)

            # 5b. Check for rule promotions
            promoted = le.check_promotions()
            rules_promoted = len(promoted)

            # 5c. Apply config tuning from promoted rules
            try:
                ct = ConfigTuner()
                config_adjustments = ct.apply_to_config(self.scanner.config)
                if config_adjustments:
                    logger.info(f"Config tuned: {len(config_adjustments)} adjustments applied")
            except Exception as tune_err:
                logger.debug(f"Config tuning error: {tune_err}")

            # 5c-ii. Run lightweight QA audit (every 20 cycles)
            if self._scan_count % 20 == 0:
                try:
                    from src.scanner.automation.qa_pipeline import QAPipeline
                    qa = QAPipeline()
                    qa_report = qa.run_full_audit()
                    critical_count = sum(1 for f in qa_report.findings if f.severity == "critical")
                    if critical_count > 0 and console:
                        console.print(f"  [red]⚠ QA: {critical_count} critical issue(s) detected[/red]")
                    elif qa_report.findings and console:
                        warning_count = sum(1 for f in qa_report.findings if f.severity == "warning")
                        if warning_count > 0:
                            console.print(f"  [yellow]QA: {warning_count} warning(s)[/yellow]")
                except Exception as qa_err:
                    logger.debug(f"QA pipeline error: {qa_err}")

            # 5d. Record session metrics
            if trades_synced > 0:
                try:
                    tracker = ImprovementTracker()
                    journal_path = Path("trained_data/trade_journal_rl.json")
                    all_entries = self._load_journal_cached(journal_path) if journal_path.exists() else []
                    tracker.record_session(
                        trades=all_entries,
                        learnings_added=learnings_added,
                        rules_promoted=rules_promoted,
                        config_adjustments=config_adjustments,
                    )
                except Exception as track_err:
                    logger.debug(f"Improvement tracking error: {track_err}")

            # 5e. Update portfolio snapshot and run alert checks
            try:
                se = StateEngine()
                snapshot = se.update_portfolio_snapshot()

                # 5e-i: Run alert system checks
                try:
                    from src.scanner.automation.alert_manager import AlertManager

                    alert_mgr = AlertManager()

                    # Prepare data for alert checks
                    nav = snapshot.get("nav", 0.0)
                    peak_nav = nav  # In production, track peak_nav separately

                    # Load recent trades for alert evaluation
                    journal_path = Path("trained_data/trade_journal_rl.json")
                    recent_trades = []
                    if journal_path.exists():
                        try:
                            recent_trades = self._load_journal_cached(journal_path)
                        except json.JSONDecodeError:
                            logger.debug("Failed to load journal for alerts")

                    # Load current and previous agent weights for stability check
                    current_weights = {}
                    try:
                        weights_path = Path("trained_data/models/agent_weights.json")
                        if weights_path.exists():
                            current_weights = json.loads(weights_path.read_text())
                    except Exception as e:
                        logger.debug(f"A/B test weights load skipped: {e}")

                    # Load previous weights from improvement log (most recent session)
                    previous_weights = {}
                    try:
                        from src.scanner.automation.improvement_tracker import ImprovementTracker
                        tracker = ImprovementTracker()
                        records = tracker._load_records()
                        if records and len(records) > 0:
                            previous_weights = records[-1].get("agent_weights_snapshot", {})
                    except Exception as e:
                        logger.debug(f"A/B test previous weights load skipped: {e}")

                    # Run all alert checks
                    fired_alerts = alert_mgr.check_all(
                        nav=nav,
                        peak_nav=peak_nav,
                        recent_trades=recent_trades,
                        current_weights=current_weights if current_weights else None,
                        previous_weights=previous_weights if previous_weights else None,
                    )

                    if fired_alerts and console:
                        for alert in fired_alerts:
                            console.print(f"  [red]⚠ ALERT [{alert.severity}]: {alert.message}[/red]")

                except Exception as alert_err:
                    logger.debug(f"Alert system error: {alert_err}")

            except Exception as state_err:
                logger.debug(f"State update error: {state_err}")

            # 5f. Audit every 10th cycle
            try:
                se = StateEngine()
                cycle = se.increment_scan_cycle()
                if cycle % 10 == 0:
                    audit_result = le.audit()
                    if audit_result.get("actions") and console:
                        console.print(f"  [dim]Audit: {audit_result['actions']}[/dim]")
            except Exception as audit_err:
                logger.debug(f"Audit error: {audit_err}")

            # 5g. Merge accuracy-gated blocked pairs into scanner config (US-013)
            try:
                from src.scanner.automation.accuracy_gate import AccuracyGate
                ag = AccuracyGate(min_accuracy=0.55, min_trades=5)
                blocked_by_accuracy = ag.get_blocked_pairs()

                if blocked_by_accuracy:
                    # Merge with existing blocked_pairs from config
                    original_blocked = set(self.scanner.config.blocked_pairs or [])
                    new_blocked = set(blocked_by_accuracy)
                    merged = list(original_blocked | new_blocked)
                    self.scanner.config.blocked_pairs = merged

                    if console and blocked_by_accuracy:
                        console.print(
                            f"  [yellow]Accuracy gate blocking {len(blocked_by_accuracy)} "
                            f"pair(s): {', '.join(blocked_by_accuracy)}[/yellow]"
                        )
            except Exception as accuracy_err:
                logger.debug(f"Accuracy gate merge error: {accuracy_err}")

            # 5h. Drift detection: record trade outcomes for retraining trigger
            if trades_synced > 0:
                try:
                    # Phase 78: Use scanner's retrain_trigger instance (preserves state)
                    trigger = getattr(self.scanner, "_retrain_trigger", None)
                    if trigger is None:
                        from src.scanner.automation.retrain_trigger import RetrainTrigger
                        from src.scanner.automation.observation_log import ObservationLog
                        obs_log = ObservationLog()
                        trigger = RetrainTrigger(observation_log=obs_log)

                    # Record prediction outcomes from closed trades
                    journal_path = Path("trained_data/trade_journal_rl.json")
                    entries = self._load_journal_cached(journal_path)
                    if not isinstance(entries, list):
                        entries = []
                    closed = entries[-trades_synced:] if trades_synced < len(entries) else entries
                    for entry in closed:
                        if entry.get("outcome"):
                            pair = entry.get("pair")
                            correct = entry.get("outcome", {}).get("prediction_correct", False)
                            trigger.record_prediction(pair, correct)

                    # Check for drift every 50 scans
                    if self._scan_count % 50 == 0:
                        drift_request = trigger.check_drift()
                        if drift_request and console:
                            console.print(
                                f"  [red]🔄 RETRAIN TRIGGER: {drift_request.reason} "
                                f"(priority={drift_request.priority})[/red]"
                            )
                            # Spawn background retrain (non-blocking)
                            self._spawn_background_retrain(drift_request.pairs, console)
                except Exception as drift_err:
                    logger.debug(f"Drift detection error: {drift_err}")

            # 5j. Per-pair model selector: check for model switches (rolling accuracy)
            try:
                from src.scanner.automation.pair_model_selector import PairModelSelector
                pms = PairModelSelector()
                switch_count = 0
                for pair in self._get_traded_pairs_from_journal():
                    new_model = pms.check_switch(pair)
                    if new_model:
                        pms.execute_switch(pair, new_model)
                        switch_count += 1
                if switch_count > 0 and console:
                    console.print(f"  [cyan]Model switches: {switch_count} pair(s)[/cyan]")
            except Exception as pms_err:
                logger.debug(f"Pair model selector error: {pms_err}")

            # Log learning activity
            if (learnings_added > 0 or rules_promoted > 0) and console:
                console.print(
                    f"  [magenta]Learning: {learnings_added} insights captured, "
                    f"{rules_promoted} rules promoted[/magenta]"
                )
            if config_adjustments and console:
                console.print(
                    f"  [magenta]Config: {len(config_adjustments)} adjustments applied[/magenta]"
                )

        except Exception as e:
            logger.debug(f"Learning loop error: {e}")

    def _load_journal_cached(self, journal_path) -> list:
        """Load trade journal with mtime-based caching.

        Re-reads disk only when the file has changed (i.e. a trade closed).
        Eliminates the 4x full json.loads(read_text()) per scan cycle.
        """
        import json as _json
        try:
            current_mtime = journal_path.stat().st_mtime if journal_path.exists() else 0.0
        except OSError:
            return self._journal_cache
        if current_mtime != self._journal_mtime:
            try:
                raw = _json.loads(journal_path.read_text())
                self._journal_cache = raw if isinstance(raw, list) else []
                self._journal_mtime = current_mtime
            except (Exception,):
                pass
        return self._journal_cache

    def _sleep_with_progress(self, minutes: int):
        """Sleep with progress indication, allowing Ctrl+C to interrupt."""
        import gc as _gc
        if console:
            console.print(f"\n[dim]Next scan in {minutes} minutes...[/dim]")

        # Gen-0 collect before sleeping — clears short-lived scan objects
        # before they age into gen-1, keeping RSS down during idle time.
        _gc.collect(0)

        # Sleep in 1-second increments to allow Ctrl+C
        for _ in range(minutes * 60):
            if not self._running:
                break
            time.sleep(1)

        # Full collect during idle — runs while waiting, not during scan.
        _gc.collect()

    def _spawn_background_retrain(
        self, pairs: list, console: Optional[Any] = None
    ) -> None:
        """Spawn retraining as a background subprocess (non-blocking).

        This allows the scan loop to continue while retraining runs. The
        retrain script handles its own validation, AccuracyGate reset, and
        cooldown tracking.
        """
        import subprocess
        from pathlib import Path  # Phase 31 (US-190): Ensure Path is available

        # Don't spawn if already running
        if getattr(self, "_retrain_process", None) and self._retrain_process.poll() is None:
            logger.debug("Background retrain already running, skipping")
            return

        pairs_str = ",".join(pairs)
        cmd = [
            sys.executable,
            str(Path("scripts/scheduled_retrain.py")),
            "--pairs",
            pairs_str,
        ]

        try:
            self._retrain_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
            )
            logger.info(
                f"Background retrain spawned (PID {self._retrain_process.pid}) "
                f"for {len(pairs)} pair(s): {pairs_str}"
            )
            if console:
                console.print(
                    f"  [yellow]Background retrain started for {pairs_str} "
                    f"(PID {self._retrain_process.pid})[/yellow]"
                )
        except Exception as e:
            logger.error(f"Failed to spawn background retrain: {e}")

    def _run_shadow_tests(self, result: "ScanResult") -> None:
        """Log incumbent and (optionally) candidate predictions to shadow_predictions.jsonl.

        This is step 1 of the true A/B framework:
        1. Record incumbent (production) prediction at each scan
        2. If enable_true_ab_testing is True: also run candidate model inference
        3. Step 2 (scoring) happens in _run_learning_loop via
           ModelManager.score_from_trade_outcome() when trades close.

        True A/B testing:
        - If enable_true_ab_testing=False: incumbent only (candidate_direction=None)
        - If enable_true_ab_testing=True: load candidate model and run both inferences
          - Candidate model failure is graceful: shadows back to incumbent-only testing
          - Candidate model is lazy-loaded on first call and unloaded when disabled
        """
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        shadow_log = Path("trained_data/models/shadow_predictions.jsonl")
        scan_timestamp = datetime.now(timezone.utc).isoformat()

        # Initialize candidate model loader if true A/B testing enabled
        candidate_loader = None
        if self.scanner.config.enable_true_ab_testing:
            try:
                from src.scanner.automation.model_manager import CandidateModelLoader
                candidate_loader = CandidateModelLoader()
            except Exception as e:
                logger.debug(f"Failed to initialize candidate loader: {e}")

        try:
            records = []
            for analysis in result.analyses:
                direction = self._get_direction_from_analysis(analysis)
                # Only log actionable (non-HOLD) predictions — these are the ones
                # that will match against actual trade outcomes
                if not direction or direction == "HOLD":
                    continue

                # True A/B: get candidate prediction if enabled
                candidate_direction = None
                if candidate_loader:
                    try:
                        # Extract features from analysis for candidate inference
                        # For now, use overall_score as a simple feature proxy
                        # In production, this would be the full feature vector
                        import numpy as np
                        # Create dummy feature array from analysis attributes
                        features = np.array([
                            float(getattr(analysis, "overall_score", 0.5)),
                            float(getattr(analysis, "confidence", 0.0)),
                            float(getattr(analysis, "momentum_score", 0.0)) if hasattr(analysis, "momentum_score") else 0.0,
                        ])
                        candidate_direction = candidate_loader.get_prediction(features)
                    except Exception as e:
                        logger.debug(f"Candidate inference failed for {analysis.pair}: {e}")
                        # Gracefully fall back to incumbent-only testing
                        candidate_direction = None

                records.append({
                    "pair": analysis.pair,
                    "scan_timestamp": scan_timestamp,
                    "incumbent_direction": direction,
                    "incumbent_confidence": float(getattr(analysis, "confidence", 0.0)),
                    "candidate_direction": candidate_direction,
                    "scored": False,
                })

            if records:
                shadow_log.parent.mkdir(parents=True, exist_ok=True)
                try:
                    from src.scanner.automation.safe_json import safe_jsonl_append
                    for r in records:
                        safe_jsonl_append(shadow_log, r)
                except Exception as _jsonl_err:
                    logger.debug(f"Shadow log append error: {_jsonl_err}")

                # Rotate shadow log if > 10MB (prevents unbounded growth)
                try:
                    if shadow_log.stat().st_size > 10 * 1024 * 1024:
                        rotated = shadow_log.with_suffix(".jsonl.old")
                        if rotated.exists():
                            rotated.unlink()
                        shadow_log.rename(rotated)
                        logger.info("Shadow predictions rotated (>10MB)")
                except Exception as e:
                    logger.debug(f"Shadow prediction rotation skipped: {e}")

                # Count true A/B tests (both predictions present)
                ab_tests = sum(1 for r in records if r["candidate_direction"] is not None)
                log_msg = f"Shadow predictions logged: {len(records)} pairs"
                if ab_tests > 0:
                    log_msg += f" ({ab_tests} with true A/B)"
                log_msg += f" at {scan_timestamp[:19]}"
                logger.debug(log_msg)

        except Exception as e:
            logger.debug(f"Shadow prediction logging error: {e}")

    def _check_model_promotion(self) -> None:
        """Check if candidate model should be promoted to production.

        Runs every 50 scans to make promotion decision based on shadow test results.
        """
        try:
            from src.scanner.automation.model_manager import ModelManager

            mm = ModelManager()

            # Check if promotion is warranted
            promotion_version = mm.check_promotion()

            if promotion_version:
                # Promotion is ready
                if mm.promote(promotion_version):
                    if console:
                        status = mm.get_status()
                        console.print(
                            f"\n[green bold]✓ MODEL PROMOTED: {promotion_version}[/green bold]\n"
                            f"  Production accuracy: {status['production']['shadow_accuracy']:.4f}\n"
                            f"  Improvement: {status['shadow_test_progress']['improvement_vs_incumbent']:.4f}"
                        )
                        logger.info(f"Model {promotion_version} promoted to production")
            else:
                # Log status at promotion checkpoints
                status = mm.get_status()
                if status.get("candidate"):
                    candidate = status["candidate"]
                    progress = status.get("shadow_test_progress", {})
                    if progress:
                        if console:
                            console.print(
                                f"  [dim]Model A/B: {candidate['version_tag']} "
                                f"{progress['progress_pct']:.0f}% complete "
                                f"({progress['scans_completed']}/{progress['scans_required']} scans) "
                                f"improvement: {progress['improvement_vs_incumbent']:+.4f}[/dim]"
                            )

        except Exception as e:
            logger.debug(f"Model promotion check error: {e}")

    def _get_direction_from_analysis(self, analysis) -> Optional[str]:
        """Extract predicted direction from a PairAnalysis object.

        Args:
            analysis: PairAnalysis from scan result

        Returns:
            "LONG", "SHORT", "HOLD", or None
        """
        try:
            # Check if analysis has a direction prediction
            if hasattr(analysis, "predicted_direction"):
                direction = analysis.predicted_direction
                if direction in ("LONG", "SHORT", "HOLD"):
                    return direction

            # Fallback: infer from score/confidence if available
            if hasattr(analysis, "overall_score"):
                score = analysis.overall_score
                if score > 0.55:
                    return "LONG"
                elif score < 0.45:
                    return "SHORT"
                else:
                    return "HOLD"

            return None

        except Exception as e:
            logger.debug(f"Error extracting direction: {e}")
            return None

    def _get_traded_pairs_from_journal(self) -> List[str]:
        """Extract list of pairs with closed trades from journal.

        Returns:
            List of unique pair codes that have closed trades
        """
        import json
        from pathlib import Path
        try:
            journal_path = Path("trained_data/trade_journal_rl.json")
            if not journal_path.exists():
                return []
            entries = json.loads(journal_path.read_text())
            closed = [e for e in entries if e.get("outcome") is not None]
            return list(set(e.get("pair", "") for e in closed if e.get("pair")))
        except Exception as e:
            logger.debug(f"Error extracting traded pairs: {e}")
            return []
