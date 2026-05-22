"""Four-layer scorecard harness for the meta-cybernetic change pipeline.

Wraps `ReplayValidator`, `QAPipeline`, and pytest into a single
`ChangeEvalHarness.score(package)` call that returns a `Scorecard` with
four `SubScore` fields populated.

Dependency injection is the design rule: every external dependency
(replay validator, QA pipeline, candle loader, pytest runner) is a
constructor argument with a sensible default. This keeps the harness
unit-testable without OANDA, MLflow, or a real test suite running.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.scanner.automation.meta_types import (
    ChangePackage,
    Scorecard,
    SubScore,
)

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_REGIME_WINDOWS_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "regime_windows.json"
_DEFAULT_SCORECARD_LOG = _PROJECT_ROOT / ".claude" / "meta" / "scorecards.jsonl"

# Canonical on-disk candle cache used by training pipelines. ChangeEvalHarness
# reads the same cache so the scorecard runs against the SAME OHLCV data the
# bot has been training on — no OANDA round trip, no network, no creds.
_DEFAULT_CANDLES_ROOT = _PROJECT_ROOT / "trained_data" / "cache" / "training_data"

# Bars per UTC day per granularity (used to convert "days" → candle count).
_BARS_PER_DAY = {"M15": 96, "M30": 48, "H1": 24, "H4": 6, "D": 1}

# Preferred granularity order when the harness has no explicit preference.
# M15 first — the scanner's primary timeframe and what we have most cached.
_PREFERRED_GRANULARITIES = ("M15", "H1", "M30", "H4")


# Trading thresholds (Stage 3.2)
TRADING_PNL_DELTA_FLOOR_R = -1.0
TRADING_SHARPE_DELTA_FLOOR = -0.1
TRADING_WIN_RATE_FLOOR = 0.5

# Governance thresholds (Stage 3.3)
GOV_MAX_NEW_WARNINGS = 2

# Regime threshold (Stage 3.4) — no single window worse than -3R
REGIME_PNL_FLOOR_R = -3.0


CandleLoader = Callable[[str, int], List[Dict[str, Any]]]
PytestRunner = Callable[[Optional[Path]], Dict[str, Any]]


def _normalize_pair(pair: str) -> str:
    """Round-trip pair symbol to the on-disk convention (`EUR_USD`).

    Accepts `EUR_USD`, `EUR/USD`, `EURUSD`, lowercase, whitespace. Returns the
    underscore form used by both the training cache and the live scanner.
    """
    if not pair:
        return ""
    s = pair.strip().upper().replace("/", "_").replace("-", "_")
    if "_" in s:
        return s
    # `EURUSD` → `EUR_USD` (only handle the 6-char major case; anything else
    # is returned as-is and will miss the cache, surfacing a real `no_candles`).
    if len(s) == 6:
        return f"{s[:3]}_{s[3:]}"
    return s


def _resolve_cached_candle_path(
    pair: str,
    *,
    candles_root: Optional[Path] = None,
    preferred_granularity: Optional[str] = None,
) -> Optional[Path]:
    """Find the largest available cached candle CSV for `pair`.

    Search order:
      1. Caller's `preferred_granularity` if supplied
      2. `_PREFERRED_GRANULARITIES` in declared order
      3. Within a granularity, the file with the highest candle-count suffix
         (`EUR_USD_M15_65000.csv` beats `EUR_USD_M15_30000.csv`).

    Returns `None` if nothing is cached for the pair.
    """
    root = Path(candles_root) if candles_root else _DEFAULT_CANDLES_ROOT
    if not root.is_dir():
        return None

    norm_pair = _normalize_pair(pair)
    if not norm_pair:
        return None

    grans: List[str] = []
    if preferred_granularity:
        grans.append(preferred_granularity)
    for g in _PREFERRED_GRANULARITIES:
        if g not in grans:
            grans.append(g)

    for gran in grans:
        candidates: List[tuple[int, Path]] = []
        for p in root.glob(f"{norm_pair}_{gran}_*.csv"):
            stem = p.stem  # e.g. EUR_USD_M15_65000
            tail = stem.rsplit("_", 1)[-1]
            try:
                count = int(tail)
            except ValueError:
                count = 0
            candidates.append((count, p))
        if candidates:
            candidates.sort(reverse=True)  # highest count first
            return candidates[0][1]
    return None


def _read_candles_csv(path: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read OHLCV rows from a training-cache CSV into the dict shape
    `ReplayValidator` consumes (`{time, open, high, low, close, volume}` with
    numeric OHLCV). Returns the LAST `limit` rows when limit is set — the
    scorecard wants the most recent history, not the oldest.
    """
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    rows.append(
                        {
                            "time": row.get("time", ""),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume", 0) or 0),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    # Skip header-shaped or malformed rows — never fail the
                    # whole load on a single dirty row.
                    continue
    except OSError as e:
        logger.warning("change_eval.csv_read_failed path=%s err=%s", path, e)
        return []

    if limit is not None and limit > 0 and len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _default_candle_loader(pair: str, days: int) -> List[Dict[str, Any]]:
    """Load historical candles for `pair` covering ~`days` of history.

    Disk-only: reads `trained_data/cache/training_data/{PAIR}_{GRAN}_*.csv`,
    the same cache the training pipelines populate. NEVER hits the network,
    NEVER needs OANDA creds — the scorecard runs in subprocess and harness
    contexts that may not have either.

    Returns `[]` only when the cache has no file for the pair OR the file is
    unreadable. Symbol mismatch is normalized away (`EUR/USD` → `EUR_USD`).
    """
    csv_path = _resolve_cached_candle_path(pair)
    if csv_path is None:
        logger.debug(
            "change_eval.candle_loader.no_cache_file pair=%s root=%s",
            pair, _DEFAULT_CANDLES_ROOT,
        )
        return []

    # Determine the bar count to slice. We don't know the granularity for sure
    # here (path resolution may have picked M15 or H1); take the per-day rate
    # of the granularity embedded in the filename so "days" stays honest.
    gran = csv_path.stem.split("_")[-2] if "_" in csv_path.stem else "M15"
    bars_per_day = _BARS_PER_DAY.get(gran, _BARS_PER_DAY["M15"])
    limit = max(int(days) * bars_per_day, 0) if days else 0
    candles = _read_candles_csv(csv_path, limit=limit if limit > 0 else None)
    logger.debug(
        "change_eval.candle_loader.loaded pair=%s path=%s gran=%s n=%d",
        pair, csv_path.name, gran, len(candles),
    )
    return candles


def _default_pytest_runner(target: Optional[Path]) -> Dict[str, Any]:
    """Run pytest -q on the provided target dir (defaults to ./tests).

    Returns {passed: bool, exit_code: int, failed: [test ids], stdout_tail: str}.

    Disk-pressure precondition (US: scanner-disk-guard, 2026-05-21): when free
    disk on the project volume is below the configured threshold (default
    500 MiB; override via `BUDDY_DISK_PRESSURE_MIB`), this function refuses to
    spawn pytest. Returns `{passed: False, skipped: True, reason:
    "low_disk_skipped", ...}` so the scorecard surface distinguishes "disk
    pressure" from "actual test failure" and the autonomous loop can alert the
    operator instead of silently rejecting every config proposal. See
    `src/scanner/runtime_guards.py` for the full diagnosis.
    """
    from src.scanner.runtime_guards import check_disk_pressure
    ok, free_mib, threshold_mib, reason = check_disk_pressure()
    if not ok:
        msg = (
            f"low_disk_skipped: {free_mib} MiB free < {threshold_mib} MiB "
            f"threshold (reason={reason}). Refusing to spawn pytest to avoid "
            f"SIGSEGV in native model loads. Free space and re-trigger."
        )
        logger.error("change_eval.pytest_runner_skipped %s", msg)
        return {
            "passed": False,
            "skipped": True,
            "reason": "low_disk_skipped",
            "exit_code": -1,
            "failed": [],
            "stdout_tail": msg,
            "free_mib": free_mib,
            "threshold_mib": threshold_mib,
        }

    cmd = ["python", "-m", "pytest", "-q"]
    if target is not None:
        cmd.append(str(target))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "exit_code": -1, "failed": [], "stdout_tail": "timeout"}
    except FileNotFoundError as e:
        return {"passed": False, "exit_code": -1, "failed": [], "stdout_tail": f"runner_missing: {e}"}
    failed = []
    for line in proc.stdout.splitlines():
        if " FAILED" in line or line.startswith("FAILED "):
            failed.append(line.strip())
    tail = "\n".join(proc.stdout.splitlines()[-20:])
    return {
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "failed": failed,
        "stdout_tail": tail,
    }


@dataclass
class _ScoreContext:
    """Per-call context bundle so sub-scorers don't reach into globals."""

    package: ChangePackage
    baseline_config: Dict[str, Any]
    candidate_config: Dict[str, Any]
    pairs: List[str]


class ChangeEvalHarness:
    """Runs the four-layer scorecard for a `ChangePackage`."""

    def __init__(
        self,
        *,
        replay_validator: Any = None,
        qa_pipeline: Any = None,
        candle_loader: Optional[CandleLoader] = None,
        pytest_runner: Optional[PytestRunner] = None,
        regime_windows_path: Optional[Path] = None,
        scorecard_log_path: Optional[Path] = None,
        baseline_config: Optional[Dict[str, Any]] = None,
        pairs: Optional[List[str]] = None,
        history_days: int = 30,
    ):
        self._replay = replay_validator
        self._qa = qa_pipeline
        self._candles = candle_loader or _default_candle_loader
        self._pytest = pytest_runner or _default_pytest_runner
        self._regime_path = Path(regime_windows_path or _DEFAULT_REGIME_WINDOWS_PATH)
        self._log_path = Path(scorecard_log_path or _DEFAULT_SCORECARD_LOG)
        self._baseline_config = dict(baseline_config or {})
        self._pairs = list(pairs or ["EUR_USD"])
        self._history_days = history_days

    def score(self, package: ChangePackage) -> Scorecard:
        candidate_config = self._candidate_config_from(package)
        ctx = _ScoreContext(
            package=package,
            baseline_config=dict(self._baseline_config),
            candidate_config=candidate_config,
            pairs=list(self._pairs),
        )
        card = Scorecard(
            software_correctness=self._score_software(ctx),
            trading_quality=self._score_trading(ctx),
            governance_quality=self._score_governance(ctx),
            regime_robustness=self._score_regime(ctx),
        )
        self._persist(package, card)
        return card

    def _candidate_config_from(self, package: ChangePackage) -> Dict[str, Any]:
        cfg = dict(self._baseline_config)
        delta = (package.proposal.config_delta if package.proposal else {}) or {}
        for k, v in delta.items():
            new_value = v.get("new") if isinstance(v, dict) and "new" in v else v
            cfg[k] = new_value
        return cfg

    def _score_software(self, ctx: _ScoreContext) -> SubScore:
        try:
            target = ctx.package.proposal.worktree_path if ctx.package.proposal else None
            target_path = Path(target) / "tests" if target else None
            result = self._pytest(target_path)
            passed = bool(result.get("passed"))
            return SubScore(
                name="software_correctness",
                passed=passed,
                score=1.0 if passed else 0.0,
                details=result,
            )
        except Exception as e:
            return SubScore(name="software_correctness", passed=False, score=0.0, error=str(e))

    def _score_trading(self, ctx: _ScoreContext) -> SubScore:
        if self._replay is None:
            return SubScore(
                name="trading_quality",
                passed=True,
                score=0.0,
                details={"skipped": True, "reason": "no_replay_validator"},
            )
        per_pair: List[Dict[str, Any]] = []
        try:
            for pair in ctx.pairs:
                candles = self._candles(pair, self._history_days)
                if not candles:
                    per_pair.append({"pair": pair, "skipped": True, "reason": "no_candles"})
                    continue
                comparison = self._replay.compare_configs(
                    ctx.baseline_config, ctx.candidate_config, pair, candles
                )
                per_pair.append({"pair": pair, **comparison})
            agg = _aggregate_trading(per_pair)
            passed = (
                agg["pnl_delta_r"] >= TRADING_PNL_DELTA_FLOOR_R
                and agg["sharpe_delta"] >= TRADING_SHARPE_DELTA_FLOOR
                and agg["win_rate"] >= TRADING_WIN_RATE_FLOOR
            )
            details = {**agg, "per_pair": per_pair}
            return SubScore(
                name="trading_quality",
                passed=passed,
                score=agg["sharpe"],
                details=details,
            )
        except Exception as e:
            return SubScore(name="trading_quality", passed=False, score=0.0, error=str(e))

    def _score_governance(self, ctx: _ScoreContext) -> SubScore:
        if self._qa is None:
            return SubScore(
                name="governance_quality",
                passed=True,
                score=1.0,
                details={"skipped": True, "reason": "no_qa_pipeline"},
            )
        try:
            report = self._qa.run_full_audit()
            findings = getattr(report, "findings", []) or []
            critical = [f for f in findings if getattr(f, "severity", "") == "critical"]
            warnings = [f for f in findings if getattr(f, "severity", "") == "warning"]
            score = float(getattr(report, "score", 0.0))
            passed = len(critical) == 0 and len(warnings) <= GOV_MAX_NEW_WARNINGS
            details = {
                "critical": len(critical),
                "warnings": len(warnings),
                "score": score,
                "findings": [
                    {"severity": getattr(f, "severity", ""), "category": getattr(f, "category", ""), "message": getattr(f, "message", "")}
                    for f in findings[:20]
                ],
            }
            return SubScore(
                name="governance_quality",
                passed=passed,
                score=score / 100.0 if score > 1 else score,
                details=details,
            )
        except Exception as e:
            return SubScore(name="governance_quality", passed=False, score=0.0, error=str(e))

    def _score_regime(self, ctx: _ScoreContext) -> SubScore:
        if self._replay is None:
            return SubScore(
                name="regime_robustness",
                passed=True,
                score=0.0,
                details={"skipped": True, "reason": "no_replay_validator"},
            )
        windows = self._load_regime_windows()
        if not windows:
            return SubScore(
                name="regime_robustness",
                passed=True,
                score=0.0,
                details={"skipped": True, "reason": "no_regime_windows_fixture"},
            )
        results: List[Dict[str, Any]] = []
        worst_pnl_r = 0.0
        try:
            for window in windows:
                pair = window.get("pair", ctx.pairs[0] if ctx.pairs else "EUR_USD")
                days = int(window.get("days", self._history_days))
                candles = self._candles(pair, days)
                if not candles:
                    results.append({"window": window.get("name"), "skipped": True})
                    continue
                replay = self._replay.replay_scan(
                    pair, candles, ctx.candidate_config, config_label=window.get("name", "regime")
                )
                metrics = replay.to_dict() if hasattr(replay, "to_dict") else {}
                pnl_r = float(metrics.get("total_pnl", 0.0))
                worst_pnl_r = min(worst_pnl_r, pnl_r)
                results.append({"window": window.get("name"), **metrics})
            passed = worst_pnl_r >= REGIME_PNL_FLOOR_R
            return SubScore(
                name="regime_robustness",
                passed=passed,
                score=worst_pnl_r,
                details={"worst_pnl_r": worst_pnl_r, "windows": results},
            )
        except Exception as e:
            return SubScore(name="regime_robustness", passed=False, score=0.0, error=str(e))

    def _load_regime_windows(self) -> List[Dict[str, Any]]:
        if not self._regime_path.exists():
            return []
        try:
            data = json.loads(self._regime_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "windows" in data:
                return list(data["windows"])
        except Exception as e:
            logger.warning("change_eval.regime_load_failed err=%s", e)
        return []

    def _persist(self, package: ChangePackage, card: Scorecard) -> None:
        record = {
            "change_id": package.change_id,
            "kind": package.kind.value,
            "scorecard": card.to_dict(),
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("change_eval.persist_failed err=%s", e)


def _aggregate_trading(per_pair: List[Dict[str, Any]]) -> Dict[str, float]:
    """Roll up `compare_configs` outputs into a single trading scorecard."""
    pnl_a = pnl_b = sharpe_a = sharpe_b = wins = trades = 0.0
    counted = 0
    for entry in per_pair:
        if entry.get("skipped"):
            continue
        a = entry.get("config_a", {}) or {}
        b = entry.get("config_b", {}) or {}
        pnl_a += float(a.get("total_pnl", 0.0))
        pnl_b += float(b.get("total_pnl", 0.0))
        sharpe_a += float(a.get("sharpe", 0.0))
        sharpe_b += float(b.get("sharpe", 0.0))
        candidate_trades = b.get("trades") or []
        wins += sum(1 for t in candidate_trades if float(t.get("pnl_pips", 0)) > 0)
        trades += len(candidate_trades)
        counted += 1
    if counted == 0:
        return {"pnl_delta_r": 0.0, "sharpe_delta": 0.0, "win_rate": 0.0, "sharpe": 0.0}
    return {
        "pnl_delta_r": (pnl_b - pnl_a) / max(counted, 1),
        "sharpe_delta": (sharpe_b - sharpe_a) / counted,
        "win_rate": wins / trades if trades else 0.0,
        "sharpe": sharpe_b / counted,
    }
