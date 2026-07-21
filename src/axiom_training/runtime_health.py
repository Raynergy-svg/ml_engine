"""Runtime health checks that must pass before AXIOM policy promotion.

This module is read-only. It inspects persisted runtime artifacts and reports
whether AXIOM's action-policy model has trustworthy, current feedback to learn
from.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
OANDA_ACCOUNT_PATH = REPO_ROOT / "trained_data" / "oanda" / "account_state.json"
OANDA_TRANSACTIONS_PATH = REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl"
TIER7_STATE_PATH = REPO_ROOT / ".claude" / "tier7_state.json"
TRADE_JOURNAL_PATH = REPO_ROOT / "trained_data" / "trade_journal_rl.json"
TRADE_FEEDBACK_PATH = REPO_ROOT / "logs" / "trade_feedback.jsonl"


@dataclass(frozen=True)
class RuntimeHealthIssue:
    severity: str
    check: str
    detail: str
    value: Any = None
    threshold: Any = None


@dataclass(frozen=True)
class RuntimeHealthReport:
    status: str
    checked_at: str
    issues: List[RuntimeHealthIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_seconds(path: Path) -> Optional[float]:
    try:
        return max(0.0, datetime.now(timezone.utc).timestamp() - Path(path).stat().st_mtime)
    except OSError:
        return None


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _closed_journal_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if entry.get("outcome") is not None]


def _pending_journal_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if entry.get("outcome") is None]


def _is_oanda_confirmed_entry(entry: Mapping[str, Any]) -> bool:
    """Return whether a journal row can represent an OANDA account trade.

    OANDA v20 trade IDs are decimal strings.  The journal also contains offline,
    synthetic, and test rows (for example ``test_1``); those remain useful
    training/audit evidence but must not be reconciled as live broker positions.
    """
    broker = str(entry.get("broker") or "oanda").strip().lower()
    asset_class = str(entry.get("asset_class") or "FX").strip().upper()
    trade_id = str(entry.get("trade_id") or "").strip()
    return broker == "oanda" and asset_class == "FX" and trade_id.isdecimal()


def _feedback_trade_ids(path: Path) -> set[str]:
    return {str(row.get("trade_id")) for row in _read_jsonl(path) if row.get("trade_id") is not None}


def _count_missing_brackets(positions: List[Mapping[str, Any]]) -> int:
    missing = 0
    for position in positions:
        if position.get("stop_loss") in (None, "", 0) or position.get("take_profit") in (None, "", 0):
            missing += 1
    return missing


def _normalized_instrument(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper().replace("/", "_")


def _position_instruments(positions: List[Mapping[str, Any]]) -> set[str]:
    instruments: set[str] = set()
    for position in positions:
        instrument = _normalized_instrument(position.get("instrument"))
        if instrument:
            instruments.add(instrument)
    return instruments


def _journal_instrument(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("pair", "instrument", "symbol"):
        instrument = _normalized_instrument(entry.get(key))
        if instrument:
            return instrument
    return None


def evaluate_runtime_health(
    *,
    account_path: Path = OANDA_ACCOUNT_PATH,
    tier7_path: Path = TIER7_STATE_PATH,
    journal_path: Path = TRADE_JOURNAL_PATH,
    feedback_path: Path = TRADE_FEEDBACK_PATH,
    transactions_path: Path = OANDA_TRANSACTIONS_PATH,
    account_fresh_s: float = 7200.0,
    tier7_fresh_s: float = 900.0,
    min_feedback_coverage: float = 0.95,
) -> RuntimeHealthReport:
    issues: List[RuntimeHealthIssue] = []
    metrics: Dict[str, Any] = {}

    account_age = _age_seconds(account_path)
    tier7_age = _age_seconds(tier7_path)
    tx_age = _age_seconds(transactions_path)
    metrics.update({
        "account_snapshot_age_s": round(account_age, 1) if account_age is not None else None,
        "tier7_snapshot_age_s": round(tier7_age, 1) if tier7_age is not None else None,
        "transaction_ledger_age_s": round(tx_age, 1) if tx_age is not None else None,
    })

    if account_age is None:
        issues.append(RuntimeHealthIssue("critical", "account_snapshot_missing", "OANDA account snapshot is missing"))
        account = {}
    else:
        account = _read_json(account_path, {})
        if account_age > account_fresh_s:
            issues.append(RuntimeHealthIssue(
                "critical", "account_snapshot_stale",
                f"OANDA account snapshot stale ({int(account_age)}s > {int(account_fresh_s)}s)",
                round(account_age, 1), account_fresh_s,
            ))

    if tier7_age is None:
        issues.append(RuntimeHealthIssue("warning", "tier7_snapshot_missing", "Tier 7 display snapshot is missing"))
        tier7 = {}
    else:
        tier7 = _read_json(tier7_path, {})
        if tier7_age > tier7_fresh_s:
            issues.append(RuntimeHealthIssue(
                "warning", "tier7_snapshot_stale",
                f"Tier 7 display snapshot stale ({int(tier7_age)}s > {int(tier7_fresh_s)}s)",
                round(tier7_age, 1), tier7_fresh_s,
            ))
        elif not bool(tier7.get("running")):
            issues.append(RuntimeHealthIssue(
                "warning", "tier7_not_running",
                "Tier 7 snapshot is fresh but the supervisor reports not running",
                tier7.get("running"), True,
            ))

    positions = account.get("positions") if isinstance(account, Mapping) else []
    positions = positions if isinstance(positions, list) else []
    position_maps = [p for p in positions if isinstance(p, Mapping)]
    account_open_count = int(account.get("open_trade_count") or len(positions) or 0) if isinstance(account, Mapping) else 0
    account_open_instruments = _position_instruments(position_maps)
    missing_brackets = _count_missing_brackets(position_maps)
    if missing_brackets:
        issues.append(RuntimeHealthIssue(
            "critical", "open_trade_missing_bracket",
            f"{missing_brackets} open OANDA position(s) missing stop_loss or take_profit",
            missing_brackets, 0,
        ))

    entries_raw = _read_json(journal_path, [])
    entries = entries_raw if isinstance(entries_raw, list) else []
    entries = [entry for entry in entries if isinstance(entry, dict)]
    pending = _pending_journal_entries(entries)
    broker_pending = [entry for entry in pending if _is_oanda_confirmed_entry(entry)]
    excluded_pending = [entry for entry in pending if not _is_oanda_confirmed_entry(entry)]
    closed = _closed_journal_entries(entries)
    pending_instruments = {
        instrument
        for instrument in (_journal_instrument(entry) for entry in broker_pending)
        if instrument is not None
    }
    journal_open_not_in_account = sorted(pending_instruments - account_open_instruments)
    account_open_not_in_journal = sorted(account_open_instruments - pending_instruments)
    feedback_ids = _feedback_trade_ids(feedback_path)
    closed_ids = {str(entry.get("trade_id")) for entry in closed if entry.get("trade_id") is not None}
    missing_feedback = sorted(closed_ids - feedback_ids)
    coverage = (len(closed_ids) - len(missing_feedback)) / len(closed_ids) if closed_ids else 1.0

    metrics.update({
        "account_open_trade_count": account_open_count,
        "account_open_instruments": sorted(account_open_instruments),
        "journal_entries": len(entries),
        "journal_pending_open": len(pending),
        "journal_broker_pending_open": len(broker_pending),
        "journal_excluded_pending": len(excluded_pending),
        "journal_excluded_pending_ids": [
            str(entry.get("trade_id")) for entry in excluded_pending[:20]
        ],
        "journal_pending_instruments": sorted(pending_instruments),
        "journal_closed": len(closed),
        "journal_open_not_in_account": journal_open_not_in_account[:20],
        "account_open_not_in_journal": account_open_not_in_journal[:20],
        "feedback_log_exists": Path(feedback_path).exists(),
        "feedback_rows": len(feedback_ids),
        "feedback_coverage": round(coverage, 4),
        "closed_without_feedback": len(missing_feedback),
        "missing_feedback_trade_ids_sample": missing_feedback[:20],
        "open_trade_missing_brackets": missing_brackets,
        "tier7_running": bool(tier7.get("running")) if isinstance(tier7, Mapping) else False,
    })

    if len(broker_pending) != account_open_count:
        issues.append(RuntimeHealthIssue(
            "warning", "journal_account_open_count_mismatch",
            "broker-confirmed journal open-entry count differs from OANDA account open-trade count",
            {"journal_pending": len(broker_pending), "account_open": account_open_count},
            "journal_broker_pending == account_open",
        ))
    if journal_open_not_in_account:
        issues.append(RuntimeHealthIssue(
            "warning", "journal_open_not_in_account",
            "journal has open instrument(s) absent from the OANDA account snapshot",
            journal_open_not_in_account[:20],
            [],
        ))
    if account_open_not_in_journal:
        issues.append(RuntimeHealthIssue(
            "warning", "account_open_not_in_journal",
            "OANDA account has open instrument(s) absent from the trade journal",
            account_open_not_in_journal[:20],
            [],
        ))
    if closed_ids and coverage < min_feedback_coverage:
        severity = "critical" if not Path(feedback_path).exists() or coverage < 0.5 else "warning"
        issues.append(RuntimeHealthIssue(
            severity, "post_trade_feedback_gap",
            f"post-trade feedback coverage {coverage:.1%} below {min_feedback_coverage:.1%}",
            round(coverage, 4), min_feedback_coverage,
        ))

    if any(issue.severity == "critical" for issue in issues):
        status = "CRITICAL"
    elif issues:
        status = "DEGRADED"
    else:
        status = "HEALTHY"
    return RuntimeHealthReport(
        status=status,
        checked_at=_utc_now(),
        issues=issues,
        metrics=metrics,
    )


def runtime_health_to_dict(report: RuntimeHealthReport) -> Dict[str, Any]:
    return asdict(report)


__all__ = [
    "RuntimeHealthIssue",
    "RuntimeHealthReport",
    "evaluate_runtime_health",
    "runtime_health_to_dict",
]
