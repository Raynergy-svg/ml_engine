"""FX day-trading guardrails (Tier-1).

This module encodes conservative, fail-safe rules so the trading loop can't
"go rogue":
- Trade only within an EST session window
- Enforce hard daily stops (loss and confidence-based profit caps)
- Limit entries and open positions

It is intentionally broker-agnostic; OANDA integration lives in main + client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import logging
import os

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _parse_hhmm(s: str) -> time:
    parts = str(s).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {s}")
    h = int(parts[0])
    m = int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid HH:MM time: {s}")
    return time(hour=h, minute=m)


@dataclass(frozen=True)
class FxCosts:
    max_spread_pips: Dict[str, float]
    slippage_pips_min: float
    slippage_pips_spread_mult: float
    spread_fallback_pips: Dict[str, float]


@dataclass(frozen=True)
class FxSession:
    timezone: str
    start: time
    end: time
    force_flat_at: time
    reset_at: time


@dataclass(frozen=True)
class FxLimits:
    max_open_positions: int
    max_entries_per_day: int


@dataclass(frozen=True)
class FxRisk:
    risk_per_trade_pct: float
    daily_loss_stop_pct: float
    atr_stop_mult: float
    rr_take_profit: float


@dataclass(frozen=True)
class FxConfidence:
    low_lt: float
    high_gte: float
    profit_stop_pct_by_band: Dict[str, float]


@dataclass(frozen=True)
class FxPolicy:
    enabled: bool
    tier: int
    instruments: Tuple[str, ...]
    granularity: str
    horizon_bars: int
    session: FxSession
    limits: FxLimits
    risk: FxRisk
    costs: FxCosts
    confidence: FxConfidence
    require_confirmation: bool
    practice_only: bool


@dataclass
class FxDailyState:
    date: str  # YYYY-MM-DD in policy timezone (session date)
    start_nav: Optional[float] = None
    start_balance: Optional[float] = None
    entries_today: int = 0
    disabled_reason: Optional[str] = None
    disabled_kind: Optional[str] = None  # "loss" | "profit" | None
    last_updated_at: Optional[str] = None


def load_fx_policy(cfg: Dict[str, Any]) -> FxPolicy:
    fx = dict((cfg or {}).get("fx") or {})

    enabled = bool(fx.get("enabled", True))
    tier = int(fx.get("tier", 1))

    instruments = tuple(str(x) for x in (fx.get("instruments") or ["EUR_USD"]))
    granularity = str(fx.get("granularity", "M5"))
    horizon_bars = int(fx.get("horizon_bars", 1))

    session_cfg = dict(fx.get("session") or {})
    tz = str(fx.get("timezone") or session_cfg.get("timezone") or "America/New_York")
    session = FxSession(
        timezone=tz,
        start=_parse_hhmm(session_cfg.get("start", "08:00")),
        end=_parse_hhmm(session_cfg.get("end", "11:30")),
        force_flat_at=_parse_hhmm(session_cfg.get("force_flat_at", "11:55")),
        reset_at=_parse_hhmm(session_cfg.get("reset_at", "08:00")),
    )

    limits_cfg = dict(fx.get("limits") or {})
    limits = FxLimits(
        max_open_positions=int(limits_cfg.get("max_open_positions", 1)),
        max_entries_per_day=int(limits_cfg.get("max_entries_per_day", 1)),
    )

    risk_cfg = dict(fx.get("risk") or {})
    risk = FxRisk(
        risk_per_trade_pct=float(risk_cfg.get("risk_per_trade_pct", 0.005)),
        daily_loss_stop_pct=float(risk_cfg.get("daily_loss_stop_pct", 0.10)),
        atr_stop_mult=float(risk_cfg.get("atr_stop_mult", 1.5)),
        rr_take_profit=float(risk_cfg.get("rr_take_profit", 1.5)),
    )

    costs_cfg = dict(fx.get("costs") or {})
    max_spread_pips = dict(costs_cfg.get("max_spread_pips") or {})
    spread_fallback_pips = dict(costs_cfg.get("spread_fallback_pips") or {})
    costs = FxCosts(
        max_spread_pips={str(k): float(v) for k, v in max_spread_pips.items()},
        slippage_pips_min=float(costs_cfg.get("slippage_pips_min", 0.2)),
        slippage_pips_spread_mult=float(costs_cfg.get("slippage_pips_spread_mult", 0.25)),
        spread_fallback_pips={str(k): float(v) for k, v in spread_fallback_pips.items()},
    )

    conf_cfg = dict(fx.get("confidence") or {})
    profit_stop = dict(conf_cfg.get("profit_stop_pct_by_band") or {"medium": 0.30, "high": 0.30})
    confidence = FxConfidence(
        low_lt=float(conf_cfg.get("low_lt", 0.60)),
        high_gte=float(conf_cfg.get("high_gte", 0.75)),
        profit_stop_pct_by_band={str(k): float(v) for k, v in profit_stop.items()},
    )

    exec_cfg = dict(fx.get("execution") or {})
    require_confirmation = bool(exec_cfg.get("require_confirmation", True))
    practice_only = bool(exec_cfg.get("practice_only", True))

    return FxPolicy(
        enabled=enabled,
        tier=tier,
        instruments=instruments,
        granularity=granularity,
        horizon_bars=horizon_bars,
        session=session,
        limits=limits,
        risk=risk,
        costs=costs,
        confidence=confidence,
        require_confirmation=require_confirmation,
        practice_only=practice_only,
    )


def _tzinfo(name: str):
    if ZoneInfo is None:
        raise RuntimeError("zoneinfo not available; requires Python 3.9+")
    return ZoneInfo(name)


def now_in_tz(tz_name: str, *, now: Optional[datetime] = None) -> datetime:
    if now is None:
        base = datetime.now(tz=_tzinfo("UTC"))
    else:
        base = now
        if base.tzinfo is None:
            base = base.replace(tzinfo=_tzinfo("UTC"))
    return base.astimezone(_tzinfo(tz_name))


def session_date_str(policy: FxPolicy, *, now: Optional[datetime] = None) -> str:
    """Return the *session date* (YYYY-MM-DD) in the policy timezone.

    Session date respects `policy.session.reset_at`: times before reset are
    attributed to the previous session day.
    """
    dt = now_in_tz(policy.session.timezone, now=now)
    t = dt.timetz().replace(tzinfo=None)
    if t < policy.session.reset_at:
        return (dt.date() - timedelta(days=1)).isoformat()
    return dt.date().isoformat()


def within_time_window(dt: datetime, start: time, end: time) -> bool:
    # Only same-day windows are supported (conservative, matches day trading).
    t = dt.timetz().replace(tzinfo=None)
    return (t >= start) and (t <= end)


def should_force_flat(policy: FxPolicy, *, now: Optional[datetime] = None) -> bool:
    dt = now_in_tz(policy.session.timezone, now=now)
    t = dt.timetz().replace(tzinfo=None)
    return t >= policy.session.force_flat_at


def can_open_new_trade(policy: FxPolicy, state: FxDailyState, *, now: Optional[datetime] = None) -> Tuple[bool, str]:
    if not policy.enabled:
        return False, "FX disabled"

    dt = now_in_tz(policy.session.timezone, now=now)

    if state.disabled_reason:
        return False, f"Trading disabled: {state.disabled_reason}"

    if not within_time_window(dt, policy.session.start, policy.session.end):
        return False, "Outside session window"

    if should_force_flat(policy, now=now):
        return False, "Past force-flat cutoff"

    if state.entries_today >= policy.limits.max_entries_per_day:
        return False, "Daily entry limit reached"

    return True, "OK"


def confidence_band(policy: FxPolicy, confidence: float) -> str:
    c = float(max(0.0, min(1.0, confidence)))
    if c < policy.confidence.low_lt:
        return "low"
    if c >= policy.confidence.high_gte:
        return "high"
    return "medium"


def profit_stop_pct_for_band(policy: FxPolicy, band: str) -> Optional[float]:
    if band == "high":
        return policy.confidence.profit_stop_pct_by_band.get("high")
    if band == "medium":
        return policy.confidence.profit_stop_pct_by_band.get("medium")
    return None


def state_path(cfg: Dict[str, Any], *, date_str: str) -> Path:
    log_dir = (
        cfg.get("LOG_DIR")
        or cfg.get("paths", {}).get("log_dir")
        or cfg.get("LOG_DIR")
        or "trained_data/logs"
    )
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"fx_state_{date_str}.json"


def load_state(cfg: Dict[str, Any], policy: FxPolicy, *, now: Optional[datetime] = None) -> FxDailyState:
    date_str = session_date_str(policy, now=now)
    path = state_path(cfg, date_str=date_str)
    if not path.exists():
        return FxDailyState(date=date_str)

    try:
        import fcntl
        with open(path, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                payload = json.loads(f.read())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        # fcntl not available — fall back to unlocked read
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("fx_guardrails: corrupted state file %s: %s — resetting", path, e)
            return FxDailyState(date=date_str)
    except json.JSONDecodeError as e:
        logger.warning("fx_guardrails: corrupted state file %s: %s — resetting", path, e)
        return FxDailyState(date=date_str)
    except Exception as e:
        logger.error("fx_guardrails: failed to load state %s: %s", path, e)
        return FxDailyState(date=date_str)

    st = FxDailyState(date=str(payload.get("date") or date_str))
    st.start_nav = _to_float(payload.get("start_nav"))
    st.start_balance = _to_float(payload.get("start_balance"))
    st.entries_today = int(payload.get("entries_today") or 0)
    st.disabled_reason = payload.get("disabled_reason")
    st.disabled_kind = payload.get("disabled_kind")
    st.last_updated_at = payload.get("last_updated_at")
    return st


def save_state(cfg: Dict[str, Any], *args) -> Path:
    """Persist daily FX state.

    Backward compatible with both:
    - save_state(cfg, state)
    - save_state(cfg, policy, state)  (policy is ignored; path is derived from cfg+state.date)
    """

    if len(args) == 1:
        state = args[0]
    elif len(args) == 2:
        # Historical signature included policy; ignore it.
        state = args[1]
    else:
        raise TypeError("save_state expects (cfg, state) or (cfg, policy, state)")

    if not isinstance(state, FxDailyState):
        raise TypeError("save_state: state must be FxDailyState")

    path = state_path(cfg, date_str=state.date)
    payload = {
        "date": state.date,
        "start_nav": state.start_nav,
        "start_balance": state.start_balance,
        "entries_today": state.entries_today,
        "disabled_reason": state.disabled_reason,
        "disabled_kind": state.disabled_kind,
        "last_updated_at": state.last_updated_at,
    }
    # H-3: atomic write — prevents guardrail state corruption on crash
    _tmp = path.with_suffix(".tmp")
    _tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(_tmp), str(path))
    return path


def update_state_from_account_summary(
    policy: FxPolicy,
    state: FxDailyState,
    account_summary: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Optional[float]]:
    dt = now_in_tz(policy.session.timezone, now=now)
    state.last_updated_at = dt.isoformat()

    acct = (account_summary or {}).get("account") or {}
    nav = _to_float(acct.get("NAV"))
    balance = _to_float(acct.get("balance"))

    if state.start_nav is None and nav is not None:
        state.start_nav = nav
    if state.start_balance is None and balance is not None:
        state.start_balance = balance

    base = state.start_nav
    if base is None or base <= 0:
        # C-3 fix: Never silently disable drawdown monitoring.
        # If start_nav is invalid, reset to current NAV so monitoring stays active.
        if nav is not None and nav > 0:
            logger.error(
                "fx_guardrails: start_nav invalid (%.2f), resetting to current NAV %.2f — "
                "drawdown monitoring was at risk of being silently disabled",
                base if base is not None else 0.0, nav,
            )
            state.start_nav = nav
            base = nav
        else:
            logger.error(
                "fx_guardrails: start_nav invalid (%.2f) and current NAV also invalid (%.2f) — "
                "returning drawdown_pct=0.0 (NOT None) to keep monitoring active",
                base if base is not None else 0.0, nav if nav is not None else 0.0,
            )
            return {"nav": nav, "balance": balance, "drawdown_pct": 0.0, "realized_pct": 0.0}

    drawdown_pct = None
    realized_pct = None
    if nav is not None:
        drawdown_pct = (nav - base) / base
    if balance is not None and state.start_balance is not None:
        realized_pct = (balance - state.start_balance) / base

    return {"nav": nav, "balance": balance, "drawdown_pct": drawdown_pct, "realized_pct": realized_pct}


def check_daily_stops(
    policy: FxPolicy,
    *,
    drawdown_pct: Optional[float],
    realized_pct: Optional[float],
    confidence_band: str,
) -> Tuple[bool, Optional[str], Optional[str]]:
    # Loss stop triggers on equity (realized + unrealized).
    if drawdown_pct is not None and drawdown_pct <= -abs(policy.risk.daily_loss_stop_pct):
        return True, f"Daily loss stop hit ({drawdown_pct:.1%})", "loss"

    # Profit stop triggers on realized-only.
    target = profit_stop_pct_for_band(policy, confidence_band)
    if target is not None and realized_pct is not None and realized_pct >= float(target):
        return (
            True,
            f"Daily profit stop hit for band={confidence_band} ({realized_pct:.1%} >= {target:.0%})",
            "profit",
        )

    return False, None, None
# — Raynergy-svg —
