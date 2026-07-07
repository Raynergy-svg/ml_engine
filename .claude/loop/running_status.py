#!/usr/bin/env python3
"""Deterministic running-status oracle (L-017) — LIVE lane + every shadow lane.

Re-derives running:YES/NO from disk + ps, fail-closed, read-only. Reports EVERY
lane distinctly LABELED so a dormant/legacy lane's "running:NO" is NEVER mistaken
for "the bot is down" (the trap that caused a false "nothing running" on
2026-06-29), and — the P0 2026-07-06 fix — so an ACTIVE shadow lane is never
mislabeled "dormant/legacy" just because it shares a code family with a
genuinely retired path.

  LIVE LANE  = the OANDA **practice** trend loop (the actual trader) + the Tier 7
               self-heal supervisor. This is what is actually live. running:YES if ANY of:
                 (a) the `run_oanda_trend.py` process is alive, OR
                 (b) `trained_data/oanda/account_state.json` is fresh (<=2h; the trend
                     loop's hourly output — a disk proxy when ps is unavailable), OR
                 (c) the Tier 7 heartbeat (`.claude/heartbeat.json`) is fresh (<=90s)
                     AND its pid is alive.
  SHADOW LANES (each a distinct `.claude/state.json` `halted_lanes` key, none can
               trade/arm — read-only research/harvester loops) = equity (the
               OANDA-based shadow harvester, `run_equity_harvester.py` — NOT the
               same thing as the retired IBKR-era path below), brain
               (`run_brain_loop.py`), crypto_momentum (`run_crypto_momentum_
               shadow.py`), track_b (`run_track_b_shadow.py`), crypto_carry
               (`run_crypto_carry_shadow.py`). running:YES if the
               process is alive OR its ledger/state artifact is fresh for its
               own cadence (each script logs its own `--interval`).
  LEGACY LANE (dormant, RETIRED IBKR-era code paths, SUPERSEDED by the lanes
               above) = `AutonomousLoop` / `embedded_scanner` / `continuous.py` /
               `buddy_scanner`. Its running:NO is EXPECTED and means NOTHING is
               wrong.

`--assert-running` asserts the LIVE lane only (exit 3 if it's down) — the shadow
and legacy lanes are informational, matching the fact that only the LIVE lane
can place an order.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root from .claude/loop/
HEARTBEAT_FRESH_S = 90
ACCOUNT_STATE_FRESH_S = 7200  # trend loop ticks hourly -> 2h tolerance
_TREND_NEEDLE = "run_oanda_trend"
_LEGACY_NEEDLES = (
    "AutonomousLoop", "embedded_scanner", "continuous.py", "buddy_scanner",
)

# Shadow lanes (P0 2026-07-06): each entry is (label, process needle, artifact
# path relative to ROOT, freshness tolerance). Artifact freshness is a
# best-effort SECONDARY signal — the daily-interval lanes (crypto_momentum,
# track_b) only touch their ledger once every ~24h by design, and brain_loop's
# ledger stays empty until it proposes its first hypothesis — so a lane is
# "running" if its PROCESS is alive, even when the artifact hasn't ticked yet.
_SHADOW_LANES = (
    ("equity", "run_equity_harvester", "trained_data/equity/cycle_ledger.jsonl", 7200),
    ("brain", "run_brain_loop", "trained_data/brain_loop/hypotheses.jsonl", 90000),
    ("crypto_momentum", "run_crypto_momentum_shadow",
     "trained_data/crypto/shadow_momentum_ledger.jsonl", 172800),
    ("track_b", "run_track_b_shadow", "trained_data/research/track_b_shadow_ledger.jsonl", 172800),
    ("crypto_carry", "run_crypto_carry_shadow",
     "trained_data/crypto_carry/shadow_carry_ledger.jsonl", 172800),
)


def _ps_lines() -> list[str] | None:
    try:
        return subprocess.run(
            ["ps", "axo", "command"], capture_output=True, text=True, timeout=5
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return None  # fail-closed: can't tell


def _proc_alive(needles, lines) -> bool:
    if lines is None:
        return False
    for ln in lines:
        if "pytest" in ln or "grep" in ln or "running_status" in ln:
            continue
        if any(n in ln for n in needles):
            return True
    return False


def _pid_alive(pid) -> bool:
    try:
        if int(pid) <= 0:   # os.kill(0,...) targets the process GROUP — never "alive"
            return False
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def _file_fresh(rel: str, max_age_s: int) -> tuple[bool, float | None]:
    p = ROOT / rel
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return (False, None)
    return (age <= max_age_s, age)


def _tier7_heartbeat() -> tuple[bool, float | None, bool]:
    try:
        hb = json.loads((ROOT / ".claude" / "heartbeat.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (False, None, False)
    ts, pid = hb.get("ts_iso"), hb.get("pid")
    age = None
    if ts:
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - t).total_seconds()
        except (ValueError, TypeError):
            age = None
    fresh = age is not None and age <= HEARTBEAT_FRESH_S
    return (fresh, age, _pid_alive(pid))


def shadow_lanes_running(lines: list[str] | None = None) -> dict:
    """Disk/ps-derived status for every shadow lane, keyed by lane name.

    ``lines`` lets ``main()`` reuse one ``ps`` snapshot across all lanes
    instead of shelling out per lane.
    """
    if lines is None:
        lines = _ps_lines()
    out: dict = {}
    for name, needle, artifact_rel, fresh_s in _SHADOW_LANES:
        proc = _proc_alive((needle,), lines)
        fresh, age = _file_fresh(artifact_rel, fresh_s)
        out[name] = {
            "running": bool(proc or fresh),
            "process_alive": proc,
            "artifact_fresh": fresh,
            "artifact_age_s": age,
        }
    return out


def live_lane_running() -> dict:
    """Disk/ps-derived LIVE-lane status (the actually-live OANDA trend + Tier 7 lane)."""
    lines = _ps_lines()
    trend_proc = _proc_alive((_TREND_NEEDLE,), lines)
    acct_fresh, acct_age = _file_fresh("trained_data/oanda/account_state.json", ACCOUNT_STATE_FRESH_S)
    hb_fresh, hb_age, hb_pid = _tier7_heartbeat()
    tier7 = bool(hb_fresh and hb_pid)
    return {
        "running": bool(trend_proc or acct_fresh or tier7),
        "oanda_trend_proc": trend_proc,
        "account_state_fresh": acct_fresh, "account_state_age_s": acct_age,
        "tier7_heartbeat_fresh": hb_fresh, "tier7_heartbeat_age_s": hb_age, "tier7_pid_alive": hb_pid,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    live = live_lane_running()

    lines = _ps_lines()
    shadow = shadow_lanes_running(lines)
    legacy_proc = _proc_alive(_LEGACY_NEEDLES, lines)

    def _ag(x):
        return "n/a" if x is None else ("%.0fs" % x)

    print(
        "LIVE LANE (OANDA trend + Tier7 — THIS is the live bot) — running: %s\n"
        "  oanda_trend_proc: %s · account_state_fresh: %s (%s) · "
        "tier7_heartbeat: %s (age %s, pid_alive %s)" % (
            "YES" if live["running"] else "NO",
            "YES" if live["oanda_trend_proc"] else "NO",
            "YES" if live["account_state_fresh"] else "NO", _ag(live["account_state_age_s"]),
            "YES" if live["tier7_heartbeat_fresh"] else "NO",
            _ag(live["tier7_heartbeat_age_s"]), "YES" if live["tier7_pid_alive"] else "NO",
        )
    )
    for name, st in shadow.items():
        print(
            "SHADOW LANE %s (read-only research/harvester — cannot trade/arm) — running: %s "
            "· process: %s · artifact_fresh: %s (%s)" % (
                name, "YES" if st["running"] else "NO",
                "YES" if st["process_alive"] else "NO",
                "YES" if st["artifact_fresh"] else "NO", _ag(st["artifact_age_s"]),
            )
        )
    print(
        "LEGACY LANE (dormant, retired IBKR-era code — SUPERSEDED; running:NO is EXPECTED, "
        "not a fault) — running: %s · process: %s" % (
            "YES" if legacy_proc else "NO", "YES" if legacy_proc else "NO",
        )
    )
    if "--assert-running" in argv and not live["running"]:
        return 3  # the LIVE lane is down
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
