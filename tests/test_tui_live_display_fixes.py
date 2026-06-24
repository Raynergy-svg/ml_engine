"""Live-dashboard display fixes (L1-L5) — found running the bot live.

L1 (HIGH): AgentPanel must NOT render fabricated DEMO_AGENTS in live mode;
           absent real data it shows an explicit "awaiting first scan".
L2: HeaderBar scanner status must reflect halt (never "ACTIVE" while halted) —
    single source of truth with the footer.
L3: SystemHealthBar OANDA dot is green only on a real successful fetch
    (connected AND latency>0), not an optimistic "● 0ms".
L4: HeaderBar NAV shows "—" (or last-known) when unfetched, never a fake "$0".
L5: SystemHealthBar hides CPU/MEM ("—") when the resource read is unavailable
    (psutil missing → 0/0), instead of a misleading "0%/0MB".

No mocks: real widgets + real DashboardSnapshot/AgentScore; render() is pure.
"""
from __future__ import annotations

from src.tui.app import DEMO_AGENTS, AgentPanel, HeaderBar, SystemHealthBar
from src.tui.data_provider import AgentScore, DashboardSnapshot


# ---------------------------------------------------------------------------
# L1 — AgentPanel never shows DEMO signal in live mode
# ---------------------------------------------------------------------------

def test_agent_panel_empty_shows_awaiting_not_demo():
    panel = AgentPanel()
    out = panel.render().plain
    assert "AWAITING FIRST SCAN" in out
    # None of the fabricated DEMO labels/values may appear.
    assert "BULLISH" not in out and "trend" not in out
    for name, _score, _sig in DEMO_AGENTS:
        assert name not in out


def test_agent_panel_shows_real_agents_after_snapshot():
    panel = AgentPanel()
    snap = DashboardSnapshot()
    snap.agents = [
        AgentScore(name="trend", score=0.40, signal="BEARISH"),
        AgentScore(name="risk", score=0.55, signal="OK"),
    ]
    panel.update_from_snapshot(snap)
    out = panel.render().plain
    assert "trend" in out and "BEARISH" in out
    assert "AWAITING" not in out


def test_agent_panel_load_demo_is_explicit_opt_in():
    panel = AgentPanel()
    panel.load_demo()
    out = panel.render().plain
    assert "trend" in out and "BULLISH" in out  # demo only when explicitly asked


# ---------------------------------------------------------------------------
# L2 — header scanner status reflects halt
# ---------------------------------------------------------------------------

def test_header_shows_halted_even_when_scanner_ready():
    h = HeaderBar()
    h.scanner_ok = True
    h.halted = True
    out = h.render().plain
    assert "HALTED" in out
    assert "ACTIVE" not in out


def test_header_active_when_running_not_halted():
    h = HeaderBar()
    h.scanner_ok = True
    h.halted = False
    assert "ACTIVE" in h.render().plain


def test_header_idle_when_not_ready_not_halted():
    h = HeaderBar()
    h.scanner_ok = False
    h.halted = False
    assert "IDLE" in h.render().plain


# ---------------------------------------------------------------------------
# L4 — NAV "—" when unknown, never a fake $0
# ---------------------------------------------------------------------------

def test_header_nav_dash_when_unknown():
    h = HeaderBar()
    h.nav = 0.0
    h.nav_known = False
    out = h.render().plain
    assert "NAV —" in out
    assert "$0" not in out


def test_header_nav_shows_value_when_known():
    h = HeaderBar()
    h.nav = 102183.0
    h.pnl = 12.0
    h.nav_known = True
    h.oanda_ok = True
    out = h.render().plain
    assert "$102,183" in out


# ---------------------------------------------------------------------------
# L3 — footer OANDA green only on a real fetch
# ---------------------------------------------------------------------------

def _health(**fields) -> SystemHealthBar:
    bar = SystemHealthBar()
    snap = DashboardSnapshot()
    for k, v in fields.items():
        setattr(snap, k, v)
    bar.update_from_snapshot(snap)
    return bar


def test_footer_oanda_off_when_connected_but_no_latency():
    out = _health(oanda_connected=True, oanda_latency_ms=0.0).render().plain
    assert "OANDA: ● OFF" in out


def test_footer_oanda_off_when_disconnected():
    out = _health(oanda_connected=False, oanda_latency_ms=42.0).render().plain
    assert "OANDA: ● OFF" in out


def test_footer_oanda_green_on_real_fetch():
    out = _health(oanda_connected=True, oanda_latency_ms=42.0).render().plain
    assert "● 42ms" in out


# ---------------------------------------------------------------------------
# L5 — CPU/MEM hidden when unavailable
# ---------------------------------------------------------------------------

def test_footer_resources_dash_when_unavailable():
    out = _health(cpu_pct=0.0, mem_mb=0.0).render().plain
    assert "CPU: —" in out and "MEM: —" in out
    assert "0MB" not in out


def test_footer_resources_shown_when_available():
    out = _health(cpu_pct=12.0, mem_mb=256.0).render().plain
    assert "CPU: 12%" in out and "MEM: 256MB" in out
