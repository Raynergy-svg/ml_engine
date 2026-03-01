from src.agent.tool_registry import _render_planner_scan_summary


def test_render_planner_scan_summary_session_block():
    summary = {
        "count": 2,
        "tradable_count": 0,
        "approved_count": 0,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "HOLD", "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)"},
            {"pair": "GBP_USD", "direction": "HOLD", "error": "Outside trading session (4:00 UTC, active: 8-21 UTC)"},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="M5").lower()
    assert "nothing is tradeable right now" in text
    assert "session filter blocked every setup" in text


def test_render_planner_scan_summary_approved_setups():
    summary = {
        "count": 5,
        "tradable_count": 2,
        "approved_count": 1,
        "all_rows": [
            {"pair": "EUR_USD", "direction": "LONG", "error": None},
            {"pair": "GBP_USD", "direction": "SHORT", "error": None},
        ],
    }

    text = _render_planner_scan_summary(summary, granularity="M5").lower()
    assert "1 approved setups" in text
    assert "best read" in text
    assert "eur/usd long" in text
