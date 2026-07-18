"""Corrections from the 2026-07-18 operator review of the promotion gate
(84d4348) — findings F8, F9 and the scorecard's missing GENUINE quadrant.

No mocks: real BookSnapshot injection, real synthetic price panels
(pd.DataFrame), real JSONL ledgers on real disk (tmp_path), real
reconcile_unresolved / run_all / _three_way_decision.

F8  ledger rows persist the book's weights, and unresolved rows (stale price
    cache at cycle time) are revisited + re-marked once forward bars exist —
    atomically, fail-closed, without ever touching resolved rows.
F9  the RL sync's regime reward-ratio floor (0.3) must NOT apply when the
    shaped reward is a residual-alpha reward: a pure-beta win's ~0 residual
    credit is the semantics, not a value to be floored away.
GEN the scorecard emits the interpretation table's "genuine strategy-specific
    signal" verdict for the raw>0 AND hedged>0 quadrant (previously fell to
    INCONCLUSIVE, so the gate's pass-set referenced an unreachable verdict).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.hedge import hedged_shadow_lane as hsl
from src.hedge.hedge_scorecard import (
    VERDICT_BETA_NOT_ALPHA,
    VERDICT_GENUINE,
    VERDICT_INCONCLUSIVE,
    VERDICT_SIGNAL_REAL_BUT_NOISY,
    VERDICT_WEAK_OR_DEAD,
    _three_way_decision,
)


# ── shared fixtures ─────────────────────────────────────────────────────────

def _panel(tickers_closes):
    """Synthetic tz-aware daily close panel, 2026-05-30..06-02."""
    idx = pd.to_datetime(["2026-05-30", "2026-06-01", "2026-06-02"]).tz_localize("UTC")
    return pd.DataFrame(tickers_closes, index=idx)


def _unresolved_row(strategy="equity_harvester", asof="2026-06-01",
                    weights=None, with_weights=True):
    row = {
        "cycle_ts": f"{asof}T00:00:00+00:00",
        "strategy": strategy,
        "asset_class": "equity",
        "asof_date": asof,
        "notional": 100_000.0,
        "raw": {"return_source": "must_compute", "net_return": None,
                "ledger_net_return": None, "gross_return": None, "cost": None,
                "notes": []},
        "exposure": {"fail_closed": False},
        "hedge": {"status": "no_valid_hedge", "applied_proposal": None,
                  "overlay_gross_pnl_dollars": None, "overlay_cost_known": False,
                  "overlay_cost_dollars": None, "notes": []},
        "hedged": {"gross_return": None, "net_return": None,
                   "return_basis": "unresolved"},
        "runtime_allowed": False, "paper_only": True, "human_review_required": True,
    }
    if with_weights:
        row["weights"] = weights if weights is not None else {"AAA": 0.5, "BBB": -0.3}
    return row


def _write_ledger(tmp_path, rows):
    p = tmp_path / "raw_vs_hedged_ledger.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return p


# ── F8a: rows persist the book's weights ────────────────────────────────────

def test_cycle_row_persists_book_weights(tmp_path):
    book = hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date="2026-06-01",
        weights={"AAA": 0.5, "BBB": -0.3}, raw_net_return=None,
        raw_gross_return=None, raw_cost=None, return_source="must_compute")
    panel = _panel({"AAA": [99.0, 100.0, 102.0], "BBB": [51.0, 50.0, 49.0]})
    row = hsl.run_cycle_for_strategy(
        "equity_harvester", book=book, price_panel=panel,
        sector_bucket_map={}, currency_bucket_map={},
        ledger_path=tmp_path / "l.jsonl", decision_log_path=tmp_path / "d.jsonl")
    assert row is not None
    assert row["weights"] == {"AAA": 0.5, "BBB": -0.3}, (
        "without persisted weights an unresolved row can never be re-marked")


# ── F8b: reconcile_unresolved re-marks stale rows ───────────────────────────

def test_reconcile_resolves_row_once_forward_bars_exist(tmp_path):
    panel = _panel({"AAA": [99.0, 100.0, 102.0], "BBB": [51.0, 50.0, 49.0]})
    stale = _unresolved_row()          # asof 06-01, unresolved at cycle time
    resolved = _unresolved_row(strategy="track_b", asof="2026-06-01")
    resolved["raw"]["net_return"] = 0.001
    resolved["hedged"] = {"gross_return": 0.001, "net_return": 0.001,
                          "return_basis": "net"}
    ledger = _write_ledger(tmp_path, [stale, resolved])

    summary = hsl.reconcile_unresolved(
        ledger, price_panel=panel, sector_bucket_map={}, currency_bucket_map={},
        today="2026-07-01")
    assert summary["checked"] == 1 and summary["resolved"] == 1

    rows = hsl._read_jsonl_rows(ledger)
    assert len(rows) == 2 and rows[0]["strategy"] == "equity_harvester"
    got = rows[0]
    # 06-01 -> 06-02 forward bar: AAA +2%, BBB -2% -> 0.5*0.02 + (-0.3)*(-0.02)
    assert got["raw"]["net_return"] == pytest.approx(0.016)
    assert "reconciled_forward_mark(F8)" in got["raw"]["notes"]
    # no hedge applied -> overlay 0 with cost known -> hedged == raw, net basis
    assert got["hedged"]["net_return"] == pytest.approx(0.016)
    assert got["hedged"]["return_basis"] == "net"
    assert "reconciled_at" in got
    # the already-resolved row is untouched
    assert "reconciled_at" not in rows[1]
    assert rows[1]["raw"]["net_return"] == pytest.approx(0.001)


def test_reconcile_fail_closed_paths(tmp_path):
    panel = _panel({"AAA": [99.0, 100.0, 102.0]})
    no_weights = _unresolved_row(with_weights=False)              # legacy row
    unmarkable = _unresolved_row(strategy="track_b",
                                 weights={"ZZZ_NOT_IN_PANEL": 1.0})
    too_fresh = _unresolved_row(strategy="crypto_momentum", asof="2026-07-01")
    ledger = _write_ledger(tmp_path, [no_weights, unmarkable, too_fresh])

    summary = hsl.reconcile_unresolved(
        ledger, price_panel=panel, sector_bucket_map={}, currency_bucket_map={},
        today="2026-07-01")
    # today-or-later rows are not even checked (forward bar cannot exist yet)
    assert summary == {"checked": 2, "resolved": 0,
                       "still_unresolved": 1, "skipped_no_weights": 1}

    rows = hsl._read_jsonl_rows(ledger)
    assert len(rows) == 3
    for r in rows:  # every row still honestly unresolved — never partial-filled
        assert r["raw"]["net_return"] is None
        assert r["hedged"]["return_basis"] == "unresolved"
        assert "reconciled_at" not in r
    # empty/missing ledger is a no-op, not a crash
    assert hsl.reconcile_unresolved(tmp_path / "missing.jsonl") == {
        "checked": 0, "resolved": 0, "still_unresolved": 0, "skipped_no_weights": 0}


def test_reconcile_preserves_corrupt_and_blank_lines_verbatim(tmp_path):
    """rev 2.1 (review): the rewrite must never launder the evidence ledger —
    unparseable and blank lines survive BYTE-FOR-BYTE even when another line
    on the same file is reconciled."""
    panel = _panel({"AAA": [99.0, 100.0, 102.0], "BBB": [51.0, 50.0, 49.0]})
    stale = _unresolved_row()
    corrupt = '{"strategy": "truncated_by_a_crash", "asof_da'
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    ledger.write_text(
        json.dumps(stale, sort_keys=True) + "\n" + corrupt + "\n" + "\n",
        encoding="utf-8")

    summary = hsl.reconcile_unresolved(
        ledger, price_panel=panel, sector_bucket_map={}, currency_bucket_map={},
        today="2026-07-01")
    assert summary["resolved"] == 1

    lines = ledger.read_text(encoding="utf-8").split("\n")
    assert len(lines) == 4 and lines[3] == ""  # 3 lines + trailing newline
    assert json.loads(lines[0])["raw"]["net_return"] == pytest.approx(0.016)
    assert lines[1] == corrupt, "corrupt evidence line must survive verbatim"
    assert lines[2] == "", "blank line must survive verbatim"


def test_run_all_wires_reconciliation_before_the_cycle(tmp_path):
    # Real repo price panel; skip when the environment lacks it. Pick a real
    # ticker + a past asof so the forward bar genuinely exists on disk.
    panel = hsl.load_equity_price_panel()
    if panel.empty or len(panel.index) < 3:
        pytest.skip("equity price panel not cached in this environment")
    asof = panel.index[-3].date().isoformat()
    ticker, expected = None, None
    for cand in panel.columns:
        ret, _notes = hsl.basket_forward_return(panel, {cand: 1.0}, asof)
        if ret is not None:
            ticker, expected = cand, ret
            break
    if ticker is None:
        pytest.skip("no panel ticker resolves a forward mark at " + asof)

    ledger = _write_ledger(
        tmp_path, [_unresolved_row(asof=asof, weights={ticker: 1.0})])
    hsl.run_all(strategies=(), ledger_path=ledger,
                decision_log_path=tmp_path / "d.jsonl", persist=True)
    rows = hsl._read_jsonl_rows(ledger)
    assert rows[0]["raw"]["net_return"] == pytest.approx(expected)
    assert "reconciled_at" in rows[0]


# ── F9: residual reward must not be floored back to 0.3 ─────────────────────

def test_regime_reward_ratio_semantics_executed():
    """The extracted helper IS the loop's ratio (see wiring test below) — so
    these are executed semantics, not source inspection."""
    from src.scanner.execution import regime_reward_ratio

    # ordinary shaped rewards keep the regime-shaping guard: floor 0.3, cap 2.0
    assert regime_reward_ratio(100.0, 10.0, False) == pytest.approx(0.3)
    assert regime_reward_ratio(100.0, 90.0, False) == pytest.approx(0.9)
    assert regime_reward_ratio(10.0, 100.0, False) == pytest.approx(2.0)
    assert regime_reward_ratio(0.0005, 5.0, False) == pytest.approx(1.0)  # tiny pnl
    assert regime_reward_ratio(100.0, None, False) == pytest.approx(0.3)
    # residual rewards: ~0 IS the semantics — no floor anywhere
    assert regime_reward_ratio(100.0, 10.0, True) == pytest.approx(0.1)
    assert regime_reward_ratio(100.0, 0.0, True) == pytest.approx(0.0)
    assert regime_reward_ratio(100.0, 1e-9, True) == pytest.approx(0.0, abs=1e-9)
    # rev 2.1 (review): the tiny-P&L branch must NOT bypass the residual rule —
    # a tiny pure-beta result previously got full ordinary credit back
    assert regime_reward_ratio(0.0005, 0.0, True) == pytest.approx(0.0)
    assert regime_reward_ratio(0.0, 5.0, True) == pytest.approx(0.0)
    # residual cap still holds
    assert regime_reward_ratio(10.0, 100.0, True) == pytest.approx(2.0)


def test_zero_residual_ratio_leaves_real_agent_weights_unchanged(tmp_path, monkeypatch):
    """Executed through the REAL updater: a pure-beta win whose residual
    reward ratio is 0.0 scales boost/penalty to 0 exactly as the sync loop
    does, and ScannerAgentTeam.update_weights_from_outcome then produces ZERO
    weight movement — on real disk (tmp cwd), no mocks. A control update with
    a nonzero ratio proves the test can detect movement."""
    from src.scanner.agents import ScannerAgentTeam
    from src.scanner.config import ScannerConfig
    from src.scanner.execution import regime_reward_ratio

    monkeypatch.chdir(tmp_path)  # _WEIGHTS_FILE is cwd-relative
    cfg = ScannerConfig()
    team = ScannerAgentTeam(config=cfg)
    verdicts = [{"name": "trend", "passed": True},
                {"name": "momentum", "passed": False}]
    before = dict(team._learned_weights.get("_global", team._BASE_WEIGHTS))

    # pure-beta win, residual reward live: shaped residual reward ~ 0
    ratio = regime_reward_ratio(120.0, 0.0, True)
    assert ratio == 0.0
    orig_boost = cfg.weight_boost_on_win
    orig_penalty = cfg.weight_penalty_on_loss
    cfg.weight_boost_on_win = round(orig_boost * ratio, 4)
    cfg.weight_penalty_on_loss = round(orig_penalty * ratio, 4)
    after = team.update_weights_from_outcome(verdicts, trade_won=True, regime="NORMAL")
    for name in ("trend", "momentum"):
        assert after[name] == pytest.approx(before[name]), (
            f"{name}: pure-beta win must move NO weight when residual credit is 0")

    # control: restore ordinary boost — the same call must now move weights
    cfg.weight_boost_on_win = orig_boost
    cfg.weight_penalty_on_loss = orig_penalty
    moved = team.update_weights_from_outcome(verdicts, trade_won=True, regime="NORMAL")
    assert moved["trend"] != pytest.approx(before["trend"])


def test_rl_sync_wires_the_helper_and_the_flag():
    """Wiring lock (the loop lives inline in a 6k-line method): rl_updates
    rows carry residual_applied, the loop delegates to regime_reward_ratio,
    and no unconditional 0.3 clamp survives anywhere."""
    src = (Path(__file__).resolve().parents[1]
           / "src" / "scanner" / "execution.py").read_text(encoding="utf-8")
    assert '"residual_applied": _residual_live' in src
    assert '_reward_ratio = regime_reward_ratio(' in src
    assert 'bool(upd.get("residual_applied"))' in src
    assert "max(0.3, _abs_shaped" not in src
    # the flag goes live ONLY on the operator-gated branch
    flag_idx = src.index("_residual_live = True")
    gate_idx = src.index('_res["applied"] and getattr(self.config, "enable_residual_alpha_rewards"')
    assert gate_idx < flag_idx < gate_idx + 600


def test_residual_rewards_flag_defaults_off():
    from src.scanner.config import ScannerConfig
    assert ScannerConfig().enable_residual_alpha_rewards is False


# ── GENUINE quadrant ────────────────────────────────────────────────────────

def test_three_way_decision_covers_the_genuine_quadrant():
    # raw>0 AND hedged>0 without the improved/dd-cut pattern: the return
    # SURVIVES systematic removal -> the interpretation table's genuine verdict.
    assert _three_way_decision(0.004, 0.05, 0.003, 0.06) == VERDICT_GENUINE
    # precedence preserved: improvement + dd cut still reads as real-but-noisy
    assert _three_way_decision(0.004, 0.05, 0.005, 0.04) == VERDICT_SIGNAL_REAL_BUT_NOISY
    # the other quadrants are untouched
    assert _three_way_decision(0.004, 0.05, -0.001, 0.06) == VERDICT_BETA_NOT_ALPHA
    assert _three_way_decision(-0.001, 0.05, -0.002, 0.04) == VERDICT_WEAK_OR_DEAD
    assert _three_way_decision(-0.001, 0.05, -0.002, 0.06) == VERDICT_INCONCLUSIVE
    assert _three_way_decision(None, 0.05, 0.003, 0.04) == VERDICT_INCONCLUSIVE
