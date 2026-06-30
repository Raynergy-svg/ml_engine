"""C4: trade-outcome labels agree across all three TUI surfaces.

Operator decision (2026-06-24): unify to the corpus/majority convention — a
sub-$1 non-win is 'loss', not 'breakeven'; a breakeven/sub-$1 outcome never
counts as a win. The three implementations are:
  * agent_vote_index._outcome_label   (SQLite audit index)
  * trade_search_index._outcome_label (FTS search corpus)
  * trade_search_modal._row_cells     (displayed trades table row)

No mocks: pure functions over plain dict entries.
"""
from __future__ import annotations

from src.tui.agent_vote_index import _outcome_label as audit_label
from src.tui.trade_search_index import _outcome_label as corpus_label
from src.tui.screens.trade_search_modal import _row_cells


def _entry(trade_won, pnl):
    return {
        "trade_id": "t1", "pair": "EUR_USD", "direction": "LONG",
        "timestamp": "2026-06-24T00:00:00Z",
        "outcome": {"trade_won": trade_won, "realized_pl": pnl},
    }


CASES = [
    # (trade_won, pnl, expected_label)
    (False, 0.5, "loss"),    # sub-$1 non-win — was 'breakeven' in the audit index
    (False, -50.0, "loss"),
    (True, 5.0, "win"),
    (True, 0.5, "win"),      # a win stays a win regardless of magnitude
]


def test_audit_and_corpus_labels_agree():
    for trade_won, pnl, expected in CASES:
        e = _entry(trade_won, pnl)
        assert audit_label(e) == expected, (trade_won, pnl)
        assert corpus_label(e) == expected, (trade_won, pnl)


def test_open_label_agrees():
    e = {"trade_id": "t2"}  # no outcome
    assert audit_label(e) == "open"
    assert corpus_label(e) == "open"


def test_modal_row_matches_label_for_sub_dollar_non_win():
    # The displayed trades-table row must read 'loss' for a sub-$1 non-win,
    # consistent with the audit index + corpus.
    cells = _row_cells(_entry(False, 0.5))
    assert "loss" in cells[4]
    assert "breakeven" not in cells[4]


def test_no_surface_emits_breakeven():
    e = _entry(False, 0.5)
    assert "breakeven" not in (audit_label(e), corpus_label(e))
    assert "breakeven" not in _row_cells(e)[4]
