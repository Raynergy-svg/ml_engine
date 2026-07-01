"""Tests for the PIT primary-document text loader.

Two layers, per the no-mock policy:

* **Offline / pure** — selection logic + HTML stripping are tested on synthetic
  data with NO network and NO mocks. These always run.
* **Integration** — real EDGAR fetches against a known issuer (Apple,
  CIK 0000320193). Marked ``@pytest.mark.integration`` and SKIP cleanly (never
  hang) when EDGAR is unreachable / non-200.

No ``unittest.mock`` anywhere — the live network IS the dependency under test.
"""
from __future__ import annotations

import requests
import pytest

from src.equity.research.contracts import FilingText
from src.equity.research.pit_text_loader import (
    DEFAULT_FORMS,
    html_to_text,
    load_pit_filing,
    load_pit_filings,
    select_pit_filing,
)

# --------------------------------------------------------------------------- #
# Offline: pure selection logic (PIT gate)
# --------------------------------------------------------------------------- #
_SYNTH_ROWS = [
    {"form": "10-K", "filingDate": "2023-11-03", "accessionNumber": "a-1",
     "primaryDocument": "k2023.htm"},
    {"form": "10-Q", "filingDate": "2024-05-03", "accessionNumber": "a-2",
     "primaryDocument": "q2024.htm"},
    {"form": "10-Q", "filingDate": "2024-08-02", "accessionNumber": "a-3",
     "primaryDocument": "q2024b.htm"},   # AFTER a 2024-06-01 as_of
    {"form": "8-K", "filingDate": "2024-02-01", "accessionNumber": "a-4",
     "primaryDocument": "8k.htm"},
]


def test_select_picks_most_recent_on_or_before_as_of():
    got = select_pit_filing(_SYNTH_ROWS, "2024-06-01", DEFAULT_FORMS)
    assert got is not None
    assert got["accessionNumber"] == "a-2"        # 2024-05-03, latest <= as_of
    assert got["filingDate"] <= "2024-06-01"


def test_select_never_returns_lookahead():
    # as_of strictly before the only post-cutoff 10-Q -> that row excluded.
    got = select_pit_filing(_SYNTH_ROWS, "2024-06-01", DEFAULT_FORMS)
    assert got["filingDate"] != "2024-08-02"


def test_select_boundary_filed_equals_as_of_is_allowed():
    got = select_pit_filing(_SYNTH_ROWS, "2024-05-03", DEFAULT_FORMS)
    assert got is not None and got["filingDate"] == "2024-05-03"


def test_select_respects_form_filter():
    got = select_pit_filing(_SYNTH_ROWS, "2024-06-01", ("8-K",))
    assert got is not None and got["form"] == "8-K"
    assert got["accessionNumber"] == "a-4"


def test_select_none_when_nothing_qualifies():
    assert select_pit_filing(_SYNTH_ROWS, "2020-01-01", DEFAULT_FORMS) is None


def test_select_skips_rows_without_primary_document():
    rows = [{"form": "10-K", "filingDate": "2024-01-01",
             "accessionNumber": "x", "primaryDocument": ""}]
    assert select_pit_filing(rows, "2024-06-01", DEFAULT_FORMS) is None


# --------------------------------------------------------------------------- #
# Offline: HTML stripping
# --------------------------------------------------------------------------- #
def test_html_to_text_strips_tags_and_collapses():
    html = (
        "<html><head><title>x</title><style>p{}</style></head>"
        "<body><p>Hello&nbsp;<b>World</b></p>"
        "<script>var a=1;</script><div>Second   line</div></body></html>"
    )
    out = html_to_text(html)
    assert "Hello World" in out
    assert "Second line" in out
    assert "var a=1" not in out      # script dropped
    assert "{" not in out            # style dropped
    assert "<" not in out and ">" not in out


def test_html_to_text_empty():
    assert html_to_text("") == ""
    assert html_to_text(None) == ""  # type: ignore[arg-type]


def test_html_to_text_plain_passthrough():
    assert html_to_text("just text") == "just text"


# --------------------------------------------------------------------------- #
# Offline: FilingText PIT invariant is enforced by the contract
# --------------------------------------------------------------------------- #
def test_filingtext_rejects_lookahead():
    with pytest.raises(ValueError):
        FilingText(
            ticker="AAPL", cik="0000320193", as_of="2024-06-01",
            form="10-Q", filed="2024-08-02", text="lookahead",
        )


# --------------------------------------------------------------------------- #
# Integration: real EDGAR (skips cleanly if unreachable)
# --------------------------------------------------------------------------- #
_AAPL_CIK = "0000320193"


def _edgar_reachable() -> bool:
    try:
        r = requests.get(
            "https://data.sec.gov/submissions/CIK0000320193.json",
            headers={"User-Agent": "ml_engine-research dcertan84@gmail.com"},
            timeout=(5, 30),
        )
        return r.status_code == 200
    except (requests.Timeout, requests.ConnectionError, requests.RequestException):
        return False


@pytest.mark.integration
def test_load_pit_filing_apple_returns_pit_text():
    if not _edgar_reachable():
        pytest.skip("EDGAR unreachable / non-200 — skipping network test")
    as_of = "2024-06-01"
    ft = load_pit_filing("AAPL", _AAPL_CIK, as_of)
    assert ft is not None, "expected an Apple filing on/before 2024-06-01"
    assert isinstance(ft, FilingText)
    assert ft.form in ("10-K", "10-Q", "8-K")
    assert ft.filed <= as_of                  # PIT invariant
    assert ft.as_of == as_of
    assert ft.cik == _AAPL_CIK
    assert ft.text and len(ft.text) > 1000     # real, non-empty document text


@pytest.mark.integration
def test_load_pit_filing_form_filter_10k_or_10q():
    if not _edgar_reachable():
        pytest.skip("EDGAR unreachable / non-200 — skipping network test")
    as_of = "2024-06-01"
    ft = load_pit_filing("AAPL", _AAPL_CIK, as_of, forms=("10-K", "10-Q"))
    assert ft is not None
    assert ft.form in ("10-K", "10-Q")
    assert ft.filed <= as_of


@pytest.mark.integration
def test_load_pit_filing_cache_roundtrip(tmp_path):
    if not _edgar_reachable():
        pytest.skip("EDGAR unreachable / non-200 — skipping network test")
    as_of = "2024-06-01"
    first = load_pit_filing("AAPL", _AAPL_CIK, as_of, cache_dir=tmp_path)
    assert first is not None
    # A cache file should now exist somewhere under tmp_path.
    assert any(tmp_path.rglob("*.txt"))
    second = load_pit_filing("AAPL", _AAPL_CIK, as_of, cache_dir=tmp_path)
    assert second is not None
    assert second.text == first.text          # served from cache, identical


@pytest.mark.integration
def test_load_pit_filing_none_before_first_filing():
    if not _edgar_reachable():
        pytest.skip("EDGAR unreachable / non-200 — skipping network test")
    # Apple's first EDGAR filings are early-1990s; well before this, none exist.
    ft = load_pit_filing("AAPL", _AAPL_CIK, "1990-01-01")
    assert ft is None


@pytest.mark.integration
def test_load_pit_filings_batch():
    if not _edgar_reachable():
        pytest.skip("EDGAR unreachable / non-200 — skipping network test")
    items = [("AAPL", _AAPL_CIK, "2024-06-01"), ("AAPL", _AAPL_CIK, "1990-01-01")]
    out = load_pit_filings(items)
    assert out[("AAPL", _AAPL_CIK, "2024-06-01")] is not None
    assert out[("AAPL", _AAPL_CIK, "1990-01-01")] is None
