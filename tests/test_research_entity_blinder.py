"""No-mock golden tests for the entity-blinder (Track B primary lookahead control).

Real ``FilingText`` + real text snippets; no mocks, no network, no LLM. These
tests are the verifier's first line of defense: they assert that the issuer's
identifying signal does NOT survive blinding, that future-dated text is scrubbed
while point-in-time-legitimate past dates are kept, that the audit counts are
correct, that the residual-risk scan actually flags an un-listed proper noun
(so leakage is VISIBLE, not silently swallowed), and that blinding is
deterministic to the byte.
"""

from __future__ import annotations

from src.equity.research.contracts import BlindedText, FilingText
from src.equity.research.entity_blinder import IssuerBundle, blind_filing


def _apple_bundle() -> IssuerBundle:
    return IssuerBundle(
        company_names=["Apple Inc.", "Apple, Inc.", "Apple"],
        tickers=["AAPL"],
        cik="0000320193",
        exchanges=["NASDAQ"],
        person_names=["Tim Cook", "Luca Maestri"],
        product_names=["iPhone", "iPad", "Mac"],
        hq_locations=["Cupertino, California", "Cupertino"],
    )


def _apple_filing(text: str, as_of: str = "2021-10-29") -> FilingText:
    return FilingText(
        ticker="AAPL",
        cik="0000320193",
        as_of=as_of,
        form="10-K",
        filed="2021-10-29",
        text=text,
    )


def test_apple_identifiers_do_not_survive_blinding():
    text = (
        "Apple Inc. (NASDAQ: AAPL), led by CEO Tim Cook and CFO Luca Maestri, "
        "designs the iPhone, iPad and Mac in Cupertino, California. "
        "Apple's revenue grew. Apple, Inc. remains profitable. CIK 0000320193."
    )
    filing = _apple_filing(text)
    blinded = blind_filing(filing, _apple_bundle())

    assert isinstance(blinded, BlindedText)
    # Contract: no ticker / cik leak back onto the blinded object.
    assert not hasattr(blinded, "ticker") or getattr(blinded, "ticker", None) is None
    assert not hasattr(blinded, "cik") or getattr(blinded, "cik", None) is None

    low = blinded.text.lower()
    for forbidden in (
        "apple", "aapl", "tim cook", "luca maestri", "iphone", "ipad",
        "cupertino", "california", "nasdaq", "0000320193", "320193",
    ):
        assert forbidden not in low, f"{forbidden!r} survived blinding: {blinded.text!r}"

    # "Mac" is a deny-listed product but a substring of nothing here; confirm it
    # is gone as a standalone token via placeholder presence.
    assert "PRODUCT_" in blinded.text
    assert "COMPANY_" in blinded.text
    assert "PERSON_" in blinded.text
    assert "PLACE_" in blinded.text
    assert "TICKER_" in blinded.text


def test_company_maps_to_stable_consistent_placeholder():
    text = "Apple Inc. did well. Later, Apple grew. Apple, Inc. shipped product."
    blinded = blind_filing(_apple_filing(text), _apple_bundle())
    # Every surface form of the company maps to the SAME placeholder.
    assert blinded.text.count("COMPANY_A") == 3
    assert "Apple" not in blinded.text


def test_possessive_form_is_redacted():
    text = "Apple's strategy is sound. Tim Cook's leadership matters."
    blinded = blind_filing(_apple_filing(text), _apple_bundle())
    low = blinded.text.lower()
    assert "apple" not in low
    assert "tim cook" not in low
    # The possessive marker should be consumed with the name.
    assert "'s strategy" not in blinded.text.lower()


def test_company_aliases_collapse_but_distinct_persons_split():
    # A bundle describes ONE issuer: its multiple company_names are ALIASES of
    # that issuer (legal name + d/b/a) and must collapse to a single COMPANY_A.
    # Persons are genuinely distinct entities and get distinct placeholders.
    bundle = IssuerBundle(
        company_names=["Acme Corporation", "Acme Corp", "Acme"],
        person_names=["Jane Doe", "John Roe"],
    )
    filing = FilingText(
        ticker="ACME", cik="111", as_of="2020-01-01", form="8-K",
        filed="2019-12-31",
        text="Acme Corporation (Acme) employs Jane Doe and John Roe. Acme Corp grew.",
    )
    blinded = blind_filing(filing, bundle)
    # All company aliases -> one placeholder; no COMPANY_B.
    assert "COMPANY_A" in blinded.text
    assert "COMPANY_B" not in blinded.text
    # Two distinct people -> two placeholders.
    assert "PERSON_1" in blinded.text and "PERSON_2" in blinded.text
    assert "Acme" not in blinded.text
    assert "Jane Doe" not in blinded.text and "John Roe" not in blinded.text


def test_unlisted_counterparty_company_surfaces_as_leak():
    # A second company that is NOT the issuer (a counterparty) is correctly NOT
    # in the issuer bundle, so it is not redacted — it surfaces honestly in the
    # residual-risk scan rather than being silently swallowed.
    bundle = IssuerBundle(company_names=["Acme Corporation"])
    filing = FilingText(
        ticker="ACME", cik="111", as_of="2020-01-01", form="8-K",
        filed="2019-12-31",
        text="Acme Corporation signed a deal with Globex Industries.",
    )
    blinded = blind_filing(filing, bundle)
    assert "Acme" not in blinded.text
    sample = blinded.audit["residual_risk"]["leak_candidate_sample"]
    assert any("Globex Industries" == c for c in sample), sample


def test_future_date_redacted_past_date_kept():
    # as_of = 2021-10-29. A date AFTER must be scrubbed; a date BEFORE preserved.
    text = (
        "The prior fiscal year ended September 25, 2021. "
        "Guidance was issued on November 15, 2021. "
        "An older milestone was March 3, 2019."
    )
    filing = _apple_filing(text, as_of="2021-10-29")
    blinded = blind_filing(filing, _apple_bundle())

    # Future date (after as_of) -> redacted.
    assert "November 15, 2021" not in blinded.text
    assert "[REDACTED_DATE]" in blinded.text
    # Past dates (before as_of) -> kept verbatim.
    assert "September 25, 2021" in blinded.text
    assert "March 3, 2019" in blinded.text


def test_iso_and_slash_future_dates_redacted():
    text = (
        "Signed 2019-05-01 (kept). Effective 2025-01-15 (future). "
        "Also 6/30/2026 (future) and 1/2/2018 (kept)."
    )
    filing = _apple_filing(text, as_of="2021-10-29")
    blinded = blind_filing(filing, _apple_bundle())
    assert "2019-05-01" in blinded.text
    assert "1/2/2018" in blinded.text
    assert "2025-01-15" not in blinded.text
    assert "6/30/2026" not in blinded.text


def test_bare_future_year_redacted_past_year_kept():
    text = "In 2018 revenue rose. By 2024 the outlook improved. The 2020 dip passed."
    filing = _apple_filing(text, as_of="2021-10-29")
    blinded = blind_filing(filing, _apple_bundle())
    # as_of year is 2021; 2024 is future -> redacted; 2018 and 2020 kept.
    assert "2018" in blinded.text
    assert "2020" in blinded.text
    assert "2024" not in blinded.text


def test_audit_counts_are_correct():
    text = "Apple Inc. and Apple grew. Tim Cook spoke. The iPhone sold."
    blinded = blind_filing(_apple_filing(text), _apple_bundle())
    audit = blinded.audit
    counts = audit["counts"]
    # Two company surface forms ("Apple Inc.", "Apple") -> 2 company redactions.
    assert counts["company"] == 2
    assert counts["person"] == 1
    assert counts["product"] == 1
    assert audit["total_redactions"] == sum(counts.values())
    assert audit["pipeline_step"] == "entity_blinder"
    # Legend maps placeholders to categories, never to real names.
    legend = audit["placeholder_legend"]
    assert legend["COMPANY_A"] == "company"
    assert "Apple" not in str(legend)


def test_residual_scan_flags_unlisted_proper_noun():
    # "Foobar Dynamics" is NOT in the deny-list, so it must survive and be
    # flagged by the residual-risk scan — this is HOW the verifier sees leakage.
    text = (
        "Apple Inc. acquired a stake in Foobar Dynamics, a private supplier. "
        "Tim Cook praised the deal."
    )
    blinded = blind_filing(_apple_filing(text), _apple_bundle())
    residual = blinded.audit["residual_risk"]
    sample = residual["leak_candidate_sample"]
    assert residual["leak_candidate_count"] >= 1
    assert any("Foobar Dynamics" == c for c in sample), sample
    # Placeholders must NOT be reported as leaks.
    assert not any("COMPANY_A" in c for c in sample)
    assert not any("PERSON_1" in c for c in sample)


def test_residual_scan_clean_when_everything_redacted():
    text = "Apple Inc. grew. Tim Cook led. The iPhone sold in Cupertino, California."
    blinded = blind_filing(_apple_filing(text), _apple_bundle())
    residual = blinded.audit["residual_risk"]
    # No un-listed proper nouns -> no leak candidates.
    assert residual["leak_candidate_count"] == 0, residual["leak_candidate_sample"]


def test_determinism_byte_identical():
    text = (
        "Apple Inc. (NASDAQ: AAPL), led by Tim Cook, makes the iPhone in "
        "Cupertino, California. Guidance issued November 15, 2025. CIK 320193. "
        "Foobar Dynamics is a supplier. In 2024 things improved; 2018 was flat."
    )
    bundle = _apple_bundle()
    first = blind_filing(_apple_filing(text), bundle)
    second = blind_filing(_apple_filing(text), bundle)
    assert first.text == second.text
    assert first.audit == second.audit


def test_filing_ticker_cik_redacted_even_if_bundle_omits_them():
    # Defensive control: caller forgot to list the ticker/cik in the bundle, but
    # the filing's own identifiers must never survive.
    bundle = IssuerBundle(company_names=["Apple"])
    text = "Apple trades as AAPL. CIK 0000320193 is on file."
    blinded = blind_filing(_apple_filing(text), bundle)
    low = blinded.text.lower()
    assert "aapl" not in low
    assert "0000320193" not in low
    assert "320193" not in low


def test_blindedtext_carries_no_ticker_or_cik_fields():
    blinded = blind_filing(_apple_filing("Apple Inc. did well."), _apple_bundle())
    field_names = set(blinded.__dataclass_fields__.keys())
    assert "ticker" not in field_names
    assert "cik" not in field_names
    assert field_names == {"as_of", "form", "filed", "text", "audit"}
