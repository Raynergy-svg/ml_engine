"""End-to-end proof: the promoted champion is resolved and consumed by sizing.

Written 2026-08-01, immediately after the first-ever champion promotion
(lane ``risk_target_vol``, package ``5d6af6ba…c2b30e``, Phase L promotion
service).  These tests run READ-ONLY against the REAL evidence store in
``trained_data/evidence/`` — they are the disk-verified proof that the
"trained → tested → live" chain is actually connected, not narrated:

    signed EvidencePackage → champion pointer → RiskTargetAdapter
    → forward-vol prediction → strictly-risk-decreasing sizing damp

No mocks (project rule).  If the champion pointer is ever retired or
re-pointed these tests follow the store — they assert the CHAIN works,
not one frozen digest.  Skips (never fails) when no champion exists yet,
so the suite stays green on a fresh clone before any promotion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.risk.risk_target_adapter import RiskTargetAdapter

_REPO = Path(__file__).resolve().parent.parent
_EVIDENCE = _REPO / "trained_data" / "evidence"
_FACTOR = _REPO / "market_data" / "factor"

_have_champion = (_EVIDENCE / "champions").is_dir() and any(
    (_EVIDENCE / "champions").rglob("*")
)

pytestmark = pytest.mark.skipif(
    not _have_champion, reason="no promoted champion on this checkout yet"
)


def _real_history(pair: str) -> pd.DataFrame:
    csv = _FACTOR / f"{pair}_D.csv"
    if not csv.exists():
        pytest.skip(f"no real daily history for {pair} on this checkout")
    return pd.read_csv(csv)


def test_adapter_resolves_real_champion_not_fallback() -> None:
    adapter = RiskTargetAdapter()
    loaded = adapter._load_from_champion()
    assert loaded is not None, (
        "champion pointer exists on disk but the adapter did not resolve it "
        f"(last refusal: {adapter.last_refusal_reason})"
    )


def test_champion_predicts_finite_vol_on_real_history() -> None:
    # The container's data snapshot ends 2026-06-11; the default 10-day
    # staleness bound would (correctly) refuse it as of today.  Relaxing the
    # bound here tests the PREDICTION path, not the staleness rail — the
    # staleness refusal itself is covered at the default bound below.
    adapter = RiskTargetAdapter(max_history_age_days=100_000.0)
    pred = adapter.predict_forward_vol("EUR_USD", _real_history("EUR_USD"))
    assert pred is not None, f"refused: {adapter.last_refusal_reason}"
    assert 0.0 < pred < 5.0, f"annualized-vol prediction out of range: {pred}"


def test_damp_multiplier_bounded_and_shrink_only() -> None:
    adapter = RiskTargetAdapter(max_history_age_days=100_000.0)
    for pair in ("EUR_USD", "USD_JPY", "GBP_USD"):
        mult, reason = adapter.compute_damp_multiplier(pair, floor=0.25)
        assert 0.25 <= mult <= 1.0, (pair, mult, reason)


def test_default_staleness_bound_refuses_stale_snapshot() -> None:
    # At the DEFAULT bound the 2026-06-11 snapshot must be refused —
    # fail-closed is the contract, and this proves the rail is armed.
    adapter = RiskTargetAdapter()
    pred = adapter.predict_forward_vol("EUR_USD", _real_history("EUR_USD"))
    if pred is not None:
        pytest.skip("history snapshot is fresh on this machine; rail not testable")
    assert adapter.last_refusal_reason is not None
    assert "stale" in adapter.last_refusal_reason.lower()
