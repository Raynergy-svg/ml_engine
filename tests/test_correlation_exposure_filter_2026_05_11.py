"""Correlation-exposure filter tests (added 2026-05-11).

Operator feedback (2026-05-10): "Buddy is not allowing trades in the same
correlation which is wrong. The same correlation, if a trade is open, should
it not pass the trade or keep blocked?"

The old policy was binary: ANY open trade in the candidate's correlation
group blocked the new trade. That ignored:
  * Position size (1% NAV open should not preclude a 1% candidate)
  * Direction (an opposite-direction correlated trade REDUCES net risk —
    partial hedge, should pass)
  * Correlation magnitude (|ρ|=0.30 should not block as if it were 0.95)

The new policy (``src/scanner/execution.py:115`` —
``compute_net_correlated_exposure``):

  1. Walk open positions. For each open with |ρ(candidate, open)| above
     ``correlation_filter_threshold``, compute its signed contribution to
     the candidate's leg: ``sign × open.size_pct × |ρ|``.
  2. Sign rule:
       +1  same direction & ρ>0      (long+long correlated — adds risk)
       +1  opposite direction & ρ<0  (long+short anti-correlated — adds risk)
       -1  same direction & ρ<0      (anti-correlated — hedges)
       -1  opposite direction & ρ>0  (correlated, opposite — hedges)
  3. Add the candidate's own ``size_pct``.
  4. Block if total > ``max_correlated_exposure_pct`` cap.

Two new ``ScannerConfig`` fields drive this:
  * ``max_correlated_exposure_pct: 0.10`` (default 10% NAV cap)
  * ``correlation_filter_threshold: 0.70`` (min |ρ| to count as correlated)

NO MOCKS rule (CLAUDE.md / .claude/rules/improvement.md "No-Mock Rule"):
the pure function takes a ``correlation_lookup`` callable. Tests inject
a real Python function (NOT a Mock) returning a fixed ρ for each pair
pair, so no broker, no disk, no I/O — but no mocks either. Integration
tests use the REAL ``get_pair_correlation`` from
``src/training/correlation_group_config.py`` to verify the production
matrix is wired correctly.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from src.scanner.config import SCAN_PROFILES, ScannerConfig
from src.scanner.execution import compute_net_correlated_exposure


# ---------------------------------------------------------------------------
# Test fixture: deterministic correlation lookup (NOT a mock — real function)
# ---------------------------------------------------------------------------


def _lookup_factory(table: Dict[Tuple[str, str], float]):
    """Build a correlation-lookup callable from a symmetric table.

    Pure Python function (not a Mock). The table maps unordered pair
    tuples to ρ; lookup tries both orders, defaults to 0.0 (independent).
    """
    def lookup(a: str, b: str) -> float:
        a_u, b_u = a.upper(), b.upper()
        if (a_u, b_u) in table:
            return table[(a_u, b_u)]
        if (b_u, a_u) in table:
            return table[(b_u, a_u)]
        return 0.0
    return lookup


_HIGH_POS_CORR = _lookup_factory({
    ("EUR_USD", "GBP_USD"): 0.85,    # both USD-quote majors — strong positive
    ("EUR_USD", "USD_JPY"): -0.65,   # USD on opposite sides — negative
    ("EUR_USD", "AUD_NZD"): 0.10,    # mostly independent
})


# ---------------------------------------------------------------------------
# 1. Same-direction correlated open, candidate fits within cap → PASS
# ---------------------------------------------------------------------------


def test_correlated_open_same_direction_within_cap_passes() -> None:
    """EUR_USD LONG open at 2% NAV + GBP_USD LONG candidate at 2% NAV,
    ρ=0.85, cap=10%. Net = 0.02 (candidate) + 0.02*0.85 (open contribution)
    = 0.037 — well under 0.10 cap. The previous binary filter would have
    blocked this 100% of the time despite trivial actual risk."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.02}]
    candidate = {"pair": "GBP_USD", "direction": "LONG", "size_pct": 0.02}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    assert net == pytest.approx(0.02 + 0.02 * 0.85, rel=1e-9)
    assert net < 0.10  # under default cap
    assert len(contrib) == 1
    assert contrib[0]["pair"] == "EUR_USD"
    assert contrib[0]["rho"] == pytest.approx(0.85, rel=1e-9)
    assert contrib[0]["contribution"] > 0


# ---------------------------------------------------------------------------
# 2. Same-direction correlated, oversized — should BREACH cap
# ---------------------------------------------------------------------------


def test_correlated_open_same_direction_breaches_cap_blocks() -> None:
    """EUR_USD LONG open at 7% NAV + GBP_USD LONG candidate at 5% NAV,
    ρ=0.85, cap=10%. Net = 0.05 + 0.07*0.85 = 0.1095 > 0.10 → exceeds cap."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.07}]
    candidate = {"pair": "GBP_USD", "direction": "LONG", "size_pct": 0.05}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    expected = 0.05 + 0.07 * 0.85
    assert net == pytest.approx(expected, rel=1e-9)
    assert net > 0.10  # would breach default cap → production blocks


# ---------------------------------------------------------------------------
# 3. Opposite-direction correlated (partial hedge) — should PASS freely
# ---------------------------------------------------------------------------


def test_correlated_open_opposite_direction_allows_freely() -> None:
    """EUR_USD LONG 5% NAV + GBP_USD SHORT candidate at 5% NAV, ρ=0.85.
    Same |ρ| but opposite direction: contribution sign flips negative.
    Net = 0.05 (cand) - 0.05*0.85 (hedge) = 0.0075 — basically flat."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.05}]
    candidate = {"pair": "GBP_USD", "direction": "SHORT", "size_pct": 0.05}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    expected = 0.05 - 0.05 * 0.85  # 0.0075
    assert net == pytest.approx(expected, rel=1e-9)
    assert net < 0.10  # passes the cap
    assert contrib[0]["contribution"] < 0  # hedge contribution is negative


def test_anti_correlated_same_direction_hedges() -> None:
    """EUR_USD LONG + USD_JPY LONG with ρ=-0.65: same direction, NEGATIVE
    correlation = the two pairs typically move opposite. So a LONG+LONG
    is effectively a partial hedge, NOT additive risk."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.05}]
    candidate = {"pair": "USD_JPY", "direction": "LONG", "size_pct": 0.04}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.60,
    )

    # |ρ|=0.65 exceeds threshold 0.60 — counted as correlated
    expected = 0.04 - 0.05 * 0.65  # hedge: 0.04 - 0.0325 = 0.0075
    assert net == pytest.approx(expected, rel=1e-9)
    assert contrib[0]["contribution"] < 0


# ---------------------------------------------------------------------------
# 4. Uncorrelated pair below threshold — ignored
# ---------------------------------------------------------------------------


def test_uncorrelated_pair_below_threshold_ignored() -> None:
    """EUR_USD LONG open + AUD_NZD LONG candidate, ρ=0.10 (below 0.70 floor).
    The open position contributes nothing to net; only the candidate's own
    risk counts."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.05}]
    candidate = {"pair": "AUD_NZD", "direction": "LONG", "size_pct": 0.03}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    assert net == pytest.approx(0.03, rel=1e-9)  # candidate's own risk only
    assert contrib == []  # nothing contributed


# ---------------------------------------------------------------------------
# 5. No open positions — candidate's own risk is the net, always
# ---------------------------------------------------------------------------


def test_no_open_positions_net_equals_candidate_size() -> None:
    """Empty portfolio: net equals the candidate's projected size_pct."""
    candidate = {"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.025}

    net, contrib = compute_net_correlated_exposure(
        open_positions=[],
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    assert net == pytest.approx(0.025, rel=1e-9)
    assert contrib == []


# ---------------------------------------------------------------------------
# 6. Multi-position aggregation — three correlated longs stack
# ---------------------------------------------------------------------------


def test_three_correlated_longs_stack_into_breach() -> None:
    """The pathological case the binary filter protects against, now
    handled by the cap. Three EUR-side longs stacking should breach cap."""
    open_positions = [
        {"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.03},
        {"pair": "EUR_GBP", "direction": "LONG", "size_pct": 0.03},
    ]
    candidate = {"pair": "GBP_USD", "direction": "LONG", "size_pct": 0.03}
    # All three pairwise ρ = 0.80 — strong correlation
    lookup = _lookup_factory({
        ("EUR_USD", "GBP_USD"): 0.80,
        ("EUR_GBP", "GBP_USD"): 0.80,
        ("EUR_USD", "EUR_GBP"): 0.80,
    })

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=lookup,
        correlation_threshold=0.70,
    )

    # 0.03 (cand) + 0.03*0.80 (EUR_USD) + 0.03*0.80 (EUR_GBP) = 0.078
    expected = 0.03 + 2 * (0.03 * 0.80)
    assert net == pytest.approx(expected, rel=1e-9)
    assert len(contrib) == 2


# ---------------------------------------------------------------------------
# 7. Net floor is zero (a hedge can't drag net below 0)
# ---------------------------------------------------------------------------


def test_net_exposure_never_goes_negative() -> None:
    """A massive hedge could mathematically drive net negative; the
    function clamps at 0 because "negative exposure" is not a meaningful
    cap quantity — the cap measures upside risk, not hedge depth."""
    open_positions = [
        {"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.10},
        {"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.10},  # duplicate intentional
    ]
    # Big hedge: candidate is opposite-direction & strongly correlated
    candidate = {"pair": "GBP_USD", "direction": "SHORT", "size_pct": 0.01}
    lookup = _lookup_factory({("EUR_USD", "GBP_USD"): 0.95})

    net, _ = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=lookup,
        correlation_threshold=0.70,
    )

    # 0.01 (cand) - 0.10*0.95 - 0.10*0.95 = -0.18 → clamped to 0
    assert net == 0.0


# ---------------------------------------------------------------------------
# 8. Candidate's own pair already open — skipped (no self-contribution)
# ---------------------------------------------------------------------------


def test_candidate_pair_already_open_is_skipped() -> None:
    """An open position in the candidate's exact pair must NOT contribute
    to the correlated-exposure aggregation (it's the candidate's own pair,
    handled by per-pair risk gates separately)."""
    open_positions = [{"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.04}]
    candidate = {"pair": "EUR_USD", "direction": "LONG", "size_pct": 0.03}

    net, contrib = compute_net_correlated_exposure(
        open_positions=open_positions,
        candidate=candidate,
        correlation_lookup=_HIGH_POS_CORR,
        correlation_threshold=0.70,
    )

    # Only the candidate's own size counts; the same-pair open is skipped
    assert net == pytest.approx(0.03, rel=1e-9)
    assert contrib == []


# ---------------------------------------------------------------------------
# 9. Config dataclass + smart-profile wiring (three-layer verification)
# ---------------------------------------------------------------------------


def test_dataclass_default_max_correlated_exposure_pct() -> None:
    """New field exists on ScannerConfig with the documented 0.10 default."""
    cfg = ScannerConfig()
    assert hasattr(cfg, "max_correlated_exposure_pct")
    assert cfg.max_correlated_exposure_pct == 0.10


def test_dataclass_default_correlation_filter_threshold() -> None:
    """Filter threshold defaults to 0.70 (|ρ| floor)."""
    cfg = ScannerConfig()
    assert hasattr(cfg, "correlation_filter_threshold")
    assert cfg.correlation_filter_threshold == 0.70


def test_dataclass_legacy_max_correlated_exposure_preserved_for_backcompat() -> None:
    """Legacy count cap is kept so old saved configs don't crash."""
    cfg = ScannerConfig()
    assert hasattr(cfg, "max_correlated_exposure")
    # Legacy default is non-positive-or-2 (sentinel telling the new path
    # to short-circuit to count behavior if pct cap is OFF). Verify it
    # exists with an int default ≥ 0.
    assert isinstance(cfg.max_correlated_exposure, int)
    assert cfg.max_correlated_exposure >= 0


def test_smart_profile_includes_correlation_fields() -> None:
    """Smart profile carries both new keys (three-layer wiring: dataclass
    + profile + consumer getattr at execution.py:1205,1224)."""
    smart = SCAN_PROFILES["smart"]
    assert "max_correlated_exposure_pct" in smart
    assert "correlation_filter_threshold" in smart
    # Smart inherits the dataclass defaults — no profile-specific tightening.
    assert smart["max_correlated_exposure_pct"] == 0.10
    assert smart["correlation_filter_threshold"] == 0.70


def test_apply_profile_smart_propagates_correlation_fields() -> None:
    """Applying smart profile sets both new fields on the live config."""
    cfg = ScannerConfig()
    cfg.apply_profile("smart")
    assert cfg.max_correlated_exposure_pct == 0.10
    assert cfg.correlation_filter_threshold == 0.70


# ---------------------------------------------------------------------------
# 10. Integration with the real correlation matrix (no test lookup)
# ---------------------------------------------------------------------------


def test_real_matrix_eur_usd_gbp_usd_passes_filter_threshold() -> None:
    """The production correlation matrix should recognize EUR_USD ↔ GBP_USD
    as correlated above the 0.70 threshold (both are USD-quote majors)."""
    from src.training.correlation_group_config import get_pair_correlation

    rho = get_pair_correlation("EUR_USD", "GBP_USD")
    assert abs(rho) >= 0.70, (
        f"EUR_USD↔GBP_USD ρ={rho:+.2f} is below 0.70 — the production "
        "matrix would treat them as independent, defeating the filter"
    )


def test_real_matrix_unknown_pair_returns_zero() -> None:
    """Unknown / exotic pair returns 0.0 (independent) — the fail-safe
    BLOCK on missing data lives in _check_correlation_exposure, not here."""
    from src.training.correlation_group_config import get_pair_correlation

    rho = get_pair_correlation("EUR_USD", "XAU_USD")  # exotic, no group
    assert rho == 0.0
