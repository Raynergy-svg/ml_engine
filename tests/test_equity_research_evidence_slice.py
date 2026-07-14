"""Phase J2 acceptance tests — equity research end-to-end evidence slice.

These are the roadmap §12 acceptance criteria made real for the equity-research
lane. NO MOCKS: every test runs the actual evidence store, real Ed25519 signing,
the real disposition chain, and the REAL analysis-only rank-IC scorecard
(``src.equity.research.rank_ic_scorecard``) over real fold-shaped rows — never a
test-double.

The unit of disposition is one *universe tier* (curated / wide / full_history);
each is produced by a DETERMINISTIC AGGREGATOR pooling per-fold rank-IC
scorecards. The roadmap J2 requirement that the attractive curated result and the
defensible wide result be reported separately is exercised directly: the curated
tier QUARANTINEs (its out-of-sample rank-IC persists) while the wide tier REJECTs
(its rank-IC fades to insignificance as the universe widens) — fully
independently, both preserved as first-class honest results.

The §12 checklist and the test that covers it:

    - Remote worker cannot read .claude/state.json ........ see
      test_equity_research_evidence_worker_no_authority.py (AST + source + closure)
    - Remote worker has no broker credentials ............. idem + capability test there
    - Candidate artifact cannot overwrite incumbent ....... test_candidate_never_overwrites_incumbent
    - Missing fold causes failure ........................ test_missing_fold_causes_aggregator_failure
                                                            + test_dropped_fold_partition_causes_hash_failure
    - Duplicate fold causes failure ...................... test_duplicate_fold_declaration_refused
    - Changed dataset partition causes hash failure ...... test_changed_fold_partition_causes_hash_failure
    - Changed artifact bytes cause signature failure ..... test_changed_artifact_bytes_are_rejected
    - Worse head is rejected independently ............... test_persists_and_fade_tiers_dispositioned_independently
    - Good head does not rescue a failed head ............ test_good_tier_does_not_rescue_failed
    - Local replay reproduces metrics within tolerance ... test_local_replay_reproduces
    - External registry outage does not stop verification  test_registry_outage_does_not_stop_verification
    - Missing declared tier head causes failure .......... test_missing_tier_head_causes_failure

Plus research-domain honesty criteria:
    - A faded OOS edge is a REJECTED honest null .......... test_fade_tier_rejected_as_honest_null
    - A no-signal tier is REJECTED ....................... test_no_signal_tier_rejected
    - Too few OOS folds are refused, never force-verdicted  test_insufficient_oos_folds_rejected
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from src.evidence.contracts import (
    AuthorityRole,
    DispositionEvent,
    DispositionState,
    GateStatus,
    ImportDecision,
)
from src.evidence.equity_research.evaluation import (
    EvaluationParams,
    evaluate_partitions,
)
from src.evidence.equity_research.local_import import import_head
from src.evidence.equity_research.manifests import (
    build_capability_profile,
    build_research_strategy_manifest,
    fold_partition_id,
)
from src.evidence.equity_research.models import lane_id_for_tier
from src.evidence.equity_research.slice import (
    build_evidence_store,
    build_slice_identities,
    equity_research_evidence_view,
    produce_worker_output,
    run_equity_research_evidence_slice,
    utc,
)
from src.evidence.equity_research.worker import DatasetHashError, MissingHeadError, run_worker
from src.equity.research.rank_ic_scorecard import (
    VERDICT_FADES_WITH_BREADTH,
    VERDICT_NO_SIGNAL,
    VERDICT_SIGNAL_PERSISTS,
    aggregate_tier_scorecard,
    compute_fold_scorecard,
)

NOW = utc(2026, 7, 14, 12, 0)
HYP = "filing_quality"
from datetime import date  # noqa: E402

SLICE_KW = dict(
    hypothesis_id=HYP,
    dataset_id="equity-research-test",
    coverage_start=date(2015, 1, 1),
    coverage_end=date(2020, 1, 1),
    retrieved_at=NOW,
    created_at=NOW,
    git_commit="0" * 40,
)

CURATED = lane_id_for_tier(HYP, "curated")
WIDE = lane_id_for_tier(HYP, "wide")
FULL = lane_id_for_tier(HYP, "full_history")


# --------------------------------------------------------------------------- #
# Deterministic fold fixtures                                                  #
# --------------------------------------------------------------------------- #

def _bitrev(n: int) -> list[int]:
    """A genuinely decorrelating permutation of range(n) (LSB-first bit reversal).

    Unlike a rotation, this destroys the rank ordering, so blending it against the
    identity order lets us dial a fold's rank-IC to a target level."""
    if n <= 1:
        return list(range(n))
    width = max(1, (n - 1).bit_length())

    def rev(x: int) -> int:
        r = 0
        for _ in range(width):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    return sorted(range(n), key=lambda i: (rev(i), i))


def _fold_rows(tier: str, idx: int, in_sample: bool, n: int, w: float) -> list[dict]:
    """One fold's cross-section. Forward-return rank = ``w*composite_rank +
    (1-|w|)*scattered_rank``; ``w`` may be negative to anti-correlate. The
    composite is monotone in the row index, so ``w`` controls the fold's rank-IC.
    """
    scat = _bitrev(n)
    rows = []
    for k in range(n):
        fq = (k / n) - 0.5
        fwd_rank = w * k + (1.0 - abs(w)) * scat[k]
        rows.append({
            "tier": tier, "fold_id": fold_partition_id(tier, idx), "fold_index": idx,
            "in_sample": in_sample, "window_start": f"201{idx}", "window_end": f"201{idx+1}",
            "ticker": f"{tier}_{k}", "fundamental_quality": fq, "accounting_red_flags": 0.0,
            "forward_outlook": 0.0, "forward_return": round(fwd_rank * 0.001, 6),
        })
    return rows


def _jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


# Per-fold blend weights (5 folds: 2 in-sample, 3 out-of-sample) that produce the
# three intended tier verdicts. Verified empirically against the scorecard bars
# (IC_BAR=0.03, IC_TSTAT_BAR=2.0).
_IS_FLAGS = (True, True, False, False, False)
_W_PERSIST = (0.80, 0.80, 0.75, 0.79, 0.83)   # high, consistent OOS IC -> persists
_W_FADE = (0.20, 0.20, 0.09, -0.05, 0.06)     # positive but noisy OOS IC (t<2) -> fades
_W_NO_SIGNAL = (0.10, 0.10, -0.04, -0.06, -0.05)  # net-negative OOS IC -> no signal


def _tier_partitions(tier: str, n: int, ws) -> dict[str, bytes]:
    return {
        fold_partition_id(tier, i): _jsonl(_fold_rows(tier, i, _IS_FLAGS[i], n, ws[i]))
        for i in range(len(ws))
    }


def _three_tier_partitions() -> dict[str, bytes]:
    parts: dict[str, bytes] = {}
    parts.update(_tier_partitions("curated", 25, _W_PERSIST))       # QUARANTINE
    parts.update(_tier_partitions("wide", 200, _W_FADE))            # REJECT (fade)
    parts.update(_tier_partitions("full_history", 150, _W_NO_SIGNAL))  # REJECT (no signal)
    return parts


def _noop_registry(experiment, metrics):
    return None


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _run(tmp_path, partitions=None, registry=_noop_registry, params=None, evaluator=evaluate_partitions):
    identities, store = _fresh(tmp_path)
    partitions = partitions if partitions is not None else _three_tier_partitions()
    result = run_equity_research_evidence_slice(
        store, identities, partitions, evaluator=evaluator,
        registry_publish=registry, params=params, **SLICE_KW,
    )
    return identities, store, result


# --------------------------------------------------------------------------- #
# Fixture sanity — the scorecard produces the three intended verdicts           #
# --------------------------------------------------------------------------- #

def _aggregate(tier, n, ws):
    cards = [compute_fold_scorecard(fold_partition_id(tier, i), _fold_rows(tier, i, _IS_FLAGS[i], n, ws[i]))
             for i in range(len(ws))]
    return aggregate_tier_scorecard(tier, cards, [c["fold_id"] for c in cards])


def test_fixtures_produce_the_three_intended_verdicts():
    assert _aggregate("curated", 25, _W_PERSIST)["decision"]["verdict"] == VERDICT_SIGNAL_PERSISTS
    assert _aggregate("wide", 200, _W_FADE)["decision"]["verdict"] == VERDICT_FADES_WITH_BREADTH
    assert _aggregate("full_history", 150, _W_NO_SIGNAL)["decision"]["verdict"] == VERDICT_NO_SIGNAL


# --------------------------------------------------------------------------- #
# Disposition semantics                                                        #
# --------------------------------------------------------------------------- #

def test_persists_and_fade_tiers_dispositioned_independently(tmp_path):
    _, _, result = _run(tmp_path)
    curated = result.outcomes[CURATED]
    wide = result.outcomes[WIDE]
    assert curated.final_state == DispositionState.QUARANTINED
    assert curated.verdict.decision == ImportDecision.ACCEPT_FOR_QUARANTINE
    assert wide.final_state == DispositionState.REJECTED
    assert wide.verdict.decision == ImportDecision.REJECT
    assert "candidate_evaluation_gate_passed" in (wide.verdict.rejection_reason or "")
    # Independent packages — one tier's verdict is not derived from another.
    assert curated.package_digest != wide.package_digest


def test_good_tier_does_not_rescue_failed(tmp_path):
    _, store, result = _run(tmp_path)
    assert result.outcomes[CURATED].final_state == DispositionState.QUARANTINED
    assert result.outcomes[WIDE].final_state == DispositionState.REJECTED
    wide_digest = result.outcomes[WIDE].package_digest
    ledger = store.load_ledger(wide_digest)
    from src.evidence.signing import verify_envelope
    states = [verify_envelope(env, DispositionEvent, store.trust_store).to_state for env in ledger]
    assert DispositionState.QUARANTINED not in states
    assert states[-1] == DispositionState.REJECTED


def test_quarantine_chain_is_exactly_the_signed_lifecycle(tmp_path):
    from src.evidence.signing import verify_envelope
    identities, store, result = _run(tmp_path)
    curated_digest = result.outcomes[CURATED].package_digest
    ledger = store.load_ledger(curated_digest)
    events = [verify_envelope(env, DispositionEvent, identities.trust_store) for env in ledger]
    assert [e.to_state for e in events] == [
        DispositionState.CREATED,
        DispositionState.RECEIVED,
        DispositionState.HASH_VERIFIED,
        DispositionState.POLICY_VERIFIED,
        DispositionState.METRIC_REPLAYED,
        DispositionState.QUARANTINED,
    ]
    assert events[0].authority == AuthorityRole.PRODUCER
    assert events[4].authority == AuthorityRole.INDEPENDENT_VERIFIER
    assert events[0].actor_id != events[4].actor_id


# --------------------------------------------------------------------------- #
# Incumbent protection                                                          #
# --------------------------------------------------------------------------- #

def test_candidate_never_overwrites_incumbent(tmp_path):
    _, store, result = _run(tmp_path)
    assert result.outcomes[CURATED].final_state == DispositionState.QUARANTINED
    for lane in (CURATED, WIDE, FULL):
        with pytest.raises(FileNotFoundError):
            store.load_champion_pointer(lane)
    view = equity_research_evidence_view(store.root)
    assert view["champions"] == {}


# --------------------------------------------------------------------------- #
# Fold-completeness rails (the roadmap's aggregator refusal)                    #
# --------------------------------------------------------------------------- #

def test_missing_fold_causes_aggregator_failure():
    # Aggregator refuses a tier whose produced folds omit a declared fold.
    parts = _tier_partitions("curated", 25, _W_PERSIST)
    expected = {"curated": tuple(sorted(parts))}
    dropped = dict(parts)
    dropped.pop(fold_partition_id("curated", 4))
    params = EvaluationParams(expected_folds_by_tier=expected)
    with pytest.raises(ValueError, match="fold set mismatch"):
        evaluate_partitions(dropped, params, hypothesis_id=HYP)


def test_duplicate_fold_declaration_refused():
    cards = [compute_fold_scorecard(fold_partition_id("curated", i),
                                    _fold_rows("curated", i, _IS_FLAGS[i], 25, _W_PERSIST[i]))
             for i in range(5)]
    expected = [c["fold_id"] for c in cards]
    with pytest.raises(ValueError, match="duplicate fold scorecards"):
        aggregate_tier_scorecard("curated", cards + [cards[0]], expected)


def test_dropped_fold_partition_causes_hash_failure(tmp_path):
    # A signed job declares every fold; handing the worker a partition set with a
    # fold removed fails the dataset-hash rail (set mismatch) before scoring.
    identities, _ = _fresh(tmp_path)
    parts = _three_tier_partitions()
    produced = produce_worker_output(identities, parts, registry_publish=_noop_registry, **SLICE_KW)
    dropped = dict(parts)
    dropped.pop(fold_partition_id("curated", 0))
    with pytest.raises(DatasetHashError, match="partition set mismatch"):
        run_worker(
            hypothesis_id=HYP,
            job_envelope=produced.worker_output.job_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            partitions=dropped,
            producer=identities.producer,
            producer_id=identities.actors[AuthorityRole.PRODUCER],
            trust_store=identities.trust_store,
            created_at=NOW,
            registry_publish=_noop_registry,
        )


def test_changed_fold_partition_causes_hash_failure(tmp_path):
    identities, _ = _fresh(tmp_path)
    parts = _three_tier_partitions()
    produced = produce_worker_output(identities, parts, registry_publish=_noop_registry, **SLICE_KW)
    tampered = dict(parts)
    fid = fold_partition_id("curated", 0)
    tampered[fid] = parts[fid] + _jsonl(_fold_rows("curated", 0, False, 25, 0.5))
    with pytest.raises(DatasetHashError, match="does not match its declared hash"):
        run_worker(
            hypothesis_id=HYP,
            job_envelope=produced.worker_output.job_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            partitions=tampered,
            producer=identities.producer,
            producer_id=identities.actors[AuthorityRole.PRODUCER],
            trust_store=identities.trust_store,
            created_at=NOW,
            registry_publish=_noop_registry,
        )


def _keep_only(lane_keys):
    """A real evaluator that scores everything then keeps only some tier heads."""
    def evaluator(partitions, params, *, hypothesis_id):
        heads = evaluate_partitions(partitions, params, hypothesis_id=hypothesis_id)
        return tuple(h for h in heads if h.lane_id in lane_keys)
    return evaluator


def test_missing_tier_head_causes_failure(tmp_path):
    identities, _ = _fresh(tmp_path)
    with pytest.raises(MissingHeadError, match="do not match declared outputs"):
        produce_worker_output(
            identities, _three_tier_partitions(), evaluator=_keep_only({CURATED}),
            registry_publish=_noop_registry, **SLICE_KW,
        )


def test_changed_artifact_bytes_are_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, _three_tier_partitions(), registry_publish=_noop_registry, **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    path = next(iter(head.files))
    mutated = bytearray(head.files[path])
    mutated[0] ^= 0xFF
    tampered_files = dict(head.files)
    tampered_files[path] = bytes(mutated)
    with pytest.raises(ValueError, match="digest mismatch"):
        store.write_package(head.package_envelope, tampered_files)


# --------------------------------------------------------------------------- #
# Independent metric replay                                                     #
# --------------------------------------------------------------------------- #

def test_local_replay_reproduces(tmp_path):
    _, _, result = _run(tmp_path)
    for outcome in result.outcomes.values():
        checks = {c.check_id: c.passed for c in outcome.verdict.checks}
        assert checks["metric_replay_reproduces"] is True
        assert checks["gate_verdict_reproduces"] is True
        assert checks["artifact_reproduces"] is True


def test_scorecard_is_deterministic_across_runs():
    parts = _three_tier_partitions()
    params = EvaluationParams(
        expected_folds_by_tier={
            t: tuple(sorted(f for f in parts if f.split("::")[0] == t))
            for t in ("curated", "wide", "full_history")
        }
    )
    first = {h.lane_id: h for h in evaluate_partitions(parts, params, hypothesis_id=HYP)}
    second = {h.lane_id: h for h in evaluate_partitions(parts, params, hypothesis_id=HYP)}
    assert set(first) == set(second)
    for key in first:
        assert first[key].verdict == second[key].verdict
        assert first[key].passed == second[key].passed
        for name in first[key].metrics:
            assert first[key].metrics[name] == pytest.approx(second[key].metrics[name], abs=1e-12)
        assert first[key].artifact_bytes == second[key].artifact_bytes


# --------------------------------------------------------------------------- #
# Registry independence                                                        #
# --------------------------------------------------------------------------- #

def test_registry_outage_does_not_stop_verification(tmp_path):
    def broken_registry(experiment, metrics):
        raise ConnectionError("external registry is down")

    _, _, result = _run(tmp_path, registry=broken_registry)
    assert result.outcomes[CURATED].final_state == DispositionState.QUARANTINED
    assert result.outcomes[WIDE].final_state == DispositionState.REJECTED


# --------------------------------------------------------------------------- #
# Recovery + view                                                              #
# --------------------------------------------------------------------------- #

def test_indexes_rebuild_to_byte_equivalent_state(tmp_path):
    import shutil

    _, store, _ = _run(tmp_path)
    before = store.current_index_bytes()
    shutil.rmtree(store.root / "indexes")
    rebuilt = store.rebuild_indexes()
    assert rebuilt == before


def test_evidence_view_reports_per_tier_dispositions(tmp_path):
    _, store, _ = _run(tmp_path)
    view = equity_research_evidence_view(store.root)
    assert view["available"] is True
    by_lane = {entry["lane_id"]: entry for entry in view["lanes"]}
    assert by_lane[CURATED]["state"] == "QUARANTINED"
    assert by_lane[CURATED]["tier"] == "curated"
    assert by_lane[WIDE]["state"] == "REJECTED"
    assert by_lane[FULL]["state"] == "REJECTED"
    # The roadmap J2 side-by-side reporting: curated vs wide vs full_history.
    div = view["tier_divergence"][HYP]
    assert div["curated"]["state"] == "QUARANTINED"
    assert div["wide"]["state"] == "REJECTED"
    assert div["full_history"]["state"] == "REJECTED"


def test_evidence_view_handles_absent_store(tmp_path):
    view = equity_research_evidence_view(tmp_path / "does-not-exist")
    assert view["available"] is False
    assert view["lanes"] == []
    assert view["tier_divergence"] == {}


# --------------------------------------------------------------------------- #
# Research-domain honesty criteria                                             #
# --------------------------------------------------------------------------- #

def test_fade_tier_rejected_as_honest_null(tmp_path):
    # A positive-but-insignificant OOS rank-IC is a first-class honest null.
    heads = {h.lane_id: h for h in _evaluate_all(_three_tier_partitions())}
    wide = heads[WIDE]
    assert wide.verdict == VERDICT_FADES_WITH_BREADTH
    assert wide.passed is False
    gate = {g.gate_id: g.status for g in wide.gates}
    assert gate["signal_persists_oos"] == GateStatus.FAIL
    _, _, result = _run(tmp_path)
    assert result.outcomes[WIDE].final_state == DispositionState.REJECTED


def test_no_signal_tier_rejected(tmp_path):
    heads = {h.lane_id: h for h in _evaluate_all(_three_tier_partitions())}
    full = heads[FULL]
    assert full.verdict == VERDICT_NO_SIGNAL
    assert full.passed is False
    _, _, result = _run(tmp_path)
    assert result.outcomes[FULL].final_state == DispositionState.REJECTED


def test_insufficient_oos_folds_rejected(tmp_path):
    # A tier with fewer than the required out-of-sample folds is refused a
    # verdict (never force-scored) and REJECTED end to end.
    # Only 2 OOS folds (< MIN_OOS_FOLDS_FOR_VERDICT = 3).
    thin_flags = (True, True, True, False, False)
    parts = {
        fold_partition_id("curated", i): _jsonl(_fold_rows("curated", i, thin_flags[i], 25, _W_PERSIST[i]))
        for i in range(5)
    }
    heads = {h.lane_id: h for h in _evaluate_all(parts)}
    curated = heads[CURATED]
    gate = {g.gate_id: g.status for g in curated.gates}
    assert gate["sufficient_oos_folds"] == GateStatus.FAIL
    assert curated.passed is False
    _, _, result = _run(tmp_path, partitions=parts)
    assert result.outcomes[CURATED].final_state == DispositionState.REJECTED


def test_non_finite_forward_return_rejected_closed():
    import math as _math
    rows = _fold_rows("curated", 0, False, 25, 0.5)
    rows[0]["forward_return"] = _math.nan
    parts = {fold_partition_id("curated", 0): _jsonl(rows)}
    with pytest.raises(ValueError, match="non-finite"):
        _evaluate_all(parts)


def _evaluate_all(parts):
    params = EvaluationParams(
        expected_folds_by_tier={
            t: tuple(sorted(f for f in parts if f.split("::")[0] == t))
            for t in {f.split("::")[0] for f in parts}
        }
    )
    return evaluate_partitions(parts, params, hypothesis_id=HYP)


# --------------------------------------------------------------------------- #
# Importer-layer integrity — the LOCAL authority's own checks in the FAILING   #
# direction (guards the risk-target F1/F4 "dead check" regression class).      #
# --------------------------------------------------------------------------- #

def _import_one(identities, store, produced, head, partitions, **overrides):
    kw = dict(
        hypothesis_id=HYP,
        job=produced.job,
        capability_profile_envelope=produced.worker_output.capability_profile_envelope,
        dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
        strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
        partitions=partitions,
        trust_store=identities.trust_store,
        importer=identities.importer,
        importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.verifier,
        verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=NOW,
    )
    kw.update(overrides)
    return import_head(store, head, **kw)


def test_import_rejects_report_not_bound_to_package(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _three_tier_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    heads = {h.lane_id: h for h in produced.worker_output.heads}
    # Swap curated's report envelope for wide's: validly signed by the producer,
    # but NOT the report digest bound in curated's package.
    tampered = dataclasses.replace(
        heads[CURATED], evaluation_report_envelope=heads[WIDE].evaluation_report_envelope
    )
    outcome = _import_one(identities, store, produced, tampered, partitions)
    assert outcome.final_state == DispositionState.REJECTED
    assert "evaluation_report_bound_to_package" in (outcome.verdict.rejection_reason or "")


def test_import_rejects_wrong_capability_profile(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _three_tier_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    head = produced.worker_output.heads[0]
    other = build_capability_profile(profile_id="some-other-worker")
    other_env = identities.producer.sign(other, created_at=NOW)
    outcome = _import_one(identities, store, produced, head, partitions,
                          capability_profile_envelope=other_env)
    assert outcome.final_state == DispositionState.REJECTED
    assert "no_forbidden_capabilities" in (outcome.verdict.rejection_reason or "")


def test_import_rejects_wrong_strategy_manifest(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _three_tier_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    head = produced.worker_output.heads[0]
    other_sm = build_research_strategy_manifest(
        "other_hyp", {"curated": (fold_partition_id("curated", 0),)},
        frozen_at=NOW, min_oos_folds=3, ic_bar=0.03, ic_tstat_bar=2.0,
    )
    other_env = identities.producer.sign(other_sm, created_at=NOW)
    outcome = _import_one(identities, store, produced, head, partitions,
                          strategy_manifest_envelope=other_env)
    assert outcome.final_state == DispositionState.REJECTED
    assert "strategy_manifest_bound_to_job" in (outcome.verdict.rejection_reason or "")


def _tamper_artifact(lane_id):
    """A real producer-side evaluator that doctors ONE tier's packaged scorecard
    bytes (leaving metrics/gates honest) — the exact attack the
    artifact_reproduces rail exists to catch."""
    def evaluator(partitions, params, *, hypothesis_id):
        out = []
        for h in evaluate_partitions(partitions, params, hypothesis_id=hypothesis_id):
            if h.lane_id == lane_id:
                h = dataclasses.replace(h, artifact_bytes=h.artifact_bytes + b" ")
            out.append(h)
        return tuple(out)
    return evaluator


def test_signed_ic_bar_is_actually_enforced():
    # A stricter, SIGNED ic_bar must change the verdict — a bar that is signed
    # into the manifest but not enforced would be a dishonest attestation.
    parts = _tier_partitions("wide", 200, (0.30, 0.30, 0.16, 0.17, 0.15))  # ~0.16 OOS IC
    expected = {"wide": tuple(sorted(parts))}
    heads_default = {h.lane_id: h for h in evaluate_partitions(
        parts, EvaluationParams(expected_folds_by_tier=expected), hypothesis_id=HYP)}
    # Default bar (0.03): persists. Stricter bar (0.25): the 0.16 IC no longer clears.
    strict = EvaluationParams(expected_folds_by_tier=expected, ic_bar=0.25)
    heads_strict = {h.lane_id: h for h in evaluate_partitions(parts, strict, hypothesis_id=HYP)}
    assert heads_default[WIDE].verdict == VERDICT_SIGNAL_PERSISTS
    assert heads_strict[WIDE].verdict == VERDICT_FADES_WITH_BREADTH
    assert heads_strict[WIDE].passed is False


def test_stricter_signed_bar_end_to_end_rejects(tmp_path):
    # The stricter ic_bar flows into the SIGNED strategy manifest and the local
    # importer reproduces against it, so the tier REJECTs end to end.
    parts = _tier_partitions("wide", 200, (0.30, 0.30, 0.16, 0.17, 0.15))
    params = EvaluationParams(ic_bar=0.25)
    _, _, result = _run(tmp_path, partitions=parts, params=params)
    assert result.outcomes[WIDE].final_state == DispositionState.REJECTED


def test_strategy_declared_folds_match_dataset_rail_is_isolated_and_load_bearing(tmp_path):
    # Adversarial producer: build a job that digest-binds BOTH an honest dataset
    # manifest (folds 0-4) AND a strategy manifest that DECLARES an extra fold5 the
    # dataset never carried, then package a head honestly scored over the real
    # folds. Because the job binds the inconsistent strategy manifest, the digest
    # rail (strategy_manifest_bound_to_job) PASSES — so this isolates the NEW
    # strategy_declared_folds_match_dataset rail as the load-bearing failure. A
    # loose OR-assertion would let this rail be silently deleted (the F1/F4
    # dead-check-test class); assert the specific check with AND semantics.
    from src.evidence.equity_research.manifests import (
        build_research_evaluation_job_manifest,
        build_research_fold_dataset_manifest,
    )
    from src.evidence.equity_research.slice import EQUITY_RESEARCH_SCORECARD_VERSION
    from src.evidence.hashing import sha256_bytes

    identities, store = _fresh(tmp_path)
    producer = identities.producer
    partitions = _tier_partitions("curated", 25, _W_PERSIST)  # real folds 0-4

    dataset = build_research_fold_dataset_manifest(
        partitions, dataset_id="er-forge", retrieved_at=NOW,
        coverage_start=date(2015, 1, 1), coverage_end=date(2020, 1, 1),
    )
    declared = tuple(sorted(partitions)) + (fold_partition_id("curated", 5),)  # lies: adds fold5
    sm = build_research_strategy_manifest(
        HYP, {"curated": declared}, frozen_at=NOW, min_oos_folds=3, ic_bar=0.03, ic_tstat_bar=2.0,
    )
    cap = build_capability_profile()
    job = build_research_evaluation_job_manifest(
        job_id="er-forge-job", dataset_manifest=dataset, strategy_manifest=sm,
        capability_profile=cap, git_commit="0" * 40, container_digest=sha256_bytes(b"forge"),
        configuration_digest=sha256_bytes(b"cfg"), scorecard_version=EQUITY_RESEARCH_SCORECARD_VERSION,
        created_at=NOW, expected_tier_lanes=[CURATED],
    )
    # Worker scores honestly over the real partitions (empty params -> folds 0-4),
    # packaging a head bound to the fold-inconsistent job.
    wout = run_worker(
        hypothesis_id=HYP,
        job_envelope=producer.sign(job, created_at=NOW),
        dataset_manifest_envelope=producer.sign(dataset, created_at=NOW),
        strategy_manifest_envelope=producer.sign(sm, created_at=NOW),
        capability_profile_envelope=producer.sign(cap, created_at=NOW),
        partitions=partitions, producer=producer,
        producer_id=identities.actors[AuthorityRole.PRODUCER],
        trust_store=identities.trust_store, created_at=NOW,
        params=EvaluationParams(), registry_publish=_noop_registry,
    )
    head = wout.heads[0]
    outcome = import_head(
        store, head, hypothesis_id=HYP, job=job,
        capability_profile_envelope=wout.capability_profile_envelope,
        dataset_manifest_envelope=wout.dataset_manifest_envelope,
        strategy_manifest_envelope=wout.strategy_manifest_envelope,
        partitions=partitions, trust_store=identities.trust_store,
        importer=identities.importer, importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.verifier, verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=NOW,
    )
    checks = {c.check_id: c.passed for c in outcome.verdict.checks}
    assert outcome.final_state == DispositionState.REJECTED
    # Digest binding is intact — isolates the new rail from strategy_manifest_bound_to_job.
    assert checks["strategy_manifest_bound_to_job"] is True
    # The new rail is the load-bearing failure (dies if the check is ever removed).
    assert checks["strategy_declared_folds_match_dataset"] is False


def test_import_rejects_doctored_scorecard_artifact(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _tier_partitions("curated", 25, _W_PERSIST)
    produced = produce_worker_output(
        identities, partitions, evaluator=_tamper_artifact(CURATED),
        registry_publish=_noop_registry, **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    outcome = _import_one(identities, store, produced, head, partitions)
    assert outcome.final_state == DispositionState.REJECTED
    assert "artifact_reproduces" in (outcome.verdict.rejection_reason or "")
