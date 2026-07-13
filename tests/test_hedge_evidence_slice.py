"""Phase J6 acceptance tests — hedge evaluation end-to-end evidence slice.

These are the roadmap §12 acceptance criteria made real for the hedge lane. NO
MOCKS: every test runs the actual evidence store, real Ed25519 signing, the real
disposition chain, and the REAL analysis-only hedge scorecard
(``src.hedge.hedge_scorecard``) over real ledger-shaped rows — never a
test-double.

Because the scorecard is deterministic pure float arithmetic, the default
evaluator (``evaluate_partitions``) is used directly: the worker and the
importer's independent replay obtain identical results from identical partition
bytes, which is exactly the reproducibility contract the store enforces. A few
structural tests inject a thin REAL wrapper that drops/keeps heads — an ordinary
callable returning real contract objects, still not a mock.

The §12 checklist and the test that covers it:

    - Remote worker cannot read .claude/state.json ........ see
      test_hedge_evidence_worker_no_authority.py (AST + source + closure scan)
    - Remote worker has no broker credentials ............. idem + capability test there
    - Candidate artifact cannot overwrite incumbent ....... test_candidate_never_overwrites_incumbent
    - Missing fold causes failure ......................... test_missing_strategy_head_causes_failure
    - Changed dataset partition causes hash failure ....... test_changed_dataset_partition_causes_hash_failure
    - Changed model bytes cause signature failure ......... test_changed_artifact_bytes_are_rejected
    - Worse head is rejected independently ................ test_beta_strategy_rejected_independently
    - Good head does not rescue a failed head ............. test_good_strategy_does_not_rescue_failed
    - Local replay reproduces metrics within tolerance .... test_local_scorecard_replay_reproduces
    - External registry outage does not stop verification . test_registry_outage_does_not_stop_verification

Plus hedge-domain honesty criteria:
    - Thin history is rejected, never force-verdicted ..... test_thin_history_strategy_rejected
    - Cost-unmodeled venue cannot clear the after-cost bar. test_cost_unmodeled_venue_cannot_pass_gate
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date

import pytest

from src.evidence.contracts import (
    AuthorityRole,
    DispositionEvent,
    DispositionState,
    GateStatus,
    ImportDecision,
)
from src.evidence.signing import verify_envelope
from src.evidence.hedge_eval.evaluation import EvaluationParams, evaluate_partitions, evaluate_strategy
from src.evidence.hedge_eval.local_import import import_head
from src.evidence.hedge_eval.manifests import build_capability_profile, build_hedge_strategy_manifest
from src.evidence.hedge_eval.slice import (
    build_evidence_store,
    build_slice_identities,
    hedge_evidence_view,
    produce_worker_output,
    run_hedge_evidence_slice,
    utc,
)
from src.evidence.hedge_eval.worker import DatasetHashError, MissingHeadError, run_worker

NOW = utc(2026, 7, 13, 12, 0)
SLICE_KW = dict(
    dataset_id="hedge-ledger-test",
    coverage_start=date(2026, 1, 1),
    coverage_end=date(2026, 3, 1),
    retrieved_at=NOW,
    created_at=NOW,
    git_commit="0" * 40,
)


def _noop_registry(experiment, metrics):
    return None


def _row(strategy, asof, raw_net, hedged_net, *, status="applied", cost_known=None, net_beta=0.2):
    cost_known = hedged_net is not None if cost_known is None else cost_known
    return {
        "strategy": strategy,
        "asof_date": asof,
        "raw": {"net_return": raw_net},
        "hedged": {"net_return": hedged_net, "gross_return": hedged_net},
        "hedge": {"status": status, "overlay_cost_known": cost_known},
        "exposure": {"net_beta_exposure": net_beta, "fail_closed": False, "concentration_warnings": []},
    }


def _jsonl(rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


# alpha: hedge lifts after-cost expectancy AND cuts drawdown -> signal_real_but_noisy -> PASS
_RAW_ALPHA = [0.01, -0.05, 0.02, -0.04, 0.01, -0.03]
_HED_ALPHA = [0.015, -0.01, 0.02, -0.008, 0.012, -0.005]
# beta: raw steadily positive, hedged ~flat/negative -> return_was_beta_not_alpha -> REJECT
_RAW_BETA = [0.02, 0.015, 0.018, 0.022, 0.019, 0.017]
_HED_BETA = [-0.001, -0.002, 0.0005, -0.0015, -0.0008, -0.001]

ALPHA_LANE = "hedge_eval_alpha_lane"
BETA_LANE = "hedge_eval_beta_lane"


def _alpha_rows():
    return [_row("alpha_lane", f"2026-01-{i+1:02d}", r, h) for i, (r, h) in enumerate(zip(_RAW_ALPHA, _HED_ALPHA))]


def _beta_rows():
    return [_row("beta_lane", f"2026-01-{i+1:02d}", r, h) for i, (r, h) in enumerate(zip(_RAW_BETA, _HED_BETA))]


def _two_strategy_partitions():
    return {"alpha_lane": _jsonl(_alpha_rows()), "beta_lane": _jsonl(_beta_rows())}


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _run(tmp_path, partitions=None, registry=_noop_registry, params=None, evaluator=evaluate_partitions):
    identities, store = _fresh(tmp_path)
    partitions = partitions if partitions is not None else _two_strategy_partitions()
    result = run_hedge_evidence_slice(
        store, identities, partitions, evaluator=evaluator,
        registry_publish=registry, params=params, **SLICE_KW,
    )
    return identities, store, result


# --------------------------------------------------------------------------- #
# Disposition semantics                                                        #
# --------------------------------------------------------------------------- #

def test_beta_strategy_rejected_independently(tmp_path):
    _, _, result = _run(tmp_path)
    alpha = result.outcomes[ALPHA_LANE]
    beta = result.outcomes[BETA_LANE]
    assert alpha.final_state == DispositionState.QUARANTINED
    assert alpha.verdict.decision == ImportDecision.ACCEPT_FOR_QUARANTINE
    assert beta.final_state == DispositionState.REJECTED
    assert beta.verdict.decision == ImportDecision.REJECT
    assert "candidate_evaluation_gate_passed" in (beta.verdict.rejection_reason or "")
    # Independent packages — one lane's verdict is not derived from the other.
    assert alpha.package_digest != beta.package_digest


def test_good_strategy_does_not_rescue_failed(tmp_path):
    _, store, result = _run(tmp_path)
    assert result.outcomes[ALPHA_LANE].final_state == DispositionState.QUARANTINED
    assert result.outcomes[BETA_LANE].final_state == DispositionState.REJECTED
    # The beta package's ledger terminates at REJECTED, never QUARANTINED.
    beta_digest = result.outcomes[BETA_LANE].package_digest
    ledger = store.load_ledger(beta_digest)
    states = [verify_envelope(env, DispositionEvent, store.trust_store).to_state for env in ledger]
    assert DispositionState.QUARANTINED not in states
    assert states[-1] == DispositionState.REJECTED


def test_quarantine_chain_is_exactly_the_signed_lifecycle(tmp_path):
    identities, store, result = _run(tmp_path)
    alpha_digest = result.outcomes[ALPHA_LANE].package_digest
    ledger = store.load_ledger(alpha_digest)
    events = [verify_envelope(env, DispositionEvent, identities.trust_store) for env in ledger]
    assert [e.to_state for e in events] == [
        DispositionState.CREATED,
        DispositionState.RECEIVED,
        DispositionState.HASH_VERIFIED,
        DispositionState.POLICY_VERIFIED,
        DispositionState.METRIC_REPLAYED,
        DispositionState.QUARANTINED,
    ]
    # Genesis authored by the remote producer; metric-replay by a DIFFERENT authority.
    assert events[0].authority == AuthorityRole.PRODUCER
    assert events[4].authority == AuthorityRole.INDEPENDENT_VERIFIER
    assert events[0].actor_id != events[4].actor_id


# --------------------------------------------------------------------------- #
# Incumbent protection                                                          #
# --------------------------------------------------------------------------- #

def test_candidate_never_overwrites_incumbent(tmp_path):
    # Even a passing (quarantined) strategy never becomes champion — the slice
    # has no promotion authority, so no champion pointer is ever written.
    _, store, result = _run(tmp_path)
    assert result.outcomes[ALPHA_LANE].final_state == DispositionState.QUARANTINED
    for lane in (ALPHA_LANE, BETA_LANE):
        with pytest.raises(FileNotFoundError):
            store.load_champion_pointer(lane)
    view = hedge_evidence_view(store.root)
    assert view["champions"] == {}


# --------------------------------------------------------------------------- #
# Integrity rails                                                               #
# --------------------------------------------------------------------------- #

def _keep_only(strategy_keys):
    """A real evaluator that scores everything then keeps only some heads."""
    def evaluator(partitions, params):
        heads = evaluate_partitions(partitions, params)
        return tuple(h for h in heads if h.head_id in strategy_keys)
    return evaluator


def test_missing_strategy_head_causes_failure(tmp_path):
    identities, _ = _fresh(tmp_path)
    with pytest.raises(MissingHeadError, match="do not match declared outputs"):
        produce_worker_output(
            identities, _two_strategy_partitions(), evaluator=_keep_only({"alpha_lane"}),
            registry_publish=_noop_registry, **SLICE_KW,
        )


def test_changed_dataset_partition_causes_hash_failure(tmp_path):
    identities, _ = _fresh(tmp_path)
    partitions = _two_strategy_partitions()
    produced = produce_worker_output(
        identities, partitions, registry_publish=_noop_registry, **SLICE_KW,
    )
    tampered = dict(partitions)
    tampered["alpha_lane"] = partitions["alpha_lane"] + b'{"strategy": "alpha_lane"}\n'
    with pytest.raises(DatasetHashError, match="does not match its declared hash"):
        run_worker(
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


def test_changed_artifact_bytes_are_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, _two_strategy_partitions(), registry_publish=_noop_registry, **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    path = next(iter(head.files))
    # Flip one byte, preserving length, so the failure is the signed-digest
    # binding itself, not a coincidental size change.
    mutated = bytearray(head.files[path])
    mutated[0] ^= 0xFF
    tampered_files = dict(head.files)
    tampered_files[path] = bytes(mutated)
    with pytest.raises(ValueError, match="digest mismatch"):
        store.write_package(head.package_envelope, tampered_files)


# --------------------------------------------------------------------------- #
# Independent metric replay                                                     #
# --------------------------------------------------------------------------- #

def test_local_scorecard_replay_reproduces(tmp_path):
    _, _, result = _run(tmp_path)
    for outcome in result.outcomes.values():
        checks = {c.check_id: c.passed for c in outcome.verdict.checks}
        assert checks["metric_replay_reproduces"] is True
        assert checks["gate_verdict_reproduces"] is True


def test_scorecard_is_deterministic_across_runs(tmp_path):
    partitions = _two_strategy_partitions()
    params = EvaluationParams()
    first = {h.head_id: h for h in evaluate_partitions(partitions, params)}
    second = {h.head_id: h for h in evaluate_partitions(partitions, params)}
    assert set(first) == set(second)
    for key in first:
        assert first[key].verdict == second[key].verdict
        assert first[key].passed == second[key].passed
        for name in first[key].metrics:
            assert first[key].metrics[name] == pytest.approx(second[key].metrics[name], abs=1e-12)
        # Byte-identical artifact -> content-addressing is stable across runs.
        assert first[key].artifact_bytes == second[key].artifact_bytes


# --------------------------------------------------------------------------- #
# Registry independence                                                        #
# --------------------------------------------------------------------------- #

def test_registry_outage_does_not_stop_verification(tmp_path):
    def broken_registry(experiment, metrics):
        raise ConnectionError("external registry is down")

    _, _, result = _run(tmp_path, registry=broken_registry)
    assert result.outcomes[ALPHA_LANE].final_state == DispositionState.QUARANTINED
    assert result.outcomes[BETA_LANE].final_state == DispositionState.REJECTED


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


def test_evidence_view_reports_per_strategy_dispositions(tmp_path):
    _, store, _ = _run(tmp_path)
    view = hedge_evidence_view(store.root)
    assert view["available"] is True
    by_lane = {entry["lane_id"]: entry for entry in view["lanes"]}
    assert by_lane[ALPHA_LANE]["state"] == "QUARANTINED"
    assert by_lane[ALPHA_LANE]["verdict"]["decision"] == "ACCEPT_FOR_QUARANTINE"
    assert by_lane[BETA_LANE]["state"] == "REJECTED"
    assert "candidate_evaluation_gate_passed" in by_lane[BETA_LANE]["verdict"]["failed_checks"]


def test_evidence_view_handles_absent_store(tmp_path):
    view = hedge_evidence_view(tmp_path / "does-not-exist")
    assert view["available"] is False
    assert view["lanes"] == []


# --------------------------------------------------------------------------- #
# Hedge-domain honesty criteria                                                #
# --------------------------------------------------------------------------- #

def test_thin_history_strategy_rejected(tmp_path):
    # Only two logged cycles: the scorecard refuses to force a verdict, so the
    # strategy fails its sufficient-history gate and is REJECTED, never
    # accepted on a thin sample.
    thin = [_row("thin_lane", "2026-01-01", 0.01, 0.005), _row("thin_lane", "2026-01-02", 0.01, 0.005)]
    # The evaluation gate that fails is sufficient_history (in the head's report).
    head = evaluate_strategy("thin_lane", thin, EvaluationParams())
    assert head.passed is False
    gate = {g.gate_id: g.status for g in head.gates}
    assert gate["sufficient_history"] == GateStatus.FAIL
    # End-to-end: the import verdict collapses the failed evaluation gate into
    # candidate_evaluation_gate_passed and REJECTS.
    _, _, result = _run(tmp_path, partitions={"thin_lane": _jsonl(thin)})
    outcome = result.outcomes["hedge_eval_thin_lane"]
    assert outcome.final_state == DispositionState.REJECTED
    checks = {c.check_id: c.passed for c in outcome.verdict.checks}
    assert checks["candidate_evaluation_gate_passed"] is False


def test_cost_unmodeled_venue_cannot_pass_gate(tmp_path):
    # An equity/ETF-style hedge with NO fill history: every applied cycle comes
    # back cost_known=False, so the scorecard reports a "cost_unmodeled:" gross
    # verdict. Even though hedging helps on the gross numbers, the after-cost
    # admissibility gate must FAIL — an unmodeled-cost win is never a genuine
    # after-cost conclusion.
    rows = [
        _row("etf_lane", f"2026-01-{i+1:02d}", r, h, cost_known=False)
        for i, (r, h) in enumerate(zip(_RAW_ALPHA, _HED_ALPHA))
    ]
    head = evaluate_strategy("etf_lane", rows, EvaluationParams())
    assert head.cost_unmodeled_for_venue is True
    assert head.verdict.startswith("cost_unmodeled:")
    assert head.passed is False
    gate = {g.gate_id: g.status for g in head.gates}
    assert gate["hedge_improves_after_cost"] == GateStatus.FAIL

    _, _, result = _run(tmp_path, partitions={"etf_lane": _jsonl(rows)})
    outcome = result.outcomes["hedge_eval_etf_lane"]
    assert outcome.final_state == DispositionState.REJECTED
    checks = {c.check_id: c.passed for c in outcome.verdict.checks}
    assert checks["candidate_evaluation_gate_passed"] is False


# --------------------------------------------------------------------------- #
# Importer-layer integrity — the LOCAL authority's own checks in the FAILING   #
# direction (guards the risk-target F1/F4 "dead check" regression class).      #
# --------------------------------------------------------------------------- #

def _import_one(identities, store, produced, head, partitions, **overrides):
    kw = dict(
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
    partitions = _two_strategy_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    heads = {h.head_id: h for h in produced.worker_output.heads}
    # Swap alpha's evaluation-report envelope for beta's: validly signed by the
    # producer, but NOT the report digest bound in alpha's package.
    tampered = dataclasses.replace(
        heads["alpha_lane"], evaluation_report_envelope=heads["beta_lane"].evaluation_report_envelope
    )
    outcome = _import_one(identities, store, produced, tampered, partitions)
    assert outcome.final_state == DispositionState.REJECTED
    assert "evaluation_report_bound_to_package" in (outcome.verdict.rejection_reason or "")


def test_import_rejects_wrong_capability_profile(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _two_strategy_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    head = produced.worker_output.heads[0]
    # A different, validly-signed capability profile the job never referenced.
    other = build_capability_profile(profile_id="some-other-worker")
    other_env = identities.producer.sign(other, created_at=NOW)
    outcome = _import_one(identities, store, produced, head, partitions,
                          capability_profile_envelope=other_env)
    assert outcome.final_state == DispositionState.REJECTED
    assert "no_forbidden_capabilities" in (outcome.verdict.rejection_reason or "")


def test_import_rejects_wrong_strategy_manifest(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = _two_strategy_partitions()
    produced = produce_worker_output(identities, partitions, registry_publish=_noop_registry, **SLICE_KW)
    head = produced.worker_output.heads[0]
    # A different, validly-signed strategy manifest the job never referenced.
    other_sm = build_hedge_strategy_manifest(["other_lane"], frozen_at=NOW, min_history=3)
    other_env = identities.producer.sign(other_sm, created_at=NOW)
    outcome = _import_one(identities, store, produced, head, partitions,
                          strategy_manifest_envelope=other_env)
    assert outcome.final_state == DispositionState.REJECTED
    assert "strategy_manifest_bound_to_job" in (outcome.verdict.rejection_reason or "")


def _tamper_artifact(strategy):
    """A real producer-side evaluator that doctors ONE strategy's packaged
    scorecard bytes (leaving metrics/gates honest) — the exact attack the
    artifact_reproduces rail exists to catch."""
    def evaluator(partitions, params):
        out = []
        for h in evaluate_partitions(partitions, params):
            if h.head_id == strategy:
                h = dataclasses.replace(h, artifact_bytes=h.artifact_bytes + b" ")
            out.append(h)
        return tuple(out)
    return evaluator


def test_import_rejects_doctored_scorecard_artifact(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions = {"alpha_lane": _jsonl(_alpha_rows())}
    # Producer packages a self-consistent, SIGNED, but doctored scorecard.json.
    produced = produce_worker_output(
        identities, partitions, evaluator=_tamper_artifact("alpha_lane"),
        registry_publish=_noop_registry, **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    # The importer reproduces with the HONEST default evaluator: metrics/gates
    # still reproduce, but the reproduced artifact digest differs -> REJECT.
    outcome = _import_one(identities, store, produced, head, partitions)
    assert outcome.final_state == DispositionState.REJECTED
    assert "artifact_reproduces" in (outcome.verdict.rejection_reason or "")


def test_non_finite_ledger_value_is_rejected_closed():
    # A crafted NaN return (json.dumps emits "NaN", json.loads reads it back)
    # must not silently flow into Canonical-JSON encoding; the byte-parse path
    # rejects it at the source so producer and importer fail closed symmetrically.
    import math as _math

    rows = _alpha_rows()
    rows[0]["raw"]["net_return"] = _math.nan
    with pytest.raises(ValueError, match="non-finite raw.net_return"):
        evaluate_partitions({"alpha_lane": _jsonl(rows)}, EvaluationParams())
