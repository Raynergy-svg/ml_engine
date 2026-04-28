"""End-to-end tests for the MetaManager pipeline coordinator.

Each test injects mock dependencies — no Claude subprocess, no OANDA,
no real pytest — so the pipeline can be exercised stage by stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scanner.automation.approval_queue import ApprovalQueue
from src.scanner.automation.change_eval import ChangeEvalHarness
from src.scanner.automation.constitution import Constitution
from src.scanner.automation.config_adjuster import ConfigAdjuster
from src.scanner.automation.meta_manager import MetaManager
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    ChangeStage,
    DeployStage,
    Proposal,
)
from src.scanner.automation.post_deploy_critic import PostDeployCritic
from src.scanner.automation.staged_deployer import StagedDeployer


SEED_CONSTITUTION = Path(__file__).resolve().parent.parent / ".claude" / "rules" / "constitution.json"


# ---------- fixtures ----------


@pytest.fixture
def tmp_layout(tmp_path):
    return SimpleNamespace(
        changes_dir=tmp_path / "changes",
        ledger=tmp_path / "changes.jsonl",
        approval_root=tmp_path / "approval",
        scorecard_log=tmp_path / "scorecards.jsonl",
        regime_fixture=tmp_path / "regimes.json",
    )


@pytest.fixture
def fake_specialist_invoker():
    """Returns canned analyst / auditor / arbiter outputs."""

    def invoke(mode: str, prompt: str) -> str:
        if mode == "incident_analyst":
            return (
                "root_cause_hypothesis: cold-start observe filter still tight\n"
                "affected_modules: [src/scanner/engine.py]\n"
                "severity: med\n"
                "proposed_intervention_kind: config\n"
            )
        if mode == "policy_auditor":
            return "agree: true\nreason: machine attestation matches my read"
        if mode == "deployment_arbiter":
            return "Tighten min_confidence by +0.05; predicted +5pip/day, worst case -1.0R."
        return ""

    return invoke


def _build_eval_harness(layout) -> ChangeEvalHarness:
    layout.regime_fixture.write_text(json.dumps({
        "windows": [{"name": "trending_test", "pair": "EUR_USD", "days": 5, "regime": "trending"}],
    }))

    class _Replay:
        def compare_configs(self, a, b, pair, candles):
            return {
                "config_a": {"total_pnl": 5.0, "sharpe": 0.4, "win_rate": 0.55, "trades": []},
                "config_b": {
                    "total_pnl": 6.0, "sharpe": 0.5, "win_rate": 0.6,
                    "trades": [{"pnl_pips": 10}, {"pnl_pips": -5}, {"pnl_pips": 12}],
                },
            }
        def replay_scan(self, pair, candles, cfg, config_label="default"):
            return SimpleNamespace(to_dict=lambda: {"total_pnl": 1.0, "sharpe": 0.3, "win_rate": 0.55, "trades": []})

    class _QA:
        def run_full_audit(self):
            return SimpleNamespace(score=92.0, findings=[])

    return ChangeEvalHarness(
        replay_validator=_Replay(),
        qa_pipeline=_QA(),
        candle_loader=lambda pair, days: [{"close": 1.0}] * 200,
        pytest_runner=lambda target: {"passed": True, "exit_code": 0, "failed": [], "stdout_tail": "ok"},
        regime_windows_path=layout.regime_fixture,
        scorecard_log_path=layout.scorecard_log,
        baseline_config={"min_confidence": 0.50},
        pairs=["EUR_USD"],
    )


def _build_manager(layout, fake_specialist_invoker, *, with_deployer=True) -> MetaManager:
    cfg = SimpleNamespace(min_confidence=0.50)
    queue = ApprovalQueue(root=layout.approval_root)
    deployer = None
    if with_deployer:
        adjuster = ConfigAdjuster(persistence_path=layout.changes_dir.parent / "adj.json")
        adjuster._history.clear()
        adjuster._pending.clear()
        deployer = StagedDeployer(config=cfg, config_adjuster=adjuster, shadow_cycles=1, canary_trades=1)
    critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 8, "win_rate": 0.6, "pnl_delta_r": 0.4},
        ledger_path=layout.ledger,
    )
    return MetaManager(
        config=cfg,
        eval_harness=_build_eval_harness(layout),
        constitution=Constitution(path=SEED_CONSTITUTION),
        approval_queue=queue,
        staged_deployer=deployer,
        post_deploy_critic=critic,
        specialist_invoker=fake_specialist_invoker,
        changes_dir=layout.changes_dir,
        ledger_path=layout.ledger,
    )


# ---------- tests ----------


def test_intake_persists_package_and_runs_diagnosis(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    pkg = mgr.intake({
        "kind": "self_heal",
        "summary": "rejection streak detected",
        "proposed_config_delta": {"min_confidence": {"old": 0.50, "new": 0.55}},
    })
    persisted = tmp_layout.changes_dir / f"{pkg.change_id}.json"
    assert persisted.exists()
    assert pkg.diagnosis is not None
    assert pkg.diagnosis.severity.value == "med"
    assert pkg.diagnosis.proposed_intervention_kind == ChangeKind.CONFIG


def test_full_pipeline_to_approval_queue(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    pkg = mgr.intake({
        "kind": "self_heal",
        "summary": "tighten confidence",
        "proposed_config_delta": {"min_confidence": {"old": 0.50, "new": 0.55}},
    })
    mgr.propose(pkg)
    mgr.evaluate(pkg)
    mgr.check_constitution(pkg)
    mgr.request_approval(pkg)
    pending_dir = tmp_layout.approval_root / "pending"
    assert (pending_dir / f"{pkg.change_id}.json").exists()
    assert pkg.dossier_summary
    assert pkg.rollback_plan
    assert pkg.scorecard is not None


def test_constitution_violation_short_circuits(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    # max_drawdown_pct is a real ScannerConfig field that matches C1's
    # `max_drawdown_*` glob. Widening from 0.10 → 0.20 trips C1
    # (no_widen_max_risk: monotonic_non_increasing) → REJECTED. Previously
    # this test used `max_portfolio_risk` which is not a real field; the
    # orphan-key validator in `_default_proposer` now drops fake keys before
    # constitution sees them (per the 2026-04-16 promoted rule about silent
    # dead-writes from orphan config keys).
    pkg = mgr.intake({
        "kind": "self_heal",
        "proposed_config_delta": {"max_drawdown_pct": {"old": 0.10, "new": 0.20}},
    })
    mgr.propose(pkg)
    mgr.evaluate(pkg)
    mgr.check_constitution(pkg)
    assert pkg.stage == ChangeStage.REJECTED
    assert "constitution_violation" in (pkg.rejection_reason or "")
    assert any(c.clause_id == "C1" for c in (pkg.constitution.clauses if pkg.constitution else []))


def test_drain_advances_approved_to_shadow(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    pkg = mgr.intake({
        "kind": "self_heal",
        "proposed_config_delta": {"min_confidence": {"old": 0.50, "new": 0.55}},
    })
    mgr.propose(pkg)
    mgr.evaluate(pkg)
    mgr.check_constitution(pkg)
    mgr.request_approval(pkg)
    queue = ApprovalQueue(root=tmp_layout.approval_root)
    queue.approve(pkg.change_id, deploy_target=DeployStage.SHADOW, reason="smoke")
    counters = mgr.drain(current_cycle=1)
    assert counters["approved_advanced"] == 1
    refreshed = mgr.get(pkg.change_id)
    assert refreshed is not None
    assert refreshed.stage == ChangeStage.DEPLOYED_SHADOW


def test_post_deploy_review_promotes_on_pass(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    pkg = mgr.intake({"kind": "self_heal", "proposed_config_delta": {"min_confidence": {"old": 0.5, "new": 0.55}}})
    mgr.propose(pkg)
    mgr.evaluate(pkg)
    mgr.check_constitution(pkg)
    mgr.request_approval(pkg)
    ApprovalQueue(root=tmp_layout.approval_root).approve(pkg.change_id, DeployStage.SHADOW)
    mgr.drain(current_cycle=1)  # advances shadow
    counters = mgr.drain(current_cycle=2)  # reviews + promotes
    assert counters["post_deploy_reviews"] >= 1
    assert counters["promotions"] >= 1
    refreshed = mgr.get(pkg.change_id)
    assert refreshed.deploy_target == DeployStage.CANARY


def test_post_deploy_review_rolls_back_on_miss(tmp_layout, fake_specialist_invoker):
    cfg = SimpleNamespace(min_confidence=0.50)
    queue = ApprovalQueue(root=tmp_layout.approval_root)
    adjuster = ConfigAdjuster(persistence_path=tmp_layout.changes_dir.parent / "adj.json")
    adjuster._history.clear(); adjuster._pending.clear()
    deployer = StagedDeployer(config=cfg, config_adjuster=adjuster)
    losing_critic = PostDeployCritic(
        metrics_slicer=lambda p, s: {"sample_size": 10, "win_rate": 0.20, "pnl_delta_r": -3.0},
        ledger_path=tmp_layout.ledger,
    )
    mgr = MetaManager(
        config=cfg,
        eval_harness=_build_eval_harness(tmp_layout),
        constitution=Constitution(path=SEED_CONSTITUTION),
        approval_queue=queue,
        staged_deployer=deployer,
        post_deploy_critic=losing_critic,
        specialist_invoker=fake_specialist_invoker,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger,
    )
    pkg = mgr.intake({"kind": "self_heal", "proposed_config_delta": {"min_confidence": {"old": 0.5, "new": 0.55}}})
    mgr.propose(pkg)
    mgr.evaluate(pkg)
    mgr.check_constitution(pkg)
    mgr.request_approval(pkg)
    queue.approve(pkg.change_id, DeployStage.SHADOW)
    mgr.drain(current_cycle=1)
    counters = mgr.drain(current_cycle=2)
    assert counters["rollbacks"] >= 1
    refreshed = mgr.get(pkg.change_id)
    assert refreshed.stage == ChangeStage.REJECTED
    assert "post_deploy_miss" in (refreshed.rejection_reason or "")


def test_max_concurrent_throttles_intake(tmp_layout, fake_specialist_invoker):
    cfg = SimpleNamespace(min_confidence=0.5)
    mgr = MetaManager(
        config=cfg,
        eval_harness=_build_eval_harness(tmp_layout),
        constitution=Constitution(path=SEED_CONSTITUTION),
        approval_queue=ApprovalQueue(root=tmp_layout.approval_root),
        staged_deployer=None,
        post_deploy_critic=PostDeployCritic(
            metrics_slicer=lambda p, s: {"sample_size": 0},
            ledger_path=tmp_layout.ledger,
        ),
        specialist_invoker=fake_specialist_invoker,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger,
        max_concurrent=1,
    )
    pkg1 = mgr.intake({"kind": "self_heal", "proposed_config_delta": {"min_confidence": {"old": 0.5, "new": 0.55}}})
    pkg2 = mgr.intake({"kind": "self_heal", "proposed_config_delta": {"min_confidence": {"old": 0.5, "new": 0.6}}})
    assert pkg1.stage != ChangeStage.ABORTED
    assert pkg2.stage == ChangeStage.ABORTED
    assert "throttled" in (pkg2.history[-1]["note"] if pkg2.history else "")


def test_persisted_package_round_trips(tmp_layout, fake_specialist_invoker):
    mgr = _build_manager(tmp_layout, fake_specialist_invoker)
    pkg = mgr.intake({"kind": "self_heal", "proposed_config_delta": {"min_confidence": {"old": 0.5, "new": 0.55}}})
    mgr.propose(pkg)
    refreshed = mgr.get(pkg.change_id)
    assert refreshed is not None
    assert refreshed.change_id == pkg.change_id
    assert refreshed.proposal is not None
    assert refreshed.proposal.config_delta == {"min_confidence": {"old": 0.5, "new": 0.55}}


def test_meta_manager_accepts_episodic_memory_dependency(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    sentinel = object()
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger,
        episodic_memory=sentinel,
    )
    assert mm._episodic_memory is sentinel


def test_meta_manager_default_episodic_memory_is_none(tmp_layout):
    from src.scanner.automation.meta_manager import MetaManager
    mm = MetaManager(
        config=None,
        changes_dir=tmp_layout.changes_dir,
        ledger_path=tmp_layout.ledger,
    )
    assert mm._episodic_memory is None


def test_build_surgeon_prompt_contains_scannerconfig_field_whitelist():
    """Surgeon prompt must enumerate valid ScannerConfig keys, not just
    say 'match ScannerConfig field names'. The LLM hallucinates less when
    given the explicit list (G6: orphan-key proposals slip through)."""
    from src.scanner.automation.meta_manager import _build_surgeon_prompt
    from src.scanner.automation.meta_types import ChangePackage
    pkg = ChangePackage(incident={"kind": "tp_too_fast", "summary": "..."})
    prompt = _build_surgeon_prompt(pkg)
    # Anchor on three known ScannerConfig field names — if any were
    # renamed or removed, this test should be updated alongside.
    assert "atr_tp_multiplier" in prompt
    assert "min_confidence" in prompt
    assert "atr_sl_multiplier" in prompt
    # Anchor on the structural marker so a surgeon LLM can find the list.
    assert "VALID_CONFIG_KEYS" in prompt


def test_intake_drops_duplicate_proposal(tmp_layout, fake_specialist_invoker):
    """G8: two incidents with identical (kind, proposed_config_delta) collapse;
    the second drops with rejection_reason='duplicate_proposal'."""
    from src.scanner.automation.meta_manager import MetaManager
    from src.scanner.automation.meta_types import ChangeStage
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    incident = {
        "kind": "tp_too_fast",
        "summary": "first",
        "proposed_config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
    }
    pkg_a = mm.intake(incident)
    incident_b = dict(incident)
    incident_b["summary"] = "duplicate of first"
    pkg_b = mm.intake(incident_b)
    assert pkg_b.stage == ChangeStage.ABORTED
    assert pkg_b.rejection_reason == "duplicate_proposal"
    # First package should reach DIAGNOSING (positive specification — catches a
    # regression where pkg_a accidentally also gets aborted by dedup).
    assert pkg_a.stage == ChangeStage.DIAGNOSING


def test_intake_attaches_historical_outcomes_when_episodic_memory_present(tmp_layout, fake_specialist_invoker):
    """G3: when episodic_memory is injected and incident.setup_features
    is present, query_similar runs and historical_outcomes attaches to
    the package's incident dict."""
    from src.scanner.automation.meta_manager import MetaManager

    class FakeEpisodicMemory:
        def __init__(self):
            self.queries = []
        def query_similar(self, **kwargs):
            self.queries.append(kwargs)
            return {"matched_episodes": 8, "loss_rate": 0.75, "sample": []}

    fake = FakeEpisodicMemory()
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    mm._episodic_memory = fake
    incident = {
        "kind": "tp_too_fast",
        "summary": "...",
        "setup_features": {
            "pair": "EUR_USD", "direction": "long", "regime": "LOW",
            "session": "LON", "news_risk_score": 0.1, "uncertainty_score": 0.3,
        },
    }
    pkg = mm.intake(incident)
    assert "historical_outcomes" in pkg.incident
    assert pkg.incident["historical_outcomes"]["loss_rate"] == 0.75
    assert len(fake.queries) == 1


def test_intake_normalizes_episodic_memory_field_names(tmp_layout, fake_specialist_invoker):
    """Regression: production EpisodicMemory.query_similar returns the dict
    key 'similar_count', but Constitution._eval_historical_loss_rate (G7)
    reads 'matched_episodes'. Without normalisation in intake() the veto
    clause would always hit the 'insufficient_history' abstain branch in
    production — the two halves never met before Phase 1's smoke wired them
    end-to-end. intake() must mirror the count under the constitution's
    expected key while preserving the original (additive, non-destructive)."""
    from src.scanner.automation.meta_manager import MetaManager

    class FakeEpisodicMemory:
        """Mimics the real EpisodicMemory.query_similar output schema."""
        def query_similar(self, **kwargs):
            return {"similar_count": 8, "loss_rate": 0.85}

    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    mm._episodic_memory = FakeEpisodicMemory()
    incident = {
        "kind": "tp_too_fast",
        "summary": "...",
        "setup_features": {
            "pair": "EUR_USD", "direction": "long", "regime": "LOW",
            "session": "LON", "news_risk_score": 0.1, "uncertainty_score": 0.3,
        },
    }
    pkg = mm.intake(incident)
    historical = pkg.incident.get("historical_outcomes") or {}
    # Normalisation: matched_episodes must be present (mirrored from
    # similar_count). This is the constitution's read key.
    assert historical.get("matched_episodes") == 8
    # Additive — original key preserved for any other reader.
    assert historical.get("similar_count") == 8
    # Untouched fields pass through unchanged.
    assert historical.get("loss_rate") == 0.85


def test_intake_no_episodic_query_when_episodic_memory_none(tmp_layout, fake_specialist_invoker):
    """When MetaManager has no episodic_memory injection, intake doesn't
    invent historical_outcomes."""
    from src.scanner.automation.meta_manager import MetaManager
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    # mm._episodic_memory should be None by default
    assert mm._episodic_memory is None
    pkg = mm.intake({"kind": "any", "summary": "..."})
    # Strict: nothing should write the key when memory is absent. (A None
    # value would hide a regression where the gate accidentally writes None.)
    assert "historical_outcomes" not in pkg.incident


def test_is_duplicate_in_flight_empty_dir_returns_false(tmp_layout, fake_specialist_invoker):
    """Helper: empty changes_dir → False (no false positives)."""
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    # Fresh layout has no persisted packages.
    assert mm._is_duplicate_in_flight("any_hash_value") is False


def test_is_duplicate_in_flight_skips_terminal_stages(tmp_layout, fake_specialist_invoker):
    """Helper: persisted packages in terminal stages (closed/aborted/rejected)
    must NOT match — terminal packages are not 'in-flight'."""
    import json
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    # Persist a synthetic ABORTED package whose dedup_hash will match anything
    # we look up — if the helper doesn't skip terminal stages, this would
    # cause a false positive.
    incident = {
        "kind": "tp_too_fast",
        "proposed_config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
    }
    blob = {
        "stage": "aborted",
        "incident": incident,
        "proposal": None,
    }
    (tmp_layout.changes_dir / "terminal_pkg.json").write_text(json.dumps(blob))
    # Compute the same hash a candidate would produce for this incident.
    from src.scanner.automation.meta_types import ChangePackage, Proposal
    candidate = ChangePackage(
        incident=incident,
        proposal=Proposal(config_delta=incident["proposed_config_delta"]),
    )
    target_hash = candidate.dedup_hash()
    # Terminal stage must be skipped — should return False even though the
    # incident.proposed_config_delta matches.
    assert mm._is_duplicate_in_flight(target_hash) is False


def test_is_duplicate_in_flight_handles_corrupt_json(tmp_layout, fake_specialist_invoker):
    """Helper: malformed JSON in changes_dir must not raise; helper logs and
    returns False (fail-open)."""
    mm = _build_manager(tmp_layout, fake_specialist_invoker)
    (tmp_layout.changes_dir / "broken.json").write_text("{not valid json}")
    # Should not raise; should return False.
    assert mm._is_duplicate_in_flight("any_hash") is False
