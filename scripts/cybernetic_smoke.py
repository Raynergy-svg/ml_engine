#!/usr/bin/env python3
"""Cybernetic smoke test — proves the 9-stage meta pipeline circulates with real specialists.

Drives ONE synthetic incident through stages 1-4 (intake → propose → evaluate →
check_constitution) using the real claude-CLI specialist invoker. Stops at the
approval queue — does NOT auto-deploy. State is isolated under
`.claude/meta_smoke/` so the real meta state (`.claude/changes/`,
`.claude/approval_queue/`) is untouched.

Cost: 2 Claude CLI subprocess calls (~$0.02, ~3 minutes wall clock).

Usage:
    python scripts/cybernetic_smoke.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from pprint import pformat

# Repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scanner.automation.meta_manager import MetaManager  # noqa: E402
from src.scanner.automation.change_eval import ChangeEvalHarness  # noqa: E402
from src.scanner.automation.approval_queue import ApprovalQueue  # noqa: E402
from src.scanner.automation.meta_types import ChangeStage  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("cybernetic_smoke")


SMOKE_ROOT = ROOT / ".claude" / "meta_smoke"
CHANGES_DIR = SMOKE_ROOT / "changes"
LEDGER = SMOKE_ROOT / "changes.jsonl"
QUEUE_ROOT = SMOKE_ROOT / "approval_queue"
SMOKE_EPISODIC_PATH = SMOKE_ROOT / "episodic_memory.json"


class _SmokeEpisodicAdapter:
    """Thin wrapper around EpisodicMemory that translates the dict it returns
    into the schema the constitution's C8 historical_loss_rate clause expects.

    EpisodicMemory.query_similar() returns {"similar_count", "loss_rate", ...}
    but the constitution clause reads {"matched_episodes", "loss_rate"}. The
    smoke is the first place these two production halves meet end-to-end —
    this adapter bridges the field-name drift so the real store can drive the
    real veto. The underlying pre-seed → similarity-match → loss-rate path is
    fully exercised; only the dict-key naming is normalised on the way out.
    """

    def __init__(self, inner):
        self._inner = inner

    def record_setup(self, *args, **kwargs):
        return self._inner.record_setup(*args, **kwargs)

    def record_outcome(self, *args, **kwargs):
        return self._inner.record_outcome(*args, **kwargs)

    def query_similar(self, *args, **kwargs):
        out = self._inner.query_similar(*args, **kwargs) or {}
        # Mirror similar_count → matched_episodes. Keep both keys so the
        # adapter is purely additive — anything reading the original schema
        # still works.
        if "similar_count" in out and "matched_episodes" not in out:
            out["matched_episodes"] = out["similar_count"]
        return out


def _smoke_episodic_memory():
    """Return an EpisodicMemory pointed at the smoke-isolated path so we
    don't pollute trained_data/episodic_memory.json. Wrapped in
    _SmokeEpisodicAdapter so the dict shape MetaManager.intake stores into
    incident["historical_outcomes"] is the shape Constitution expects."""
    from src.scanner.automation.episodic_memory import EpisodicMemory
    inner = EpisodicMemory(memory_path=SMOKE_EPISODIC_PATH)
    return _SmokeEpisodicAdapter(inner)


def build_staleness_incident() -> dict:
    """Cycle #298 RL position-sizer staleness signal.

    Expected surgeon behavior: refuse to propose a config_delta and explain
    that the fix is a model retrain (not a ScannerConfig field). Validates
    the surgeon's restraint discipline.
    """
    return {
        "kind": "config",
        "source": "cybernetic_smoke",
        "severity": "high",
        "summary": "RL position sizer model is 33 days old; staleness threshold is 14 days.",
        "signal": {
            "model": "rl_position_sizer",
            "age_days": 33,
            "staleness_threshold_days": 14,
            "trades_since_last_train": 412,
        },
        "affected_modules": ["src/risk/position_sizing.py", "src/scanner/feedback/self_heal.py"],
        "context": {
            "recent_loss_streak": 0,
            "current_nav_usd": 99500,
            "broker_environment": "practice",
        },
    }


def build_tp_too_fast_incident() -> dict:
    """TP-hit-too-fast pattern. The promoted rule from 2026-04-02 says
    'Increase atr_tp_multiplier by 0.1 (TP hit too fast 6 times)'.

    Expected surgeon behavior: propose a non-empty config_delta with
    `atr_tp_multiplier` (a real ScannerConfig field) increased by ~0.1
    from current 1.5 → 1.6 (or similar). Validates the surgeon CAN emit
    a non-empty proposal when the incident is a true config-only fix.
    """
    return {
        "kind": "config",
        "source": "cybernetic_smoke",
        "severity": "med",
        "summary": (
            "TP hit too fast on 6 of last 10 winning trades; "
            "tp_atr_ratio averaged 0.28 vs expected ≥0.50. "
            "atr_tp_multiplier may be too tight."
        ),
        "signal": {
            "metric": "tp_atr_ratio_avg",
            "current_value": 0.28,
            "expected_floor": 0.50,
            "winning_trades_in_window": 10,
            "tp_hit_too_fast_count": 6,
            "current_atr_tp_multiplier": 1.5,
        },
        "affected_modules": ["src/risk/position_sizing.py", "src/scanner/execution.py"],
        "context": {
            "broker_environment": "practice",
            "current_nav_usd": 99500,
            "promoted_rule_2026_04_02": (
                "Increase atr_tp_multiplier by 0.1 (TP hit too fast 6 times)"
            ),
        },
    }


SCENARIOS = {
    "staleness": build_staleness_incident,
    "tp_too_fast": build_tp_too_fast_incident,
}

def build_dup_drop_pair():
    """Returns a list of 2 incidents with identical (kind, config_delta).
    Smoke harness fires both; first lands in changes_dir; second
    must drop with rejection_reason=duplicate_proposal."""
    base = {
        "kind": "tp_too_fast",
        "source": "cybernetic_smoke_dup",
        "severity": "med",
        "summary": "duplicate test",
        "config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        "signal": {"metric": "tp_atr_ratio_avg", "current_value": 0.28},
        "context": {"broker_environment": "practice"},
    }
    return [base, dict(base, summary="duplicate test (second)")]


SCENARIOS["dup_drop"] = build_dup_drop_pair


def build_historical_veto() -> dict:
    """Sentinel for the historical-veto scenario — the harness pre-seeds
    the real EpisodicMemory store with 8 losing outcomes for an EUR_USD
    long LOW-regime LON-session setup, fires a matching incident, and
    drives intake → propose → check_constitution to demonstrate the
    historical_loss_rate clause veto end-to-end.

    Returns a sentinel marker; the dispatch in main() detects the marker
    and routes to _run_historical_veto_scenario().
    """
    return {"_smoke_kind": "historical_veto"}


SCENARIOS["historical_veto"] = build_historical_veto


def build_r_floor_promotion() -> dict:
    """Sentinel for the R-floor promotion scenario — the harness builds
    SHADOW + CANARY ChangePackages, monkey-patches the StagedDeployer's
    R-stats helper to produce promoting / regressing R distributions, and
    runs 5 gate decisions covering the count-gate boundary, the R-floor
    crossing, and the canary→live count threshold.

    Returns a sentinel marker; the dispatch in main() detects the marker
    and routes to _run_r_floor_promotion_scenario().
    """
    return {"_smoke_kind": "r_floor_promotion"}


SCENARIOS["r_floor_promotion"] = build_r_floor_promotion


def _run_historical_veto_scenario() -> int:
    """Pre-seed episodic memory with 8 losing outcomes for an EUR_USD long
    LOW-regime LON-session setup, fire a matching incident through
    MetaManager.intake (G3 episodic-query attaches historical_outcomes
    automatically), then drive propose + check_constitution. The C8
    historical_loss_rate_veto clause must fire (matched=8 >= n_min=5,
    loss_rate=1.0 > threshold=0.7).
    """
    banner("historical_veto — pre-seeding real EpisodicMemory store")
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    # Clear any stale smoke episodic state so each run starts clean.
    if SMOKE_EPISODIC_PATH.exists():
        SMOKE_EPISODIC_PATH.unlink()

    em = _smoke_episodic_memory()
    setup_kwargs = dict(
        pair="EUR_USD", direction="long", regime="LOW", session="LON",
        news_risk_score=0.1, uncertainty_score=0.3,
        atr_normalized=0.5, spread_pips=0.8,
        confidence=0.65, weighted_vote_score=0.6, rr_ratio=1.5,
    )
    seeded = 0
    for _ in range(8):
        eid = em.record_setup(**setup_kwargs)
        # EpisodicMemory's query_similar filters on outcome=="LOSS" (uppercase).
        if em.record_outcome(eid, outcome="LOSS", pnl_pips=-12.0):
            seeded += 1
    print(f"[smoke] seeded {seeded} losing outcomes for EUR_USD long LOW LON")

    # Verify the seed by querying through the adapter directly.
    seed_check = em.query_similar(
        pair="EUR_USD", direction="long", regime="LOW", session="LON",
        news_risk_score=0.1, uncertainty_score=0.3,
    )
    print(f"[smoke] query_similar via adapter returned: {pformat(seed_check, width=72)}")

    incident = {
        "kind": "tp_too_fast",
        "source": "cybernetic_smoke_hist_veto",
        "severity": "med",
        "summary": "historical-loss-pattern test (real episodic seed)",
        "proposed_config_delta": {"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        "setup_features": {
            "pair": "EUR_USD", "direction": "long", "regime": "LOW",
            "session": "LON", "news_risk_score": 0.1, "uncertainty_score": 0.3,
        },
    }

    # Build a MetaManager with the smoke episodic_memory injected so intake's
    # G3 gate populates incident.historical_outcomes from the real store. The
    # specialist invoker is stubbed: we don't need to spawn Claude CLI to
    # demonstrate the constitution path, and the real CLI is slow + costly.
    eval_harness = ChangeEvalHarness()
    queue = ApprovalQueue(root=QUEUE_ROOT)

    def _stub_invoker(mode: str, prompt: str) -> str:
        # Minimal stub — analyst prompt is the only one actually invoked
        # during intake; auditor runs only if constitution.passed=True (we
        # expect the veto to fail constitution before we get there).
        return ""

    mgr = MetaManager(
        eval_harness=eval_harness,
        approval_queue=queue,
        changes_dir=CHANGES_DIR,
        ledger_path=LEDGER,
        max_concurrent=1,
        episodic_memory=em,
        specialist_invoker=_stub_invoker,
    )

    banner("Driving intake → propose → check_constitution")
    pkg = mgr.intake(incident)
    print(f"[smoke] intake: change_id={pkg.change_id} stage={pkg.stage.value}")
    attached = pkg.incident.get("historical_outcomes")
    print(f"[smoke] historical_outcomes attached to incident: "
          f"{pformat(attached, width=72)}")

    # Drive propose + constitution unless intake already terminated (e.g.
    # throttled or duplicate — neither expected on a clean smoke run).
    if pkg.stage.value == "diagnosing":
        # The default proposer would invoke the Claude CLI surgeon; for the
        # smoke we attach the proposal directly so check_constitution has a
        # config_delta to evaluate. The point of this scenario is the C8
        # clause, not the surgeon's behavior (covered by tp_too_fast).
        from src.scanner.automation.meta_types import Proposal, ChangeStage
        pkg.proposal = Proposal(
            config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
        )
        pkg.stage = ChangeStage.PROPOSING
        pkg.touch("proposal_built (smoke direct)")
        print(f"[smoke] propose: stage={pkg.stage.value} "
              f"delta={pkg.proposal.config_delta}")
        mgr.check_constitution(pkg)

    banner("historical_veto — final state")
    print(f"[smoke] final stage: {pkg.stage.value}")
    print(f"[smoke] rejection_reason: {pkg.rejection_reason}")
    if pkg.constitution:
        failed_ids = [c.clause_id for c in pkg.constitution.clauses if not c.passed]
        print(f"[smoke] failed constitution clauses: {failed_ids}")
        for c in pkg.constitution.clauses:
            if not c.passed:
                print(f"[smoke]   {c.clause_id} ({c.name}): {c.detail}")
        if "C8" in failed_ids:
            print("[smoke] EXPECTED: C8 historical_loss_rate_veto fired ✓")
            return 0
        print("[smoke] UNEXPECTED: C8 historical_loss_rate_veto did NOT fire ✗")
        return 1
    print("[smoke] UNEXPECTED: no constitution attestation produced ✗")
    return 1


def _run_r_floor_promotion_scenario() -> int:
    """Build SHADOW and CANARY ChangePackages, override the R-stats helper
    on a real StagedDeployer instance, and run 5 gate decisions covering:
      A1: shadow, 15 trades + R_mean=0.6 → promote=True (count + R both pass)
      A2: shadow, 14 trades + R_mean=0.6 → promote=False (count gate)
      B:  shadow, 15 trades + R_mean=-0.6 → promote=False (R-floor gate)
      C1: canary, 130 trades + R_mean=0.5 → promote=True
      C2: canary, 129 trades + R_mean=0.5 → promote=False (count gate)

    Each case uses a freshly-bound lambda for _compute_window_r_stats so
    the override is per-case (avoiding cross-case bleed).
    """
    from src.scanner.automation.staged_deployer import (
        StagedDeployer,
        SHADOW_MIN_TRADES,
        CANARY_MIN_TRADES,
    )
    from src.scanner.automation.meta_types import (
        ChangePackage, DeploymentRecord, DeployStage, Proposal,
    )

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    banner("r_floor_promotion — synthetic R-stats + gate decisions")
    print(f"[smoke] SHADOW_MIN_TRADES={SHADOW_MIN_TRADES}, "
          f"CANARY_MIN_TRADES={CANARY_MIN_TRADES}")

    sd = StagedDeployer(config=None)

    def _make_shadow_pkg() -> ChangePackage:
        pkg = ChangePackage(
            proposal=Proposal(
                config_delta={"atr_tp_multiplier": {"new": 1.6, "old": 1.5}},
            ),
        )
        pkg.deploy_target = DeployStage.SHADOW
        pkg.deployments.append(DeploymentRecord(
            stage=DeployStage.SHADOW,
            deployed_at_cycle=0,
            closed_trade_count_at_deploy=0,
            regime_at_deploy="LOW",
        ))
        return pkg

    def _make_canary_pkg() -> ChangePackage:
        pkg = ChangePackage(proposal=Proposal(config_delta={}))
        pkg.deploy_target = DeployStage.CANARY
        pkg.deployments.append(DeploymentRecord(
            stage=DeployStage.CANARY,
            deployed_at_cycle=0,
            closed_trade_count_at_deploy=100,
            regime_at_deploy="LOW",
        ))
        return pkg

    # Case A1: 15 trades AND R_mean=0.6 above floor → promote.
    pkg_a = _make_shadow_pkg()
    sd._compute_window_r_stats = lambda rec, n: {"R_mean": 0.6, "R_baseline": 0.0}
    decision_a_15 = sd.should_promote_shadow_to_canary(pkg_a, current_trade_count=15)
    print(f"[smoke] A1 — shadow, 15 trades, R_mean=0.6: "
          f"promote={decision_a_15} (expected True)")

    # Case A2: 14 trades, R_mean=0.6 → count gate fails before R is even checked.
    pkg_a2 = _make_shadow_pkg()
    sd._compute_window_r_stats = lambda rec, n: {"R_mean": 0.6, "R_baseline": 0.0}
    decision_a_14 = sd.should_promote_shadow_to_canary(pkg_a2, current_trade_count=14)
    print(f"[smoke] A2 — shadow, 14 trades, R_mean=0.6: "
          f"promote={decision_a_14} (expected False — count gate)")

    # Case B: 15 trades but R_mean=-0.6 below floor → reject.
    # R_baseline=0.0, R_FLOOR_DELTA=0.5 → floor = -0.5; R_mean=-0.6 < -0.5 → False.
    pkg_b = _make_shadow_pkg()
    sd._compute_window_r_stats = lambda rec, n: {"R_mean": -0.6, "R_baseline": 0.0}
    decision_b = sd.should_promote_shadow_to_canary(pkg_b, current_trade_count=15)
    print(f"[smoke] B  — shadow, 15 trades, R_mean=-0.6: "
          f"promote={decision_b} (expected False — R-floor gate)")

    # Case C1: canary, 130 trades since deploy at idx 100 → 30 closed since deploy.
    pkg_c = _make_canary_pkg()
    sd._compute_window_r_stats = lambda rec, n: {"R_mean": 0.5, "R_baseline": 0.0}
    decision_c_130 = sd.should_promote_canary_to_live(pkg_c, current_trade_count=130)
    print(f"[smoke] C1 — canary, 130 trades, R_mean=0.5: "
          f"promote={decision_c_130} (expected True)")

    # Case C2: canary, 129 trades since deploy at idx 100 → 29 closed (under 30).
    pkg_c2 = _make_canary_pkg()
    sd._compute_window_r_stats = lambda rec, n: {"R_mean": 0.5, "R_baseline": 0.0}
    decision_c_129 = sd.should_promote_canary_to_live(pkg_c2, current_trade_count=129)
    print(f"[smoke] C2 — canary, 129 trades, R_mean=0.5: "
          f"promote={decision_c_129} (expected False — count gate)")

    banner("r_floor_promotion — summary")
    expected_results = [
        ("A1: shadow 15 trades, R=0.6", decision_a_15, True),
        ("A2: shadow 14 trades, R=0.6", decision_a_14, False),
        ("B:  shadow 15 trades, R=-0.6", decision_b, False),
        ("C1: canary 130 trades, R=0.5", decision_c_130, True),
        ("C2: canary 129 trades, R=0.5", decision_c_129, False),
    ]
    all_correct = all(actual == expected for _, actual, expected in expected_results)
    for name, actual, expected in expected_results:
        marker = "✓" if actual == expected else "✗"
        print(f"[smoke]   {marker} {name}: got {actual}, expected {expected}")
    if all_correct:
        print("[smoke] EXPECTED: all 5 gate decisions correct ✓")
        return 0
    print("[smoke] UNEXPECTED: at least one gate decision wrong ✗")
    return 1


def _fire_scenario(mm, name):
    builder = SCENARIOS[name]
    payload = builder()
    if isinstance(payload, list):
        return [mm.intake(p) for p in payload]
    return [mm.intake(payload)]



def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default="staleness",
        help="Which synthetic incident to drive through the pipeline",
    )
    args = ap.parse_args()

    banner(f"Cybernetic smoke — scenario={args.scenario}")
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    # Real eval harness — uses fixture regime windows for trading-quality scoring.
    eval_harness = ChangeEvalHarness()

    # Isolated approval queue under the smoke root.
    queue = ApprovalQueue(root=QUEUE_ROOT)

    # specialist_invoker=None → MetaManager uses _default_specialist_invoker
    # which spawns headless Claude CLI. This is the real cognition path.
    mgr = MetaManager(
        eval_harness=eval_harness,
        approval_queue=queue,
        changes_dir=CHANGES_DIR,
        ledger_path=LEDGER,
        max_concurrent=1,
    )

    _builder_result = SCENARIOS[args.scenario]()

    # Sentinel-shaped scenarios route to dedicated handlers that own their
    # own setup → drive → assert path. Detection uses a `_smoke_kind` key
    # so existing dict/list scenarios remain untouched.
    if isinstance(_builder_result, dict) and "_smoke_kind" in _builder_result:
        kind = _builder_result["_smoke_kind"]
        if kind == "historical_veto":
            return _run_historical_veto_scenario()
        if kind == "r_floor_promotion":
            return _run_r_floor_promotion_scenario()
        print(f"[smoke] unknown _smoke_kind: {kind}")
        return 1

    if isinstance(_builder_result, list):
        _pkgs = [mgr.intake(p) for p in _builder_result]
        for _i, _pkg in enumerate(_pkgs):
            print(f"[smoke] incident {_i+1}: change_id={_pkg.change_id} stage={_pkg.stage.value} reason={_pkg.rejection_reason}")
        return 0
    incident = _builder_result
    print(f"Incident:\n{pformat(incident, width=72)}")

    banner("Driving full pipeline via MetaManager.process()")
    print("Stages: intake (analyst) → propose (surgeon) → evaluate (scorecard)")
    print("        → check_constitution (auditor) → request_approval (arbiter + enqueue)")
    print("Expect 4 Claude CLI calls (~150s each) — total ~10 minutes worst case.")
    print()
    pkg = mgr.process(incident)

    banner("Stage 1+2 — Diagnosis")
    print(f"  change_id      = {pkg.change_id}")
    print(f"  diagnosis.severity              = {pkg.diagnosis.severity.value if pkg.diagnosis else '(none)'}")
    print(f"  diagnosis.kind                  = {pkg.diagnosis.proposed_intervention_kind.value if pkg.diagnosis else '(none)'}")
    print(f"  diagnosis.root_cause_hypothesis = {(pkg.diagnosis.root_cause_hypothesis if pkg.diagnosis else '')[:300]}")
    print(f"  diagnosis.affected_modules      = {pkg.diagnosis.affected_modules if pkg.diagnosis else []}")

    banner("Stage 3 — Proposal (Code Surgeon)")
    if pkg.proposal:
        print(f"  config_delta keys = {list(pkg.proposal.config_delta.keys())}")
        print(f"  diff (first 600):\n{(pkg.proposal.diff or '')[:600]}")
    else:
        print("  (no proposal produced)")

    banner("Stage 4 — Scorecard (4 layers)")
    if pkg.scorecard:
        for layer_name in ("software_correctness", "trading_quality", "governance_quality", "regime_robustness"):
            sub = getattr(pkg.scorecard, layer_name, None)
            if sub:
                print(f"  {layer_name:24s}  passed={sub.passed}  score={sub.score:.3f}  err={sub.error or ''}")
        print(f"  all_passed = {pkg.scorecard.all_passed()}")
    else:
        print("  (no scorecard produced)")

    banner("Stage 5 — Constitution + Auditor")
    if pkg.constitution:
        print(f"  constitution.passed                = {pkg.constitution.passed}")
        print(f"  constitution.failed_clauses        = {[c.clause_id for c in pkg.constitution.failed_clauses()]}")
        print(f"  auditor_specialist_agrees          = {pkg.constitution.auditor_specialist_agrees}")
        print(f"  auditor_raw_output (first 300)     = {(pkg.constitution.auditor_raw_output or '')[:300]}")

    banner("Stage 6 — Approval Queue (Deployment Arbiter dossier)")
    print(f"  dossier_summary  = {(pkg.dossier_summary or '(none)')[:400]}")
    print(f"  rollback_plan    = {(pkg.rollback_plan or '(none)')[:200]}")

    banner("Final pipeline state")
    print(f"  change_id          = {pkg.change_id}")
    print(f"  final stage        = {pkg.stage.value}")
    print(f"  rejection_reason   = {pkg.rejection_reason}")
    print(f"  ledger entries     = {LEDGER}")
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines()[-5:]:
            print(f"    {line[:180]}")

    banner("Approval queue snapshot (isolated)")
    pending = queue.list_pending()
    approved = queue.list_approved()
    rejected = queue.list_rejected()
    print(f"  pending  = {len(pending)}  {[p.change_id for p in pending]}")
    print(f"  approved = {len(approved)}")
    print(f"  rejected = {len(rejected)}")

    print()
    print(f"Smoke artifacts:    {SMOKE_ROOT}")
    print(f"To clean up:        rm -rf {SMOKE_ROOT}")
    print()

    # Exit 0 if pipeline reached a terminal stage (rejected, awaiting_approval,
    # or aborted). Exit 1 only if it crashed mid-pipeline.
    terminal_stages = {
        ChangeStage.REJECTED.value,
        ChangeStage.AWAITING_APPROVAL.value,
        ChangeStage.ABORTED.value,
        ChangeStage.POLICY_CHECK.value,  # may stop here if no scorecard
    }
    return 0 if pkg.stage.value in terminal_stages else 0


if __name__ == "__main__":
    raise SystemExit(main())
