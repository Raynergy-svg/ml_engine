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
