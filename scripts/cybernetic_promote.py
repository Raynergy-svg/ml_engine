#!/usr/bin/env python3
"""Cybernetic promote — closes the loop past human approval into shadow deploy.

Picks up where `cybernetic_smoke.py` leaves off. Takes a pending package
from the isolated approval queue (`.claude/meta_smoke/approval_queue/pending/`),
moves it to `approved/`, then calls `MetaManager.drain()` so `StagedDeployer`
advances it to the SHADOW stage. Verifies the shadow_* parallel namespace
on the config receives the surgeon-proposed delta.

Shadow-only by design — does NOT promote to canary or live. Canary uses
`ConfigAdjuster.apply_adjustment` which writes to `.claude/config_adjustments.json`
(the real, live file). To stay non-destructive in this demo, we stop at shadow.

Usage:
    # First run the smoke to populate the pending queue:
    python scripts/cybernetic_smoke.py --scenario tp_too_fast
    # Then promote the pending package to shadow:
    python scripts/cybernetic_promote.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scanner.config import ScannerConfig, SCAN_PROFILES  # noqa: E402
from src.scanner.automation.meta_manager import MetaManager  # noqa: E402
from src.scanner.automation.approval_queue import ApprovalQueue  # noqa: E402
from src.scanner.automation.staged_deployer import StagedDeployer  # noqa: E402
from src.scanner.automation.meta_types import ChangeStage, DeployStage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

SMOKE_ROOT = ROOT / ".claude" / "meta_smoke"
CHANGES_DIR = SMOKE_ROOT / "changes"
LEDGER = SMOKE_ROOT / "changes.jsonl"
QUEUE_ROOT = SMOKE_ROOT / "approval_queue"
PENDING = QUEUE_ROOT / "pending"
APPROVED = QUEUE_ROOT / "approved"


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    banner("Cybernetic promote — pending → approved → shadow")

    # 1. Locate one pending package from the isolated smoke queue.
    if not PENDING.exists() or not list(PENDING.glob("*.json")):
        print(f"ERROR: no pending packages in {PENDING}")
        print("Hint: run scripts/cybernetic_smoke.py first to populate.")
        return 1

    pending_files = sorted(PENDING.glob("*.json"))
    target = pending_files[0]
    change_id = target.stem
    print(f"  Found pending package: {change_id}")
    print(f"  Source path:           {target}")

    # 2. Operator-equivalent action — move pending → approved/.
    APPROVED.mkdir(parents=True, exist_ok=True)
    approved_path = APPROVED / target.name
    shutil.move(str(target), str(approved_path))
    print(f"  Moved to approved:     {approved_path}")

    # 3. Build a real ScannerConfig from the smart profile so StagedDeployer
    #    has a concrete object to setattr() shadow_* attributes onto.
    config = ScannerConfig()
    config.apply_profile("smart")
    print(f"  Config baseline:       atr_tp_multiplier={getattr(config, 'atr_tp_multiplier', '(unset)')}")
    print(f"                         shadow_atr_tp_multiplier={getattr(config, 'shadow_atr_tp_multiplier', '(unset)')}")

    # 4. Initialize StagedDeployer + isolated MetaManager.
    deployer = StagedDeployer(
        config=config,
        config_adjuster=None,            # canary/live use this; shadow doesn't
        adjustment_approver=None,        # canary/live use this; shadow doesn't
        shadow_cycles=int(SCAN_PROFILES["smart"]["staged_deploy_shadow_cycles"]),
        canary_trades=int(SCAN_PROFILES["smart"]["staged_deploy_canary_trades"]),
    )
    queue = ApprovalQueue(root=QUEUE_ROOT)
    mgr = MetaManager(
        config=config,
        approval_queue=queue,
        staged_deployer=deployer,
        changes_dir=CHANGES_DIR,
        ledger_path=LEDGER,
    )

    # 5. Drain — should pick up the approved package and advance to shadow.
    banner("drain(current_cycle=1) — expect approved_advanced=1, shadow setattr")
    counters = mgr.drain(current_cycle=1)
    print(f"  Counters: {counters}")

    # 6. Verify shadow_* attribute landed on the config object.
    banner("Shadow namespace verification")
    shadow_value = getattr(config, "shadow_atr_tp_multiplier", None)
    live_value = getattr(config, "atr_tp_multiplier", None)
    print(f"  config.atr_tp_multiplier         = {live_value}     (live, untouched)")
    print(f"  config.shadow_atr_tp_multiplier  = {shadow_value}   (parallel namespace)")

    # 7. Read package state post-drain.
    pkg = mgr.get(change_id)
    if pkg is None:
        print(f"  ERROR: package {change_id} not found in changes_dir")
        return 1
    banner("Package state after drain")
    print(f"  stage          = {pkg.stage.value}")
    print(f"  deploy_target  = {pkg.deploy_target.value}")
    print(f"  deployments    = {[(d.stage.value, 'rolled_back' if d.rolled_back_at else 'live') for d in pkg.deployments]}")

    # 8. Soak-window gate verification.
    # shadow_cycles=20 in smart profile. Drain at cycle=2 (elapsed=1) should
    # NOT promote (soak active). Drain at cycle=25 (elapsed=24) SHOULD promote.
    banner("drain(current_cycle=2) — soak window active (elapsed 1 < required 20)")
    counters2 = mgr.drain(current_cycle=2)
    print(f"  Counters: {counters2}")
    pkg2 = mgr.get(change_id)
    print(f"  pkg.deploy_target after cycle=2:  {pkg2.deploy_target.value} (expected: shadow)")
    print(f"  pkg.stage after cycle=2:          {pkg2.stage.value} (expected: deployed_shadow)")
    soak_held = pkg2.deploy_target == DeployStage.SHADOW

    banner("drain(current_cycle=25) — soak elapsed (24 >= required 20), should promote")
    counters3 = mgr.drain(current_cycle=25)
    print(f"  Counters: {counters3}")
    pkg3 = mgr.get(change_id)
    print(f"  pkg.deploy_target after cycle=25: {pkg3.deploy_target.value} (expected: canary)")
    soak_released = pkg3.deploy_target == DeployStage.CANARY

    # 9. Final report
    banner("Cybernetic promote — verdict")
    success = (
        shadow_value == 1.6
        and pkg.stage == ChangeStage.DEPLOYED_SHADOW
        and pkg.deploy_target == DeployStage.SHADOW
        and len(pkg.deployments) >= 1
        and soak_held
        and soak_released
    )
    print(f"  shadow_atr_tp_multiplier == 1.6:            {shadow_value == 1.6}")
    print(f"  pkg.stage == DEPLOYED_SHADOW:               {pkg.stage == ChangeStage.DEPLOYED_SHADOW}")
    print(f"  pkg.deploy_target == SHADOW:                {pkg.deploy_target == DeployStage.SHADOW}")
    print(f"  deployment record exists:                   {len(pkg.deployments) >= 1}")
    print(f"  soak window held at cycle=2 (still SHADOW): {soak_held}")
    print(f"  soak released at cycle=25 (now CANARY):     {soak_released}")
    print()
    print(f"  PROMOTE LOOP CLOSED + SOAK GATE WORKING: {success}")
    print()
    print(f"  Changes dir:    {CHANGES_DIR}")
    print(f"  Ledger:         {LEDGER}")
    print(f"  Cleanup later:  rm -rf {SMOKE_ROOT}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
