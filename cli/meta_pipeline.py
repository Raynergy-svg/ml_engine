"""CLI for the meta-cybernetic change pipeline.

Usage:
    python -m cli.meta_pipeline list [--state pending|approved|rejected|all]
    python -m cli.meta_pipeline show <change_id>
    python -m cli.meta_pipeline approve <change_id> [--target shadow|canary|live] [--reason ...]
    python -m cli.meta_pipeline reject <change_id> --reason "..."
    python -m cli.meta_pipeline test-eval --change-id <id>

Designed for one-shot human use. Prints to stdout. Exit code 0 on success,
1 on bad input or missing change_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# When invoked as `python cli/meta_pipeline.py`, the project root isn't on
# sys.path. Insert it so `from src...` works without forcing the user to
# import the heavy cli package via `python -m`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scanner.automation.approval_queue import ApprovalQueue
from src.scanner.automation.meta_manager import MetaManager
from src.scanner.automation.meta_types import ChangePackage, ChangeStage, DeployStage


def _print_header(pkg: ChangePackage) -> None:
    print(f"=== Change {pkg.change_id} ===")
    print(f"  stage:        {pkg.stage.value}")
    print(f"  kind:         {pkg.kind.value}")
    print(f"  deploy_target:{pkg.deploy_target.value}")
    print(f"  created_at:   {pkg.created_at}")
    print(f"  updated_at:   {pkg.updated_at}")
    if pkg.diagnosis:
        print(f"  severity:     {pkg.diagnosis.severity.value}")
        print(f"  root_cause:   {pkg.diagnosis.root_cause_hypothesis}")
    if pkg.dossier_summary:
        print("\n--- dossier ---")
        print(pkg.dossier_summary)


def _cmd_list(args) -> int:
    queue = ApprovalQueue()
    state = args.state
    buckets = {
        "pending": queue.list_pending,
        "approved": queue.list_approved,
        "rejected": queue.list_rejected,
    }
    targets = list(buckets.keys()) if state == "all" else [state]
    found = 0
    for t in targets:
        pkgs = buckets[t]()
        if not pkgs:
            continue
        print(f"[{t}]  {len(pkgs)} package(s)")
        for p in pkgs:
            sev = p.diagnosis.severity.value if p.diagnosis else "-"
            print(f"  {p.change_id}  kind={p.kind.value:<6} target={p.deploy_target.value:<6} sev={sev:<4} {p.created_at}")
            if p.dossier_summary:
                first_line = p.dossier_summary.splitlines()[0][:100]
                print(f"      → {first_line}")
        found += len(pkgs)
    if found == 0:
        print("(no packages)")
    return 0


def _cmd_show(args) -> int:
    queue = ApprovalQueue()
    pkg = queue.get(args.change_id)
    if pkg is None:
        # Try the meta_manager changes dir as fallback
        mgr = MetaManager()
        pkg = mgr.get(args.change_id)
    if pkg is None:
        print(f"error: no package with id {args.change_id}", file=sys.stderr)
        return 1
    _print_header(pkg)
    if pkg.scorecard:
        print("\n--- scorecard ---")
        print(json.dumps(pkg.scorecard.to_dict(), indent=2, default=str))
    if pkg.constitution:
        print("\n--- constitution ---")
        print(json.dumps(pkg.constitution.to_dict(), indent=2, default=str))
    if pkg.proposal:
        print("\n--- proposal ---")
        print(f"  changed_files: {pkg.proposal.changed_files}")
        if pkg.proposal.diff:
            diff_preview = "\n".join(pkg.proposal.diff.splitlines()[:200])
            print(diff_preview)
    return 0


def _cmd_approve(args) -> int:
    queue = ApprovalQueue()
    try:
        target = DeployStage(args.target)
    except ValueError:
        print(f"error: invalid --target {args.target}", file=sys.stderr)
        return 1
    pkg = queue.approve(args.change_id, deploy_target=target, reason=args.reason or "")
    if pkg is None:
        print(f"error: no pending package with id {args.change_id}", file=sys.stderr)
        return 1
    print(f"approved {pkg.change_id} target={target.value}")
    return 0


def _cmd_reject(args) -> int:
    queue = ApprovalQueue()
    pkg = queue.reject(args.change_id, reason=args.reason)
    if pkg is None:
        print(f"error: no pending package with id {args.change_id}", file=sys.stderr)
        return 1
    print(f"rejected {pkg.change_id} reason={args.reason}")
    return 0


def _cmd_test_eval(args) -> int:
    """Smoke test the eval harness against an existing package or a synthetic one."""
    mgr = MetaManager()
    pkg = mgr.get(args.change_id) if args.change_id else None
    if pkg is None:
        print(f"warning: no package with id {args.change_id}; using a synthetic one")
        from src.scanner.automation.meta_types import Proposal, ChangeKind
        pkg = ChangePackage(kind=ChangeKind.CONFIG)
        pkg.proposal = Proposal(config_delta={"min_confidence": {"old": 0.5, "new": 0.55}})
    if mgr._eval is None:  # type: ignore[attr-defined]
        from src.scanner.automation.change_eval import ChangeEvalHarness
        try:
            from src.scanner.automation.replay_validator import ReplayValidator
            from src.scanner.automation.qa_pipeline import QAPipeline
            harness = ChangeEvalHarness(
                replay_validator=ReplayValidator(),
                qa_pipeline=QAPipeline(),
            )
        except Exception:
            harness = ChangeEvalHarness()
        card = harness.score(pkg)
    else:
        card = mgr._eval.score(pkg)  # type: ignore[attr-defined]
    print(json.dumps(card.to_dict(), indent=2, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meta_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--state", choices=["pending", "approved", "rejected", "all"], default="pending")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show")
    p_show.add_argument("change_id")
    p_show.set_defaults(func=_cmd_show)

    p_app = sub.add_parser("approve")
    p_app.add_argument("change_id")
    p_app.add_argument("--target", default="shadow")
    p_app.add_argument("--reason", default="")
    p_app.set_defaults(func=_cmd_approve)

    p_rej = sub.add_parser("reject")
    p_rej.add_argument("change_id")
    p_rej.add_argument("--reason", required=True)
    p_rej.set_defaults(func=_cmd_reject)

    p_test = sub.add_parser("test-eval")
    p_test.add_argument("--change-id", default=None)
    p_test.set_defaults(func=_cmd_test_eval)

    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
