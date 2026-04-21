"""
PRD Agent Chain — Event-driven pipeline that does REAL WORK:

  PRD Watcher (filesystem)
      ↓ detects all stories pass
  Gap Wirer Agent
      ↓ generates a NEW prd.json from detected gaps
  Ralph (claude/amp CLI)
      ↓ implements the gap stories autonomously
  Code Reviewer Agent (spawns claude --print)
      ↓ reviews the changes, generates follow-up PRD if blockers found
  Ralph (again, if follow-up exists)
      ↓ fixes the blockers
  Chain Complete → state.json updated

This is NOT a linter. Each step spawns real Claude agent instances
that write code, create files, and modify the codebase.

Phase 28 (US-172): PRD agent chain orchestrator.
"""

import json
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.background_activity import get_background_activity_tracker
from src.scanner.automation.event_bus import (
    Event, PrdEventBus as EventBus, EventResult, EventType, get_prd_event_bus as get_event_bus,
)
from src.scanner.automation.prd_watcher import PRDWatcher, PRDCompletionChecker
from src.scanner.automation.gap_wirer import GapWirer
from src.scanner.automation.code_reviewer import CodeReviewer
from src.scanner.automation.chain_memory import get_chain_memory

logger = logging.getLogger(__name__)


class PRDAgentChain:
    """
    Top-level orchestrator that chains real agent work:

      1. PRDWatcher detects PRD completion (filesystem event)
      2. GapWirer analyzes gaps, generates a new prd.json
      3. Ralph (claude/amp) implements the generated stories
      4. CodeReviewer spawns Claude to review the changes
      5. If blockers found → generates follow-up PRD → Ralph again
      6. Chain completion → state.json + report

    The key: steps 3 and 4 spawn real CLI agent processes that do work.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        ralph_dir: Optional[Path] = None,
        event_bus: Optional[EventBus] = None,
        poll_interval_secs: float = 10.0,
        ralph_tool: str = "claude",
        ralph_max_iterations: int = 10,
        auto_spawn_ralph: bool = True,
        auto_review: bool = True,
        auto_fix_blockers: bool = True,
    ):
        self._root = project_root or Path(".")
        self._ralph_dir = ralph_dir or self._root / ".claude" / "ralph"
        self._bus = event_bus or get_event_bus(
            log_path=self._ralph_dir / "event_log.jsonl"
        )
        self._poll_interval = poll_interval_secs
        self._ralph_tool = ralph_tool
        self._ralph_max_iters = ralph_max_iterations
        self._auto_spawn = auto_spawn_ralph
        self._auto_review = auto_review
        self._auto_fix = auto_fix_blockers
        self._max_review_cycles = 3  # Hard cap: review→fix→re-review max iterations
        self._current_review_cycle = 0  # Tracks how many review→fix loops we've done

        self._report_dir = self._ralph_dir / "reports"
        self._report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize agents
        self._gap_wirer = GapWirer(
            project_root=self._root,
            event_bus=self._bus,
            auto_spawn_ralph=False,  # Chain controls Ralph spawning
            ralph_tool=self._ralph_tool,
            ralph_max_iterations=self._ralph_max_iters,
        )
        self._code_reviewer = CodeReviewer(
            project_root=self._root,
            event_bus=self._bus,
            auto_generate_followup=True,
        )
        self._watcher = PRDWatcher(
            ralph_dir=self._ralph_dir,
            event_bus=self._bus,
            poll_interval_secs=self._poll_interval,
        )

        # Chain state
        self._chain_runs: List[dict] = []
        self._memory = get_chain_memory(project_root=self._root)

        # Wire the event chain
        self._register_handlers()

    def _register_handlers(self) -> None:
        """
        Wire the event chain. Handler execution order is controlled by priority.

        PRD_COMPLETE (priority 10): gap_wirer.handle_prd_complete
            → generates gaps, writes new prd.json, emits GAPS_IDENTIFIED

        GAPS_IDENTIFIED (priority 10): chain._handle_gaps_identified
            → spawns Ralph to implement the gap PRD
            → waits for Ralph to finish
            → triggers code review

        REVIEW_COMPLETE (priority 10): chain._handle_review_complete
            → if blockers + follow-up PRD exists, spawns Ralph again
            → persists report, updates state.json
        """
        # Step 1: Gap wirer generates PRD from gaps
        self._gap_wirer.register(self._bus)

        # Step 2: Chain spawns Ralph when gaps are identified
        self._bus.subscribe(
            EventType.GAPS_IDENTIFIED,
            self._handle_gaps_identified,
            name="chain_ralph_spawner",
            priority=20,  # After gap wirer writes the PRD
        )

        # Step 3: Chain handles review completion
        self._bus.subscribe(
            EventType.REVIEW_COMPLETE,
            self._handle_review_complete,
            name="chain_finalizer",
            priority=10,
        )

        handler_count = self._bus.handler_count()
        logger.info(f"PRDAgentChain: registered {handler_count} handlers")

    # ─── Event Handlers (the real work happens here) ────────────────

    def _handle_gaps_identified(self, event: Event) -> EventResult:
        """
        After gap wirer generates a PRD, spawn Ralph to implement it,
        then trigger code review on the results.
        """
        start = time.monotonic()
        # New chain run — reset the review cycle counter
        self._current_review_cycle = 0
        gap_data = event.payload
        generated_prd = gap_data.get("generated_prd")
        total_gaps = gap_data.get("total_gaps", 0)

        if not generated_prd or total_gaps == 0:
            logger.info("PRDAgentChain: no gaps to implement — skipping Ralph")
            return EventResult(success=True, handler_name="chain_ralph_spawner")

        try:
            tracker = get_background_activity_tracker()
            # Memory: record chain start + sync PRD state
            phase = gap_data.get("branch_name", gap_data.get("phase", "unknown"))
            self._memory.record_chain_start(phase=phase, gaps=total_gaps)
            self._memory.sync_from_prd()
            self._active_chain_activity_id = tracker.start_activity(
                kind="agent_chain",
                title=f"PRD agent chain for {phase}",
                source="prd_agent_chain",
                metadata={"phase": phase, "gaps": total_gaps, "generated_prd": generated_prd},
            )
            tracker.append_event(self._active_chain_activity_id, f"Detected {total_gaps} gap(s) and started chain")

            # Step 2a: Spawn Ralph to implement the gap PRD
            if self._auto_spawn:
                logger.info(f"🚀 PRDAgentChain: spawning Ralph for {total_gaps} gaps")
                self._memory.set_flag("ralph_in_progress", True)
                tracker.append_event(self._active_chain_activity_id, "Spawning Ralph for gap implementation")
                ralph_success = self._run_ralph(generated_prd)
                self._memory.set_flag("ralph_in_progress", False)
                self._memory.sync_from_prd()  # Refresh after Ralph finishes
                logger.info(
                    f"{'✓' if ralph_success else '✗'} Ralph finished "
                    f"({'success' if ralph_success else 'incomplete'})"
                )
            else:
                logger.info(
                    f"PRDAgentChain: auto_spawn=False — PRD ready at {generated_prd}. "
                    f"Run: ./scripts/ralph.sh --tool {self._ralph_tool}"
                )

            # Step 2b: Trigger code review on the implemented changes
            if self._auto_review:
                self._memory.set_flag("review_in_progress", True)
                logger.info("🔍 PRDAgentChain: spawning code review agent")
                tracker.append_event(self._active_chain_activity_id, "Starting code review")
                review_result = self._code_reviewer.review_prd_changes(
                    generated_prd, trigger="chain_post_ralph"
                )
                # review_complete event is emitted by the reviewer itself

            return EventResult(
                success=True,
                handler_name="chain_ralph_spawner",
                duration_secs=time.monotonic() - start,
                output={
                    "ralph_spawned": self._auto_spawn,
                    "review_triggered": self._auto_review,
                    "gaps": total_gaps,
                },
            )
        except Exception as e:
            logger.error(f"PRDAgentChain: Ralph/review error: {e}")
            if getattr(self, "_active_chain_activity_id", None):
                tracker = get_background_activity_tracker()
                tracker.fail_activity(
                    self._active_chain_activity_id,
                    error=str(e),
                    summary="PRD agent chain failed before review completion",
                )
            return EventResult(
                success=False,
                handler_name="chain_ralph_spawner",
                duration_secs=time.monotonic() - start,
                error=str(e),
            )

    def _handle_review_complete(self, event: Event) -> EventResult:
        """
        After code review completes:
        - If PASSED → finalize chain, reset watcher for next PRD
        - If FAILED + follow-up PRD → spawn Ralph to fix → re-review
        - Re-review loop capped at max_review_cycles to prevent infinite loops
        """
        start = time.monotonic()
        review_data = event.payload

        try:
            tracker = get_background_activity_tracker()
            passed = review_data.get("passed", True)
            followup_prd = review_data.get("followup_prd")
            blockers = review_data.get("blockers", 0)

            # ── Review→Fix→Re-review loop (with hard cap) ──────────
            if not passed and followup_prd and self._auto_fix:
                self._current_review_cycle += 1

                if self._current_review_cycle > self._max_review_cycles:
                    logger.warning(
                        f"⚠️ PRDAgentChain: hit max review cycles ({self._max_review_cycles}). "
                        f"Still {blockers} blockers but stopping to prevent infinite loop. "
                        f"Follow-up PRD left at {followup_prd} for manual resolution."
                    )
                    # Fall through to finalization — don't loop forever
                else:
                    logger.info(
                        f"🔧 PRDAgentChain: review cycle {self._current_review_cycle}/{self._max_review_cycles} — "
                        f"{blockers} blockers — spawning Ralph to fix via {followup_prd}"
                    )
                    if getattr(self, "_active_chain_activity_id", None):
                        tracker.append_event(
                            self._active_chain_activity_id,
                            f"Review found {blockers} blocker(s); spawning Ralph fix cycle {self._current_review_cycle}",
                        )
                    ralph_success = self._run_ralph(followup_prd)

                    if ralph_success and self._auto_review:
                        # Re-review the fixed code — this is the closed loop
                        logger.info(
                            f"🔍 PRDAgentChain: re-reviewing after fix cycle {self._current_review_cycle}"
                        )
                        if getattr(self, "_active_chain_activity_id", None):
                            tracker.append_event(
                                self._active_chain_activity_id,
                                f"Fix cycle {self._current_review_cycle} complete; re-reviewing",
                            )
                        re_review = self._code_reviewer.review_prd_changes(
                            followup_prd, trigger=f"fix_cycle_{self._current_review_cycle}"
                        )
                        # The reviewer emits REVIEW_COMPLETE which re-enters this handler.
                        # Since we incremented the cycle counter, infinite loops are capped.
                        # We return here — the re-emitted event will handle finalization.
                        return EventResult(
                            success=True,
                            handler_name="chain_finalizer",
                            duration_secs=time.monotonic() - start,
                            output={
                                "action": "re_review_triggered",
                                "cycle": self._current_review_cycle,
                                "ralph_success": ralph_success,
                            },
                        )
                    elif not ralph_success:
                        logger.warning(
                            f"⚠️ PRDAgentChain: Ralph failed to fix blockers on cycle "
                            f"{self._current_review_cycle} — finalizing with blockers"
                        )

            # ── Finalization (reached when PASSED or max cycles hit) ──
            # Reset cycle counter for the next chain run
            cycles_used = self._current_review_cycle
            self._current_review_cycle = 0

            # Build chain completion record
            completion = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "review_passed": passed,
                "blockers": blockers,
                "suggestions": review_data.get("suggestions", 0),
                "followup_generated": followup_prd is not None,
                "followup_ralph_spawned": not passed and followup_prd is not None and self._auto_fix,
                "review_cycles_used": cycles_used,
                "files_reviewed": review_data.get("files_reviewed", []),
                "agent_spawned": review_data.get("agent_spawned", False),
            }
            self._chain_runs.append(completion)

            # Persist report
            self._persist_report(completion, review_data)

            # Update state.json
            self._update_state(completion)

            # Update memory — record chain complete + sync PRD progress
            self._memory.record_chain_complete(completion)
            prd_status = self._memory.sync_from_prd()
            if prd_status.get("all_complete"):
                self._memory.record_phase_complete(
                    phase_name=prd_status.get("phase", "unknown"),
                    stories_count=prd_status.get("total_stories", 0),
                )

            status = "✓ PASSED" if passed else f"✗ {blockers} BLOCKERS (max cycles reached)"
            logger.info(f"🏁 CHAIN COMPLETE: review {status} | memory: {self._memory.summary()}")
            if getattr(self, "_active_chain_activity_id", None):
                final_summary = "PRD agent chain completed cleanly" if passed else f"PRD agent chain stopped with {blockers} blocker(s)"
                if passed:
                    tracker.complete_activity(self._active_chain_activity_id, summary=final_summary, metadata={"blockers": blockers})
                else:
                    tracker.fail_activity(self._active_chain_activity_id, error=f"{blockers} blocker(s) remain", summary=final_summary)

            # Re-entry: reset watcher fired state so the next poll picks up
            # PRDs that Ralph just marked as complete. This closes the outer loop:
            # PRD completes → chain runs → Ralph implements gap PRD → watcher
            # sees gap PRD now complete → chain runs again with fresh gaps.
            self._watcher.reset_fired()
            logger.info("🔄 PRDAgentChain: watcher reset — next poll will re-scan all PRDs")

            self._bus.emit(Event(
                event_type=EventType.CHAIN_COMPLETE,
                source="prd_agent_chain",
                payload=completion,
            ))

            return EventResult(
                success=True,
                handler_name="chain_finalizer",
                duration_secs=time.monotonic() - start,
                output=completion,
            )
        except Exception as e:
            logger.error(f"PRDAgentChain: finalization error: {e}")
            self._current_review_cycle = 0  # Reset on error to not corrupt state
            if getattr(self, "_active_chain_activity_id", None):
                tracker = get_background_activity_tracker()
                tracker.fail_activity(self._active_chain_activity_id, error=str(e), summary="PRD agent chain finalization failed")
            return EventResult(
                success=False,
                handler_name="chain_finalizer",
                duration_secs=time.monotonic() - start,
                error=str(e),
            )

    # ─── Ralph Execution ────────────────────────────────────────────

    def _run_ralph(self, prd_path: str) -> bool:
        """
        Run Ralph synchronously to implement a PRD.
        Returns True if Ralph reports completion.
        """
        ralph_script = self._root / "scripts" / "ralph.sh"

        if not ralph_script.exists():
            # Fallback: run claude directly with the PRD as context
            return self._run_claude_direct(prd_path)

        logger.info(f"Running Ralph: {ralph_script} --tool {self._ralph_tool} {self._ralph_max_iters}")
        tracker = get_background_activity_tracker()
        activity_id = getattr(self, "_active_chain_activity_id", None)

        try:
            proc = subprocess.run(
                [str(ralph_script), "--tool", self._ralph_tool, str(self._ralph_max_iters)],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._ralph_max_iters * 180,  # ~3min per iteration
            )
            success = proc.returncode == 0
            if "<promise>COMPLETE</promise>" in (proc.stdout or ""):
                success = True
            if activity_id:
                tracker.append_event(
                    activity_id,
                    "Ralph subprocess completed" if success else "Ralph subprocess returned non-zero",
                    details={"returncode": proc.returncode},
                )
            return success
        except subprocess.TimeoutExpired:
            logger.warning("PRDAgentChain: Ralph timed out")
            if activity_id:
                tracker.append_event(activity_id, "Ralph subprocess timed out", level="warning")
            return False
        except Exception as e:
            logger.error(f"PRDAgentChain: Ralph execution error: {e}")
            if activity_id:
                tracker.append_event(activity_id, f"Ralph execution error: {e}", level="error")
            return False

    def _run_claude_direct(self, prd_path: str) -> bool:
        """
        Fallback: run Claude CLI directly with the PRD content as prompt.
        Used when ralph.sh is not available.
        """
        try:
            prd_content = Path(prd_path).read_text()
            prompt = (
                f"You are Ralph, an autonomous development agent. "
                f"Implement ALL user stories in this PRD. "
                f"For each story, implement the code changes described in the acceptance criteria. "
                f"Mark each story as passes: true in the PRD when done. "
                f"Work through them in priority order.\n\n"
                f"PRD:\n{prd_content}"
            )

            proc = subprocess.run(
                ["claude", "--dangerously-skip-permissions", "--print", "-p", prompt],
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"PRDAgentChain: direct Claude execution failed: {e}")
            return False

    # ─── Persistence ────────────────────────────────────────────────

    def _persist_report(self, completion: dict, review_data: dict) -> None:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = self._report_dir / f"chain_{ts}.json"
            report_file.write_text(json.dumps({
                "summary": completion,
                "review_detail": review_data,
            }, indent=2))
            logger.info(f"PRDAgentChain: report → {report_file}")
        except Exception as e:
            logger.error(f"PRDAgentChain: report persist error: {e}")

    def _update_state(self, completion: dict) -> None:
        state_file = self._root / ".claude" / "state.json"
        try:
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            state["last_chain_completion"] = {
                "timestamp": completion["timestamp"],
                "review_passed": completion["review_passed"],
                "blockers": completion["blockers"],
                "followup_generated": completion["followup_generated"],
            }
            state["last_updated"] = datetime.now(timezone.utc).isoformat()
            state_file.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"PRDAgentChain: state update error: {e}")

    # ─── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the filesystem watcher."""
        logger.info("PRDAgentChain: starting watcher...")
        self._watcher.start()

    def stop(self) -> None:
        self._watcher.stop()
        logger.info("PRDAgentChain: stopped")

    def scan_now(self) -> List[dict]:
        """One-shot scan of all PRDs."""
        return [s.__dict__ for s in self._watcher.scan_all()]

    def run_manual(self, prd_filepath: str) -> dict:
        """
        Manually trigger the full chain for a specific completed PRD.
        """
        logger.info(f"PRDAgentChain: manual run for {prd_filepath}")

        status = PRDCompletionChecker.check_file(Path(prd_filepath))
        if not status.is_complete:
            return {
                "error": "PRD not complete",
                "progress": f"{status.passed_stories}/{status.total_stories}",
                "pending": status.failed_story_ids,
            }

        # Fire the event — the chain handles everything
        event = Event(
            event_type=EventType.PRD_COMPLETE,
            source="manual_trigger",
            payload={
                "filepath": prd_filepath,
                "project": status.project,
                "branch_name": status.branch_name,
                "description": status.description,
                "total_stories": status.total_stories,
                "passed_stories": status.passed_stories,
                "completed_at": status.completed_at,
            },
        )
        results = self._bus.emit(event)

        return {
            "prd": prd_filepath,
            "branch": status.branch_name,
            "stories": status.total_stories,
            "event_results": [
                {"handler": r.handler_name, "success": r.success,
                 "duration": round(r.duration_secs, 3), "error": r.error}
                for r in results
            ],
            "chain_runs": self._chain_runs,
        }

    def get_status(self) -> dict:
        return {
            "watcher": self._watcher.get_status(),
            "config": {
                "ralph_tool": self._ralph_tool,
                "ralph_max_iterations": self._ralph_max_iters,
                "auto_spawn_ralph": self._auto_spawn,
                "auto_review": self._auto_review,
                "auto_fix_blockers": self._auto_fix,
            },
            "chain_runs": len(self._chain_runs),
            "last_run": self._chain_runs[-1] if self._chain_runs else None,
        }

    def run_daemon(self) -> None:
        """Run as a long-lived daemon."""
        def _shutdown(signum, frame):
            logger.info(f"PRDAgentChain: signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self.start()
        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            self.stop()


# ─── CLI Entry Point ────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PRD Agent Chain — Watch, Wire, Review, Fix")
    parser.add_argument("--mode", choices=["watch", "scan", "manual", "status"],
                        default="watch")
    parser.add_argument("--prd", type=str, default=None)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--ralph-tool", choices=["amp", "claude"], default="claude")
    parser.add_argument("--ralph-iters", type=int, default=10)
    parser.add_argument("--no-spawn", action="store_true",
                        help="Don't auto-spawn Ralph (just generate PRDs)")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip code review step")
    parser.add_argument("--no-fix", action="store_true",
                        help="Don't auto-fix blockers from review")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = Path(args.project_root).resolve()
    chain = PRDAgentChain(
        project_root=root,
        poll_interval_secs=args.interval,
        ralph_tool=args.ralph_tool,
        ralph_max_iterations=args.ralph_iters,
        auto_spawn_ralph=not args.no_spawn,
        auto_review=not args.no_review,
        auto_fix_blockers=not args.no_fix,
    )

    if args.mode == "watch":
        print(f"🔍 PRD Agent Chain — watching {root / '.claude' / 'ralph'}")
        print(f"   Ralph: {args.ralph_tool} (max {args.ralph_iters} iters)")
        print(f"   Auto-spawn: {'YES' if not args.no_spawn else 'NO'}")
        print(f"   Auto-review: {'YES' if not args.no_review else 'NO'}")
        print(f"   Auto-fix: {'YES' if not args.no_fix else 'NO'}")
        print(f"   Chain: PRD Complete → Gap PRD → Ralph → Review → Fix")
        print(f"   Ctrl+C to stop\n")
        chain.run_daemon()

    elif args.mode == "scan":
        for s in chain.scan_now():
            c = "✓" if s.get("is_complete") else f"⏳ {s.get('passed_stories',0)}/{s.get('total_stories',0)}"
            print(f"  {Path(s.get('filepath','')).name}: {c}")

    elif args.mode == "manual":
        if not args.prd:
            print("Error: --prd required for manual mode")
            sys.exit(1)
        result = chain.run_manual(args.prd)
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "status":
        print(json.dumps(chain.get_status(), indent=2, default=str))


if __name__ == "__main__":
    main()
