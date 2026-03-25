"""
Code Review Agent — Spawns a real Claude agent to review code changes
from a completed PRD/Ralph run, then generates follow-up stories for
any issues found.

Flow:
  1. Receive GAPS_IDENTIFIED or CHAIN_COMPLETE event
  2. Determine which files were changed (git diff or PRD module list)
  3. Spawn `claude` CLI with the code-reviewer agent personality
  4. Parse the review output for blockers/suggestions
  5. If blockers found: generate a follow-up PRD with fix stories
  6. Emit review_complete event with results

Phase 28 (US-171): Code review agent that spawns real Claude instances.
"""

import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.event_bus import Event, EventBus, EventResult, EventType, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result from a Claude code review agent run."""
    trigger: str = ""
    files_reviewed: List[str] = field(default_factory=list)
    raw_output: str = ""
    blockers: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    nits: List[str] = field(default_factory=list)
    passed: bool = True
    followup_prd_path: Optional[str] = None
    agent_spawned: bool = False
    duration_secs: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "files_reviewed": self.files_reviewed,
            "blockers": len(self.blockers),
            "suggestions": len(self.suggestions),
            "nits": len(self.nits),
            "passed": self.passed,
            "followup_prd": self.followup_prd_path,
            "agent_spawned": self.agent_spawned,
            "duration_secs": round(self.duration_secs, 2),
            "error": self.error,
            "blocker_details": self.blockers[:10],  # Cap for payload size
            "suggestion_details": self.suggestions[:10],
        }


class CodeReviewer:
    """
    Spawns a real Claude agent with the code-reviewer personality to review
    files touched by a completed PRD. If blockers are found, generates a
    follow-up PRD for Ralph to fix them.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        event_bus: Optional[EventBus] = None,
        auto_generate_followup: bool = True,
        review_timeout_secs: int = 300,
    ):
        self._root = project_root or Path(".")
        self._bus = event_bus or get_event_bus()
        self._auto_followup = auto_generate_followup
        self._timeout = review_timeout_secs

        self._agents_dir = self._root / ".claude" / "agents"
        self._ralph_dir = self._root / ".claude" / "ralph"
        self._reviewer_personality = self._agents_dir / "engineering-code-reviewer.md"
        self._automation_dir = self._root / "src" / "scanner" / "automation"

    def review_prd_changes(self, prd_filepath: str, trigger: str = "prd_complete") -> ReviewResult:
        """
        Review files referenced in a completed PRD by spawning a Claude agent.
        """
        start = time.monotonic()
        result = ReviewResult(trigger=trigger)

        try:
            # 1. Determine files to review
            files = self._extract_files_from_prd(Path(prd_filepath))
            if not files:
                # Fall back to git diff if no files found in PRD
                files = self._get_recent_changed_files()

            result.files_reviewed = [str(f) for f in files]

            if not files:
                logger.info("CodeReviewer: no files to review")
                result.duration_secs = time.monotonic() - start
                return result

            # 2. Spawn Claude with code-reviewer personality
            review_output = self._spawn_review_agent(files, prd_filepath)
            result.raw_output = review_output
            result.agent_spawned = True

            # 3. Parse the output for findings
            result.blockers = self._extract_findings(review_output, "🔴")
            result.suggestions = self._extract_findings(review_output, "🟡")
            result.nits = self._extract_findings(review_output, "💭")
            result.passed = len(result.blockers) == 0

            logger.info(
                f"CodeReviewer: {len(result.blockers)} blockers, "
                f"{len(result.suggestions)} suggestions, {len(result.nits)} nits → "
                f"{'PASSED' if result.passed else 'FAILED'}"
            )

            # 4. Generate follow-up PRD if blockers found
            if not result.passed and self._auto_followup:
                followup_path = self._generate_followup_prd(
                    result.blockers, result.suggestions, prd_filepath
                )
                result.followup_prd_path = str(followup_path) if followup_path else None

        except Exception as e:
            result.error = str(e)
            logger.error(f"CodeReviewer: failed: {e}")

        result.duration_secs = time.monotonic() - start

        self._bus.emit(Event(
            event_type=EventType.REVIEW_COMPLETE,
            source="code_reviewer",
            payload=result.to_dict(),
        ))

        return result

    # ─── Agent Spawning ─────────────────────────────────────────────

    def _spawn_review_agent(self, files: List[Path], prd_filepath: str) -> str:
        """
        Spawn a Claude CLI agent with the code-reviewer personality.
        Pipes in a review prompt and captures the output.
        """
        # Load the code reviewer personality
        personality = ""
        if self._reviewer_personality.exists():
            personality = self._reviewer_personality.read_text()

        # Build file content for review
        file_contents = []
        for f in files[:15]:  # Cap at 15 files to stay in context window
            try:
                content = f.read_text()
                file_contents.append(f"### {f.relative_to(self._root)}\n```python\n{content}\n```")
            except Exception:
                continue

        files_block = "\n\n".join(file_contents)

        # Build the review prompt
        prompt = f"""You are a code reviewer for the ML Engine (Buddy) FX trading bot.

{personality}

## Review Context
A PRD was just completed: {prd_filepath}
The following files were created or modified. Review them for:
1. 🔴 Blockers: Security, crashes, data loss, trading rule violations
2. 🟡 Suggestions: Missing error handling, untested paths, unclear logic
3. 💭 Nits: Style, naming, minor improvements

## Trading Domain Rules (MUST enforce)
- NEVER hardcode SL/TP pip values — use ATR-based calculations
- NEVER execute trades without R:R ratio >= 1.2:1
- ALWAYS wrap JSON parsing in try/except with graceful defaults
- ALWAYS call save_state() on modules during graceful shutdown
- NEVER skip RL sync after trade close

## Files to Review

{files_block}

## Output Format
List each finding as:
🔴 BLOCKER: [file:line] description
🟡 SUGGESTION: [file:line] description
💭 NIT: [file:line] description

End with:
VERDICT: PASS or VERDICT: FAIL (if any blockers)
"""

        logger.info(f"CodeReviewer: spawning Claude agent to review {len(files)} files...")

        try:
            # Use stdin pipe instead of -p flag (avoids arg list too long)
            proc = subprocess.run(
                ["claude", "--print"],
                input=prompt,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = proc.stdout or ""
            if proc.returncode != 0 and proc.stderr:
                logger.warning(f"CodeReviewer: Claude stderr: {proc.stderr[:500]}")
                # If Claude CLI fails, fall back to static analysis
                if not output.strip():
                    logger.info("CodeReviewer: Claude returned empty, using static fallback")
                    return self._static_review_fallback(files)
            return output
        except subprocess.TimeoutExpired:
            logger.warning(f"CodeReviewer: Claude agent timed out after {self._timeout}s, using static fallback")
            return self._static_review_fallback(files)
        except FileNotFoundError:
            logger.warning("CodeReviewer: 'claude' CLI not found, falling back to static analysis")
            return self._static_review_fallback(files)
        except Exception as e:
            logger.warning(f"CodeReviewer: agent spawn failed ({e}), using static fallback")
            return self._static_review_fallback(files)

    def _static_review_fallback(self, files: List[Path]) -> str:
        """
        Fallback static analysis when Claude CLI is not available.
        Checks for the most critical patterns.
        """
        import ast as ast_mod
        findings = []

        for filepath in files:
            try:
                content = filepath.read_text()
                tree = ast_mod.parse(content)
                rel_path = filepath.relative_to(self._root)

                # Check for bare excepts
                for node in ast_mod.walk(tree):
                    if isinstance(node, ast_mod.ExceptHandler) and node.type is None:
                        findings.append(f"🔴 BLOCKER: [{rel_path}:{node.lineno}] Bare 'except:' catches SystemExit/KeyboardInterrupt")

                # Check for hardcoded SL/TP
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(r'(sl|stop_loss|take_profit|tp)\s*=\s*\d+\.?\d*\s*$', line, re.IGNORECASE):
                        if "atr" not in line.lower() and "#" not in line:
                            findings.append(f"🔴 BLOCKER: [{rel_path}:{i}] Hardcoded SL/TP — must use ATR-based")

                # Check for json.load without try/except
                for node in ast_mod.walk(tree):
                    if isinstance(node, ast_mod.Call):
                        if isinstance(node.func, ast_mod.Attribute):
                            if isinstance(node.func.value, ast_mod.Name) and node.func.value.id == "json":
                                if node.func.attr in ("load", "loads"):
                                    # Simple check: is it inside a try?
                                    findings.append(f"🟡 SUGGESTION: [{rel_path}:{node.lineno}] Verify json.{node.func.attr}() is wrapped in try/except")

            except Exception:
                continue

        verdict = "VERDICT: FAIL" if any("🔴" in f for f in findings) else "VERDICT: PASS"
        return "\n".join(findings + ["", verdict])

    # ─── Follow-up PRD Generation ───────────────────────────────────

    def _generate_followup_prd(
        self, blockers: List[str], suggestions: List[str], source_prd: str
    ) -> Optional[Path]:
        """
        Generate a follow-up PRD from review findings for Ralph to fix.
        Only generated if there are blockers.
        """
        if not blockers:
            return None

        stories = []

        # Each blocker becomes a story
        for i, blocker in enumerate(blockers[:10], 1):  # Cap at 10 stories
            # Parse file:line from blocker
            file_match = re.search(r'\[([^\]]+)\]', blocker)
            file_ref = file_match.group(1) if file_match else "unknown"
            description = re.sub(r'🔴\s*BLOCKER:\s*\[[^\]]*\]\s*', '', blocker).strip()

            stories.append({
                "id": f"US-{i:03d}",
                "title": f"Fix blocker in {file_ref}: {description[:60]}",
                "description": f"Code review found a blocker: {description}",
                "acceptanceCriteria": [
                    f"Fix the issue in {file_ref}",
                    "Verify the fix doesn't break existing tests",
                    "Non-blocking — wrap in try/except if fixing near critical path",
                    "Syntax validation passes",
                ],
                "priority": i,
                "passes": False,
                "notes": f"Auto-generated from code review of {source_prd}",
            })

        # Batch suggestions into one story
        if suggestions:
            suggestion_criteria = [
                re.sub(r'🟡\s*SUGGESTION:\s*\[[^\]]*\]\s*', '', s).strip()
                for s in suggestions[:8]
            ]
            suggestion_criteria.append("Syntax validation passes")
            stories.append({
                "id": f"US-{len(stories) + 1:03d}",
                "title": "Address code review suggestions",
                "description": f"Code review found {len(suggestions)} suggestions to improve code quality.",
                "acceptanceCriteria": suggestion_criteria,
                "priority": len(stories) + 1,
                "passes": False,
                "notes": "",
            })

        prd = {
            "project": "ML_Engine",
            "branchName": "ralph/buddy-review-fixes",
            "description": (
                f"Auto-generated from code review findings. "
                f"{len(blockers)} blockers, {len(suggestions)} suggestions. "
                f"Source: {Path(source_prd).name}. "
                f"DIRECTIVE: NEVER STOP. DONT ASK. ONLY ASK WHEN DESTRUCTIVE."
            ),
            "userStories": stories,
        }

        # Write to a separate file (don't overwrite gap_wirer's PRD if it exists)
        followup_path = self._ralph_dir / "prd_review_fixes.json"
        followup_path.write_text(json.dumps(prd, indent=2))
        logger.info(f"CodeReviewer: follow-up PRD written to {followup_path} ({len(stories)} stories)")

        return followup_path

    # ─── File Discovery ─────────────────────────────────────────────

    def _extract_files_from_prd(self, prd_path: Path) -> List[Path]:
        """Get files referenced in a PRD's user stories."""
        files = []
        try:
            data = json.loads(prd_path.read_text())
            text = json.dumps(data).lower()
            for match in re.findall(r'(\w+)\.py', text):
                for search_dir in [self._automation_dir, self._root / "src" / "scanner",
                                   self._root / "src" / "scanner" / "agents"]:
                    candidate = search_dir / f"{match}.py"
                    if candidate.exists() and candidate not in files:
                        files.append(candidate)
                        break
        except Exception:
            pass
        return files

    def _get_recent_changed_files(self) -> List[Path]:
        """Get recently changed Python files via git diff."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~5", "--", "*.py"],
                cwd=str(self._root),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return [
                    self._root / f.strip()
                    for f in result.stdout.splitlines()
                    if f.strip() and (self._root / f.strip()).exists()
                ]
        except Exception:
            pass
        return []

    @staticmethod
    def _extract_findings(output: str, marker: str) -> List[str]:
        """Extract findings by marker emoji from review output."""
        findings = []
        for line in output.splitlines():
            if marker in line:
                findings.append(line.strip())
        return findings

    # ─── EventBus Handler ───────────────────────────────────────────

    def handle_gaps_identified(self, event: Event) -> EventResult:
        """After gap wirer generates a PRD, review the code it references."""
        start = time.monotonic()
        try:
            gap_data = event.payload
            generated_prd = gap_data.get("generated_prd", "")
            if generated_prd:
                result = self.review_prd_changes(generated_prd, trigger="gap_wirer")
            else:
                # Review the source PRD's modules instead
                source = gap_data.get("prd_file", "")
                result = self.review_prd_changes(source, trigger="gap_wirer_source")

            return EventResult(
                success=result.passed,
                handler_name="code_reviewer",
                duration_secs=time.monotonic() - start,
                output=result.to_dict(),
            )
        except Exception as e:
            return EventResult(
                success=False,
                handler_name="code_reviewer",
                duration_secs=time.monotonic() - start,
                error=str(e),
            )

    def register(self, bus: Optional[EventBus] = None) -> None:
        target = bus or self._bus
        target.subscribe(
            EventType.GAPS_IDENTIFIED,
            self.handle_gaps_identified,
            name="code_reviewer",
            priority=20,
        )
        logger.info("CodeReviewer: registered on EventBus for GAPS_IDENTIFIED")
