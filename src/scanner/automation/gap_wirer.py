"""
Gap Wirer Agent — Analyzes completed PRDs, detects unwired integrations,
and GENERATES a real PRD (prd.json) for Ralph to execute autonomously.

Flow:
  1. AST-scan the codebase for integration gaps
  2. Prioritize gaps by severity
  3. Convert gaps into user stories (Ralph-compatible prd.json format)
  4. Write the PRD to .claude/ralph/prd.json
  5. Optionally kick off Ralph to implement the stories

Phase 28 (US-170): Gap-wirer agent that produces actionable PRDs.
"""

import ast
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.scanner.automation.event_bus import Event, PrdEventBus as EventBus, EventResult, EventType, get_prd_event_bus as get_event_bus

logger = logging.getLogger(__name__)


# ─── Gap Detection (kept from v1, actually useful) ─────────────────

@dataclass
class Gap:
    """A detected integration gap."""
    gap_type: str
    severity: str  # critical | high | medium | low
    module_name: str
    description: str
    file_path: str
    suggested_fix: str
    line_hint: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class GapReport:
    """Full gap analysis report."""
    prd_file: str
    branch_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gaps: List[Gap] = field(default_factory=list)
    modules_scanned: int = 0
    duration_secs: float = 0.0
    generated_prd_path: Optional[str] = None
    ralph_spawned: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for g in self.gaps if g.severity == "critical")

    def to_dict(self) -> dict:
        return {
            "prd_file": self.prd_file,
            "branch": self.branch_name,
            "timestamp": self.timestamp,
            "total_gaps": len(self.gaps),
            "critical": self.critical_count,
            "modules_scanned": self.modules_scanned,
            "duration_secs": round(self.duration_secs, 2),
            "generated_prd": self.generated_prd_path,
            "ralph_spawned": self.ralph_spawned,
            "gaps": [g.to_dict() for g in self.gaps],
            "errors": self.errors,
        }


class GapWirer:
    """
    Detects integration gaps and generates a real PRD for Ralph to implement.

    The key difference from v1: this agent PRODUCES WORK, not just reports.
    It writes a prd.json and optionally spawns Ralph to execute it.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        event_bus: Optional[EventBus] = None,
        auto_spawn_ralph: bool = False,
        ralph_tool: str = "claude",
        ralph_max_iterations: int = 10,
    ):
        self._root = project_root or Path(".")
        self._bus = event_bus or get_event_bus()
        self._auto_spawn = auto_spawn_ralph
        self._ralph_tool = ralph_tool
        self._ralph_max_iters = ralph_max_iterations

        self._automation_dir = self._root / "src" / "scanner" / "automation"
        self._ralph_dir = self._root / ".claude" / "ralph"
        self._agents_dir = self._root / ".claude" / "agents"

        # Key files to cross-reference
        self._orchestrator = self._automation_dir / "orchestrator.py"
        self._continuous = self._automation_dir / "continuous.py"
        self._dispatcher = self._automation_dir / "module_dispatcher.py"
        self._config = self._root / "src" / "scanner" / "config.py"

    def analyze_and_generate(self, source_prd: str, branch_name: str = "") -> GapReport:
        """
        Full pipeline: detect gaps → generate PRD → (optionally) spawn Ralph.
        """
        start = time.monotonic()
        report = GapReport(prd_file=source_prd, branch_name=branch_name)

        try:
            # 1. Detect gaps
            all_modules = self._inventory_modules()
            report.modules_scanned = len(all_modules)

            orchestrator_refs = self._parse_refs(self._orchestrator)
            continuous_refs = self._parse_refs(self._continuous)
            dispatcher_refs = self._parse_dispatcher_refs()
            shutdown_refs = self._parse_shutdown_hooks()

            gaps = []
            gaps.extend(self._find_unregistered_modules(all_modules, dispatcher_refs, orchestrator_refs))
            gaps.extend(self._find_unwired_state_persistence(all_modules, shutdown_refs))
            gaps.extend(self._find_disabled_config_flags())
            gaps.extend(self._find_dead_feedback_loops(all_modules, orchestrator_refs, continuous_refs))
            report.gaps = gaps

            if not gaps:
                logger.info("GapWirer: no gaps detected — nothing to generate")
                report.duration_secs = time.monotonic() - start
                return report

            # 2. Check if prd.json is currently in-progress (Ralph working on it)
            #    If so, don't overwrite — wait for Ralph to finish
            current_prd = self._ralph_dir / "prd.json"
            if current_prd.exists():
                try:
                    existing = json.loads(current_prd.read_text())
                    stories = existing.get("userStories", [])
                    in_progress = any(not s.get("passes", False) for s in stories)
                    if in_progress and stories:
                        passed = sum(1 for s in stories if s.get("passes", False))
                        logger.info(
                            f"GapWirer: prd.json has {passed}/{len(stories)} stories done — "
                            f"Ralph still working, skipping overwrite"
                        )
                        report.duration_secs = time.monotonic() - start
                        return report
                except Exception as e:
                    logger.debug(f"GapWirer: existing PRD parse failed (will overwrite): {e}")

            # 3. Convert gaps → user stories → prd.json
            prd_data = self._gaps_to_prd(gaps, source_prd, branch_name)
            prd_path = self._write_prd(prd_data)
            report.generated_prd_path = str(prd_path)

            logger.info(
                f"GapWirer: generated PRD with {len(prd_data['userStories'])} stories → {prd_path}"
            )

            # 3. Spawn Ralph if configured
            if self._auto_spawn:
                self._spawn_ralph(prd_path)
                report.ralph_spawned = True

        except Exception as e:
            report.errors.append(f"Gap analysis error: {e}")
            logger.error(f"GapWirer: failed: {e}")

        report.duration_secs = time.monotonic() - start

        self._bus.emit(Event(
            event_type=EventType.GAPS_IDENTIFIED,
            source="gap_wirer",
            payload=report.to_dict(),
        ))

        return report

    # ─── PRD Generation ─────────────────────────────────────────────

    def _gaps_to_prd(self, gaps: List[Gap], source_prd: str, branch_name: str) -> dict:
        """
        Convert detected gaps into Ralph-compatible prd.json format.
        Each gap becomes a user story with verifiable acceptance criteria.
        """
        # Group gaps by type for batching into stories
        by_type: Dict[str, List[Gap]] = {}
        for g in gaps:
            by_type.setdefault(g.gap_type, []).append(g)

        stories = []
        story_id = 1

        # Priority order: state_persistence > config_flag > module_registration > feedback_loop
        type_priority = {
            "state_persistence": 1,
            "config_flag": 2,
            "module_registration": 3,
            "feedback_loop": 4,
        }

        for gap_type in sorted(by_type.keys(), key=lambda t: type_priority.get(t, 99)):
            type_gaps = by_type[gap_type]

            # Batch small gaps into single stories (max 5 per story for one-context-window)
            for batch_start in range(0, len(type_gaps), 5):
                batch = type_gaps[batch_start:batch_start + 5]
                story = self._batch_to_story(batch, gap_type, story_id)
                stories.append(story)
                story_id += 1

        # Derive branch name from source
        source_branch = branch_name or "unknown"
        phase_match = re.search(r'phase(\d+)', source_branch)
        next_phase = int(phase_match.group(1)) + 1 if phase_match else 29

        prd = {
            "project": "ML_Engine",
            "branchName": f"ralph/buddy-gap-resolution-phase{next_phase}",
            "description": (
                f"Auto-generated gap resolution PRD. Source: {Path(source_prd).name}. "
                f"Detected {len(gaps)} integration gaps across {len(by_type)} categories. "
                f"DIRECTIVE: NEVER STOP. DONT ASK. ONLY ASK WHEN DESTRUCTIVE."
            ),
            "userStories": stories,
        }

        return prd

    def _batch_to_story(self, gaps: List[Gap], gap_type: str, story_id: int) -> dict:
        """Convert a batch of gaps into a single user story."""
        modules = [g.module_name for g in gaps]
        modules_str = ", ".join(modules[:5])

        # Build acceptance criteria from each gap's suggested fix
        criteria = []
        for g in gaps:
            criteria.append(g.suggested_fix)
        criteria.append("Non-blocking — never crash on module init failure (wrap in try/except)")
        criteria.append("Syntax validation passes")

        title_map = {
            "state_persistence": f"Wire save_state() into graceful shutdown for: {modules_str}",
            "config_flag": f"Enable config flags for wired modules: {modules_str}",
            "module_registration": f"Register modules in dispatcher/orchestrator: {modules_str}",
            "feedback_loop": f"Connect feedback consumers for: {modules_str}",
        }

        desc_map = {
            "state_persistence": "Modules have save_state() methods but aren't called during shutdown. State is lost between sessions.",
            "config_flag": "Modules are wired into the loop but their config flags default to False, so they never activate.",
            "module_registration": "Modules exist in automation/ but aren't registered in the dispatcher or orchestrator.",
            "feedback_loop": "Modules produce data that nothing consumes — dead-end outputs that should feed into the learning loop.",
        }

        return {
            "id": f"US-{story_id:03d}",
            "title": title_map.get(gap_type, f"Resolve {gap_type} gaps for: {modules_str}"),
            "description": desc_map.get(gap_type, f"Fix {len(gaps)} {gap_type} integration gaps."),
            "acceptanceCriteria": criteria,
            "priority": story_id,
            "passes": False,
            "notes": "",
        }

    def _write_prd(self, prd_data: dict) -> Path:
        """
        Write the generated PRD to .claude/ralph/prd.json.
        Archives the existing prd.json first (same logic as ralph.sh).
        """
        prd_path = self._ralph_dir / "prd.json"
        self._ralph_dir.mkdir(parents=True, exist_ok=True)

        # Archive existing if different branch
        if prd_path.exists():
            try:
                existing = json.loads(prd_path.read_text())
                existing_branch = existing.get("branchName", "")
                new_branch = prd_data.get("branchName", "")
                if existing_branch and existing_branch != new_branch:
                    archive_dir = self._ralph_dir / "archive"
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    folder_name = existing_branch.replace("ralph/", "")
                    dest = archive_dir / f"{date_str}-{folder_name}"
                    dest.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(prd_path, dest / "prd.json")
                    progress = self._ralph_dir / "progress.txt"
                    if progress.exists():
                        shutil.copy2(progress, dest / "progress.txt")
                    logger.info(f"GapWirer: archived previous PRD to {dest}")
            except Exception as e:
                logger.warning(f"GapWirer: archive failed: {e}")

        # Write the new PRD (atomic: tmp + rename)
        _tmp_prd = prd_path.with_suffix(".json.tmp")
        _tmp_prd.write_text(json.dumps(prd_data, indent=2, sort_keys=True))
        _tmp_prd.replace(prd_path)

        # Reset progress file (atomic: tmp + rename)
        progress = self._ralph_dir / "progress.txt"
        _progress_content = (
            f"# Ralph Progress Log\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Source: gap_wirer auto-generated\n"
            f"---\n"
        )
        _tmp_progress = progress.with_suffix(".txt.tmp")
        _tmp_progress.write_text(_progress_content)
        _tmp_progress.replace(progress)

        return prd_path

    # ─── Ralph Spawning ─────────────────────────────────────────────

    def _spawn_ralph(self, prd_path: Path) -> Optional[subprocess.Popen]:
        """
        Spawn Ralph as a background process to implement the generated PRD.
        Uses the same mechanism as scripts/ralph.sh.
        """
        ralph_script = self._root / "scripts" / "ralph.sh"
        if not ralph_script.exists():
            logger.error(f"GapWirer: ralph.sh not found at {ralph_script}")
            return None

        logger.info(
            f"🚀 GapWirer: spawning Ralph (tool={self._ralph_tool}, "
            f"max_iters={self._ralph_max_iters}) for {prd_path}"
        )

        try:
            proc = subprocess.Popen(
                [
                    str(ralph_script),
                    "--tool", self._ralph_tool,
                    str(self._ralph_max_iters),
                ],
                cwd=str(self._root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # Detach from parent
            )
            logger.info(f"GapWirer: Ralph spawned as PID {proc.pid}")
            return proc
        except Exception as e:
            logger.error(f"GapWirer: failed to spawn Ralph: {e}")
            return None

    # ─── Gap Detection (rewritten for clarity) ──────────────────────

    def _inventory_modules(self) -> Dict[str, Path]:
        modules = {}
        if self._automation_dir.exists():
            for py in self._automation_dir.glob("*.py"):
                name = py.stem
                if not name.startswith("__") and not name.startswith("test"):
                    modules[name] = py
        return modules

    def _parse_refs(self, filepath: Path) -> Set[str]:
        """Extract module/attribute references from a Python file."""
        refs = set()
        try:
            content = filepath.read_text()
            for m in re.findall(r'from\s+\S*automation\S*\s+import\s+(\w+)', content):
                refs.add(self._class_to_snake(m))
            for m in re.findall(r'self\.(\w+)\s*=\s*\w+\(', content):
                refs.add(m)
            for m in re.findall(r'self\.(\w+)\.\w+\(', content):
                refs.add(m)
        except Exception as e:
            logger.debug(f"GapWirer: engine ref extraction failed: {e}")
        return refs

    def _parse_dispatcher_refs(self) -> Set[str]:
        refs = set()
        try:
            content = self._dispatcher.read_text()
            for m in re.findall(r'register_module\s*\(\s*["\'](\w+)["\']', content):
                refs.add(m)
        except Exception as e:
            logger.debug(f"GapWirer: dispatcher ref extraction failed: {e}")
        return refs

    def _parse_shutdown_hooks(self) -> Set[str]:
        refs = set()
        for filepath in [self._continuous, self._orchestrator]:
            try:
                content = filepath.read_text()
                for m in re.findall(r'(\w+)\.save_state\s*\(', content):
                    refs.add(m)
            except Exception as e:
                logger.debug(f"GapWirer: shutdown hook extraction failed for {filepath}: {e}")
        return refs

    def _find_unregistered_modules(
        self, all_modules: Dict[str, Path],
        dispatcher_refs: Set[str], orchestrator_refs: Set[str]
    ) -> List[Gap]:
        exempt = {
            "safe_json", "event_bus", "prd_watcher", "gap_wirer",
            "code_reviewer", "prd_agent_chain", "__init__", "maintenance",
        }
        registered = dispatcher_refs | orchestrator_refs
        gaps = []
        for name, path in all_modules.items():
            if name in exempt:
                continue
            if not self._has_class(path):
                continue
            if name not in registered and not any(name in r for r in registered):
                gaps.append(Gap(
                    gap_type="module_registration",
                    severity="medium",
                    module_name=name,
                    description=f"Module '{name}' exists but not registered",
                    file_path=str(path),
                    suggested_fix=f"Register '{name}' in module_dispatcher or orchestrator._init_modules()",
                ))
        return gaps

    def _find_unwired_state_persistence(
        self, all_modules: Dict[str, Path], shutdown_refs: Set[str]
    ) -> List[Gap]:
        gaps = []
        for name, path in all_modules.items():
            if self._has_method(path, "save_state"):
                if not any(name in r or r in name for r in shutdown_refs):
                    gaps.append(Gap(
                        gap_type="state_persistence",
                        severity="high",
                        module_name=name,
                        description=f"'{name}' has save_state() but not called in shutdown",
                        file_path=str(path),
                        suggested_fix=f"Add self.{name}.save_state() to graceful shutdown in continuous.py",
                    ))
        return gaps

    def _find_disabled_config_flags(self) -> List[Gap]:
        gaps = []
        try:
            content = self._config.read_text()
            for m in re.finditer(r'(enable_\w+)\s*:\s*bool\s*=\s*False', content):
                flag = m.group(1)
                mod_name = flag.replace("enable_", "")
                gaps.append(Gap(
                    gap_type="config_flag",
                    severity="high",
                    module_name=mod_name,
                    description=f"Config flag '{flag}' defaults to False",
                    file_path=str(self._config),
                    suggested_fix=f"Change {flag} default to True in ScannerConfig",
                ))
        except Exception as e:
            logger.debug(f"GapWirer: disabled config flag scan failed: {e}")
        return gaps

    def _find_dead_feedback_loops(
        self, all_modules: Dict[str, Path],
        orchestrator_refs: Set[str], continuous_refs: Set[str]
    ) -> List[Gap]:
        integrated = orchestrator_refs | continuous_refs
        gaps = []
        exempt = {"safe_json", "event_bus", "prd_watcher", "gap_wirer", "code_reviewer", "prd_agent_chain"}
        for name, path in all_modules.items():
            if name in exempt:
                continue
            try:
                content = path.read_text()
            except Exception:
                continue
            writes = bool(re.search(r'(write_text|json\.dump\s*\(|open.*["\'][wa]["\'])', content))
            if not writes:
                continue
            has_consumer = name in " ".join(integrated)
            if not has_consumer:
                for other_name, other_path in all_modules.items():
                    if other_name == name:
                        continue
                    try:
                        if name in other_path.read_text():
                            has_consumer = True
                            break
                    except Exception:
                        continue
            if not has_consumer:
                gaps.append(Gap(
                    gap_type="feedback_loop",
                    severity="low",
                    module_name=name,
                    description=f"'{name}' writes data but nothing consumes it",
                    file_path=str(path),
                    suggested_fix=f"Wire '{name}' output into observation_consumer or learning_engine",
                ))
        return gaps

    def _has_class(self, path: Path) -> bool:
        try:
            return any(isinstance(n, ast.ClassDef) for n in ast.walk(ast.parse(path.read_text())))
        except Exception:
            return False

    def _has_method(self, path: Path, method: str) -> bool:
        try:
            return any(
                isinstance(n, ast.FunctionDef) and n.name == method
                for n in ast.walk(ast.parse(path.read_text()))
            )
        except Exception:
            return False

    @staticmethod
    def _class_to_snake(name: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    # ─── EventBus Handler ───────────────────────────────────────────

    def handle_prd_complete(self, event: Event) -> EventResult:
        """EventBus handler: when a PRD completes, analyze gaps and generate next PRD."""
        start = time.monotonic()
        try:
            filepath = event.payload.get("filepath", "")
            branch = event.payload.get("branch_name", "")
            report = self.analyze_and_generate(filepath, branch)
            return EventResult(
                success=True,
                handler_name="gap_wirer",
                duration_secs=time.monotonic() - start,
                output=report.to_dict(),
            )
        except Exception as e:
            return EventResult(
                success=False,
                handler_name="gap_wirer",
                duration_secs=time.monotonic() - start,
                error=str(e),
            )

    def register(self, bus: Optional[EventBus] = None) -> None:
        target = bus or self._bus
        target.subscribe(
            EventType.PRD_COMPLETE,
            self.handle_prd_complete,
            name="gap_wirer",
            priority=10,
        )
        logger.info("GapWirer: registered on EventBus for PRD_COMPLETE")
