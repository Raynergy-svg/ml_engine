"""Pure-Python invariant engine for the meta-cybernetic change pipeline.

Loads `.claude/rules/constitution.json` and evaluates every clause against a
`ChangePackage`. The Constitution is the *machine half* of stage 4 — the
qualitative `Policy Auditor` specialist runs separately on the same package and
their disagreement triggers a HARD STOP.

Clause `kind` vocabulary (matches the seed file):
    config_delta — invariant on `proposal.config_delta`
    scorecard    — invariant on `scorecard.<sub>.details.<field>`
    diff         — invariant on `proposal.changed_files`
    package      — invariant on top-level `ChangePackage` fields

Rule vocabulary:
    monotonic_non_increasing             — new value must be <= old for matching keys
    no_true_to_false                     — boolean field cannot flip True -> False
    le_constant / ge_constant            — numeric bound (constant lives in the clause)
    if_any_changed:<glob>:then_any_changed:<glob> — paired-glob requirement
    if:<expr>:then_nonempty:<field>       — conditional non-empty requirement
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.scanner.automation.meta_types import (
    ChangePackage,
    ClauseResult,
    ConstitutionAttestation,
    DeployStage,
)
from src.scanner.automation.safe_json import safe_json_read

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(".claude/rules/constitution.json")


class Constitution:
    """Evaluator for the seed clauses defined in constitution.json."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self._clauses: List[Dict[str, Any]] = []
        self._version: int = 0
        self._load()

    def _load(self) -> None:
        data = safe_json_read(self.path, default={})
        if not isinstance(data, dict):
            logger.warning("constitution.invalid_root path=%s", self.path)
            return
        self._version = int(data.get("version", 0))
        clauses = data.get("clauses", [])
        if isinstance(clauses, list):
            self._clauses = [c for c in clauses if isinstance(c, dict) and "id" in c and "rule" in c]
        if not self._clauses:
            logger.warning("constitution.no_clauses path=%s", self.path)

    @property
    def clauses(self) -> List[Dict[str, Any]]:
        return list(self._clauses)

    def check(self, package: ChangePackage) -> ConstitutionAttestation:
        results: List[ClauseResult] = []
        for clause in self._clauses:
            try:
                results.append(self._evaluate_clause(clause, package))
            except Exception as e:
                logger.warning(
                    "constitution.clause_eval_error clause=%s err=%s",
                    clause.get("id"),
                    e,
                )
                results.append(
                    ClauseResult(
                        clause_id=clause.get("id", "?"),
                        name=clause.get("name", "?"),
                        passed=False,
                        detail=f"evaluation_error: {e}",
                    )
                )
        passed = all(r.passed for r in results)
        return ConstitutionAttestation(passed=passed, clauses=results)

    def _evaluate_clause(self, clause: Dict[str, Any], pkg: ChangePackage) -> ClauseResult:
        kind = clause.get("kind")
        rule = clause.get("rule", "")
        cid = clause.get("id", "?")
        name = clause.get("name", "?")

        if kind == "config_delta":
            return self._check_config_delta(cid, name, clause, pkg)
        if kind == "scorecard":
            return self._check_scorecard(cid, name, clause, pkg)
        if kind == "diff":
            return self._check_diff(cid, name, clause, pkg)
        if kind == "package":
            return self._check_package(cid, name, clause, pkg)
        return ClauseResult(cid, name, False, f"unknown clause kind: {kind}")

    def _check_config_delta(
        self, cid: str, name: str, clause: Dict[str, Any], pkg: ChangePackage
    ) -> ClauseResult:
        rule = clause["rule"]
        delta = (pkg.proposal.config_delta if pkg.proposal else {}) or {}
        if not delta:
            return ClauseResult(cid, name, True, "no config delta to evaluate")

        if rule == "monotonic_non_increasing":
            pattern = clause.get("field_pattern", "")
            patterns = pattern.split("|") if pattern else []
            for key, change in delta.items():
                if patterns and not any(fnmatch.fnmatch(key, p) for p in patterns):
                    continue
                old, new = _extract_old_new(change)
                if old is None or new is None:
                    continue
                try:
                    if float(new) > float(old):
                        return ClauseResult(
                            cid, name, False,
                            f"{key}: {old} -> {new} widens risk",
                        )
                except (TypeError, ValueError):
                    continue
            return ClauseResult(cid, name, True, "all matching fields non-increasing")

        if rule == "no_true_to_false":
            field = clause.get("field", "")
            if field not in delta:
                return ClauseResult(cid, name, True, f"{field} unchanged")
            old, new = _extract_old_new(delta[field])
            if old is True and new is False:
                return ClauseResult(cid, name, False, f"{field} flipped True -> False")
            return ClauseResult(cid, name, True, f"{field} ok")

        if rule == "le_constant":
            field = clause.get("field", "")
            constant = clause.get("constant")
            if field not in delta:
                return ClauseResult(cid, name, True, f"{field} unchanged")
            old, new = _extract_old_new(delta[field])
            try:
                if new is not None and float(new) > float(constant):
                    return ClauseResult(
                        cid, name, False,
                        f"{field}={new} exceeds floor {constant}",
                    )
            except (TypeError, ValueError):
                return ClauseResult(cid, name, False, f"non-numeric {field}={new}")
            return ClauseResult(cid, name, True, f"{field}={new} within bound")

        if rule == "ge_constant":
            field = clause.get("field", "")
            constant = clause.get("constant")
            if field not in delta:
                return ClauseResult(cid, name, True, f"{field} unchanged")
            old, new = _extract_old_new(delta[field])
            try:
                if new is not None and float(new) < float(constant):
                    return ClauseResult(
                        cid, name, False,
                        f"{field}={new} below floor {constant}",
                    )
            except (TypeError, ValueError):
                return ClauseResult(cid, name, False, f"non-numeric {field}={new}")
            return ClauseResult(cid, name, True, f"{field}={new} within bound")

        return ClauseResult(cid, name, False, f"unknown config_delta rule: {rule}")

    def _check_scorecard(
        self, cid: str, name: str, clause: Dict[str, Any], pkg: ChangePackage
    ) -> ClauseResult:
        if pkg.scorecard is None:
            return ClauseResult(cid, name, True, "no scorecard yet — deferred")
        rule = clause["rule"]
        path = clause.get("field", "")
        value = _resolve_scorecard_path(pkg.scorecard, path)
        constant = clause.get("constant")
        if value is None:
            return ClauseResult(cid, name, True, f"{path} not present in scorecard")
        try:
            v = float(value)
            c = float(constant)
        except (TypeError, ValueError):
            return ClauseResult(cid, name, False, f"non-numeric scorecard value at {path}")
        if rule == "ge_constant":
            if v < c:
                return ClauseResult(cid, name, False, f"{path}={v} below floor {c}")
            return ClauseResult(cid, name, True, f"{path}={v} ok")
        if rule == "le_constant":
            if v > c:
                return ClauseResult(cid, name, False, f"{path}={v} above ceiling {c}")
            return ClauseResult(cid, name, True, f"{path}={v} ok")
        return ClauseResult(cid, name, False, f"unknown scorecard rule: {rule}")

    def _check_diff(
        self, cid: str, name: str, clause: Dict[str, Any], pkg: ChangePackage
    ) -> ClauseResult:
        files = (pkg.proposal.changed_files if pkg.proposal else []) or []
        rule = clause["rule"]
        if rule.startswith("if_any_changed:"):
            parts = rule.split(":")
            if len(parts) != 4 or parts[2] != "then_any_changed":
                return ClauseResult(cid, name, False, "malformed paired-glob rule")
            trigger_glob, follow_glob = parts[1], parts[3]
            triggered = any(fnmatch.fnmatch(f, trigger_glob) for f in files)
            if not triggered:
                return ClauseResult(cid, name, True, f"no file matches {trigger_glob}")
            satisfied = any(fnmatch.fnmatch(f, follow_glob) for f in files)
            if not satisfied:
                return ClauseResult(
                    cid, name, False,
                    f"{trigger_glob} changed but no {follow_glob} change",
                )
            return ClauseResult(cid, name, True, "paired requirement satisfied")
        return ClauseResult(cid, name, False, f"unknown diff rule: {rule}")

    def _check_package(
        self, cid: str, name: str, clause: Dict[str, Any], pkg: ChangePackage
    ) -> ClauseResult:
        rule = clause["rule"]
        if rule.startswith("if:"):
            parts = rule.split(":")
            if len(parts) != 4 or parts[2] != "then_nonempty":
                return ClauseResult(cid, name, False, "malformed package rule")
            condition_expr, target_field = parts[1], parts[3]
            if not _evaluate_package_condition(condition_expr, pkg):
                return ClauseResult(cid, name, True, f"condition `{condition_expr}` not met")
            target_value = getattr(pkg, target_field, None)
            if not target_value:
                return ClauseResult(cid, name, False, f"{target_field} is empty")
            return ClauseResult(cid, name, True, f"{target_field} present")
        return ClauseResult(cid, name, False, f"unknown package rule: {rule}")


def _extract_old_new(change: Any) -> tuple:
    """A config_delta value can be either {old, new} or just the new value."""
    if isinstance(change, dict) and "new" in change:
        return change.get("old"), change.get("new")
    return None, change


def _resolve_scorecard_path(scorecard: Any, dotted: str) -> Any:
    """Resolve `trading_quality.win_rate` against a Scorecard.

    Looks first at the SubScore's `details` dict, falling back to its `score`.
    """
    if not dotted:
        return None
    head, _, tail = dotted.partition(".")
    sub = getattr(scorecard, head, None)
    if sub is None:
        return None
    if not tail:
        return sub.score
    details = getattr(sub, "details", None) or {}
    cur: Any = details
    for part in tail.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _evaluate_package_condition(expr: str, pkg: ChangePackage) -> bool:
    """Tiny condition language: only `<lhs>==<rhs>` for now."""
    if "==" not in expr:
        return False
    lhs, rhs = expr.split("==", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if lhs == "deploy_target":
        try:
            return pkg.deploy_target == DeployStage(rhs)
        except ValueError:
            return False
    if lhs == "kind":
        return pkg.kind.value == rhs
    if lhs == "stage":
        return pkg.stage.value == rhs
    return False
