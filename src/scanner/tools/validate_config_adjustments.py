"""Standalone validator for config_adjustments.json key integrity.

Promoted rule date: 2026-04-16 — "Config Adjustment Consumer Verification"
(see .claude/rules/improvement.md).

Source evidence: $3,527 loss over 14 consecutive trades. Root cause: the
self-tuner proposed keys like `min_confidence_threshold` and
`atr_sl_multiplier_low_regime` that do NOT exist on the ScannerConfig
dataclass. ConfigAdjuster.apply_adjustments() called setattr(config, key,
value) for each pending key, which silently created orphan attributes that
no code ever read. The self-tuner appeared "successful" while the real
fields (`min_confidence`, `atr_sl_multiplier`) never changed.

This script is a CI-safe surface-only validator. It does NOT mutate state.

WHAT IT DOES
    1. AST-parses src/scanner/config.py to extract ScannerConfig dataclass
       field names (plus any nested dataclasses defined in the same file).
       AST is used instead of regex because the regex approach is fragile
       against type annotations with generics (Dict[str, Dict[str, float]]).
    2. Loads .claude/config_adjustments.json and (if present)
       logs/reflection_staging/config_adjustments.json. Both locations are
       checked because the staging file has a different shape (`proposals`
       list) than the applied file (`pending`/`history`/`last_applied`).
    3. Walks every shape variant:
        - pending:            {key: {value, source, reason, ...}}
        - last_applied:       {key: cycle_int}
        - history:            [{key, old_value, new_value, ...}] OR
                              [{changes: {key: value, ...}}]
        - proposals (staging): [{key, old_value, new_value, ...}]
    4. For each discovered key, classifies VALID vs ORPHAN. Orphan keys get
       a nearest-match suggestion via difflib.get_close_matches (cutoff 0.6).
    5. Prints a table and exits 0 if all valid, 1 if any orphan found.

USAGE
    python3 src/scanner/tools/validate_config_adjustments.py

    In CI:
        python3 src/scanner/tools/validate_config_adjustments.py || exit 1

CONSTRAINTS
    - stdlib only (ast, json, sys, pathlib, difflib) — no heavy imports.
    - Does NOT import ScannerConfig. Parsing source is faster and runs on
      machines without OANDA creds / model files.
    - Idempotent, read-only.
"""

from __future__ import annotations

import ast
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at: <root>/src/scanner/tools/validate_config_adjustments.py
# parents: [0]=tools, [1]=scanner, [2]=src, [3]=<root>
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_CONFIG_SOURCE = _PROJECT_ROOT / "src" / "scanner" / "config.py"

# Primary location (written by ConfigAdjuster).
_APPLIED_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"

# Secondary location (self-heal reflection staging, different shape).
_STAGING_PATH = (
    _PROJECT_ROOT / "logs" / "reflection_staging" / "config_adjustments.json"
)

# Pending file — used by the approval flow.
_PENDING_PATH = _PROJECT_ROOT / ".claude" / "pending_adjustments.json"

# Target dataclass whose field names are the source of truth.
_TARGET_DATACLASS = "ScannerConfig"


# ---------------------------------------------------------------------------
# AST field extraction
# ---------------------------------------------------------------------------


def _extract_dataclass_fields(source_path: Path, target: str = _TARGET_DATACLASS) -> set[str]:
    """Return declared field names for ``target`` dataclass.

    Parses the full source file with :mod:`ast` and walks every ClassDef.
    For each class whose name matches ``target``, collects every AnnAssign
    whose target is an ast.Name — that is the canonical dataclass field
    declaration shape (``name: type = default``). Private fields (leading
    underscore) are included because ConfigAdjuster can technically tune
    them, and we want to catch any such key.

    Also inlines fields from any other ClassDef in the same file whose name
    ends with "Config" — this covers inner/sibling dataclasses whose field
    names may show up in adjustment proposals.

    Args:
        source_path: Path to config.py.
        target: Name of the primary dataclass to extract.

    Returns:
        Set of all declared field names (lowercase preserved).

    Raises:
        FileNotFoundError: If source_path does not exist.
        SyntaxError: If the source file has syntax errors (surfaced — never
            swallowed, per improvement.md silent-exception rule).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Config source not found: {source_path}")

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    fields: set[str] = set()
    found_target = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        is_target = node.name == target
        is_sibling_config = node.name.endswith("Config") and node.name != target

        if not (is_target or is_sibling_config):
            continue

        if is_target:
            found_target = True

        for item in node.body:
            # Only ``name: type = default`` or ``name: type`` count as fields.
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields.add(item.target.id)

    if not found_target:
        raise ValueError(
            f"Target dataclass '{target}' not found in {source_path}. "
            f"AST walk found classes but none matched."
        )

    return fields


# ---------------------------------------------------------------------------
# JSON loaders (graceful-fallback per improvement.md)
# ---------------------------------------------------------------------------


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON file, returning None on missing/corrupt file.

    Per improvement.md JSON Safety Gates: always wrap JSON parsing in
    try/except with a graceful fallback. Returns None (sentinel) rather
    than raising so callers can distinguish "file not present" from
    "file present but empty dict".
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"[warn] {path} is present but not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}. "
            f"Treating as empty.",
            file=sys.stderr,
        )
        return None
    except OSError as exc:
        print(f"[warn] cannot read {path}: {exc}. Treating as empty.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Key extraction from every known JSON shape
# ---------------------------------------------------------------------------


def _collect_keys_from_applied(
    doc: dict[str, Any],
    source_label: str,
) -> list[tuple[str, str]]:
    """Collect (key, origin) tuples from a config_adjustments.json doc.

    Handles the ConfigAdjuster shape:
        {
          "version": ...,
          "pending":      {key: {...}},
          "last_applied": {key: cycle_int},
          "history":      [{key, ...} | {changes: {key: value}}],
        }

    Returns a list so duplicates from different sections surface clearly
    in the output table.
    """
    out: list[tuple[str, str]] = []

    pending = doc.get("pending") or {}
    if isinstance(pending, dict):
        for k in pending.keys():
            out.append((str(k), f"{source_label}:pending"))

    last_applied = doc.get("last_applied") or {}
    if isinstance(last_applied, dict):
        for k in last_applied.keys():
            out.append((str(k), f"{source_label}:last_applied"))

    history = doc.get("history") or []
    if isinstance(history, list):
        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                continue
            # Shape A: {key: "foo", ...}
            if "key" in entry and isinstance(entry["key"], str):
                out.append((entry["key"], f"{source_label}:history[{idx}]"))
            # Shape B: {changes: {key: value, ...}, ...}
            changes = entry.get("changes")
            if isinstance(changes, dict):
                for k in changes.keys():
                    out.append((str(k), f"{source_label}:history[{idx}].changes"))

    return out


def _collect_keys_from_staging(
    doc: dict[str, Any],
    source_label: str,
) -> list[tuple[str, str]]:
    """Collect (key, origin) tuples from reflection-staging shape.

    Staging shape:
        {
          "version": 1,
          "proposals": [{key, old_value, new_value, ...}, ...],
          "kill_switch": {"key": "...", ...}
        }
    """
    out: list[tuple[str, str]] = []

    proposals = doc.get("proposals") or []
    if isinstance(proposals, list):
        for idx, entry in enumerate(proposals):
            if isinstance(entry, dict) and isinstance(entry.get("key"), str):
                out.append((entry["key"], f"{source_label}:proposals[{idx}]"))

    # kill_switch.key is intentionally NOT validated — it can reference
    # virtual control flags (e.g., "scanner_paused") that aren't dataclass
    # fields. If the operator wants that validated, remove this skip.

    return out


def _collect_keys_from_pending(
    doc: dict[str, Any],
    source_label: str,
) -> list[tuple[str, str]]:
    """Collect (key, origin) tuples from pending_adjustments.json.

    Pending shape (US-508):
        {proposal_id: {key, value, source, reason, cycle, ...}, ...}
    """
    out: list[tuple[str, str]] = []
    for proposal_id, entry in doc.items():
        if isinstance(entry, dict) and isinstance(entry.get("key"), str):
            out.append((entry["key"], f"{source_label}:pending[{proposal_id}]"))
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(
    found: Iterable[tuple[str, str]],
    valid_fields: set[str],
) -> list[dict[str, str]]:
    """Return one row per found key with status + suggestion."""
    rows: list[dict[str, str]] = []
    valid_list = sorted(valid_fields)

    for key, origin in found:
        if key in valid_fields:
            rows.append({"key": key, "origin": origin, "status": "VALID", "suggestion": ""})
        else:
            matches = difflib.get_close_matches(key, valid_list, n=1, cutoff=0.6)
            suggestion = matches[0] if matches else "(no close match)"
            rows.append({"key": key, "origin": origin, "status": "ORPHAN", "suggestion": suggestion})

    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_table(rows: list[dict[str, str]]) -> str:
    """Fixed-width 4-column table: key | origin | status | suggestion."""
    if not rows:
        return "(no keys found in any config_adjustments.json variant)"

    headers = ("KEY", "ORIGIN", "STATUS", "SUGGESTION")
    col_key = max(len(headers[0]), max(len(r["key"]) for r in rows))
    col_origin = max(len(headers[1]), max(len(r["origin"]) for r in rows))
    col_status = max(len(headers[2]), max(len(r["status"]) for r in rows))
    col_suggestion = max(len(headers[3]), max(len(r["suggestion"]) for r in rows))

    def fmt(row_values: tuple[str, str, str, str]) -> str:
        return (
            f"{row_values[0]:<{col_key}}  "
            f"{row_values[1]:<{col_origin}}  "
            f"{row_values[2]:<{col_status}}  "
            f"{row_values[3]:<{col_suggestion}}"
        )

    sep = "-" * (col_key + col_origin + col_status + col_suggestion + 6)
    lines = [fmt(headers), sep]
    # Sort orphans first to make CI failures easy to spot at the top.
    rows_sorted = sorted(rows, key=lambda r: (r["status"] != "ORPHAN", r["key"]))
    for r in rows_sorted:
        lines.append(fmt((r["key"], r["origin"], r["status"], r["suggestion"])))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run() -> int:
    """Entry point. Returns exit code (0 valid, 1 orphans found)."""
    print("config_adjustments.json orphan-key validator")
    print(f"  project root:   {_PROJECT_ROOT}")
    print(f"  config source:  {_CONFIG_SOURCE}")
    print()

    # 1. Source of truth — ScannerConfig fields.
    try:
        valid_fields = _extract_dataclass_fields(_CONFIG_SOURCE)
    except (FileNotFoundError, SyntaxError, ValueError) as exc:
        print(f"[fatal] cannot extract ScannerConfig fields: {exc}", file=sys.stderr)
        return 2  # distinct code — infra error, not orphan detection

    print(f"Discovered {len(valid_fields)} ScannerConfig fields.")

    # 2. Collect keys from every known location / shape.
    found: list[tuple[str, str]] = []
    files_checked = 0
    files_present = 0

    applied = _safe_load_json(_APPLIED_PATH)
    files_checked += 1
    if applied is not None:
        files_present += 1
        print(f"  + loaded {_APPLIED_PATH.relative_to(_PROJECT_ROOT)}")
        if isinstance(applied, dict):
            found.extend(_collect_keys_from_applied(applied, ".claude/config_adjustments.json"))

    staging = _safe_load_json(_STAGING_PATH)
    files_checked += 1
    if staging is not None:
        files_present += 1
        print(f"  + loaded {_STAGING_PATH.relative_to(_PROJECT_ROOT)}")
        if isinstance(staging, dict):
            found.extend(
                _collect_keys_from_staging(staging, "logs/reflection_staging/config_adjustments.json")
            )
            # Some staging files may reuse the applied shape — cover that too.
            found.extend(
                _collect_keys_from_applied(staging, "logs/reflection_staging/config_adjustments.json")
            )

    pending = _safe_load_json(_PENDING_PATH)
    files_checked += 1
    if pending is not None:
        files_present += 1
        print(f"  + loaded {_PENDING_PATH.relative_to(_PROJECT_ROOT)}")
        if isinstance(pending, dict):
            found.extend(_collect_keys_from_pending(pending, ".claude/pending_adjustments.json"))

    print(f"Checked {files_checked} locations, {files_present} present.")
    print(f"Collected {len(found)} key occurrences across all shapes.")
    print()

    if not found:
        print("No keys found. Nothing to validate. Exit 0.")
        return 0

    # 3. Classify.
    rows = _classify(found, valid_fields)

    # 4. Render.
    print(_render_table(rows))
    print()

    orphans = [r for r in rows if r["status"] == "ORPHAN"]
    unique_orphan_keys = {r["key"] for r in orphans}

    valid_count = len(rows) - len(orphans)
    print(f"Summary: {valid_count} VALID occurrence(s), {len(orphans)} ORPHAN occurrence(s).")
    print(f"Unique orphan keys: {len(unique_orphan_keys)}")

    if orphans:
        print()
        print("ORPHAN KEYS (need operator decision — rename or delete):")
        # Dedupe by key for the operator-facing list.
        seen: set[str] = set()
        for r in sorted(orphans, key=lambda x: x["key"]):
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            print(f"  - {r['key']!r:40s} -> suggested: {r['suggestion']!r}")
        return 1

    print("All keys map to real ScannerConfig fields. Exit 0.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
