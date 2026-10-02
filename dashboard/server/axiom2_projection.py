"""Read-only display projection. Cached state is never an authority receipt.

No EvidenceStore construction (it writes), signing, broker transport or holdout
access. Only fixed files under service-selected roots are read; requests cannot
select paths. Digests identify displayed bytes, not their authenticity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.axiom2.execution.admission import BLOCKED_PREREQUISITES

REPO_ROOT = Path(__file__).resolve().parents[2]
router = APIRouter()
MAX_BYTES = 2 * 1024 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _read(root: Path, relative: str) -> tuple[dict | None, dict]:
    provenance = dict(source=relative, digest=None, source_time=None,
                      local_file_modified_at=None, status="UNAVAILABLE",
                      authenticity="NOT_VERIFIED_BY_DASHBOARD")
    try:
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("outside root")
        with path.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
            stat = os.fstat(handle.fileno())
        if len(raw) > MAX_BYTES:
            raise ValueError("oversized")
        value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(value, dict):
            raise ValueError("not object")
        provenance.update(digest=hashlib.sha256(raw).hexdigest(), status="AVAILABLE",
                          local_file_modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
        return value, provenance
    except FileNotFoundError:
        return None, provenance
    except (OSError, ValueError, RecursionError):
        provenance["status"] = "INVALID"
        return None, provenance


def build_projection(repo_root: Path = REPO_ROOT, evidence_root: Path | None = None) -> dict:
    receipt = datetime.now(timezone.utc).isoformat()
    root = evidence_root if evidence_root is not None else Path(os.environ.get("AXIOM_EVIDENCE_ROOT", repo_root / "trained_data/evidence"))
    index, evidence_source = _read(root, "indexes/current.json")
    rows = []
    if index is not None:
        packages = index.get("packages")
        valid = index.get("schema_version") == "1.0.0" and isinstance(packages, dict) and len(packages) <= 2000
        if valid:
            for digest, row in sorted(packages.items()):
                if not DIGEST.fullmatch(digest) or not isinstance(row, dict):
                    valid = False
                    break
                lane = row.get("lane_id")
                package = row.get("package_id")
                state = row.get("state")
                head = row.get("head_event_digest")
                if not isinstance(lane, str) or len(lane) > 256 or not isinstance(package, str) or len(package) > 256 or (state is not None and (not isinstance(state, str) or len(state) > 64)) or (head is not None and (not isinstance(head, str) or not DIGEST.fullmatch(head))):
                    valid = False
                    break
                # This surface is equity-first; legacy caches remain on /legacy.
                if lane.startswith("equity_research_"):
                    rows.append(dict(lane_id=lane, package_id=package, package_digest=digest,
                                     cached_state=state, head_event_digest=head,
                                     evidence_class="UNVERIFIED", capital_eligible=False))
        if not valid:
            rows = []
            evidence_source["status"] = "INVALID"
    boundaries = []
    for name in ("research", "portfolio", "execution"):
        config, source = _read(repo_root, f"config/axiom2/{name}_boundary.json")
        if config is not None and (config.get("schema_version") != 1 or not isinstance(config.get("profile"), str) or len(config["profile"]) > 128 or type(config.get("execution_enabled")) is not bool):
            config = None
            source["status"] = "INVALID"
        boundaries.append(dict(name=name, provenance=source,
                               profile=config.get("profile") if config else None,
                               declared_execution_enabled=config.get("execution_enabled") if config else None))
    return dict(schema_version="axiom.dashboard.v1", mode="READ_ONLY_LOCAL_PROJECTION",
                received_at=receipt, source_time=None,
                provenance="Committed evidence cache; signatures and live qualification are not verified here.",
                execution_enabled=False, capital_authorized=False,
                evidence=dict(provenance=evidence_source, rows=rows), boundaries=boundaries,
                market=dict(status="UNAVAILABLE", reason="No authenticated market observation producer is bound to this dashboard.", source_time=None, received_at=None),
                portfolio=dict(status="UNAVAILABLE", account_alias=None, account_value=None,
                               reported_cash=None, buying_power=None, settled_cash=None,
                               source_time=None, received_at=None),
                qualification=dict(holdout="UNVERIFIED", live_shadow="UNQUALIFIED", reconciliation="UNVERIFIED", operational_boundary="UNVERIFIED"),
                blocked_prerequisites=list(BLOCKED_PREREQUISITES))


@router.get("/api/axiom2/overview")
def overview():
    return JSONResponse(build_projection(), headers={"Cache-Control": "no-store"})
