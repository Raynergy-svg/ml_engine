"""Read-only Phase-M training and evidence cockpit projection.

The module composes existing evidence readers, immutable package/index files,
the canonical forward-capture readiness contract, and a dashboard job journal.
It owns no signing key and exposes no mutation function.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import sha256_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DOMAINS = ("fx", "equity", "crypto", "filings", "multi_asset", "portfolio", "outcomes", "evidence")
DATA_TIERS = ("raw", "normalized", "snapshots", "features", "history")
_DATA_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
_DATA_CACHE_LOCK = threading.Lock()
_DATA_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class ReaderSpec:
    family: str
    module: str
    function: str
    lane_prefixes: tuple[str, ...]
    exact_lanes: tuple[str, ...] = ()

    def owns(self, lane_id: str) -> bool:
        return lane_id in self.exact_lanes or lane_id.startswith(self.lane_prefixes)


READER_SPECS = (
    ReaderSpec("risk_target", "src.evidence.risk_target.dashboard", "risk_target_evidence_view", ("risk_target_",)),
    ReaderSpec("hedge_eval", "src.evidence.hedge_eval.dashboard", "hedge_evidence_view", ("hedge_eval_",)),
    ReaderSpec("equity_research", "src.evidence.equity_research.dashboard", "equity_research_evidence_view", ("equity_research_",)),
    ReaderSpec("crypto_momentum", "src.evidence.crypto_momentum.dashboard", "crypto_momentum_evidence_view", ("crypto_momentum_",)),
    ReaderSpec("crypto_carry", "src.evidence.crypto_carry.dashboard", "crypto_carry_evidence_view", ("crypto_carry_",)),
    ReaderSpec("track_b", "src.evidence.track_b.dashboard", "track_b_evidence_view", ("track_b_",)),
    ReaderSpec("execution_cost", "src.evidence.execution_cost.dashboard", "execution_cost_evidence_view", (), ("execution_cost_model",)),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def _age_seconds(path: Path, now: datetime) -> float | None:
    try:
        return max(0.0, now.timestamp() - path.stat().st_mtime)
    except OSError:
        return None


def _timestamp_age_seconds(value: object, now: datetime) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds())
    except ValueError:
        return None


def _capture_stream_summary(
    data_root: Path,
    domain: str,
    family: str,
    source_key: str,
) -> dict[str, Any]:
    """Read a bounded summary from one canonical append-only history stream."""
    stream_id = f"forward-capture:{family}:{source_key}"
    component = hashlib.sha256(stream_id.encode("utf-8")).hexdigest()[:32]
    root = data_root / domain / "history" / component
    try:
        segments = sorted(root.glob("*.jsonl"))
    except OSError:
        segments = []
    if not segments:
        return {
            "source_key": source_key,
            "canonical_batches": 0,
            "last_canonical_at": None,
        }
    last = _json(segments[-1], {})
    if not isinstance(last, Mapping):
        last = {}
    return {
        "source_key": source_key,
        "canonical_batches": len(segments),
        "last_canonical_at": last.get("captured_at_utc") or last.get("observed_at_utc"),
    }


def read_live_capture_status(
    repo_root: str | Path,
    *,
    data_root: str | Path,
    required_pairs: list[str] | tuple[str, ...],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Project current writer and canonical-accrual truth independently of P2."""
    repo_root = Path(repo_root)
    data_root = Path(data_root)
    now = (now or _utc_now()).astimezone(timezone.utc)
    pair_rows = [
        _capture_stream_summary(data_root, "fx", "tick_spread", str(pair))
        for pair in required_pairs
    ]
    latest_tick = max(
        (str(row["last_canonical_at"]) for row in pair_rows if row.get("last_canonical_at")),
        default=None,
    )
    latest_tick_age = _timestamp_age_seconds(latest_tick, now)
    pairs_with_records = sum(row["canonical_batches"] > 0 for row in pair_rows)
    current_pairs = sum(
        (age := _timestamp_age_seconds(row.get("last_canonical_at"), now)) is not None
        and age <= 900
        for row in pair_rows
    )

    health_path = repo_root / "trained_data" / "ticks" / "_health.json"
    health = _json(health_path, {})
    if not isinstance(health, Mapping):
        health = {}
    health_age = _age_seconds(health_path, now)
    try:
        health_updated_at = _iso(
            datetime.fromtimestamp(health_path.stat().st_mtime, timezone.utc)
        )
    except OSError:
        health_updated_at = None
    writer_status = str(health.get("stream_status") or "unavailable")
    if health_age is None or health_age > 120:
        writer_status = "health_stale"

    writer_current = writer_status in {"running", "slow"}
    if not pair_rows or pairs_with_records == 0:
        tick_status = "unavailable"
    elif (
        latest_tick_age is not None
        and latest_tick_age <= 120
        and current_pairs == len(pair_rows)
        and writer_current
    ):
        tick_status = "accruing"
    elif latest_tick_age is not None and latest_tick_age <= 900 and writer_status != "dead":
        tick_status = "delayed"
    else:
        tick_status = "stalled"

    exposure = _capture_stream_summary(
        data_root,
        "portfolio",
        "exposure",
        "oanda_fx_trend_lane",
    )
    exposure_age = _timestamp_age_seconds(exposure.get("last_canonical_at"), now)
    if exposure["canonical_batches"] == 0:
        exposure_status = "unavailable"
    # The resident writer checks every 15 minutes, while the source account
    # snapshot normally advances about hourly. Dedupe is expected between
    # source updates and must not be presented as a stopped writer.
    elif exposure_age is not None and exposure_age <= 4_500:
        exposure_status = "accruing"
    elif exposure_age is not None and exposure_age <= 9_000:
        exposure_status = "delayed"
    else:
        exposure_status = "stalled"

    return {
        "tick": {
            "status": tick_status,
            "writer_status": writer_status,
            "canonical_batches": sum(int(row["canonical_batches"]) for row in pair_rows),
            "pairs_with_records": pairs_with_records,
            "pairs_current": current_pairs,
            "required_pairs": len(pair_rows),
            "last_canonical_at": latest_tick,
            "last_canonical_age_seconds": latest_tick_age,
            "health_updated_at": health_updated_at,
            "health_age_seconds": health_age,
            "ticks_received_session": health.get("ticks_received_total"),
            "ticks_written_session": health.get("ticks_written"),
            "buffered_ticks": health.get("buffered_ticks"),
            "ticks_per_minute": health.get("ticks_per_minute"),
            "flushes": health.get("flushes"),
            "flush_errors": health.get("flush_errors"),
            "last_flush_at": health.get("last_flush_at"),
            "last_flush_error": health.get("last_flush_error"),
        },
        "exposure": {
            "status": exposure_status,
            "canonical_snapshots": exposure["canonical_batches"],
            "last_canonical_at": exposure.get("last_canonical_at"),
            "last_canonical_age_seconds": exposure_age,
        },
    }


def _resolve_axiom_uri(data_root: Path, uri: object) -> Path | None:
    if not isinstance(uri, str) or not uri.startswith("axiom-data://"):
        return None
    relative = uri.removeprefix("axiom-data://")
    candidate = (data_root / relative).resolve()
    try:
        candidate.relative_to(data_root.resolve())
    except ValueError:
        return None
    return candidate


def read_data_status(
    repo_root: str | Path = REPO_ROOT,
    *,
    data_root: str | Path | None = None,
    control_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Summarize canonical data manifests and the P2 forward-only gate."""
    repo_root = Path(repo_root)
    data_root = Path(data_root or os.getenv("AXIOM_DATA_ROOT", repo_root / "axiom-data"))
    control_root = Path(control_root or repo_root / "trained_data" / "forward_capture")
    now = (now or _utc_now()).astimezone(timezone.utc)
    failures: list[dict[str, str]] = []
    domains: list[dict[str, Any]] = []

    for domain in DATA_DOMAINS:
        domain_root = data_root / domain
        tier_rows: list[dict[str, Any]] = []
        for tier in DATA_TIERS:
            tier_root = domain_root / tier
            manifests = (
                sorted(
                    path for path in tier_root.rglob("manifest.json")
                    if not any(part.startswith(".") for part in path.relative_to(tier_root).parts)
                )
                if tier_root.is_dir()
                else []
            )
            latest = max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None
            partition_count = 0
            for manifest_path in manifests:
                payload = _json(manifest_path)
                if not isinstance(payload, Mapping):
                    failures.append({"kind": "invalid_manifest", "path": str(manifest_path.relative_to(data_root))})
                    continue
                manifest_payload = payload.get("payload") if isinstance(payload.get("payload"), Mapping) else payload
                partitions = manifest_payload.get("partitions", []) if isinstance(manifest_payload, Mapping) else []
                if not isinstance(partitions, list):
                    failures.append({"kind": "invalid_partitions", "path": str(manifest_path.relative_to(data_root))})
                    continue
                partition_count += len(partitions)
                for partition in partitions:
                    if not isinstance(partition, Mapping):
                        failures.append({"kind": "invalid_partition", "path": str(manifest_path.relative_to(data_root))})
                        continue
                    resolved = _resolve_axiom_uri(data_root, partition.get("uri"))
                    if resolved is not None and not resolved.is_file():
                        failures.append({"kind": "missing_partition", "path": str(resolved.relative_to(data_root))})
            tier_rows.append({
                "tier": tier,
                "manifest_count": len(manifests),
                "partition_count": partition_count,
                "latest_manifest": str(latest.relative_to(data_root)) if latest else None,
                "latest_manifest_at": _iso(datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc)) if latest else None,
                "freshness_age_seconds": _age_seconds(latest, now) if latest else None,
            })
        domains.append({"domain": domain, "available": domain_root.is_dir(), "tiers": tier_rows})

    readiness_path = control_root / "p2_readiness.json"
    readiness_raw = _json(readiness_path)
    readiness: dict[str, Any]
    if isinstance(readiness_raw, Mapping):
        try:
            from src.data_platform.forward_capture import P2ReadinessReport

            report = P2ReadinessReport.model_validate_json(json.dumps(dict(readiness_raw)))
            readiness = report.model_dump(mode="json", round_trip=True)
            readiness["available"] = True
            readiness["source"] = str(readiness_path)
            readiness["age_seconds"] = _age_seconds(readiness_path, now)
        except Exception as exc:  # noqa: BLE001 - corrupt status is a visible quality failure
            readiness = {"available": False, "ready": False, "source": str(readiness_path), "error": f"invalid P2 readiness report: {exc}"}
            failures.append({"kind": "invalid_p2_readiness", "path": str(readiness_path)})
    else:
        readiness = {"available": False, "ready": False, "source": str(readiness_path), "error": "P2 readiness report is absent"}

    return {
        "data_root": str(data_root),
        "domains": domains,
        "quality_failure_count": len(failures),
        "quality_failures": failures[:100],
        "quality_failures_truncated": len(failures) > 100,
        "p2_readiness": readiness,
    }


def _reader(spec: ReaderSpec) -> Callable[[str | Path], dict]:
    module = importlib.import_module(spec.module)
    function = getattr(module, spec.function)
    if not callable(function):
        raise TypeError(f"{spec.module}.{spec.function} is not callable")
    return function


def _package_metadata(evidence_root: Path, package_digest: str) -> dict[str, Any]:
    envelope = _json(evidence_root / "packages" / package_digest / "envelope.json", {})
    payload = envelope.get("payload", {}) if isinstance(envelope, Mapping) else {}
    if not isinstance(payload, Mapping):
        return {}
    return {
        "created_at": payload.get("created_at"),
        "job_manifest_digest": payload.get("job_manifest_digest"),
        "dataset_manifest_digests": payload.get("dataset_manifest_digests", []),
        "derived_from_package_digests": payload.get("derived_from_package_digests", []),
        "artifacts": payload.get("artifacts", []),
        "safety_assertions": payload.get("safety_assertions", []),
    }


def _verdicts(evidence_root: Path) -> dict[str, dict[str, Any]]:
    by_package: dict[str, dict[str, Any]] = {}
    verdict_root = evidence_root / "verdicts"
    if not verdict_root.is_dir():
        return by_package
    for path in sorted(verdict_root.glob("*.json")):
        envelope = _json(path, {})
        payload = envelope.get("payload", {}) if isinstance(envelope, Mapping) else {}
        if not isinstance(payload, Mapping):
            continue
        digest = payload.get("package_digest")
        checks = payload.get("checks", [])
        if not isinstance(digest, str) or not isinstance(checks, list):
            continue
        normalized_checks = [dict(check) for check in checks if isinstance(check, Mapping)]
        by_package[digest] = {
            "decision": payload.get("decision"),
            "rejection_reason": payload.get("rejection_reason"),
            "checks": normalized_checks,
            "failed_checks": [check.get("check_id") for check in normalized_checks if check.get("passed") is False],
            "verdict_digest": envelope.get("payload_digest"),
        }
    return by_package


def read_evidence_status(evidence_root: str | Path) -> dict[str, Any]:
    """Normalize every installed slice reader into one lane/package view."""
    evidence_root = Path(evidence_root)
    index = _json(evidence_root / "indexes" / "current.json", {})
    packages = index.get("packages", {}) if isinstance(index, Mapping) else {}
    champions = index.get("champions", {}) if isinstance(index, Mapping) else {}
    if not isinstance(packages, Mapping):
        packages = {}
    if not isinstance(champions, Mapping):
        champions = {}
    verdicts = _verdicts(evidence_root)
    reader_health: list[dict[str, Any]] = []
    specialized: dict[str, dict[str, Any]] = {}

    for spec in READER_SPECS:
        try:
            view = _reader(spec)(evidence_root)
            if not isinstance(view, Mapping):
                raise TypeError("reader returned a non-object")
            reader_health.append({"family": spec.family, "available": bool(view.get("available")), "reason": view.get("reason")})
            for row in view.get("lanes", []) if isinstance(view.get("lanes", []), list) else []:
                if not isinstance(row, Mapping):
                    continue
                lane_id = str(row.get("lane_id") or "")
                digest = str(row.get("package_digest") or "")
                if lane_id and digest and spec.owns(lane_id):
                    specialized[digest] = {**dict(row), "family": spec.family}
        except Exception as exc:  # noqa: BLE001 - one optional lane cannot blank the cockpit
            reader_health.append({"family": spec.family, "available": False, "reason": f"{type(exc).__name__}: {exc}"})

    lanes: list[dict[str, Any]] = []
    for package_digest, raw in sorted(packages.items()):
        if not isinstance(package_digest, str) or not isinstance(raw, Mapping):
            continue
        lane_id = str(raw.get("lane_id") or "unknown")
        row = dict(specialized.get(package_digest, {}))
        family = row.get("family")
        if not family:
            family = next((spec.family for spec in READER_SPECS if spec.owns(lane_id)), "other")
        verdict = verdicts.get(package_digest, {})
        state = raw.get("state")
        lane = {
            **row,
            "family": family,
            "lane_id": lane_id,
            "package_id": raw.get("package_id"),
            "package_digest": package_digest,
            "state": state,
            "sequence": raw.get("sequence"),
            "disposition_head_digest": raw.get("head_event_digest"),
            "artifact_digests": raw.get("artifact_digests", {}),
            "verification": verdict,
            "gate_results": verdict.get("checks", []),
            "negative_result": state == "REJECTED" or verdict.get("decision") == "REJECT",
            "is_champion": isinstance(champions.get(lane_id), Mapping) and champions[lane_id].get("package_digest") == package_digest,
            **_package_metadata(evidence_root, package_digest),
        }
        lanes.append(lane)
    lanes.sort(key=lambda row: (str(row.get("family")), str(row.get("lane_id")), str(row.get("created_at") or "")), reverse=False)

    by_lane: dict[str, list[dict[str, Any]]] = {}
    for row in lanes:
        by_lane.setdefault(str(row["lane_id"]), []).append(row)
    lane_summaries = []
    for lane_id, rows in sorted(by_lane.items()):
        ordered = sorted(rows, key=lambda row: str(row.get("created_at") or ""), reverse=True)
        champion = champions.get(lane_id) if isinstance(champions.get(lane_id), Mapping) else None
        retired = [row for row in ordered if row.get("state") == "RETIRED"]
        lane_summaries.append({
            "lane_id": lane_id,
            "family": ordered[0].get("family"),
            "current": ordered[0],
            "champion": dict(champion) if champion else None,
            "prior_champion": retired[0] if retired else None,
            "package_count": len(rows),
            "negative_count": sum(bool(row.get("negative_result")) for row in rows),
        })

    state_counts: dict[str, int] = {}
    for row in lanes:
        state = str(row.get("state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1
    return {
        "available": isinstance(index, Mapping) and bool(index),
        "source": str(evidence_root / "indexes" / "current.json"),
        "readers": reader_health,
        "packages": lanes,
        "lanes": lane_summaries,
        "state_counts": state_counts,
        "negative_result_count": sum(bool(row.get("negative_result")) for row in lanes),
        "champions": dict(champions),
    }


def _journal_jobs(job_root: Path) -> list[dict[str, Any]]:
    jobs = []
    if job_root.is_dir():
        for path in sorted(job_root.glob("*.json")):
            row = _json(path)
            if isinstance(row, Mapping):
                jobs.append(dict(row))
    return jobs


def read_jobs(evidence: Mapping[str, Any], job_root: str | Path) -> dict[str, Any]:
    """Merge transient dashboard jobs with completed immutable package lineage."""
    jobs_by_id = {
        str(row.get("job_id")): row
        for row in _journal_jobs(Path(job_root))
        if row.get("job_id")
    }
    by_manifest: dict[str, dict[str, Any]] = {}
    for package in evidence.get("packages", []) if isinstance(evidence.get("packages", []), list) else []:
        if not isinstance(package, Mapping):
            continue
        digest = package.get("job_manifest_digest")
        if not isinstance(digest, str) or not digest:
            continue
        job = by_manifest.setdefault(digest, {
            "job_id": f"observed-{digest[:12]}",
            "source": "immutable_evidence",
            "status": "completed",
            "manifest_digest": digest,
            "container_digest": None,
            "resource_class": None,
            "resource_usage": None,
            "cost": None,
            "submitted_at": package.get("created_at"),
            "started_at": package.get("created_at"),
            "completed_at": package.get("created_at"),
            "lanes": [],
            "package_digests": [],
        })
        job["lanes"].append(package.get("lane_id"))
        job["package_digests"].append(package.get("package_digest"))
        if package.get("negative_result"):
            job.setdefault("negative_lanes", []).append(package.get("lane_id"))
    for digest, observed in by_manifest.items():
        if not any(row.get("manifest_digest") == digest for row in jobs_by_id.values()):
            jobs_by_id[observed["job_id"]] = observed
    jobs = sorted(jobs_by_id.values(), key=lambda row: str(row.get("submitted_at") or ""), reverse=True)
    counts = {status: sum(row.get("status") == status for row in jobs) for status in ("submitted", "running", "failed", "completed")}
    return {"jobs": jobs, "counts": counts, "total": len(jobs), "source": str(job_root)}


def read_forward_monitoring(evidence: Mapping[str, Any], monitor_root: str | Path) -> dict[str, Any]:
    """Read optional, factual per-lane forward telemetry without fabricating it."""
    monitor_root = Path(monitor_root)
    by_lane: dict[str, dict[str, Any]] = {}
    now = _utc_now()
    if monitor_root.is_dir():
        for path in sorted(monitor_root.glob("*.json")):
            row = _json(path)
            if not isinstance(row, Mapping) or not isinstance(row.get("lane_id"), str):
                continue
            by_lane[row["lane_id"]] = {
                **dict(row),
                "available": True,
                "source": str(path),
                "shadow_age_seconds": _timestamp_age_seconds(row.get("shadow_started_at"), now),
            }
    for lane in evidence.get("lanes", []) if isinstance(evidence.get("lanes", []), list) else []:
        if not isinstance(lane, Mapping):
            continue
        lane_id = str(lane.get("lane_id") or "")
        current = lane.get("current", {}) if isinstance(lane.get("current"), Mapping) else {}
        if lane_id and lane_id not in by_lane:
            by_lane[lane_id] = {
                "lane_id": lane_id,
                "package_digest": current.get("package_digest"),
                "state": current.get("state"),
                "available": False,
                "status": "not_reporting",
                "shadow_started_at": None,
                "shadow_age_seconds": None,
                "updated_at": None,
                "forward_sharpe": None,
                "drawdown": None,
                "turnover": None,
                "cost": None,
                "exposure": None,
                "baseline_deviation": None,
                "retirement_warnings": [],
            }
    rows = [by_lane[key] for key in sorted(by_lane)]
    return {"lanes": rows, "reporting_count": sum(bool(row.get("available")) for row in rows), "total": len(rows), "source": str(monitor_root)}


def read_control_readiness(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = Path(repo_root)
    descriptor_path = Path(os.getenv(
        "AXIOM_RISK_TARGET_DATASET",
        repo_root / "market_data" / "factor" / "axiom_risk_target_dataset.json",
    ))
    trust_path = Path(os.getenv("AXIOM_OPERATOR_TRUST", repo_root / "trained_data" / "axiom" / "operator_trust.json"))
    descriptor_digest = None
    request_template = None
    request_subject_digest = None
    if descriptor_path and descriptor_path.is_file():
        try:
            descriptor_digest = sha256_bytes(descriptor_path.read_bytes())
            request_template = {"dataset_sha256": descriptor_digest, "lane": "risk_target"}
            request_subject_digest = sha256_bytes(canonical_bytes(request_template))
        except OSError:
            descriptor_digest = None
    enabled = os.getenv("AXIOM_CONTROL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return {
        "enabled": enabled,
        "operator_trust_configured": trust_path.is_file(),
        "dataset_configured": bool(descriptor_digest),
        "dataset_descriptor": str(descriptor_path),
        "dataset_sha256": descriptor_digest,
        "run_request_template": request_template,
        "run_subject_digest": request_subject_digest,
        "operator_private_key_on_server": False,
        "actions": ["run_risk_target_to_quarantine", "relay_operator_signed_disposition"],
        "invariants": [
            "Ed25519 operator request credential is bound to the exact action and subject",
            "single-use nonce and expiry are enforced before work begins",
            "run terminates at QUARANTINED or REJECTED",
            "the dashboard cannot sign a disposition or write a champion pointer",
            "LiveGate, lane halts, and champion verification remain downstream promotion-service gates",
        ],
    }


def read_training_cockpit(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = Path(repo_root)
    evidence_root = Path(os.getenv("AXIOM_EVIDENCE_ROOT", repo_root / "trained_data" / "evidence"))
    job_root = Path(os.getenv("AXIOM_JOB_JOURNAL", evidence_root / "dashboard_jobs"))
    monitor_root = Path(os.getenv("AXIOM_FORWARD_MONITOR_ROOT", evidence_root / "forward_monitoring"))
    data_root = str(Path(os.getenv("AXIOM_DATA_ROOT", repo_root / "axiom-data")))
    control_root = str(repo_root / "trained_data" / "forward_capture")
    cache_key = (str(repo_root), data_root, control_root)
    with _DATA_CACHE_LOCK:
        cached = _DATA_CACHE.get(cache_key)
        if cached is not None and time.monotonic() - cached[0] < _DATA_CACHE_TTL_SECONDS:
            data = cached[1]
        else:
            data = None
    if data is None:
        data = read_data_status(repo_root, data_root=data_root, control_root=control_root)
        with _DATA_CACHE_LOCK:
            _DATA_CACHE[cache_key] = (time.monotonic(), data)
    # Canonical live accrual is deliberately outside the 30-second manifest
    # cache so a page refresh cannot report a dead or advancing writer from an
    # old data-inventory snapshot.
    readiness = data.get("p2_readiness", {})
    required_pairs = readiness.get("required_pairs", []) if isinstance(readiness, Mapping) else []
    data = {
        **data,
        "capture": read_live_capture_status(
            repo_root,
            data_root=data_root,
            required_pairs=[str(pair) for pair in required_pairs],
        ),
    }
    evidence = read_evidence_status(evidence_root)
    return {
        "connected": True,
        "generated_at": _iso(_utc_now()),
        "data": data,
        "jobs": read_jobs(evidence, job_root),
        "evidence": evidence,
        "forward_monitoring": read_forward_monitoring(evidence, monitor_root),
        "controls": read_control_readiness(repo_root),
    }


__all__ = [
    "READER_SPECS",
    "read_data_status",
    "read_live_capture_status",
    "read_evidence_status",
    "read_jobs",
    "read_forward_monitoring",
    "read_control_readiness",
    "read_training_cockpit",
]
