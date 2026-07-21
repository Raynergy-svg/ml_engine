"""Governed HTTP boundary for the Phase-M Training tab."""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from dashboard.server.training_cockpit import REPO_ROOT, read_training_cockpit
from dashboard.server.training_identity import require_runtime_git_commit
from dashboard.server.training_jobs import LocalTrainingJobRunner, TrainingJobJournal
from src.evidence.hashing import sha256_bytes

router = APIRouter(prefix="/api/axiom_training", tags=["axiom_training"])


class OperatorCredentialBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    subject: str
    nonce: str
    issued_at: str
    expires_at: str
    key_id: str
    signature_b64: str


class RunRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lane: Literal["risk_target"] = "risk_target"
    program: Literal["risk_target_baseline"]


class AxiomTrainingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential: OperatorCredentialBody
    request: RunRequestBody


class AxiomPromotionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    credential: OperatorCredentialBody
    disposition: Dict[str, Any]


def _enabled() -> bool:
    return os.getenv("AXIOM_CONTROL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(status_code=503, detail="AXIOM governed controls are disabled")


def _control():
    from dashboard.server.axiom_evidence_control import EvidenceControlPlane, OperatorTrustUnavailable

    try:
        return EvidenceControlPlane.from_config()
    except OperatorTrustUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"operator authority not configured: {exc}") from exc


def _safe_partition(base: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("dataset partition paths must be non-empty and relative")
    path = (base / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError("dataset partition path escapes the descriptor directory") from exc
    if not path.is_file():
        raise ValueError(f"dataset partition is absent: {relative}")
    return path


def _configured_cpu_hourly_rate() -> Decimal:
    raw = os.getenv("AXIOM_LOCAL_CPU_USD_PER_HOUR", "0")
    try:
        rate = Decimal(raw)
    except InvalidOperation as exc:
        raise HTTPException(status_code=503, detail="invalid local training cost policy") from exc
    if not rate.is_finite() or rate < 0:
        raise HTTPException(status_code=503, detail="invalid local training cost policy")
    return rate


def _load_run_dataset(request: RunRequestBody) -> tuple[dict[str, bytes], dict[str, Any], str]:
    descriptor_path = Path(os.getenv(
        "AXIOM_RISK_TARGET_DATASET",
        REPO_ROOT / "market_data" / "factor" / "axiom_risk_target_dataset.json",
    ))
    if not descriptor_path.is_file():
        raise HTTPException(status_code=503, detail="risk-target run dataset descriptor is absent")
    try:
        import json

        descriptor_bytes = descriptor_path.read_bytes()
        actual_digest = sha256_bytes(descriptor_bytes)
        if request.dataset_sha256 != actual_digest:
            raise HTTPException(status_code=422, detail="signed request does not bind the configured dataset descriptor")
        descriptor = json.loads(descriptor_bytes.decode("utf-8"))
        if not isinstance(descriptor, dict):
            raise ValueError("dataset descriptor must be an object")
        declared_partitions = descriptor["partitions"]
        if not isinstance(declared_partitions, dict) or not declared_partitions:
            raise ValueError("dataset descriptor must declare at least one partition")
        base = descriptor_path.resolve().parent
        partitions = {
            str(instrument): _safe_partition(base, relative).read_bytes()
            for instrument, relative in sorted(declared_partitions.items())
        }
        now = datetime.now(timezone.utc)
        job_id = str(descriptor.get("job_id") or f"dashboard-risk-target-{now.strftime('%Y%m%dT%H%M%SZ')}")
        kwargs = {
            "dataset_id": str(descriptor["dataset_id"]),
            "coverage_start": date.fromisoformat(str(descriptor["coverage_start"])),
            "coverage_end": date.fromisoformat(str(descriptor["coverage_end"])),
            "git_commit": require_runtime_git_commit(REPO_ROOT),
            "retrieved_at": now,
            "created_at": now,
            "job_id": job_id,
        }
    except HTTPException:
        raise
    except (KeyError, OSError, RuntimeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=f"risk-target run dataset is unreadable: {exc}") from exc
    return partitions, kwargs, job_id


def _journal() -> TrainingJobJournal:
    evidence_root = Path(os.getenv("AXIOM_EVIDENCE_ROOT", REPO_ROOT / "trained_data" / "evidence"))
    return TrainingJobJournal(os.getenv("AXIOM_JOB_JOURNAL", evidence_root / "dashboard_jobs"))


@lru_cache(maxsize=8)
def _runner(journal_root: str) -> LocalTrainingJobRunner:
    return LocalTrainingJobRunner(TrainingJobJournal(journal_root))


def _completion_fields(
    result: Dict[str, Any], elapsed: float, cpu_hourly_rate: Decimal = Decimal("0")
) -> Dict[str, Any]:
    manifest = result.get("job_manifest", {})
    cost = (Decimal(str(elapsed)) * cpu_hourly_rate / Decimal("3600")).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    return {
        "manifest_digest": result.get("job_manifest_digest"),
        "git_commit": manifest.get("git_commit"),
        "container_digest": manifest.get("container_digest"),
        "configuration_digest": manifest.get("configuration_digest"),
        "feature_pipeline_version": manifest.get("feature_pipeline_version"),
        "assigned_unit": manifest.get("assigned_unit"),
        "trial_budget": manifest.get("trial_budget"),
        "resource_class": manifest.get("resource_class"),
        "resource_usage": {"wall_seconds": elapsed},
        "cost": float(cost),
        "cost_currency": "USD",
        "cost_basis": (
            "local_unmetered_no_incremental_provider_charge"
            if cpu_hourly_rate == 0
            else "configured_local_cpu_hourly_rate"
        ),
        "cost_rate_per_hour": float(cpu_hourly_rate),
        "outcomes": result.get("outcomes", {}),
    }


@router.get("")
def training_cockpit() -> Dict[str, Any]:
    try:
        return read_training_cockpit()
    except Exception as exc:  # noqa: BLE001 - return a truthful degraded state
        return {"connected": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/run", status_code=202)
def run_training(body: AxiomTrainingRunRequest) -> Dict[str, Any]:
    from dashboard.server.axiom_evidence_control import AuthorizationError, OperatorCredential

    # Authorization and nonce consumption happen before the 202 response.  The
    # queued work item has evidence-production authority only and is executed
    # by one bounded local worker; there is no distributed orchestrator here.
    _require_enabled()
    cpu_hourly_rate = _configured_cpu_hourly_rate()
    control = _control()
    request = body.request.model_dump(mode="json")
    partitions, slice_kwargs, base_job_id = _load_run_dataset(body.request)
    job_id = f"{base_job_id}-{sha256_bytes(body.credential.nonce.encode('utf-8'))[:12]}"
    slice_kwargs = {
        **slice_kwargs,
        "job_id": job_id,
        "cost_rate_per_hour": float(cpu_hourly_rate),
    }
    try:
        credential = OperatorCredential.from_mapping(body.credential.model_dump(mode="json"))
        journal = _journal()
        work_item = control.prepare_run(
            credential,
            request_body=request,
            partitions=partitions,
            slice_kwargs=slice_kwargs,
        )
        journal.transition(
            job_id,
            "submitted",
            fields={
                "lane": "risk_target",
                "program": body.request.program,
                "source": "dashboard",
            },
        )
        _runner(str(journal.root.resolve())).submit(
            job_id,
            work_item,
            completion_fields=lambda result, elapsed: _completion_fields(
                result, elapsed, cpu_hourly_rate
            ),
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - persist queue failure and fail closed
        try:
            _journal().transition(
                job_id,
                "failed",
                fields={"error": f"{type(exc).__name__}: {exc}"},
            )
        except ValueError:
            pass
        raise HTTPException(status_code=503, detail="authorized training job could not be queued") from exc
    return {"ok": True, "job_id": job_id, "status": "submitted"}


@router.post("/promote")
def relay_promotion(body: AxiomPromotionRequest) -> Dict[str, Any]:
    from dashboard.server.axiom_evidence_control import AuthorizationError, OperatorCredential
    from src.evidence.store import EvidenceStoreError

    _require_enabled()
    control = _control()
    try:
        credential = OperatorCredential.from_mapping(body.credential.model_dump(mode="json"))
        result = control.promote(credential, disposition=body.disposition)
    except AuthorizationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except (EvidenceStoreError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "result": result}


__all__ = ["router"]
