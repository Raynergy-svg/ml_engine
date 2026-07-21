"""Deterministic Track B filing ingestion, blinding and score-lineage records."""

from __future__ import annotations

import json
import math
import re
from datetime import date
from typing import Any, Mapping, Sequence

from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import sha256_bytes

SCORER_MODEL = "track-b-frozen-lexicon"
SCORER_VERSION = "1.0.0"
PROMPT_TEMPLATE = (
    "Score only the blinded filing text. Return quality, red flags, outlook, "
    "conviction, rationale and exact supporting spans. No outside knowledge."
)
PROMPT_DIGEST = sha256_bytes(PROMPT_TEMPLATE.encode("utf-8"))

POSITIVE = ("record revenue", "strong demand", "margin expansion", "free cash flow")
NEGATIVE = ("going concern", "material weakness", "restatement", "impairment")
OUTLOOK_POSITIVE = ("expect growth", "improving demand", "raised guidance")
OUTLOOK_NEGATIVE = ("expect decline", "weakening demand", "lowered guidance")

REQUIRED_SCORE_LINEAGE = (
    "filing_accession", "ticker", "composite", "document_hash", "filing_date", "as_of_date",
    "model_cutoff", "scorer_model", "scorer_version", "prompt_digest",
    "parameters", "rationale", "extracted_spans", "score_validation", "batch_id",
)


def _iso(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string") from exc
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _blind(text: str, entities: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    blinded = text
    normalized = sorted(
        {entity.strip() for entity in entities if entity.strip()},
        key=lambda value: (-len(value), value.casefold(), value),
    )
    if not normalized:
        raise ValueError("entities_to_blind must contain at least one identity")
    for entity in normalized:
        pattern = rf"(?<!\w){re.escape(entity)}(?!\w)"
        blinded = re.sub(pattern, "[ENTITY]", blinded, flags=re.IGNORECASE)
    residual = tuple(
        entity for entity in normalized
        if re.search(rf"(?<!\w){re.escape(entity)}(?!\w)", blinded, flags=re.IGNORECASE)
    )
    return blinded, residual


def _score_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    positive_hits = [phrase for phrase in POSITIVE if phrase in lower]
    negative_hits = [phrase for phrase in NEGATIVE if phrase in lower]
    outlook_positive = [phrase for phrase in OUTLOOK_POSITIVE if phrase in lower]
    outlook_negative = [phrase for phrase in OUTLOOK_NEGATIVE if phrase in lower]
    quality = max(-1.0, min(1.0, 0.25 * (len(positive_hits) - len(negative_hits))))
    red_flags = min(1.0, 0.25 * len(negative_hits))
    outlook = max(-1.0, min(1.0, 0.25 * (len(outlook_positive) - len(outlook_negative))))
    all_hits = positive_hits + negative_hits + outlook_positive + outlook_negative
    conviction = min(1.0, len(all_hits) / 4.0)
    spans: list[str] = []
    for phrase in all_hits:
        match = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
        if match:
            spans.append(text[match.start():match.end()])
    rationale = (
        f"Frozen lexical scorer found {len(positive_hits)} positive, "
        f"{len(negative_hits)} red-flag and "
        f"{len(outlook_positive) + len(outlook_negative)} outlook span(s)."
    )
    return {
        "fundamental_quality": quality,
        "accounting_red_flags": red_flags,
        "forward_outlook": outlook,
        "conviction": conviction,
        "rationale": rationale,
        "extracted_spans": spans,
        "composite": 0.5 * quality - 0.5 * red_flags,
    }


def score_filing(record: Mapping[str, Any], *, batch_id: str) -> dict[str, Any]:
    """Validate, blind and score one raw filing into a full §J5 lineage record."""
    accession = record.get("filing_accession")
    ticker = record.get("ticker")
    issuer_name = record.get("issuer_name")
    form = record.get("form")
    text = record.get("document_text")
    for field, value in (
        ("filing_accession", accession), ("ticker", ticker),
        ("issuer_name", issuer_name), ("form", form), ("document_text", text),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    declared_hash = record.get("document_hash")
    actual_hash = sha256_bytes(text.encode("utf-8"))
    if declared_hash != actual_hash:
        raise ValueError(f"filing {accession!r} document hash mismatch")
    filing_date = _iso(record.get("filing_date"), "filing_date")
    as_of_date = _iso(record.get("as_of_date"), "as_of_date")
    model_cutoff = _iso(record.get("model_cutoff"), "model_cutoff")
    if filing_date > as_of_date:
        raise ValueError(f"filing {accession!r} violates PIT cutoff")
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    entities = record.get("entities_to_blind")
    if not isinstance(entities, list) or not all(isinstance(value, str) for value in entities):
        raise ValueError("entities_to_blind must be a string list")
    normalized_entities = {value.strip().casefold() for value in entities}
    if ticker.casefold() not in normalized_entities or issuer_name.casefold() not in normalized_entities:
        raise ValueError("entities_to_blind must include both ticker and issuer_name")
    blinded, residual = _blind(text, entities)
    score = _score_text(blinded)
    validation_checks = {
        "document_hash_matches": True,
        "pit_cutoff_valid": True,
        "entity_blinding_complete": not residual,
        "spans_grounded": all(span in blinded for span in score["extracted_spans"]),
    }
    manual_review = bool(residual) or score["conviction"] == 0.0 or any(
        abs(_finite(score[field], field)) == 1.0
        for field in ("fundamental_quality", "forward_outlook")
    ) or score["accounting_red_flags"] == 1.0
    output = {
        "filing_accession": accession,
        "ticker": ticker,
        "issuer_identity_digest": sha256_bytes(f"{ticker}\0{issuer_name}".encode("utf-8")),
        "form": form,
        "document_hash": actual_hash,
        "filing_date": filing_date,
        "as_of_date": as_of_date,
        "model_cutoff": model_cutoff,
        "post_model_cutoff": filing_date > model_cutoff,
        "scorer_model": SCORER_MODEL,
        "scorer_version": SCORER_VERSION,
        "prompt_digest": PROMPT_DIGEST,
        "parameters": dict(parameters),
        "rationale": score["rationale"],
        "extracted_spans": score["extracted_spans"],
        "score_validation": {
            "status": "manual_review" if manual_review else "valid",
            "checks": validation_checks,
        },
        "batch_id": batch_id,
        "blinded_document_digest": sha256_bytes(blinded.encode("utf-8")),
        "fundamental_quality": score["fundamental_quality"],
        "accounting_red_flags": score["accounting_red_flags"],
        "forward_outlook": score["forward_outlook"],
        "conviction": score["conviction"],
        "composite": score["composite"],
    }
    output["score_record_digest"] = sha256_bytes(canonical_bytes(output))
    return output


def score_batch(partition_id: str, data: bytes, *, batch_id: str) -> tuple[bytes, bytes, dict[str, float]]:
    """Score a batch deterministically; duplicate accessions fail closed."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"batch {partition_id!r} is not valid UTF-8") from exc
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"batch {partition_id!r} line {line_number} is invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"batch {partition_id!r} line {line_number} is not an object")
        scored = score_filing(raw, batch_id=batch_id)
        accession = scored["filing_accession"]
        if accession in seen:
            raise ValueError(f"duplicate filing accession {accession!r} in batch")
        seen.add(accession)
        records.append(scored)
    if not records:
        raise ValueError(f"batch {partition_id!r} has no filings")
    ordered = sorted(records, key=lambda row: (row["filing_accession"], row["ticker"]))
    artifact = b"".join(canonical_bytes(row) + b"\n" for row in ordered)
    manual = [row for row in ordered if row["score_validation"]["status"] == "manual_review"]
    queue = b"".join(canonical_bytes(row) + b"\n" for row in manual)
    metrics = {
        "n_scored": float(len(ordered)),
        "n_manual_review": float(len(manual)),
        "n_post_model_cutoff": float(sum(bool(row["post_model_cutoff"]) for row in ordered)),
    }
    return artifact, queue, metrics


def validate_score_lineage(record: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_SCORE_LINEAGE if field not in record]
    if missing:
        raise ValueError(f"score record missing lineage fields: {missing}")
    supplied = record.get("score_record_digest")
    payload = dict(record)
    payload.pop("score_record_digest", None)
    if supplied != sha256_bytes(canonical_bytes(payload)):
        raise ValueError("score record digest mismatch")
    validation = record.get("score_validation")
    if not isinstance(validation, dict) or validation.get("status") not in {"valid", "manual_review"}:
        raise ValueError("score validation is malformed")


__all__ = [
    "SCORER_MODEL", "SCORER_VERSION", "PROMPT_DIGEST", "REQUIRED_SCORE_LINEAGE",
    "score_filing", "score_batch", "validate_score_lineage",
]
