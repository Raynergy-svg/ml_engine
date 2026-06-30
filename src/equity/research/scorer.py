"""§3.1 scorer layer of the Track B research-alpha pipeline.

See ``docs/experiment-equity-research-alpha-prereg-2026-06-30.md`` §3.1 and the
frozen :mod:`src.equity.research.contracts` spine.

This module is the *offline* scorer layer. There is intentionally NO Anthropic /
network / LLM call here: in this environment ``ANTHROPIC_API_KEY`` is absent and
the research fan-out is driven by SEPARATE subagents that read a prompt, emit a
JSON ``ResearchScore``, and write a versioned **scores artifact** to disk. The
harness (:mod:`src.equity.research.harness`) consumes that artifact. This module
therefore owns four things and nothing else:

1. **Task preparation** — :func:`prepare_scoring_task` COMPOSES the existing
   loader (:class:`FilingText`) and blinder
   (:func:`src.equity.research.entity_blinder.blind_filing`) into a
   :class:`ScoringTask`. It does not reimplement either.

   * ``blind=True`` is the protocol arm: the text fed to the scorer is the
     entity-blinded text, the ticker is bookkeeping only and never rendered into
     the prompt.
   * ``blind=False`` is an UN-BLINDED arm used to DEMONSTRATE lookahead
     contamination — raw text (which names the issuer) is passed through so a
     pretrained model can recall the company's known future. It exists to
     produce the contaminated baseline the pre-registration's clean arms are
     compared against.

2. **Prompt rendering** — :func:`scoring_task_to_prompt` renders the instruction
   a subagent follows to emit the §3.1 JSON. When ``blind=True`` the prompt
   carries NO issuer identity.

3. **JSON ingest** — :func:`research_score_from_json` parses + validates a
   subagent's JSON output into a :class:`ResearchScore`, calling ``.validate()``
   (raises on out-of-range) and raising a specific :class:`ScoreParseError` on
   malformed input. It NEVER silently defaults a missing or malformed field.

4. **Deterministic baseline + artifact I/O** — :class:`BaselineScorer` is a
   no-LLM, no-RNG control implementing the :class:`ResearchScorer` Protocol from
   fixed lexical features. :func:`write_scores_artifact` /
   :func:`read_scores_artifact` round-trip a versioned, atomically-written JSON
   artifact; the reader REFUSES (raises) on a pipeline-version mismatch, mirroring
   the inference-contract version discipline elsewhere in the repo.

Pure offline + deterministic: no network, no LLM, no RNG, atomic writes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.equity.research.contracts import (
    RESEARCH_PIPELINE_VERSION,
    BlindedText,
    FilingText,
    ResearchScore,
)
from src.equity.research.entity_blinder import IssuerBundle, blind_filing

# Artifact format version — independent of RESEARCH_PIPELINE_VERSION's meaning but
# tagged WITH it: an artifact written under pipeline version X must refuse to load
# under pipeline version Y (see read_scores_artifact).
SCORES_ARTIFACT_SCHEMA = "research_scores"


class ScoreParseError(ValueError):
    """Raised when an LLM/subagent JSON payload cannot be parsed into a
    schema-valid :class:`ResearchScore`. A specific type so callers can
    distinguish a malformed score (abstain / drop the name) from an unexpected
    bug — and so a malformed score is NEVER silently defaulted."""


class ArtifactVersionError(ValueError):
    """Raised by :func:`read_scores_artifact` when the on-disk artifact's
    ``pipeline_version`` does not match the runtime ``RESEARCH_PIPELINE_VERSION``.
    Cross-version scores carry silent skew (the blinder ruleset / scorer schema
    may have changed) and must not be consumed."""


# --------------------------------------------------------------------------- #
# 1. Task preparation — compose loader + blinder.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringTask:
    """One unit of work for the offline research pass.

    Carries everything a subagent needs to emit a :class:`ResearchScore`, plus
    the bookkeeping the harness needs to re-attach the score to an issuer.

    Invariants:

    * ``ticker`` is BOOKKEEPING ONLY. When ``blinded`` is True it must never be
      rendered into the prompt (:func:`scoring_task_to_prompt` enforces this) —
      it exists so the harness can map the returned score back to a name.
    * ``text_to_score`` is the blinded text when ``blinded`` is True, or the raw
      (issuer-naming) text when ``blinded`` is False (the contamination-demo arm).
    * ``audit`` is the blinder's audit dict when ``blinded`` is True; ``None``
      otherwise (raw passthrough has nothing to audit).
    * ``truncated`` / ``original_chars`` / ``scored_chars`` record the head-keep
      truncation applied so downstream LLM scoring stays affordable on full 10-Ks.
    """

    ticker: str
    as_of: str
    form: str
    text_to_score: str
    blinded: bool
    audit: Optional[Dict[str, object]] = None
    truncated: bool = False
    original_chars: int = 0
    scored_chars: int = 0


def _truncate_head(text: str, max_chars: int) -> Tuple[str, bool]:
    """Keep the HEAD of ``text`` up to ``max_chars``; report whether truncated.

    The head of a 10-K carries the business overview / MD&A / risk factors that
    the §3.1 score reads — keeping the head (not a middle window) is the right
    affordability cut. Returns ``(kept_text, was_truncated)``.
    """
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0, got {max_chars!r}")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def prepare_scoring_task(
    filing: FilingText,
    bundle: IssuerBundle,
    *,
    blind: bool = True,
    max_chars: int = 12000,
) -> ScoringTask:
    """Compose loader output + blinder into a :class:`ScoringTask`.

    Args:
        filing: a PIT :class:`FilingText` from the loader (raw primary-doc text).
        bundle: issuer deny-list metadata for the blinder.
        blind: ``True`` (default, the protocol arm) runs
            :func:`blind_filing` and scores the entity-blinded text. ``False``
            passes the RAW issuer-naming text through — the UN-BLINDED arm that
            DEMONSTRATES lookahead contamination (a pretrained model can recall
            the named issuer's future). ``bundle`` is unused when ``blind`` is
            False.
        max_chars: head-keep truncation ceiling applied to the (blinded or raw)
            text so full 10-Ks stay affordable for the LLM pass.

    Returns:
        A :class:`ScoringTask`. ``ticker`` is bookkeeping only and is omitted
        from the prompt when ``blind`` is True.
    """
    if blind:
        blinded: BlindedText = blind_filing(filing, bundle)
        source_text = blinded.text
        audit: Optional[Dict[str, object]] = blinded.audit
    else:
        source_text = filing.text
        audit = None

    original_chars = len(source_text)
    text_to_score, truncated = _truncate_head(source_text, max_chars)

    return ScoringTask(
        ticker=filing.ticker,
        as_of=filing.as_of,
        form=filing.form,
        text_to_score=text_to_score,
        blinded=blind,
        audit=audit,
        truncated=truncated,
        original_chars=original_chars,
        scored_chars=len(text_to_score),
    )


# --------------------------------------------------------------------------- #
# 2. Prompt rendering.
# --------------------------------------------------------------------------- #
_SCHEMA_BLOCK = (
    "{\n"
    '  "fundamental_quality": <float in [-1, 1]>,\n'
    '  "accounting_red_flags": <float in [0, 1]>,\n'
    '  "forward_outlook": <float in [-1, 1]>,\n'
    '  "conviction": <float in [0, 1]>,\n'
    '  "rationale": "<one or two sentences, <= 400 chars>",\n'
    '  "spans_used": ["<verbatim quote from the text>", "..."]\n'
    "}"
)

_FIELD_GUIDE = (
    "- fundamental_quality [-1, 1]: balance-sheet / margin / cash-flow strength "
    "as evidenced IN THE TEXT. +1 = exceptionally strong, 0 = neutral/unclear, "
    "-1 = severely impaired.\n"
    "- accounting_red_flags [0, 1]: weight of going-concern language, "
    "restatements, impairments, control weaknesses, aggressive recognition. "
    "0 = none evident, 1 = severe.\n"
    "- forward_outlook [-1, 1]: management's forward guidance / demand signals. "
    "+1 = strongly positive, -1 = strongly negative. (Logged; zero-weighted in "
    "the primary composite.)\n"
    "- conviction [0, 1]: your confidence that the text supports the scores. "
    "0 = the text is too thin to judge, 1 = the evidence is decisive.\n"
    "- rationale: a short justification grounded only in the provided text.\n"
    "- spans_used: short VERBATIM quotes from the provided text that drove the "
    "scores. Quote only what is present."
)


def scoring_task_to_prompt(task: ScoringTask) -> str:
    """Render the instruction a subagent follows to emit a §3.1 JSON score.

    The prompt makes the JSON schema explicit and instructs the scorer to ground
    every field in the provided text only. When ``task.blinded`` is True the
    prompt contains NO issuer identity (no ticker, no company name) — the scorer
    sees only entity-blinded text, which is the primary lookahead control.

    Output is a single string ready to hand to a subagent.
    """
    header_lines = [
        "You are a buy-side fundamental analyst scoring ONE primary filing "
        "document.",
        "Read ONLY the text provided below. Do not use outside knowledge, do "
        "not guess the issuer's identity, and do not reference any company by "
        "name.",
        "",
        f"Filing form: {task.form}",
        f"As-of (point-in-time) date: {task.as_of}",
    ]
    if not task.blinded:
        # UN-BLINDED contamination-demo arm: the raw text already names the
        # issuer, so there is no identity to withhold here. We deliberately do
        # NOT inject the ticker as a field, but the text itself is un-redacted.
        header_lines.append(
            "(Un-blinded arm: the text is raw and may name the issuer.)"
        )
    if task.truncated:
        header_lines.append(
            "(The document was truncated to its head for length; score on what "
            "is shown.)"
        )

    body = [
        "\n".join(header_lines),
        "",
        "Emit EXACTLY one JSON object and nothing else, matching this schema:",
        _SCHEMA_BLOCK,
        "",
        "Field definitions and ranges (values OUT of range are rejected):",
        _FIELD_GUIDE,
        "",
        "If the text is too thin to support ANY judgement, still emit valid "
        "JSON with conviction near 0 and neutral scores — do not invent "
        "evidence.",
        "",
        "=== BEGIN FILING TEXT ===",
        task.text_to_score,
        "=== END FILING TEXT ===",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------- #
# 3. JSON ingest — parse + validate, never silently default.
# --------------------------------------------------------------------------- #
_REQUIRED_NUMERIC = (
    "fundamental_quality",
    "accounting_red_flags",
    "forward_outlook",
    "conviction",
)


def _coerce_payload(payload: Union[dict, str]) -> dict:
    """Return a dict from a dict or a JSON string; raise on anything else."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ScoreParseError(
                f"payload is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ScoreParseError(
                f"payload JSON must be an object, got {type(parsed).__name__}"
            )
        return parsed
    raise ScoreParseError(
        f"payload must be a dict or JSON string, got {type(payload).__name__}"
    )


def _require_float(payload: dict, key: str) -> float:
    if key not in payload:
        raise ScoreParseError(f"missing required field {key!r}")
    val = payload[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ScoreParseError(
            f"field {key!r} must be a number, got {type(val).__name__}: {val!r}"
        )
    return float(val)


def research_score_from_json(
    ticker: str,
    as_of: str,
    payload: Union[dict, str],
) -> ResearchScore:
    """Parse + validate a subagent's JSON output into a :class:`ResearchScore`.

    ``ticker`` / ``as_of`` are supplied by the harness (the bookkeeping the
    blinded scorer never saw) and stamped onto the score. The four numeric fields
    plus ``rationale`` are read from ``payload`` (a dict or JSON string); a
    missing or wrong-typed field raises :class:`ScoreParseError`. After
    construction the score's own ``.validate()`` runs, raising ``ValueError`` on
    any out-of-range value. Nothing is silently defaulted.
    """
    data = _coerce_payload(payload)

    values = {key: _require_float(data, key) for key in _REQUIRED_NUMERIC}

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        raise ScoreParseError(
            f"field 'rationale' must be a string, got {type(rationale).__name__}"
        )

    spans = data.get("spans_used", [])
    if not isinstance(spans, list) or not all(isinstance(s, str) for s in spans):
        raise ScoreParseError("field 'spans_used' must be a list of strings")

    score = ResearchScore(
        ticker=ticker,
        as_of=as_of,
        fundamental_quality=values["fundamental_quality"],
        accounting_red_flags=values["accounting_red_flags"],
        forward_outlook=values["forward_outlook"],
        conviction=values["conviction"],
        rationale=rationale,
        spans_used=list(spans),
    )
    # Raises ValueError on out-of-range — surfaced, never defaulted.
    score.validate()
    return score


# --------------------------------------------------------------------------- #
# 4. Deterministic baseline scorer (no LLM / network / RNG).
# --------------------------------------------------------------------------- #
# Fixed lexicons. These are intentionally small and frozen — the baseline is a
# plumbing control + non-LLM arm, not a good analyst. It must be deterministic
# and schema-valid, nothing more.
POSITIVE_FUNDAMENTALS_LEXICON: Tuple[str, ...] = (
    "record revenue", "revenue growth", "strong demand", "margin expansion",
    "free cash flow", "cash flow", "profitability", "profitable",
    "increased earnings", "earnings growth", "robust", "resilient",
    "outperformed", "exceeded", "strong balance sheet", "improved", "growth",
    "expansion", "increase", "increased", "gain", "surplus",
)

RISK_LEXICON: Tuple[str, ...] = (
    "going concern", "substantial doubt", "material weakness", "restatement",
    "impairment", "default", "covenant breach", "liquidity risk", "bankruptcy",
    "decline", "declined", "loss", "losses", "deteriorated", "weakness",
    "adverse", "litigation", "investigation", "write-down", "writedown",
    "shortfall", "downturn",
)

# Forward-looking sentiment lexicons (drive forward_outlook only).
OUTLOOK_POSITIVE_LEXICON: Tuple[str, ...] = (
    "we expect growth", "expect to grow", "positive outlook", "anticipate growth",
    "raised guidance", "strong pipeline", "favorable outlook", "expect to increase",
    "optimistic",
)
OUTLOOK_NEGATIVE_LEXICON: Tuple[str, ...] = (
    "lowered guidance", "expect a decline", "headwinds", "uncertain outlook",
    "expect to decrease", "challenging environment", "cautious", "soft demand",
    "reduced guidance",
)


def _count_terms(text_low: str, lexicon: Tuple[str, ...]) -> int:
    """Total word-bounded occurrences of every term in ``lexicon``.

    Deterministic: case-insensitive (caller lowercases), word-boundary anchored
    so 'loss' does not match 'glossary'. Multi-word terms match across single
    spaces.
    """
    total = 0
    for term in lexicon:
        pattern = r"(?<![\w])" + re.escape(term) + r"(?![\w])"
        total += len(re.findall(pattern, text_low))
    return total


def _signed_ratio(pos: int, neg: int) -> float:
    """Map (positive, negative) counts to a [-1, 1] signed balance.

    ``(pos - neg) / (pos + neg)``; 0 when neither fires. Pure, deterministic.
    """
    denom = pos + neg
    if denom == 0:
        return 0.0
    return (pos - neg) / denom


def _redflag_ratio(risk: int, positive: int) -> float:
    """Map risk vs positive counts to a [0, 1] red-flag intensity.

    ``risk / (risk + positive)``; 0 when no risk terms fire. Deterministic.
    """
    denom = risk + positive
    if denom == 0:
        return 0.0
    return risk / denom


def _conviction_from_signal(total_hits: int) -> float:
    """Map total lexicon hits to a [0, 1] conviction via a saturating curve.

    More evidence -> higher conviction, capped at 1.0. Deterministic, no RNG.
    Uses a simple bounded ramp: 0 hits -> 0.0, saturating toward 1.0.
    """
    if total_hits <= 0:
        return 0.0
    # Saturating: each hit adds less. Caps below 1.0 until many hits.
    return min(1.0, total_hits / (total_hits + 10.0))


class BaselineScorer:
    """Deterministic, no-LLM implementation of the :class:`ResearchScorer`
    Protocol.

    Derives the §3.1 fields from fixed lexical features of the blinded text:

    * ``fundamental_quality`` — signed balance of positive-fundamentals vs risk
      terms, in [-1, 1].
    * ``accounting_red_flags`` — risk-term intensity relative to total fundamental
      signal, in [0, 1].
    * ``forward_outlook`` — signed balance of forward positive vs negative terms,
      in [-1, 1].
    * ``conviction`` — saturating function of total lexicon hits, in [0, 1].

    It is a plumbing control + non-LLM arm: it does not need to be good, only
    deterministic and schema-valid. It ABSTAINS (raises ``ValueError``) on empty
    text rather than emitting a fabricated neutral score, mirroring the
    inference-contract refusal rule.
    """

    def score(self, blinded: BlindedText, *, ticker: str) -> ResearchScore:
        text = blinded.text or ""
        if not text.strip():
            raise ValueError(
                f"BaselineScorer: empty blinded text for {ticker} @ "
                f"{blinded.as_of} — abstaining (no fabricated score)"
            )
        text_low = text.lower()

        pos = _count_terms(text_low, POSITIVE_FUNDAMENTALS_LEXICON)
        risk = _count_terms(text_low, RISK_LEXICON)
        out_pos = _count_terms(text_low, OUTLOOK_POSITIVE_LEXICON)
        out_neg = _count_terms(text_low, OUTLOOK_NEGATIVE_LEXICON)

        fundamental_quality = _signed_ratio(pos, risk)
        accounting_red_flags = _redflag_ratio(risk, pos)
        forward_outlook = _signed_ratio(out_pos, out_neg)
        conviction = _conviction_from_signal(pos + risk + out_pos + out_neg)

        rationale = (
            f"baseline lexical scorer: positive={pos}, risk={risk}, "
            f"outlook_pos={out_pos}, outlook_neg={out_neg}"
        )

        score = ResearchScore(
            ticker=ticker,
            as_of=blinded.as_of,
            fundamental_quality=fundamental_quality,
            accounting_red_flags=accounting_red_flags,
            forward_outlook=forward_outlook,
            conviction=conviction,
            rationale=rationale,
            spans_used=[],
        )
        score.validate()
        return score


# --------------------------------------------------------------------------- #
# 5. Scores artifact I/O — versioned, atomic, version-refusing.
# --------------------------------------------------------------------------- #
def _score_to_dict(s: ResearchScore) -> Dict[str, object]:
    return {
        "ticker": s.ticker,
        "as_of": s.as_of,
        "fundamental_quality": s.fundamental_quality,
        "accounting_red_flags": s.accounting_red_flags,
        "forward_outlook": s.forward_outlook,
        "conviction": s.conviction,
        "rationale": s.rationale,
        "spans_used": list(s.spans_used),
    }


def _score_from_dict(d: Dict[str, object]) -> ResearchScore:
    """Reconstruct + validate a :class:`ResearchScore` from an artifact row."""
    try:
        score = ResearchScore(
            ticker=str(d["ticker"]),
            as_of=str(d["as_of"]),
            fundamental_quality=float(d["fundamental_quality"]),
            accounting_red_flags=float(d["accounting_red_flags"]),
            forward_outlook=float(d["forward_outlook"]),
            conviction=float(d["conviction"]),
            rationale=str(d.get("rationale", "")),
            spans_used=[str(x) for x in d.get("spans_used", [])],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreParseError(
            f"malformed score row in artifact: {exc}"
        ) from exc
    score.validate()
    return score


def write_scores_artifact(
    scores: List[ResearchScore],
    path: Union[str, Path],
    *,
    meta: Dict[str, object],
) -> Path:
    """Atomically write a versioned scores artifact to ``path``.

    The artifact is a JSON object tagged with ``RESEARCH_PIPELINE_VERSION`` and
    carrying the caller's ``meta`` (arm id, blinding ruleset hash, model id,
    etc.) plus the list of scores. Written atomically (tmp file + ``os.replace``)
    so a crash never leaves a half-written artifact the harness might read.

    Returns the resolved ``Path`` written.
    """
    path = Path(path)
    document = {
        "schema": SCORES_ARTIFACT_SCHEMA,
        "pipeline_version": RESEARCH_PIPELINE_VERSION,
        "meta": dict(meta),
        "score_count": len(scores),
        "scores": [_score_to_dict(s) for s in scores],
    }
    serialized = json.dumps(document, indent=2, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic
    return path


def read_scores_artifact(
    path: Union[str, Path],
) -> Tuple[List[ResearchScore], Dict[str, object]]:
    """Read a scores artifact, REFUSING on a pipeline-version mismatch.

    Returns ``(scores, meta)``. Raises:

    * :class:`ArtifactVersionError` if the artifact's ``pipeline_version`` differs
      from the runtime ``RESEARCH_PIPELINE_VERSION`` — cross-version scores carry
      silent skew and must not be consumed.
    * :class:`ScoreParseError` on missing/corrupt structure or a malformed score
      row (never a silent default).
    """
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoreParseError(f"cannot read artifact {path}: {exc}") from exc

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScoreParseError(
            f"artifact {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise ScoreParseError(
            f"artifact {path} must be a JSON object, got "
            f"{type(document).__name__}"
        )

    version = document.get("pipeline_version")
    if version != RESEARCH_PIPELINE_VERSION:
        raise ArtifactVersionError(
            f"artifact {path} pipeline_version={version!r} != runtime "
            f"{RESEARCH_PIPELINE_VERSION!r} — refusing to load cross-version "
            f"scores"
        )

    rows = document.get("scores")
    if not isinstance(rows, list):
        raise ScoreParseError(
            f"artifact {path} missing a 'scores' list"
        )

    scores = [_score_from_dict(_require_dict_row(r, path)) for r in rows]

    meta = document.get("meta", {})
    if not isinstance(meta, dict):
        raise ScoreParseError(
            f"artifact {path} 'meta' must be an object, got "
            f"{type(meta).__name__}"
        )
    return scores, dict(meta)


def _require_dict_row(row: object, path: Path) -> Dict[str, object]:
    if not isinstance(row, dict):
        raise ScoreParseError(
            f"artifact {path} score row must be an object, got "
            f"{type(row).__name__}"
        )
    return row


__all__ = [
    "ScoringTask",
    "prepare_scoring_task",
    "scoring_task_to_prompt",
    "research_score_from_json",
    "BaselineScorer",
    "write_scores_artifact",
    "read_scores_artifact",
    "ScoreParseError",
    "ArtifactVersionError",
    "SCORES_ARTIFACT_SCHEMA",
    "POSITIVE_FUNDAMENTALS_LEXICON",
    "RISK_LEXICON",
    "OUTLOOK_POSITIVE_LEXICON",
    "OUTLOOK_NEGATIVE_LEXICON",
]
