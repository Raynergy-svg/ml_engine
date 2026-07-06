"""No-mock tests for the §3.1 scorer layer (src/equity/research/scorer.py).

Everything is REAL: real FilingText / IssuerBundle / BlindedText / ResearchScore
built in-process, real on-disk artifacts via pytest ``tmp_path``. No network is
needed — the scorer module is fully offline — and there are NO mocks / patches
(project No-Mock rule).
"""

from __future__ import annotations

import json

import pytest

from src.equity.research.contracts import (
    RESEARCH_PIPELINE_VERSION,
    BlindedText,
    FilingText,
    ResearchScore,
)
from src.equity.research.entity_blinder import IssuerBundle, blind_filing
from src.equity.research.scorer import (
    ArtifactVersionError,
    BaselineScorer,
    ScoreParseError,
    prepare_scoring_task,
    read_scores_artifact,
    research_score_from_json,
    scoring_task_to_prompt,
    write_scores_artifact,
)

# A small filing whose raw text NAMES the issuer (so the blinded vs un-blinded
# distinction is observable) and contains lexicon terms for the baseline scorer.
_RAW_TEXT = (
    "Apple Inc. (ticker AAPL) reported record revenue and strong demand this "
    "fiscal year, with margin expansion and robust free cash flow. Tim Cook, "
    "based in Cupertino, California, noted improved profitability. "
    "There was no going concern doubt and no material weakness. We expect "
    "growth to continue with a strong pipeline. Filed September 25, 2021."
)


def _filing(text: str = _RAW_TEXT) -> FilingText:
    return FilingText(
        ticker="AAPL",
        cik="0000320193",
        as_of="2021-10-29",
        form="10-K",
        filed="2021-09-25",
        text=text,
    )


def _bundle() -> IssuerBundle:
    return IssuerBundle(
        company_names=("Apple Inc.", "Apple"),
        tickers=("AAPL",),
        cik="0000320193",
        person_names=("Tim Cook",),
        hq_locations=("Cupertino, California", "Cupertino"),
    )


# --------------------------------------------------------------------------- #
# prepare_scoring_task — blinding vs raw passthrough.
# --------------------------------------------------------------------------- #
def test_prepare_scoring_task_blinds_when_blind_true():
    task = prepare_scoring_task(_filing(), _bundle(), blind=True)
    assert task.blinded is True
    assert task.ticker == "AAPL"          # bookkeeping retained on the task
    low = task.text_to_score.lower()
    # Identity must NOT survive into the scored text.
    assert "apple" not in low
    assert "aapl" not in low
    assert "tim cook" not in low
    assert "cupertino" not in low
    # The blinder audit is carried for the verifier.
    assert isinstance(task.audit, dict)
    assert task.audit.get("pipeline_step") == "entity_blinder"


def test_prepare_scoring_task_passes_raw_when_blind_false():
    task = prepare_scoring_task(_filing(), _bundle(), blind=False)
    assert task.blinded is False
    # Un-blinded arm: raw issuer-naming text passes through unredacted.
    assert "Apple" in task.text_to_score
    assert "AAPL" in task.text_to_score
    assert task.audit is None


def test_prepare_scoring_task_truncates_to_max_chars():
    long_text = "A " * 20000  # 40000 chars
    task = prepare_scoring_task(
        _filing(long_text), _bundle(), blind=False, max_chars=500
    )
    assert task.truncated is True
    assert task.scored_chars == 500
    assert len(task.text_to_score) == 500
    assert task.original_chars == len(long_text)


def test_prepare_scoring_task_no_truncation_when_short():
    task = prepare_scoring_task(_filing(), _bundle(), blind=True, max_chars=100000)
    assert task.truncated is False
    assert task.scored_chars == len(task.text_to_score)


# --------------------------------------------------------------------------- #
# scoring_task_to_prompt — schema present; identity withheld when blinded.
# --------------------------------------------------------------------------- #
def test_prompt_blinded_has_no_identity_and_explicit_schema():
    task = prepare_scoring_task(_filing(), _bundle(), blind=True)
    prompt = scoring_task_to_prompt(task)
    low = prompt.lower()
    assert "aapl" not in low
    assert "apple" not in low
    # Explicit JSON schema fields present.
    for field_name in (
        "fundamental_quality",
        "accounting_red_flags",
        "forward_outlook",
        "conviction",
        "rationale",
        "spans_used",
    ):
        assert field_name in prompt
    assert task.text_to_score in prompt


def test_prompt_unblinded_arm_marks_raw():
    task = prepare_scoring_task(_filing(), _bundle(), blind=False)
    prompt = scoring_task_to_prompt(task)
    assert "Un-blinded arm" in prompt


# --------------------------------------------------------------------------- #
# research_score_from_json — validate ranges + reject malformed.
# --------------------------------------------------------------------------- #
def _good_payload() -> dict:
    return {
        "fundamental_quality": 0.7,
        "accounting_red_flags": 0.1,
        "forward_outlook": 0.3,
        "conviction": 0.8,
        "rationale": "Strong margins and cash flow.",
        "spans_used": ["record revenue", "margin expansion"],
    }


def test_research_score_from_json_accepts_dict_and_string():
    payload = _good_payload()
    s1 = research_score_from_json("AAPL", "2021-10-29", payload)
    s2 = research_score_from_json("AAPL", "2021-10-29", json.dumps(payload))
    assert isinstance(s1, ResearchScore)
    assert s1.ticker == "AAPL"
    assert s1.fundamental_quality == pytest.approx(0.7)
    assert s2.spans_used == ["record revenue", "margin expansion"]


def test_research_score_from_json_rejects_out_of_range():
    bad = _good_payload()
    bad["fundamental_quality"] = 1.5  # out of [-1, 1]
    with pytest.raises(ValueError):
        research_score_from_json("AAPL", "2021-10-29", bad)


def test_research_score_from_json_rejects_negative_redflag():
    bad = _good_payload()
    bad["accounting_red_flags"] = -0.2  # out of [0, 1]
    with pytest.raises(ValueError):
        research_score_from_json("AAPL", "2021-10-29", bad)


def test_research_score_from_json_rejects_malformed_json():
    with pytest.raises(ScoreParseError):
        research_score_from_json("AAPL", "2021-10-29", "{not valid json")


def test_research_score_from_json_rejects_missing_field():
    bad = _good_payload()
    del bad["conviction"]
    with pytest.raises(ScoreParseError):
        research_score_from_json("AAPL", "2021-10-29", bad)


def test_research_score_from_json_rejects_non_numeric_field():
    bad = _good_payload()
    bad["conviction"] = "high"
    with pytest.raises(ScoreParseError):
        research_score_from_json("AAPL", "2021-10-29", bad)


def test_research_score_from_json_rejects_bool_as_number():
    # bool is an int subclass — must NOT be accepted as a numeric score.
    bad = _good_payload()
    bad["fundamental_quality"] = True
    with pytest.raises(ScoreParseError):
        research_score_from_json("AAPL", "2021-10-29", bad)


# --------------------------------------------------------------------------- #
# BaselineScorer — deterministic, schema-valid, abstains on empty.
# --------------------------------------------------------------------------- #
def _blinded(text: str) -> BlindedText:
    return BlindedText(
        as_of="2021-10-29", form="10-K", filed="2021-09-25", text=text
    )


def test_baseline_scorer_is_deterministic_and_schema_valid():
    scorer = BaselineScorer()
    blinded = blind_filing(_filing(), _bundle())
    s1 = scorer.score(blinded, ticker="AAPL")
    s2 = scorer.score(blinded, ticker="AAPL")
    # Deterministic: identical input -> identical output fields.
    assert s1 == s2
    # Schema-valid (validate raises on out-of-range; would have raised in score).
    s1.validate()
    assert -1.0 <= s1.fundamental_quality <= 1.0
    assert 0.0 <= s1.accounting_red_flags <= 1.0
    assert -1.0 <= s1.forward_outlook <= 1.0
    assert 0.0 <= s1.conviction <= 1.0


def test_baseline_scorer_satisfies_protocol_signature():
    # Positive-fundamentals text should not be red-flag dominated.
    scorer = BaselineScorer()
    pos = scorer.score(
        _blinded("record revenue, strong demand, margin expansion, growth"),
        ticker="X",
    )
    neg = scorer.score(
        _blinded("going concern, material weakness, impairment, default, loss"),
        ticker="Y",
    )
    assert pos.fundamental_quality > neg.fundamental_quality
    assert neg.accounting_red_flags > pos.accounting_red_flags


def test_baseline_scorer_raises_on_empty_text():
    scorer = BaselineScorer()
    with pytest.raises(ValueError):
        scorer.score(_blinded("   "), ticker="AAPL")


# --------------------------------------------------------------------------- #
# Artifact I/O — round-trip + version refusal.
# --------------------------------------------------------------------------- #
def _sample_scores() -> list[ResearchScore]:
    return [
        research_score_from_json("AAPL", "2021-10-29", _good_payload()),
        research_score_from_json(
            "MSFT",
            "2021-10-29",
            {
                "fundamental_quality": -0.2,
                "accounting_red_flags": 0.4,
                "forward_outlook": -0.1,
                "conviction": 0.5,
                "rationale": "Mixed.",
                "spans_used": [],
            },
        ),
    ]


def test_artifact_round_trips(tmp_path):
    scores = _sample_scores()
    path = tmp_path / "scores" / "artifact.json"
    meta = {"arm": "full", "model": "offline-subagent", "ruleset_hash": "abc123"}

    written = write_scores_artifact(scores, path, meta=meta)
    assert written == path
    assert path.exists()

    back, back_meta = read_scores_artifact(path)
    assert len(back) == 2
    assert back[0] == scores[0]
    assert back[1] == scores[1]
    assert back_meta == meta


def test_artifact_is_versioned_and_sorted_json(tmp_path):
    path = tmp_path / "artifact.json"
    write_scores_artifact(_sample_scores(), path, meta={})
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["pipeline_version"] == RESEARCH_PIPELINE_VERSION
    assert document["score_count"] == 2
    # sort_keys=True: top-level keys come out alphabetically ordered.
    keys = list(document.keys())
    assert keys == sorted(keys)


def test_artifact_refuses_wrong_version(tmp_path):
    path = tmp_path / "artifact.json"
    write_scores_artifact(_sample_scores(), path, meta={})

    # Tamper the version on disk, then confirm the reader REFUSES.
    document = json.loads(path.read_text(encoding="utf-8"))
    document["pipeline_version"] = "0.0.1-bogus"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactVersionError):
        read_scores_artifact(path)


def test_artifact_rejects_malformed_score_row(tmp_path):
    path = tmp_path / "artifact.json"
    write_scores_artifact(_sample_scores(), path, meta={})
    document = json.loads(path.read_text(encoding="utf-8"))
    # Corrupt a row: red_flags out of range -> validate() must reject on read.
    document["scores"][0]["accounting_red_flags"] = 5.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises((ScoreParseError, ValueError)):
        read_scores_artifact(path)


def test_artifact_rejects_non_json_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(ScoreParseError):
        read_scores_artifact(path)
