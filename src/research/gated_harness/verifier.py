"""Independent deterministic replay of a producer's evaluation report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .report import ResearchEvaluationReport


@dataclass(frozen=True)
class ReplayVerdict:
    accepted: bool
    expected_digest: str
    replayed_digest: str
    verifier_id: str
    reason: str


class IndependentReplayVerifier:
    def __init__(self, verifier_id: str) -> None:
        if not verifier_id:
            raise ValueError("verifier_id is required")
        self.verifier_id = verifier_id

    def verify(
        self,
        produced: ResearchEvaluationReport,
        replay: Callable[[], ResearchEvaluationReport],
    ) -> ReplayVerdict:
        if self.verifier_id == produced.producer_id:
            raise ValueError("producer cannot independently verify its own report")
        replayed = replay()
        accepted = produced.report_digest == replayed.report_digest
        return ReplayVerdict(
            accepted=accepted,
            expected_digest=produced.report_digest,
            replayed_digest=replayed.report_digest,
            verifier_id=self.verifier_id,
            reason="byte-equivalent replay" if accepted else "report digest mismatch",
        )
