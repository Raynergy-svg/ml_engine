"""Bounded, durable ResearchWakeup consumption."""

from __future__ import annotations

from .contracts import ResearchResult, ResearchWakeup
from .journal import CandidateJournal, WakeupResult


class ResearchWakeupConsumer:
    def __init__(self, journal: CandidateJournal) -> None:
        self.journal = journal

    def pending(self) -> tuple[ResearchWakeup, ...]:
        return self.journal.pending_wakeups()

    def claim(self, wakeup_id: str) -> WakeupResult:
        return self.journal.claim_research_wakeup(wakeup_id)

    def complete(self, result: ResearchResult) -> WakeupResult:
        return self.journal.complete_research_wakeup(result)

    def consume(self, result: ResearchResult) -> WakeupResult:
        return self.journal.consume_research_wakeup(result)
