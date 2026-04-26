"""Trade Homework System — Buddy studies past trades, operator grades.

Public API:
    from src.scanner.automation.homework import (
        HomeworkEntry, Heuristic, TrainingSignal,
        HomeworkStore, HomeworkGenerator, HomeworkReviewer,
        HEURISTIC_CATALOG,
    )

See docs/superpowers/specs/2026-04-25-trade-homework-system-design.md.
"""

from src.scanner.automation.homework.generator import HomeworkGenerator
from src.scanner.automation.homework.heuristics import HEURISTIC_CATALOG
from src.scanner.automation.homework.reviewer import HomeworkReviewer
from src.scanner.automation.homework.store import HomeworkStore
from src.scanner.automation.homework.types import (
    HomeworkEntry,
    Heuristic,
    TrainingSignal,
)

__all__ = [
    "HomeworkEntry",
    "Heuristic",
    "TrainingSignal",
    "HomeworkStore",
    "HomeworkGenerator",
    "HomeworkReviewer",
    "HEURISTIC_CATALOG",
]
