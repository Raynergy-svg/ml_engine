"""Deterministic Axiom 2.0 market monitoring without broker or order authority."""

from .contracts import (
    CandidateRegistration,
    CandidateState,
    MarketObservation,
    MaterialEventKind,
    ReconciliationReceipt,
    ResearchResponse,
    ResearchWakeup,
    ThresholdRule,
)
from .lean_adapter import LeanEventSource, LeanMarketEvent, LeanObservationAdapter
from .monitor import AxiomMonitor, MonitorTransitionError, ResearchWakeupSink
from .store import MonitorStore, MonitorStoreCorruption, MonitorStoreError, MonitorStoreOrderingError

__all__ = (
    "AxiomMonitor",
    "CandidateRegistration",
    "CandidateState",
    "LeanEventSource",
    "LeanMarketEvent",
    "LeanObservationAdapter",
    "MarketObservation",
    "MaterialEventKind",
    "MonitorStore",
    "MonitorStoreCorruption",
    "MonitorStoreError",
    "MonitorStoreOrderingError",
    "MonitorTransitionError",
    "ReconciliationReceipt",
    "ResearchResponse",
    "ResearchWakeup",
    "ResearchWakeupSink",
    "ThresholdRule",
)
