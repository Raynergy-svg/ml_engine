"""Canonical immutable data-platform storage for AXIOM research and training."""

from .contracts import (
    FeatureBundle,
    FeatureOutputRef,
    NormalizedPartitionRef,
    NormalizedVersion,
    RawObjectReceipt,
    StorageDomain,
    StorageFormat,
)
from .platform import (
    ControlStateStore,
    DataPlatform,
    FeatureWriter,
    IngestionWriter,
    PartitionPayload,
    TrainingDataView,
)
from .forward_capture import (
    CaptureFamily,
    CaptureMode,
    ForwardCaptureRecord,
    ForwardCaptureService,
    P2ReadinessReport,
)

__all__ = [
    "ControlStateStore",
    "CaptureFamily",
    "CaptureMode",
    "DataPlatform",
    "FeatureBundle",
    "FeatureOutputRef",
    "FeatureWriter",
    "ForwardCaptureRecord",
    "ForwardCaptureService",
    "IngestionWriter",
    "NormalizedPartitionRef",
    "NormalizedVersion",
    "PartitionPayload",
    "P2ReadinessReport",
    "RawObjectReceipt",
    "StorageDomain",
    "StorageFormat",
    "TrainingDataView",
]
