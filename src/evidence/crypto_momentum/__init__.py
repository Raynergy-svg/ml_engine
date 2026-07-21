"""Phase J3 crypto-momentum worker evidence slice.

Workers evaluate one ``construction × fold × stress`` cell from supplied bytes;
the deterministic aggregator refuses missing/duplicate cells. The local
authority independently replays complete packages and preserves the frozen
construction's insignificant result as a signed rejection. No module here can
promote a lane, write a champion pointer, or reach execution.
"""

from .models import (
    ConstructionHeadResult,
    ConstructionImportOutcome,
    CryptoMomentumWorkerOutput,
    PackagedConstructionHead,
    lane_id_for_construction,
)

__all__ = [
    "ConstructionHeadResult",
    "ConstructionImportOutcome",
    "CryptoMomentumWorkerOutput",
    "PackagedConstructionHead",
    "lane_id_for_construction",
]
