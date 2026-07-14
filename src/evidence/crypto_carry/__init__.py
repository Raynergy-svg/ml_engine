"""Phase J4 crypto carry evidence workers.

Workers score ``venue set × cost model × regime`` cells. The deterministic
aggregator refuses missing/duplicate cells, and the local authority separately
requires return, tail-risk and counterparty evidence before quarantine.
"""

from .models import (
    CarryHeadResult,
    CarryImportOutcome,
    CryptoCarryWorkerOutput,
    PackagedCarryHead,
    lane_id_for_carry,
)

__all__ = [
    "CarryHeadResult",
    "CarryImportOutcome",
    "CryptoCarryWorkerOutput",
    "PackagedCarryHead",
    "lane_id_for_carry",
]
