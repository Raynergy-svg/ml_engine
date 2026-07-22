"""Row-level source-package lineage binding for sleeve return partitions.

Every row of a sleeve's return-series partition must carry the digest of the
already-QUARANTINED source EvidencePackage it was drawn from. This binds the
digest INTO the hash-verified partition bytes: the signed
``StrategyManifest.parameters["source_package_digests"]`` binding cannot be
swapped for an unrelated-but-quarantined package's digest without also
rewriting every row of the partition, which changes its hash and is caught by
the existing dataset-hash verification (mirrors Track B's row-level
``validate_score_lineage``, adapted to a cross-lane, heterogeneous-artifact
setting).

This closes the "attach a legitimate-but-unrelated quarantined digest"
substitution this module is specifically scoped to prevent. It does NOT (yet)
prove the return values themselves were computed from that package's content —
that requires every upstream lane to emit the standardized strategy-return
contract named in roadmap §14 ("every lane produces a standardized
strategy-return and exposure contract"), which is not yet built for the other
7 evidence slices. That is the honest next step, not a gap papered over here.
"""

from __future__ import annotations

import json
from typing import Mapping


class LineageError(ValueError):
    pass


def verify_source_lineage(partitions: Mapping[str, bytes], source_package_digests: Mapping[str, str]) -> None:
    if set(partitions) != set(source_package_digests):
        raise LineageError("every sleeve partition requires exactly one bound source_package_digest")
    for sleeve_id, data in partitions.items():
        expected = source_package_digests[sleeve_id]
        for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("source_package_digest") != expected:
                raise LineageError(
                    f"sleeve {sleeve_id!r} row {line_number} does not carry its bound source_package_digest "
                    f"{expected!r}"
                )


__all__ = ["LineageError", "verify_source_lineage"]
