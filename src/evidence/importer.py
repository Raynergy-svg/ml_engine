"""Pure Phase-A construction of local import verdicts.

Artifact I/O, quarantine writes, and metric replay execution arrive in later
phases. This module converts completed check evidence into the strict verdict
contract without granting promotion authority.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .contracts import ImportCheck, ImportDecision, LocalImportVerdict


def build_import_verdict(
    *,
    verdict_id: str,
    package_digest: str,
    importer_id: str,
    created_at: datetime,
    checks: Iterable[ImportCheck],
) -> LocalImportVerdict:
    frozen_checks = tuple(checks)
    if not frozen_checks:
        raise ValueError("at least one import check is required")
    failed = [check.check_id for check in frozen_checks if not check.passed]
    if failed:
        return LocalImportVerdict(
            verdict_id=verdict_id,
            package_digest=package_digest,
            importer_id=importer_id,
            created_at=created_at,
            checks=frozen_checks,
            decision=ImportDecision.REJECT,
            rejection_reason=f"failed checks: {', '.join(failed)}",
        )
    return LocalImportVerdict(
        verdict_id=verdict_id,
        package_digest=package_digest,
        importer_id=importer_id,
        created_at=created_at,
        checks=frozen_checks,
        decision=ImportDecision.ACCEPT_FOR_QUARANTINE,
    )
