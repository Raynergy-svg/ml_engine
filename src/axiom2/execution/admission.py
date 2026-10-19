"""Local denial diagnostics; these are not signed readiness evidence."""
from dataclasses import dataclass, field


BLOCKED_PREREQUISITES = (
    'EXECUTION_IMPLEMENTATION_DISABLED',
    'PROVIDER_REQUEST_IDENTITY_UNVERIFIED',
    'PROVIDER_APPROVAL_CAPABILITY_UNVERIFIED',
    'IMMUTABLE_ARTIFACT_SIGNATURE_RESOLUTION_NOT_IMPLEMENTED',
    'RECEIPT_TIME_ACTOR_TRUST_NOT_VERIFIED',
    'GENUINE_HOLDOUT_NOT_VERIFIED',
    'PROMOTED_MODEL_AND_CALIBRATION_NOT_VERIFIED',
    'LIVE_SHADOW_AND_RECONCILIATION_NOT_QUALIFIED',
    'OPERATIONAL_BOUNDARY_NOT_VERIFIED',
    'APPROVED_BUILD_NOT_VERIFIED',
    'BOUNDED_OPERATOR_AUTHORIZATION_NOT_VERIFIED',
    'DURABLE_NONCE_RESERVATION_AND_FENCING_NOT_IMPLEMENTED',
    'FRESH_ACCOUNT_QUOTE_CAPABILITY_AND_KILL_NOT_VERIFIED',
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionReceipt:
    """Immutable local denial only; no accepted constructor/state exists."""

    action: str
    status: str = field(default='BLOCKED', init=False)
    reason_codes: tuple[str, ...] = field(default=BLOCKED_PREREQUISITES, init=False)
    execution_enabled: bool = field(default=False, init=False)
    capital_authorized: bool = field(default=False, init=False)

    def __post_init__(self):
        if type(self.action) is not str or self.action not in ('REVIEW', 'SUBMIT', 'CANCEL'):
            raise ValueError('unsupported execution action')
