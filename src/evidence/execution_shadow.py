"""Signed shadow events and authenticated operator role history."""
from datetime import datetime
from typing import Any, Literal
from pydantic import Field
from src.evidence.contracts import StrictContract, SignedEnvelope

class AuthorityBindingEvent(StrictContract):
    sequence: int = Field(ge=0)
    previous_digest: str | None
    action: Literal['REGISTER', 'REVOKE']
    actor_id: str = Field(min_length=1)
    key_id: str = Field(min_length=1)
    occurred_at: datetime
    role: Literal['SHADOW_EXECUTION'] = 'SHADOW_EXECUTION'

class ExecutionEvent(StrictContract):
    sequence: int = Field(ge=0)
    previous_digest: str | None
    actor_id: str = Field(min_length=1)
    occurred_at: datetime
    intent_id: str = Field(min_length=1)
    intent_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    context_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    state: Literal['RESERVED', 'COMPLETED']
    intent: dict[str, Any]
    result: dict[str, Any] | None = None
    mode: Literal['SHADOW'] = 'SHADOW'
    execution_enabled: Literal[False] = False
    capital_authorized: Literal[False] = False

class ShadowJournalReceipt(StrictContract):
    envelope: SignedEnvelope
    received_at: datetime
    stream: Literal['shadow-authority', 'shadow-execution', 'shadow-portfolio', 'shadow-fences','shadow-provider-rehearsal']
