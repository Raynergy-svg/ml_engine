"""Disabled ingress must never treat preparation or caller claims as authority."""
import importlib
from dataclasses import FrozenInstanceError
import pytest


def modules():
    names = ('authority', 'admission', 'authorization')
    try:
        return tuple(importlib.import_module('src.axiom2.execution.' + name) for name in names)
    except ModuleNotFoundError:
        pytest.fail('disabled execution admission interface is missing')


@pytest.mark.parametrize('action', ('review', 'submit', 'cancel'))
@pytest.mark.parametrize('claim', (None, True, {'status': 'PROMOTED', 'capital_authorized': True}, 'signed-operator-approval'))
def test_every_action_denied_for_unverified_claims(action, claim):
    authority, admission, _ = modules()
    receipt = getattr(authority.ExecutionAuthority(), action)(claim, claim)
    assert type(receipt) is admission.AdmissionReceipt
    assert receipt.status == 'BLOCKED'
    assert receipt.action == action.upper()
    assert receipt.execution_enabled is False
    assert receipt.capital_authorized is False
    assert 'EXECUTION_IMPLEMENTATION_DISABLED' in receipt.reason_codes
    assert 'PROVIDER_REQUEST_IDENTITY_UNVERIFIED' in receipt.reason_codes
    assert 'GENUINE_HOLDOUT_NOT_VERIFIED' in receipt.reason_codes


def test_facade_does_not_evaluate_caller_code_or_dispatch_transport():
    authority, _, _ = modules()
    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError('untrusted caller object accessed')
    gateway = authority.ExecutionAuthority()
    for action in ('review', 'submit', 'cancel'):
        assert getattr(gateway, action)(Hostile(), Hostile()).status == 'BLOCKED'
    with pytest.raises(TypeError):
        authority.ExecutionAuthority(transport=Hostile())
    with pytest.raises(TypeError):
        authority.ExecutionAuthority(enabled=True)


def test_denial_is_immutable_and_cannot_be_constructed_as_approval():
    _, admission, _ = modules()
    receipt = admission.AdmissionReceipt(action='REVIEW')
    with pytest.raises(FrozenInstanceError):
        receipt.status = 'ACCEPTED'
    with pytest.raises(TypeError):
        admission.AdmissionReceipt(action='REVIEW', capital_authorized=True)
    with pytest.raises(ValueError):
        admission.AdmissionReceipt(action='PLACE')


def test_signed_authorization_declaration_is_still_not_permission():
    authority, _, authorization = modules()
    # Signed bytes are opaque preparation material, never verified admission.
    declaration = authorization.CapitalAuthorization(envelope_bytes=b'pretend signature')
    assert authority.ExecutionAuthority().review(object(), declaration).status == 'BLOCKED'
    with pytest.raises(TypeError):
        authorization.CapitalAuthorization(envelope_bytes=bytearray(b'mutable'))
    with pytest.raises(ValueError):
        authorization.CapitalAuthorization(envelope_bytes=b'')


@pytest.mark.parametrize('role', ('research', 'llm', 'development', 'healing', 'operator', 'execution'))
def test_roles_signatures_and_replays_cannot_enable_actions(role):
    authority, _, _ = modules()
    gateway = authority.ExecutionAuthority()
    claim = {'role': role, 'signature': 'valid-looking', 'nonce': 'replayed',
             'genuine_holdout': True, 'boundary_verified': True,
             'mode': 'LIVE', 'approval_setting': 'ON'}
    for _ in range(2):
        for action in ('review', 'submit', 'cancel'):
            assert getattr(gateway, action)(claim, claim).execution_enabled is False


def test_fake_transport_has_zero_calls_and_cannot_be_attached():
    authority, _, _ = modules()
    class FakeTransport:
        def __init__(self):
            self.calls = []
        def review(self, *args):
            self.calls.append('review')
        def submit(self, *args):
            self.calls.append('submit')
        def cancel(self, *args):
            self.calls.append('cancel')
    transport = FakeTransport()
    gateway = authority.ExecutionAuthority()
    with pytest.raises(AttributeError):
        gateway.transport = transport
    for action in ('review', 'submit', 'cancel'):
        getattr(gateway, action)(transport, transport)
    assert transport.calls == []
