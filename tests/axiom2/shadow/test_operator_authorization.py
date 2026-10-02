"""Ephemeral operator signatures and durable offline nonce fencing; no broker."""
from datetime import timedelta
import pytest
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore, ImmutableConflictError
from src.evidence.transition_policy import AuthorityRegistry
from tests.axiom2.shadow.test_shadow_journal import NOW
from src.axiom2.execution import authorization as auth


def fixture(tmp_path):
    operator=Ed25519Signer.generate(); gateway=Ed25519Signer.generate()
    trust=TrustStore();trust.add(operator.trusted_key(valid_from=NOW-timedelta(days=1)))
    receipt_trust=TrustStore();receipt_trust.add(gateway.trusted_key(valid_from=NOW-timedelta(days=1)))
    expected=dict(account_alias='dedicated',candidate_digest='a'*64,model_digest='b'*64,
        risk_policy_digest='c'*64,intent_digest='d'*64,build_digest='e'*64,capability_digest='f'*64)
    declaration=auth.OperatorAuthorization(**expected,nonce='proof-one',not_before=NOW,
        expires_at=NOW+timedelta(minutes=5),maximum_order_quantity=5,maximum_capital_cents=1000)
    store=EvidenceStore(tmp_path,trust_store=receipt_trust,authorities=AuthorityRegistry(),trusted_clock=lambda:NOW)
    return operator,gateway,trust,receipt_trust,expected,declaration,store


def test_real_operator_signature_exact_scope_and_expiry(tmp_path):
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path)
    signed=operator.sign(declaration,created_at=NOW)
    assert auth.verify_authorization(signed,trust,expected=expected,now=NOW,quantity=5,capital_cents=1000)==declaration
    for changes in ({'model_digest':'0'*64},{'account_alias':'other'}):
        with pytest.raises(ValueError): auth.verify_authorization(signed,trust,expected={**expected,**changes},now=NOW,quantity=1,capital_cents=100)
    for clock in (NOW-timedelta(seconds=1),declaration.expires_at):
        with pytest.raises(ValueError): auth.verify_authorization(signed,trust,expected=expected,now=clock,quantity=1,capital_cents=100)
    for quantity,capital in ((6,100),(1,1001),(True,100),(1,True)):
        with pytest.raises((ValueError,TypeError)): auth.verify_authorization(signed,trust,expected=expected,now=NOW,quantity=quantity,capital_cents=capital)


def test_wrong_role_and_revoked_key_fail_at_receipt(tmp_path):
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path)
    with pytest.raises(ValueError):auth.verify_authorization(gateway.sign(declaration,created_at=NOW),trust,expected=expected,now=NOW,quantity=1,capital_cents=1)
    signed=operator.sign(declaration,created_at=NOW)
    trust.revoke(operator.key_id,NOW+timedelta(seconds=1))
    with pytest.raises(ValueError):auth.verify_authorization(signed,trust,expected=expected,now=NOW+timedelta(seconds=2),quantity=1,capital_cents=1)


def test_nonce_fencing_survives_restart_and_rejects_changed_request(tmp_path):
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path)
    signed=operator.sign(declaration,created_at=NOW)
    journal=auth.AuthorizationJournal(store,signer=gateway,operator_trust_store=trust)
    first=journal.reserve(signed,expected=expected,quantity=1,capital_cents=100)
    restarted=auth.AuthorizationJournal(store,signer=gateway,operator_trust_store=trust)
    assert restarted.reserve(signed,expected=expected,quantity=1,capital_cents=100)==first
    with pytest.raises(ImmutableConflictError):restarted.reserve(signed,expected=expected,quantity=2,capital_cents=100)
    assert first.execution_enabled is False and first.capital_authorized is False


def test_nonce_receipt_tampering_is_rejected(tmp_path):
    from src.evidence.store import StoreCorruptionError
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path)
    signed=operator.sign(declaration,created_at=NOW)
    journal=auth.AuthorizationJournal(store,signer=gateway,operator_trust_store=trust)
    journal.reserve(signed,expected=expected,quantity=1,capital_cents=100)
    path=next((tmp_path/'offline-authorizations').glob('*.json'))
    path.write_bytes(path.read_bytes().replace(b'RESERVED_OFFLINE',b'FORGED_OFFLINE'))
    with pytest.raises(StoreCorruptionError):journal.reserve(signed,expected=expected,quantity=1,capital_cents=100)


def nonce_worker(root,private,operator_private,queue):
    from pathlib import Path
    operator=Ed25519Signer.from_private_bytes(operator_private);gateway=Ed25519Signer.from_private_bytes(private)
    trust=TrustStore();trust.add(operator.trusted_key(valid_from=NOW-timedelta(days=1)))
    receipts=TrustStore();receipts.add(gateway.trusted_key(valid_from=NOW-timedelta(days=1)))
    expected=dict(account_alias='dedicated',candidate_digest='a'*64,model_digest='b'*64,risk_policy_digest='c'*64,
        intent_digest='d'*64,build_digest='e'*64,capability_digest='f'*64)
    declaration=auth.OperatorAuthorization(**expected,nonce='proof-one',not_before=NOW,expires_at=NOW+timedelta(minutes=5),
        maximum_order_quantity=5,maximum_capital_cents=1000)
    store=EvidenceStore(Path(root),trust_store=receipts,authorities=AuthorityRegistry(),trusted_clock=lambda:NOW)
    try:
        reserved=auth.AuthorizationJournal(store,signer=gateway,operator_trust_store=trust).reserve(
            operator.sign(declaration,created_at=NOW),expected=expected,quantity=1,capital_cents=100)
        queue.put(('OK',reserved.request_digest))
    except Exception as exc:queue.put(('ERROR',repr(exc)))


def test_real_multiprocess_nonce_fencing(tmp_path):
    import multiprocessing
    operator,gateway,trust,receipts,expected,declaration,store=fixture(tmp_path)
    context=multiprocessing.get_context('spawn');queue=context.Queue()
    workers=[context.Process(target=nonce_worker,args=(str(tmp_path),gateway.private_bytes(),operator.private_bytes(),queue)) for _ in range(4)]
    for worker in workers:worker.start()
    for worker in workers:worker.join(20);assert worker.exitcode==0
    outcomes=[queue.get(timeout=2) for _ in workers]
    assert all(row[0]=='OK' for row in outcomes),outcomes
    assert len({row[1] for row in outcomes})==1
    assert len(list((tmp_path/'offline-authorizations').glob('*.json')))==1
