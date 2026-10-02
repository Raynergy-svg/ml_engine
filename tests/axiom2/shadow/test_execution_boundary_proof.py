import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('data', [None, {}, {'status': 'PASS'}, {'signed': True}, {'fixture': False, 'operational_denials': True}])
def test_boundary_denies_claimed_evidence(data):
    result = load('axiom2_verify_execution_boundary').verify_boundary(data)
    assert result['status'] == 'BLOCKED'
    assert result['operational_isolation_verified'] is False
    assert result['execution_enabled'] is False

def proposal():
    return dict(account_alias='fixture-account', instrument_id='fixture-equity', side='BUY', quantity=1,
                limit_price_cents=100, capital_cap_cents=150, expires_at='2026-10-02T14:00:00Z',
                kill_plan='Disable gateway, reconcile fills and revoke authorization.')

@pytest.mark.parametrize('readiness', [None, {}, {'approval_on': True}, {'genuine_promotion': True, 'operator_authorized': True}])
def test_preparation_always_disabled(readiness):
    result = load('axiom2_controlled_execution_proof').prepare_proof(proposal(), readiness)
    assert result['status'] == 'BLOCKED'
    assert result['execution_enabled'] is False
    assert result['capital_authorized'] is False
    assert result['broker_actions_created'] == []

@pytest.mark.parametrize('field,value', [('quantity', True), ('quantity', 0), ('capital_cap_cents', 99),
    ('limit_price_cents', 1.5), ('side', 'SELL'), ('expires_at', 'invalid'), ('account_alias', ''), ('kill_plan', '')])
def test_invalid_proposal(field, value):
    data = proposal()
    data[field] = value
    with pytest.raises(ValueError):
        load('axiom2_controlled_execution_proof').prepare_proof(data)

def test_exact_schema_and_no_broker_surface():
    module = load('axiom2_controlled_execution_proof')
    data = proposal()
    data['submit'] = True
    with pytest.raises(ValueError):
        module.prepare_proof(data)
    with pytest.raises(ValueError):
        module.prepare_proof({})
    for name in ('axiom2_controlled_execution_proof', 'axiom2_verify_execution_boundary'):
        assert not any(hasattr(load(name), action) for action in ('submit', 'cancel', 'review', 'approve'))

@pytest.mark.parametrize('expiry', ['2026-10-02Z', '2026-10-02T14:00:00+01:00', '2026-10-02T14:00:00+01:00Z'])
def test_expiry_requires_utc_datetime(expiry):
    data = proposal()
    data['expires_at'] = expiry
    with pytest.raises(ValueError):
        load('axiom2_controlled_execution_proof').prepare_proof(data)

def test_preparation_copies_proposal_without_authority():
    data = proposal()
    result = load('axiom2_controlled_execution_proof').prepare_proof(data)
    data['quantity'] = 999
    assert result['proposal']['quantity'] == 1
    assert result['readiness_verified'] is False

def test_cli_denials(tmp_path):
    import json
    import subprocess
    import sys
    cases = [('axiom2_verify_execution_boundary', []),
             ('axiom2_verify_execution_boundary', ['--evidence', str(tmp_path / 'missing')])]
    proposal_path = tmp_path / 'proposal.json'
    proposal_path.write_text(json.dumps(proposal()))
    cases.append(('axiom2_controlled_execution_proof', ['--proposal', str(proposal_path)]))
    proposal_path_bad = tmp_path / 'bad.json'
    proposal_path_bad.write_text('{')
    cases.append(('axiom2_controlled_execution_proof', ['--proposal', str(proposal_path_bad)]))
    for name, args in cases:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / (name + '.py')), *args],
                                capture_output=True, text=True, timeout=10, check=False)
        assert result.returncode == 2
        receipt = json.loads(result.stdout)
        assert receipt['status'] == 'BLOCKED'
        assert receipt['execution_enabled'] is False
