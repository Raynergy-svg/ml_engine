import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from dashboard.server.axiom2_projection import build_projection, router


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_empty_read_does_not_create_store(tmp_path):
    before = list(tmp_path.rglob('*'))
    result = build_projection(tmp_path, tmp_path / 'absent')
    assert result['evidence']['provenance']['status'] == 'UNAVAILABLE'
    assert result['portfolio']['settled_cash'] is None
    assert result['market']['source_time'] is None
    assert result['execution_enabled'] is False
    assert result['capital_authorized'] is False
    assert list(tmp_path.rglob('*')) == before


def test_index_is_cache_not_capital_authority(tmp_path):
    digest = 'a' * 64
    index = {'schema_version': '1.0.0', 'packages': {digest: {'package_id': 'eq-test', 'lane_id': 'equity_research_test_wide', 'state': 'CHAMPION', 'head_event_digest': 'b' * 64}, 'c' * 64: {'package_id': 'fx-test', 'lane_id': 'risk_target', 'state': 'CHAMPION'}}}
    path = put(tmp_path, 'indexes/current.json', index)
    raw = path.read_bytes()
    result = build_projection(tmp_path, tmp_path)
    rows = result['evidence']['rows']
    assert len(rows) == 1
    assert rows[0]['cached_state'] == 'CHAMPION'
    assert rows[0]['evidence_class'] == 'UNVERIFIED'
    assert rows[0]['capital_eligible'] is False
    assert result['evidence']['provenance']['source_time'] is None
    assert result['evidence']['provenance']['digest'] is not None
    assert path.read_bytes() == raw


@pytest.mark.parametrize('value', [[], {'schema_version': 'unknown'}, {'schema_version': '1.0.0', 'packages': []}, {'schema_version': '1.0.0', 'packages': {'bad': {}}}])
def test_invalid_cache_fails_closed(tmp_path, value):
    put(tmp_path, 'indexes/current.json', value)
    result = build_projection(tmp_path, tmp_path)
    assert result['evidence']['provenance']['status'] == 'INVALID'
    assert result['evidence']['rows'] == []


@pytest.mark.parametrize('profile,enabled', [({}, False), ('x', 'false'), ('x' * 129, False), (float('nan'), False)])
def test_malformed_boundary_is_unavailable(tmp_path, profile, enabled):
    put(tmp_path, 'config/axiom2/execution_boundary.json', {'schema_version': 1, 'profile': profile, 'execution_enabled': enabled})
    result = build_projection(tmp_path, tmp_path)
    boundary = result['boundaries'][2]
    assert boundary['provenance']['status'] == 'INVALID'
    assert boundary['profile'] is None
    json.dumps(result, allow_nan=False)


def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('{"packages":{}}')
    root = tmp_path / 'root'
    (root / 'indexes').mkdir(parents=True)
    (root / 'indexes/current.json').symlink_to(outside)
    assert build_projection(tmp_path, root)['evidence']['provenance']['status'] == 'INVALID'


def test_router_is_read_only(monkeypatch, tmp_path):
    monkeypatch.setenv('AXIOM_EVIDENCE_ROOT', str(tmp_path))
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get('/api/axiom2/overview').headers['cache-control'] == 'no-store'
    for method in ['post', 'put', 'delete', 'patch']:
        assert getattr(client, method)('/api/axiom2/overview').status_code == 405
