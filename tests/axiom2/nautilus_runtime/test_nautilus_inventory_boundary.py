"""Exact replay source coexists with research but cannot enter its artifact."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
POLICY = 'config/axiom2/nautilus_boundary.json'
FILES = tuple('src/axiom2/nautilus_runtime/' + name for name in (
    '__init__.py', 'contracts.py', 'journal.py', 'order_replay.py',
    'policy.py', 'runtime.py', 'wakeup.py',
))


def setup_tree(tmp_path, include_nautilus=True):
    spec = importlib.util.spec_from_file_location(
        'nautilus_inventory_gate', ROOT / 'scripts/axiom2_verify_isolation.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    tree = tmp_path / 'checkout'
    names = (*gate.SOURCE_FILES, 'config/axiom2/research_boundary.json')
    if include_nautilus:
        names += (*FILES, POLICY)
    for name in names:
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return gate, tree


def rehash(tree, name, policy_path=POLICY):
    path = tree / policy_path
    policy = json.loads(path.read_text())
    policy['source_sha256'][name] = hashlib.sha256((tree / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(policy))


def test_exact_registered_replay_files_never_enter_research_bundle(tmp_path):
    gate, tree = setup_tree(tmp_path)
    snapshot = gate.audit_sources(tree)
    assert set(snapshot) == set(gate.SOURCE_FILES)
    assert not set(snapshot).intersection(FILES)
    bundle = tmp_path / 'research.zip'
    gate.build_bundle(snapshot, bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(gate.SOURCE_FILES)
    code = ('import sys,importlib.util;sys.path.insert(0,sys.argv[1]);'
            'assert importlib.util.find_spec("src.axiom2.nautilus_runtime") is None')
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(bundle)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_research_only_checkout_still_requires_no_native_profile(tmp_path):
    gate, tree = setup_tree(tmp_path, include_nautilus=False)
    assert set(gate.audit_sources(tree)) == set(gate.SOURCE_FILES)


@pytest.mark.parametrize('name', FILES)
def test_each_registered_native_file_rejects_digest_drift(tmp_path, name):
    gate, tree = setup_tree(tmp_path)
    with (tree / name).open('a') as stream:
        stream.write('\n# unreviewed change\n')
    with pytest.raises(ValueError, match='Nautilus source digest changed'):
        gate.audit_sources(tree)


@pytest.mark.parametrize('name', FILES)
def test_each_registered_native_file_is_required(tmp_path, name):
    gate, tree = setup_tree(tmp_path)
    (tree / name).unlink()
    with pytest.raises(ValueError, match='missing source'):
        gate.audit_sources(tree)


@pytest.mark.parametrize('name', (
    'src/axiom2/nautilus_runtime/unreviewed.py',
    'src/axiom2/nautilus_runtime/native.so',
    'src/axiom2/nautilus_runtime/subdir/resource.json',
    'src/axiom2/unreviewed.py',
))
def test_unknown_source_or_resource_still_fails(tmp_path, name):
    gate, tree = setup_tree(tmp_path)
    target = tree / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('unreviewed')
    with pytest.raises(ValueError, match='inventory drift'):
        gate.audit_sources(tree)


@pytest.mark.parametrize('field,value', (
    ('execution_enabled', True), ('capital_authorized', True),
    ('execution_enabled', 0), ('capital_authorized', 0),
    ('profile', 'production'), ('schema_version', True), ('unknown', False),
))
def test_native_profile_cannot_claim_authority(tmp_path, field, value):
    gate, tree = setup_tree(tmp_path)
    policy = json.loads((tree / POLICY).read_text())
    policy[field] = value
    (tree / POLICY).write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='unsupported Nautilus exclusion profile'):
        gate.audit_sources(tree)


def test_known_files_require_policy(tmp_path):
    gate, tree = setup_tree(tmp_path)
    (tree / POLICY).unlink()
    with pytest.raises(ValueError, match='missing source'):
        gate.audit_sources(tree)


@pytest.mark.parametrize('name', (POLICY, *FILES))
def test_native_file_and_policy_symlinks_fail(tmp_path, name):
    gate, tree = setup_tree(tmp_path)
    target = tree / name
    real = tmp_path / 'outside'
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    with pytest.raises(ValueError, match='symlink'):
        gate.audit_sources(tree)


def test_repinning_cannot_register_an_arbitrary_path(tmp_path):
    gate, tree = setup_tree(tmp_path)
    policy = json.loads((tree / POLICY).read_text())
    policy['source_sha256']['../outside.py'] = '0' * 64
    (tree / POLICY).write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='Nautilus source inventory requires explicit review'):
        gate.audit_sources(tree)


def test_repinning_cannot_grant_native_network_imports(tmp_path):
    gate, tree = setup_tree(tmp_path)
    name = FILES[-1]
    with (tree / name).open('a') as stream:
        stream.write('\nimport socket\n')
    rehash(tree, name)
    with pytest.raises(ValueError, match='dependency outside research profile'):
        gate.audit_sources(tree)


@pytest.mark.parametrize('statement', (
    'import src.axiom2.nautilus_runtime',
    'from src.axiom2.nautilus_runtime import NautilusReplayRuntime',
    'from ..nautilus_runtime import NautilusReplayRuntime',
    'import nautilus_trader',
))
def test_research_cannot_import_registered_replay_or_native_dependency(tmp_path, statement):
    gate, tree = setup_tree(tmp_path)
    name = 'src/axiom2/research/__init__.py'
    with (tree / name).open('a') as stream:
        stream.write('\n' + statement + '\n')
    rehash(tree, name, 'config/axiom2/research_boundary.json')
    with pytest.raises(ValueError, match='dependency outside research profile'):
        gate.audit_sources(tree)
