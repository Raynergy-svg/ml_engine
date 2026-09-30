"""Research-only artifact boundary: real files, real subprocesses, no broker calls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/axiom2_verify_isolation.py"
POLICY = "config/axiom2/research_boundary.json"
SOURCE_FILES = (
    "src/__init__.py",
    "src/axiom2/__init__.py",
    "src/axiom2/contracts/__init__.py",
    "src/axiom2/contracts/research_proposal.py",
    "src/axiom2/data/__init__.py",
    "src/axiom2/data/temporal.py",
    "src/axiom2/data/universe.py",
)


def test_research_isolation_gate_exists():
    assert (ROOT / SCRIPT).is_file(), "Research isolation gate is missing"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("axiom2_isolation_gate", ROOT / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    checkout = tmp_path / "checkout"
    for name in (*SOURCE_FILES, SCRIPT, POLICY):
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return checkout


def rehash(tree, filename):
    policy = json.loads((tree / POLICY).read_text())
    policy["source_sha256"][filename] = hashlib.sha256((tree / filename).read_bytes()).hexdigest()
    (tree / POLICY).write_text(json.dumps(policy))


def test_current_audited_contracts_have_a_complete_pinned_inventory(gate, tree):
    snapshot = gate.audit_sources(tree)
    assert set(snapshot) == set(SOURCE_FILES)
    assert all(snapshot[name] == (tree / name).read_bytes() for name in SOURCE_FILES)


def test_standalone_bundle_exercises_real_contracts_without_legacy_modules(gate, tree, tmp_path):
    snapshot = gate.audit_sources(tree)
    bundle = tmp_path / "research.zip"
    gate.build_bundle(snapshot, bundle)
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES)
    result = gate.verify_bundle(bundle, snapshot)
    assert result["exercised"] == ["research_proposal", "temporal", "universe"]
    assert result["blocked_side_effects"] == []
    assert "src.scanner.execution" in result["unavailable_modules"]
    assert "src.training.correlation_group_config" in result["unavailable_modules"]
    assert result["loaded_project_modules"] == sorted(gate.module_names(snapshot))


@pytest.mark.parametrize("filename", SOURCE_FILES)
def test_any_audited_source_change_requires_review_and_repinning(gate, tree, filename):
    with (tree / filename).open("a") as stream:
        stream.write("\n# source drift\n")
    with pytest.raises(ValueError, match="source digest"):
        gate.audit_sources(tree)


@pytest.mark.parametrize(
    "payload",
    [
        "\nimport src.scanner.execution\n",
        "\nif False:\n    import src.brokers.oanda\n",
        "\nfrom .. import scanner\n",
        "\ndef hidden():\n    from src.training.correlation_group_config import get_pair_correlation\n",
        "\nimport socket\n",
        "\nimport subprocess\n",
        "\nimport importlib\n",
        "\nfrom dataclasses import *\n",
        "\ndef hidden():\n    return __import__('src.scanner.execution')\n",
        "\ndef hidden():\n    return eval('1 + 1')\n",
        "\ndef hidden():\n    return open('state.json')\n",
    ],
)
def test_forbidden_dependencies_fail_even_if_their_hash_is_updated(gate, tree, payload):
    filename = "src/axiom2/__init__.py"
    with (tree / filename).open("a") as stream:
        stream.write(payload)
    rehash(tree, filename)
    with pytest.raises(ValueError, match="dependency|dynamic capability|wildcard"):
        gate.audit_sources(tree)


def test_new_axiom_module_cannot_be_silently_left_out_of_the_gate(gate, tree):
    (tree / "src/axiom2/new_module.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="inventory"):
        gate.audit_sources(tree)


def test_removed_source_is_not_ignored(gate, tree):
    (tree / SOURCE_FILES[-1]).unlink()
    with pytest.raises(ValueError, match="inventory|missing"):
        gate.audit_sources(tree)


@pytest.mark.parametrize("entry", ["src/axiom2/link.py", "src/axiom2/linked_package"])
def test_symlinked_sources_or_packages_are_rejected(gate, tree, tmp_path, entry):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree / entry).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        gate.audit_sources(tree)


@pytest.mark.parametrize("entry", ["src/axiom2/native.so", "src/axiom2/hidden.pyc"])
def test_unreviewed_loadable_code_is_rejected(gate, tree, entry):
    (tree / entry).write_bytes(b"not audited")
    with pytest.raises(ValueError, match="inventory"):
        gate.audit_sources(tree)


def test_cache_files_are_never_packaged(gate, tree):
    cache = tree / "src/axiom2/__pycache__"
    cache.mkdir()
    (cache / "unused.cpython-311.pyc").write_bytes(b"cache")
    assert set(gate.audit_sources(tree)) == set(SOURCE_FILES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "production"),
        ("execution_enabled", True),
        ("schema_version", 2),
        ("implemented_tasks", [1, 2, 3, 4, 5]),
    ],
)
def test_policy_cannot_silently_claim_a_later_or_executable_release(gate, tree, field, value):
    data = json.loads((tree / POLICY).read_text())
    data[field] = value
    (tree / POLICY).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="profile"):
        gate.audit_sources(tree)


def test_source_paths_cannot_escape_the_reviewed_namespace(gate, tree):
    data = json.loads((tree / POLICY).read_text())
    data["source_sha256"]["../outside.py"] = "0" * 64
    (tree / POLICY).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inventory"):
        gate.audit_sources(tree)


@pytest.mark.parametrize("kind", ["extra", "modified", "missing", "duplicate"])
def test_artifact_contents_are_verified_not_assumed(gate, tree, tmp_path, kind):
    snapshot = gate.audit_sources(tree)
    bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, content in snapshot.items():
            if kind == "missing" and name == SOURCE_FILES[-1]:
                continue
            archive.writestr(name, content + (b"\n# tampered" if kind == "modified" else b""))
        if kind == "extra":
            archive.writestr("src/scanner/execution.py", "raise RuntimeError('must not import')\n")
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr(SOURCE_FILES[0], snapshot[SOURCE_FILES[0]])
    with pytest.raises(ValueError, match="artifact"):
        gate.verify_bundle(bundle, snapshot)


def test_bundles_are_byte_reproducible_and_do_not_overwrite_existing_files(gate, tree, tmp_path):
    snapshot = gate.audit_sources(tree)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    gate.build_bundle(snapshot, first)
    gate.build_bundle(snapshot, second)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        gate.build_bundle(snapshot, first)


def test_cli_reports_only_current_boundary_not_task10_or_live_readiness(tree, tmp_path):
    bundle = tmp_path / "cli.zip"
    # An unrelated legacy module exists on disk; isolated bundle must not see it.
    scanner = tree / "src/scanner"
    scanner.mkdir()
    (scanner / "__init__.py").write_text("raise AssertionError('legacy scanner loaded')\n")
    result = subprocess.run(
        [sys.executable, str(tree / SCRIPT), "--repo-root", str(tree), "--bundle", str(bundle)],
        cwd=tmp_path,
        env={"PATH": os.defpath, "PYTHONPATH": str(tree)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["scope"] == "research-contracts-only"
    assert receipt["execution_enabled"] is False
    assert receipt["research_kernel_complete"] is False
    assert receipt["implemented_tasks"] == [1, 2, 3, 4]
    assert receipt["bundle_sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()


def test_failed_probe_does_not_publish_an_artifact(tree, tmp_path):
    filename = "src/axiom2/__init__.py"
    with (tree / filename).open("a") as stream:
        stream.write("\nraise RuntimeError('broken candidate')\n")
    rehash(tree, filename)
    bundle = tmp_path / "must-not-exist.zip"
    result = subprocess.run(
        [sys.executable, str(tree / SCRIPT), "--repo-root", str(tree), "--bundle", str(bundle)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "PASS" not in result.stdout
    assert not bundle.exists(), "failed verification published an artifact"


@pytest.mark.parametrize(
    "payload",
    [
        "\nimport socket\nsocket.socket().connect(('127.0.0.1', 9))\n",
        "\nimport subprocess\nsubprocess.run(['must-not-execute'])\n",
    ],
)
def test_runtime_probe_detects_side_effect_attempts_in_disposable_mutants(gate, tree, tmp_path, payload):
    snapshot = gate.audit_sources(tree)
    # Intentionally bypass the source audit only to test the probe's separate defense.
    snapshot["src/axiom2/__init__.py"] += payload.encode()
    bundle = tmp_path / "effect-mutant.zip"
    gate.build_bundle(snapshot, bundle)
    with pytest.raises(ValueError, match="external effect"):
        gate.verify_bundle(bundle, snapshot)


def test_updated_source_with_late_import_is_rejected_before_any_execution(gate, tree, tmp_path):
    filename = "src/axiom2/__init__.py"
    sentinel = tmp_path / "should-not-be-written"
    with (tree / filename).open("a") as stream:
        stream.write(f"\nfrom pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n")
    rehash(tree, filename)
    with pytest.raises(ValueError, match="dependency"):
        gate.audit_sources(tree)
    assert not sentinel.exists()
