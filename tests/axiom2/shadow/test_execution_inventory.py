"""Exact disabled execution inventory is excluded from research artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
ROOT = Path(__file__).resolve().parents[3]

def gate_module():
    spec = importlib.util.spec_from_file_location("execution_inventory_gate", ROOT / "scripts/axiom2_verify_isolation.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert hasattr(gate, "EXECUTION_FILES"), "explicit reviewed execution inventory missing"
    return gate

def tree(tmp_path):
    gate = gate_module()
    root = tmp_path / "tree"
    for name in (*gate.SOURCE_FILES, *gate.PORTFOLIO_FILES, *gate.EXECUTION_FILES,
                 gate.POLICY, gate.PORTFOLIO_POLICY, gate.EXECUTION_POLICY):
        dst = root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
    assert set(gate.audit_sources(root)) == set(gate.SOURCE_FILES)
    return gate, root

def test_execution_inventory_requires_explicit_review():
    gate_module()

def test_execution_files_are_pinned_and_not_research(tmp_path):
    gate, root = tree(tmp_path)
    assert not set(gate.EXECUTION_FILES).intersection(gate.audit_sources(root))

@pytest.mark.parametrize("change", ["drift", "missing", "extra", "network", "privilege", "partial_manifest", "research_import"])
def test_execution_boundary_fails_closed(tmp_path, change):
    gate, root = tree(tmp_path)
    source = "src/axiom2/execution/authority.py"
    path = root / source
    policy_path = root / gate.EXECUTION_POLICY
    policy = json.loads(policy_path.read_text())
    if change == "missing": path.unlink()
    elif change == "extra": (path.parent / "unreviewed.py").write_text("# unsafe\n")
    elif change == "privilege":
        policy["execution_enabled"] = True
        policy_path.write_text(json.dumps(policy))
    elif change == "partial_manifest":
        del policy["source_sha256"][source]
        policy_path.write_text(json.dumps(policy))
    elif change == "research_import":
        source = "src/axiom2/__init__.py"
        path = root / source
        path.write_text(path.read_text()+"\nimport src.axiom2.execution.authority\n")
        policy_path = root / gate.POLICY
        policy = json.loads(policy_path.read_text())
        policy["source_sha256"][source] = hashlib.sha256(path.read_bytes()).hexdigest()
        policy_path.write_text(json.dumps(policy))
    else:
        path.write_text(path.read_text()+("\nimport socket\n" if change == "network" else "\n# drift\n"))
        if change == "network":
            policy["source_sha256"][source] = hashlib.sha256(path.read_bytes()).hexdigest()
            policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError): gate.audit_sources(root)
