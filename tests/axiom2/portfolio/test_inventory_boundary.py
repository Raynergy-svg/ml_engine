"""Every excluded authority file is reviewed, pinned, and never packaged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[3]
POLICY = "config/axiom2/portfolio_boundary.json"
PORTFOLIO_FILES = (
    "src/axiom2/portfolio/__init__.py",
    "src/axiom2/portfolio/contracts.py",
    "src/axiom2/portfolio/authority.py",
    "src/axiom2/contracts/equity_orders.py",
    "src/axiom2/portfolio/intents.py",
)


def setup_tree(tmp_path):
    spec = importlib.util.spec_from_file_location("portfolio_inventory_gate", ROOT / "scripts/axiom2_verify_isolation.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    tree = tmp_path / "checkout"
    names = (*gate.SOURCE_FILES, *PORTFOLIO_FILES, "config/axiom2/research_boundary.json")
    if (ROOT / POLICY).is_file():
        names += (POLICY,)
    for name in names:
        target = tree / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return gate, tree


def admitted_tree(tmp_path):
    gate, tree = setup_tree(tmp_path)
    assert set(gate.audit_sources(tree)) == set(gate.SOURCE_FILES)
    return gate, tree


def test_reviewed_portfolio_is_pinned_but_absent_from_research_snapshot(tmp_path):
    gate, tree = admitted_tree(tmp_path)
    assert not set(PORTFOLIO_FILES).intersection(gate.audit_sources(tree))


@pytest.mark.parametrize("name", PORTFOLIO_FILES)
def test_excluded_source_changes_still_require_review(tmp_path, name):
    gate, tree = admitted_tree(tmp_path)
    with (tree / name).open("a") as stream:
        stream.write("\n# unreviewed change\n")
    with pytest.raises(ValueError, match="source digest"):
        gate.audit_sources(tree)


@pytest.mark.parametrize("name", ["new.py", "native.so", "resource.json"])
def test_no_blanket_exclusion_of_portfolio_namespace(tmp_path, name):
    gate, tree = admitted_tree(tmp_path)
    (tree / "src/axiom2/portfolio" / name).write_text("unreviewed")
    with pytest.raises(ValueError, match="inventory"):
        gate.audit_sources(tree)


def test_missing_exclusion_policy_is_not_a_read_only_default(tmp_path):
    gate, tree = admitted_tree(tmp_path)
    (tree / POLICY).unlink()
    with pytest.raises(ValueError, match="missing|profile|policy"):
        gate.audit_sources(tree)


def test_partial_portfolio_inventory_is_rejected(tmp_path):
    gate, tree = admitted_tree(tmp_path)
    (tree / PORTFOLIO_FILES[-1]).unlink()
    with pytest.raises(ValueError, match="missing|inventory"):
        gate.audit_sources(tree)


@pytest.mark.parametrize("field,value", [
    ("execution_enabled", True), ("capital_authorized", True),
    ("profile", "production"), ("schema_version", True),
])
def test_portfolio_profile_cannot_claim_capital(tmp_path, field, value):
    gate, tree = admitted_tree(tmp_path)
    policy = json.loads((tree / POLICY).read_text())
    policy[field] = value
    (tree / POLICY).write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="profile"):
        gate.audit_sources(tree)


def test_hash_repinning_does_not_authorize_portfolio_network_imports(tmp_path):
    gate, tree = admitted_tree(tmp_path)
    name = PORTFOLIO_FILES[-1]
    with (tree / name).open("a") as stream:
        stream.write("\nimport socket\n")
    policy = json.loads((tree / POLICY).read_text())
    policy["source_sha256"][name] = hashlib.sha256((tree / name).read_bytes()).hexdigest()
    (tree / POLICY).write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="dependency"):
        gate.audit_sources(tree)


def test_research_cannot_import_reviewed_portfolio_authority(tmp_path):
    gate, tree = admitted_tree(tmp_path)
    name = "src/axiom2/__init__.py"
    with (tree / name).open("a") as stream:
        stream.write("\nfrom src.axiom2.portfolio.authority import evaluate_portfolio\n")
    policy_path = tree / "config/axiom2/research_boundary.json"
    policy = json.loads(policy_path.read_text())
    policy["source_sha256"][name] = hashlib.sha256((tree / name).read_bytes()).hexdigest()
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="dependency"):
        gate.audit_sources(tree)
