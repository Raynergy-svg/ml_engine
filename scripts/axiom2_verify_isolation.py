#!/usr/bin/env python3
"""Build and verify the CURRENT research-evidence bundle, never a live release.

This dev/CI gate pins audited bytes and checks their dependency closure. It is
not a Python sandbox or provenance/promotion authority. Expansion requires
review of this gate, its source inventory, tests and shared dependencies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import sysconfig
import zipfile

POLICY = "config/axiom2/research_boundary.json"
SOURCE_FILES = frozenset(
    [
        "src/__init__.py",
        "src/axiom2/__init__.py",
        "src/axiom2/contracts/__init__.py",
        "src/axiom2/contracts/research_proposal.py",
        "src/axiom2/data/__init__.py",
        "src/axiom2/data/temporal.py",
        "src/axiom2/data/universe.py",
        "src/evidence/__init__.py",
        "src/evidence/canonical.py",
        "src/evidence/contracts/__init__.py",
        "src/evidence/contracts/base.py",
        "src/evidence/contracts/models.py",
        "src/evidence/equity_research/__init__.py",
        "src/evidence/equity_research/experiment_registry.py",
        "src/evidence/equity_research/holdout.py",
        "src/evidence/equity_research/models.py",
        "src/evidence/event_store.py",
        "src/evidence/hashing.py",
        "src/evidence/signing.py",
        "src/evidence/store.py",
        "src/evidence/transition_policy.py",
    ]
)
STDLIB_IMPORTS = frozenset({"__future__", "dataclasses", "datetime", "re", "zoneinfo"})
DYNAMIC_CAPABILITIES = frozenset({"__import__", "eval", "exec", "compile", "open"})
DEPENDENCY_VERSIONS = {"pydantic": "2.12.5", "cryptography": "46.0.7"}
IMPORTS_BY_SOURCE = {
    "src/__init__.py": ["__future__"],
    "src/axiom2/__init__.py": ["__future__"],
    "src/axiom2/contracts/__init__.py": ["__future__"],
    "src/axiom2/contracts/research_proposal.py": ["__future__", "dataclasses", "re"],
    "src/axiom2/data/__init__.py": ["__future__"],
    "src/axiom2/data/temporal.py": ["__future__", "dataclasses", "datetime", "zoneinfo"],
    "src/axiom2/data/universe.py": ["__future__", "dataclasses", "datetime", "src.axiom2.data.temporal"],
    "src/evidence/__init__.py": ["src.evidence.canonical", "src.evidence.hashing", "src.evidence.store"],
    "src/evidence/canonical.py": [
        "__future__",
        "collections.abc",
        "datetime",
        "enum",
        "json",
        "math",
        "pathlib",
        "pydantic",
        "typing",
    ],
    "src/evidence/contracts/__init__.py": ["src.evidence.contracts.base", "src.evidence.contracts.models"],
    "src/evidence/contracts/base.py": ["__future__", "collections.abc", "copy", "json", "pydantic", "typing"],
    "src/evidence/contracts/models.py": [
        "__future__",
        "datetime",
        "enum",
        "pydantic",
        "src.evidence.contracts.base",
        "typing",
    ],
    "src/evidence/equity_research/__init__.py": ["__future__", "src.evidence.equity_research.models"],
    "src/evidence/equity_research/experiment_registry.py": [
        "__future__",
        "dataclasses",
        "datetime",
        "json",
        "pydantic",
        "re",
        "src.axiom2.contracts.research_proposal",
        "src.axiom2.data.temporal",
        "src.evidence.canonical",
        "src.evidence.contracts",
        "src.evidence.contracts.models",
        "src.evidence.equity_research.holdout",
        "src.evidence.signing",
        "src.evidence.store",
        "src.evidence.transition_policy",
        "types",
        "typing",
    ],
    "src/evidence/equity_research/holdout.py": [
        "__future__",
        "dataclasses",
        "datetime",
        "src.axiom2.data.temporal",
        "src.evidence.equity_research.experiment_registry",
        "src.evidence.signing",
        "src.evidence.store",
    ],
    "src/evidence/equity_research/models.py": ["__future__", "dataclasses", "src.evidence.contracts", "typing"],
    "src/evidence/event_store.py": [
        "__future__",
        "dataclasses",
        "datetime",
        "src.evidence.contracts",
        "src.evidence.signing",
        "src.evidence.transition_policy",
        "typing",
    ],
    "src/evidence/hashing.py": ["__future__", "hashlib", "pathlib", "src.evidence.canonical", "typing"],
    "src/evidence/signing.py": [
        "__future__",
        "base64",
        "collections.abc",
        "cryptography.exceptions",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.asymmetric.ed25519",
        "dataclasses",
        "datetime",
        "src.evidence.canonical",
        "src.evidence.contracts",
        "src.evidence.hashing",
        "typing",
    ],
    "src/evidence/store.py": [
        "__future__",
        "collections.abc",
        "contextlib",
        "datetime",
        "fcntl",
        "os",
        "pathlib",
        "pydantic",
        "shutil",
        "src.evidence.canonical",
        "src.evidence.contracts",
        "src.evidence.equity_research.experiment_registry",
        "src.evidence.event_store",
        "src.evidence.hashing",
        "src.evidence.signing",
        "src.evidence.transition_policy",
        "typing",
        "uuid",
    ],
    "src/evidence/transition_policy.py": ["__future__", "collections.abc", "dataclasses", "src.evidence.contracts"],
}


def module_names(snapshot: dict[str, bytes]) -> dict[str, str]:
    """Map module identities to their artifact-relative paths."""
    result = {}
    for name in snapshot:
        module = name.removesuffix(".py").replace("/", ".")
        result[module.removesuffix(".__init__")] = name
    return result


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate policy key")
        result[name] = value
    return result


def _regular_source(root: Path, name: str) -> Path:
    candidate = root / name
    for item in (candidate, *candidate.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"symlink is outside the audited boundary: {name}")
    if not candidate.is_file():
        raise ValueError(f"missing source: {name}")
    return candidate


def _audit_imports(snapshot: dict[str, bytes]) -> None:
    modules = module_names(snapshot)
    packages = {name for name, path in modules.items() if path.endswith("/__init__.py")}
    exports = {}
    for package in packages:
        names = set()
        for node in ast.parse(snapshot[modules[package]]).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
        exports[package] = names
    for module, name in modules.items():
        package = module if module in packages else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(snapshot[name], filename=name)):
            if isinstance(node, (ast.Name, ast.Attribute)):
                token = node.id if isinstance(node, ast.Name) else node.attr
                allowed_io = token == "open" and name in {"src/evidence/store.py", "src/evidence/hashing.py"}
                if token in DYNAMIC_CAPABILITIES and not allowed_io:
                    raise ValueError(f"dynamic capability outside audited profile: {name}:{node.lineno}")
            if isinstance(node, ast.Import):
                dependencies = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if any(alias.name == "*" for alias in node.names):
                    raise ValueError(f"wildcard dependency is not audited: {name}:{node.lineno}")
                try:
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                except (ImportError, ValueError) as exc:
                    raise ValueError(f"unresolved dependency: {name}:{node.lineno}") from exc
                dependencies = [base]
                if base in packages:
                    for alias in node.names:
                        target = f"{base}.{alias.name}"
                        if target not in modules and alias.name not in exports[base]:
                            raise ValueError(f"dependency outside research profile: {target}")
            else:
                continue
            if any(dependency not in IMPORTS_BY_SOURCE[name] for dependency in dependencies):
                raise ValueError(f"dependency outside research profile: {name}:{node.lineno}: {dependencies}")


def audit_sources(root: Path) -> dict[str, bytes]:
    """Capture exact audited bytes; fail before executing any candidate source."""
    root = root.resolve()
    policy = json.loads(_regular_source(root, POLICY).read_text(), object_pairs_hook=_no_duplicates)
    expected_fields = {
        "schema_version",
        "profile",
        "execution_enabled",
        "implemented_tasks",
        "source_sha256",
        "dependency_versions",
    }
    if (
        type(policy) is not dict
        or set(policy) != expected_fields
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
        or policy["profile"] != "research-evidence-only"
        or policy["dependency_versions"] != DEPENDENCY_VERSIONS
        or policy["execution_enabled"] is not False
        or policy["implemented_tasks"] != [1, 2, 3, 4, 5]
        or any(type(task) is not int for task in policy["implemented_tasks"])
    ):
        raise ValueError("unsupported research boundary profile")
    digests = policy["source_sha256"]
    if type(digests) is not dict or set(digests) != SOURCE_FILES:
        raise ValueError("source inventory requires explicit review")
    _regular_source(root, "src/__init__.py")
    _regular_source(root, "src/axiom2/__init__.py")
    observed = {"src/__init__.py"}
    for directory, dirs, files in os.walk(root / "src/axiom2", followlinks=False):
        for name in (*dirs, *files):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"symlink is outside the audited boundary: {path.relative_to(root)}")
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            # No resources/native modules are needed by the current profile.
            observed.add((Path(directory) / name).relative_to(root).as_posix())
    if observed != {name for name in SOURCE_FILES if name == "src/__init__.py" or name.startswith("src/axiom2/")}:
        raise ValueError("source inventory drift: missing or unreviewed files")
    snapshot = {}
    for name in sorted(SOURCE_FILES):
        content = _regular_source(root, name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digests[name]:
            raise ValueError(f"source digest changed; review required: {name}")
        snapshot[name] = content
    _audit_imports(snapshot)
    return snapshot


def build_bundle(snapshot: dict[str, bytes], destination: Path) -> None:
    """Write only captured source bytes; never reread a changing checkout."""
    if set(snapshot) != SOURCE_FILES:
        raise ValueError("artifact source inventory is not the audited profile")
    with destination.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(snapshot.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


_PROBE = r"""
import importlib, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
root = Path(sys.argv[1]).resolve()
# Explicit dependency locations only: -I -S prevents repository/PYTHONPATH/.pth injection.
sys.path.extend(json.loads(sys.argv[3]))
from importlib.metadata import version
for distribution, expected_version in json.loads(sys.argv[4]).items():
    if version(distribution) != expected_version:
        raise RuntimeError('unreviewed runtime dependency version: ' + distribution)
expected = json.loads(sys.argv[2])
sys.path.insert(0, str(root))
violations = []
def audit(event, args):
    if event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn'):
        violations.append(event)
        raise RuntimeError('research contract attempted an external effect')
sys.addaudithook(audit)
for name, relative in expected.items():
    loaded = importlib.import_module(name)
    if Path(loaded.__file__).resolve() != root / relative:
        raise RuntimeError('module origin is outside the verified bundle')
from src.axiom2.contracts.research_proposal import DataDependency, ResearchProposal, validate_research_proposal
from src.axiom2.data.temporal import TemporalRecord, assert_trainable
from src.axiom2.data.universe import UniverseMembership, UniverseManifest, build_universe_as_of
start = datetime(2024, 1, 1, tzinfo=timezone.utc)
end = start + timedelta(days=10)
proposal = ResearchProposal(
    experiment_id='isolation-proof', hypothesis='synthetic boundary exercise', universe_id='fixture-universe',
    feature_families=('price_return',), label_id='fixture-label', holding_horizon=5,
    benchmark_id='fixture-benchmark', cost_model_id='fixture-costs', model_id='fixture-model',
    baseline_ids=('fixture-baseline',), primary_metrics=('net-return',), promotion_rule_id='fixture-rule',
    code_commit='0'*40, data_dependencies=(DataDependency(dataset_id='fixture-data', source='fixture', schema_version='v1'),),
)
if validate_research_proposal(proposal) is not None:
    raise RuntimeError('proposal API failed')
record = TemporalRecord(event_time=start, published_time=None, available_to_axiom_time=start,
    ingested_time=end, source='fixture', revision=None, schema_version='v1')
if assert_trainable(record, start) is not None:
    raise RuntimeError('temporal API failed')
unknown = TemporalRecord(event_time=start, published_time=None, available_to_axiom_time=None,
    ingested_time=end, source='fixture', revision=None, schema_version='v1')
try:
    assert_trainable(unknown, end)
except ValueError:
    pass
else:
    raise RuntimeError('unknown availability was accepted')
manifest = UniverseManifest(universe_id='fixture-universe', source='fixture', evidence_id='fixture-evidence',
    schema_version='v1', membership_basis='sp500', coverage_start=start, coverage_end=end,
    memberships=(UniverseMembership(instrument_id='FIXTURE:SEC', asset_class='equity',
        member_from=start, member_until=end),))
if build_universe_as_of(start, manifest) != ('FIXTURE:SEC',):
    raise RuntimeError('universe API failed')
from dataclasses import replace
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.transition_policy import AuthorityRegistry
from src.evidence.store import EvidenceStore
from src.evidence.equity_research.experiment_registry import ExperimentRegistry
from src.evidence.equity_research.holdout import HoldoutAuthority
trust, authorities = TrustStore(), AuthorityRegistry()
keys = {}
for role in (AuthorityRole.LOCAL_IMPORTER, AuthorityRole.OPERATOR, AuthorityRole.INDEPENDENT_VERIFIER):
    signer = Ed25519Signer.generate()
    keys[role] = signer
    trust.add(signer.trusted_key(valid_from=start-timedelta(days=1)))
    authorities.register(actor_id=role.value, role=role, key_ids=(signer.key_id,))
store = EvidenceStore(root/'_probe_evidence', trust_store=trust, authorities=authorities, trusted_clock=lambda: start)
registry = ExperimentRegistry(store, signer=keys[AuthorityRole.LOCAL_IMPORTER], actor_id='local_importer')
operator = HoldoutAuthority(store, signer=keys[AuthorityRole.OPERATOR], actor_id='operator')
evaluator = HoldoutAuthority(store, signer=keys[AuthorityRole.INDEPENDENT_VERIFIER], actor_id='independent_verifier')
registry.register_experiment(replace(proposal, experiment_id='failed-attempt'))
registry.record_failure('failed-attempt', 'synthetic failure retained')
registry.register_experiment(proposal)
operator.register_holdout('sealed', 'f'*64, start, end)
candidate = registry.freeze_candidate(proposal.experiment_id, '1'*64, proposal.code_commit)
if candidate.attempt_count != 2:
    raise RuntimeError('registry lost attempted-search context')
evaluator.open_holdout(candidate.candidate_id, 'sealed')
consumed = evaluator.consume_holdout(candidate.candidate_id, 'sealed', '2'*64)
if consumed.status != 'CONSUMED':
    raise RuntimeError('holdout was not consumed')
try:
    evaluator.assert_untouched('sealed')
except ValueError:
    pass
else:
    raise RuntimeError('consumed holdout was reported untouched')
unavailable = []
for name in ('src.scanner.execution', 'src.brokers.oanda', 'src.training.correlation_group_config',
             'src.sota_core', 'src.axiom_operator', 'src.evidence.crypto_carry',
             'src.evidence.equity_research.worker', 'src.axiom2.execution'):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if not (name == exc.name or name.startswith(exc.name + '.')):
            raise
        unavailable.append(name)
    else:
        raise RuntimeError('excluded execution dependency was reachable: ' + name)
loaded = sorted(name for name in sys.modules if name == 'src' or name.startswith('src.'))
if loaded != sorted(expected) or violations:
    raise RuntimeError('unexpected project module or external effect')
print(json.dumps({'exercised': ['research_proposal', 'temporal', 'universe', 'experiment_registry', 'sealed_holdout'],
    'unavailable_modules': unavailable, 'loaded_project_modules': loaded, 'blocked_side_effects': violations}))
"""


def verify_bundle(bundle: Path, snapshot: dict[str, bytes]) -> dict:
    """Exercise the exact archive with pinned evidence dependencies, without ambient repository imports."""
    if set(snapshot) != SOURCE_FILES:
        raise ValueError("artifact source inventory is not the audited profile")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(snapshot):
            raise ValueError("artifact inventory mismatch")
        contents = {name: archive.read(name) for name in names}
        if contents != snapshot:
            raise ValueError("artifact source digest mismatch")
    with tempfile.TemporaryDirectory(prefix="axiom2-isolation-") as temporary:
        stage = Path(temporary)
        for name, content in contents.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        sites = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _PROBE,
            str(stage),
            json.dumps(module_names(snapshot)),
            json.dumps(sites),
            json.dumps(DEPENDENCY_VERSIONS),
        ]
        result = subprocess.run(
            command,
            cwd=stage,
            env={"PATH": os.defpath, "TZ": "UTC"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"isolated artifact probe failed ({result.returncode}): {result.stdout}\n{result.stderr}")
        return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--bundle", type=Path, required=True, help="New output ZIP; existing files are never overwritten"
    )
    args = parser.parse_args()
    try:
        snapshot = audit_sources(args.repo_root)
        # Publish only after verification; an existing destination is never replaced.
        with tempfile.TemporaryDirectory(prefix="axiom2-verified-", dir=args.bundle.parent) as temporary:
            candidate = Path(temporary) / "candidate.zip"
            build_bundle(snapshot, candidate)
            result = verify_bundle(candidate, snapshot)
            os.link(candidate, args.bundle)
        receipt = {
            "status": "PASS",
            "scope": "research-evidence-only",
            "execution_enabled": False,
            "research_kernel_complete": False,
            "implemented_tasks": [1, 2, 3, 4, 5],
            "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in snapshot.items()},
            "probe": result,
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        print(f"Research isolation FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
