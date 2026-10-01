"""Task 2: immutable, provider-neutral research proposal boundary."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest


MODULE = "src.axiom2.contracts.research_proposal"
FEATURES = (
    "price_return",
    "volume_liquidity",
    "cross_sectional_relative_strength",
    "market_sector_regime",
)
TEXT_FIELDS = (
    "experiment_id",
    "hypothesis",
    "universe_id",
    "label_id",
    "benchmark_id",
    "cost_model_id",
    "model_id",
    "promotion_rule_id",
    "code_commit",
)
SEQUENCE_FIELDS = ("feature_families", "baseline_ids", "primary_metrics")
REQUIRED_FIELDS = TEXT_FIELDS + SEQUENCE_FIELDS + ("holding_horizon", "data_dependencies")


def test_contract_module_is_available() -> None:
    assert importlib.util.find_spec(MODULE) is not None, "Task 2 contract implementation is missing"


@pytest.fixture(scope="module")
def api():
    return importlib.import_module(MODULE)


@pytest.fixture
def valid(api):
    return {
        "experiment_id": "equity-ranking-001",
        "hypothesis": "Relative strength ranks excess returns after conservative costs.",
        "universe_id": "liquid-us-equities-etfs.pit.v1",
        "feature_families": FEATURES,
        "label_id": "sector-adjusted-forward-return.daily-bars.v1",
        "holding_horizon": 5,
        "benchmark_id": "market-sector-blend.v1",
        "cost_model_id": "conservative-equity-costs.v1",
        "model_id": "lightgbm-ranker.v1",
        "baseline_ids": ("naive-momentum.v1", "market-sector-factor.v1"),
        "primary_metrics": ("net-excess-return", "rank-ic"),
        "promotion_rule_id": "equity-promotion.v1",
        "code_commit": "79b2b6cc350acfd107980b648255f60b488e92ba",
        "data_dependencies": (
            api.DataDependency(dataset_id="ohlcv-snapshot-001", source="historical-bars", schema_version="ohlcv.v1"),
            api.DataDependency(
                dataset_id="universe-snapshot-001", source="historical-membership", schema_version="membership.v1"
            ),
        ),
    }


def test_valid_proposal_has_version_and_validates_without_side_effects(api, valid) -> None:
    proposal = api.ResearchProposal(**valid)
    before = asdict(proposal)
    assert proposal.schema_version == "1.0.0"
    assert api.validate_research_proposal(proposal) is None
    assert asdict(proposal) == before
    assert json.loads(json.dumps(before))["experiment_id"] == valid["experiment_id"]


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_required_proposal_field_must_be_supplied(api, valid, field) -> None:
    valid.pop(field)
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", TEXT_FIELDS)
@pytest.mark.parametrize("value", ["", " \t\n", None, 123])
def test_required_text_is_nonempty_and_not_coerced(api, valid, field, value) -> None:
    valid[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", [field for field in TEXT_FIELDS if field != "hypothesis"])
def test_reference_strings_must_not_have_surrounding_whitespace(api, valid, field) -> None:
    valid[field] = " " + valid[field]
    with pytest.raises(ValueError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "5", None])
def test_holding_horizon_is_a_positive_integer_not_boolean(api, valid, value) -> None:
    valid["holding_horizon"] = value
    with pytest.raises((TypeError, ValueError), match="holding_horizon"):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS)
@pytest.mark.parametrize("value", [(), [], "price_return", None, ("",), (" \t",), (123,)])
def test_collections_are_nonempty_immutable_string_tuples(api, valid, field, value) -> None:
    valid[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS)
def test_duplicate_entries_are_rejected(api, valid, field) -> None:
    item = valid[field][0]
    valid[field] = (item, item)
    with pytest.raises(ValueError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS)
def test_whitespace_cannot_create_an_alias_in_a_collection(api, valid, field) -> None:
    valid[field] = (" " + valid[field][0],)
    with pytest.raises(ValueError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("family", FEATURES)
def test_each_approved_feature_family_is_accepted(api, valid, family) -> None:
    valid["feature_families"] = (family,)
    assert api.validate_research_proposal(api.ResearchProposal(**valid)) is None


@pytest.mark.parametrize(
    "family",
    [
        "news",
        "fundamentals",
        "earnings",
        "l2",
        "order_book",
        "llm_sentiment",
        "transformers",
        "multi_agent_consensus",
        "options",
        "crypto",
        "unknown",
        "Price_Return",
        "price_return.news",
    ],
)
def test_quarantined_and_unknown_phase1_features_fail_closed(api, valid, family) -> None:
    valid["feature_families"] = ("price_return", family)
    with pytest.raises(ValueError, match="feature_families"):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("commit", ["main", "79b2b6c", "g" * 40, "A" * 40, "a" * 39, "a" * 41, "a" * 65])
def test_code_commit_requires_full_canonical_git_identity(api, valid, commit) -> None:
    valid["code_commit"] = commit
    with pytest.raises(ValueError, match="code_commit"):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("commit", ["a" * 40, "b" * 64])
def test_full_sha1_or_sha256_identity_is_accepted(api, valid, commit) -> None:
    valid["code_commit"] = commit
    assert api.ResearchProposal(**valid).code_commit == commit


@pytest.mark.parametrize("version", ["2.0.0", "1.0", "", 1, True, None])
def test_unsupported_proposal_versions_are_rejected(api, valid, version) -> None:
    with pytest.raises((TypeError, ValueError), match="schema_version"):
        api.ResearchProposal(**valid, schema_version=version)


@pytest.mark.parametrize("field", ["dataset_id", "source", "schema_version"])
@pytest.mark.parametrize("value", ["", " \t", " padded ", None, 123])
def test_data_dependency_requires_unambiguous_reference_fields(api, field, value) -> None:
    fields = {"dataset_id": "snapshot-1", "source": "bars", "schema_version": "ohlcv.v1"}
    fields[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        api.DataDependency(**fields)


@pytest.mark.parametrize("field", ["dataset_id", "source", "schema_version"])
def test_data_dependency_fields_are_required(api, field) -> None:
    fields = {"dataset_id": "snapshot-1", "source": "bars", "schema_version": "ohlcv.v1"}
    fields.pop(field)
    with pytest.raises(TypeError, match=field):
        api.DataDependency(**fields)


@pytest.mark.parametrize("value", [(), [], None, "snapshot-1", ({"dataset_id": "snapshot-1"},), (object(),)])
def test_data_dependencies_require_nonempty_tuple_of_contract_records(api, valid, value) -> None:
    valid["data_dependencies"] = value
    with pytest.raises((TypeError, ValueError), match="data_dependencies"):
        api.ResearchProposal(**valid)


def test_duplicate_dataset_id_is_rejected_even_with_conflicting_metadata(api, valid) -> None:
    original = valid["data_dependencies"][0]
    conflict = api.DataDependency(dataset_id=original.dataset_id, source="other", schema_version="ohlcv.v2")
    valid["data_dependencies"] = (original, conflict)
    with pytest.raises(ValueError, match="data_dependencies"):
        api.ResearchProposal(**valid)


def test_proposal_and_nested_dependencies_reject_ordinary_mutation(api, valid) -> None:
    proposal = api.ResearchProposal(**valid)
    with pytest.raises(FrozenInstanceError):
        proposal.holding_horizon = 10
    with pytest.raises(FrozenInstanceError):
        proposal.data_dependencies[0].source = "changed"
    with pytest.raises(AttributeError):
        proposal.baseline_ids.append("another")
    assert not hasattr(proposal, "__dict__")
    assert not hasattr(proposal.data_dependencies[0], "__dict__")


def test_independent_producers_generate_equivalent_plain_contracts(api, valid) -> None:
    first = api.ResearchProposal(**valid)
    payload = json.loads(json.dumps(asdict(first)))
    payload["data_dependencies"] = tuple(api.DataDependency(**item) for item in payload["data_dependencies"])
    for name in SEQUENCE_FIELDS:
        payload[name] = tuple(payload[name])
    second = api.ResearchProposal(**payload)
    assert second == first
    assert hash(second) == hash(first)


class ProviderAgent:
    """A provider object must never be coerced, introspected or invoked."""

    def __str__(self):
        raise AssertionError("provider coercion is forbidden")

    def __repr__(self):
        raise AssertionError("provider representation is forbidden")


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_provider_objects_are_rejected_in_every_proposal_field(api, valid, field) -> None:
    valid[field] = ProviderAgent()
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)


def test_provider_object_is_not_a_proposal(api) -> None:
    with pytest.raises(TypeError, match="ResearchProposal"):
        api.validate_research_proposal(ProviderAgent())


def test_unknown_agent_or_execution_fields_are_forbidden(api, valid) -> None:
    for field in ("agent", "broker", "execute", "promoted", "approval_override"):
        with pytest.raises(TypeError, match=field):
            api.ResearchProposal(**valid, **{field: ProviderAgent()})


def test_explicit_validator_rechecks_a_tampered_instance(api, valid) -> None:
    proposal = api.ResearchProposal(**valid)
    object.__setattr__(proposal, "feature_families", ("news",))
    with pytest.raises(ValueError, match="feature_families"):
        api.validate_research_proposal(proposal)


def test_contract_import_has_no_broker_provider_or_evidence_side_effects() -> None:
    root = Path(__file__).resolve().parents[3]
    script = f"""
import importlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
module = importlib.import_module({MODULE!r})
assert Path(module.__file__).resolve() == Path(sys.argv[1]) / 'src/axiom2/contracts/research_proposal.py'
for prefix in ('src.brokers', 'src.scanner', 'src.evidence', 'requests', 'openai', 'anthropic', 'tensorflow', 'torch'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(root)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class ProviderString(str):
    def strip(self, *args):
        raise AssertionError("provider string hooks must not be invoked")


class ProviderTuple(tuple):
    def __iter__(self):
        raise AssertionError("provider tuple hooks must not be invoked")


class ProviderInteger(int):
    pass


@pytest.mark.parametrize("field", TEXT_FIELDS)
def test_string_subclasses_are_not_plain_reference_values(api, valid, field) -> None:
    valid[field] = ProviderString(valid[field])
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS + ("data_dependencies",))
def test_tuple_subclasses_are_rejected_before_iteration(api, valid, field) -> None:
    valid[field] = ProviderTuple(valid[field])
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)


def test_integer_subclasses_cannot_supply_holding_horizon(api, valid) -> None:
    valid["holding_horizon"] = ProviderInteger(5)
    with pytest.raises(TypeError, match="holding_horizon"):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS)
def test_nested_provider_objects_are_rejected_without_representation(api, valid, field) -> None:
    valid[field] = (ProviderAgent(),)
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)


@pytest.mark.parametrize("field", REQUIRED_FIELDS + ("schema_version",))
def test_admission_revalidates_all_fields_after_reflective_tampering(api, valid, field) -> None:
    proposal = api.ResearchProposal(**valid)
    object.__setattr__(proposal, field, ProviderAgent())
    with pytest.raises(TypeError, match=field):
        api.validate_research_proposal(proposal)


@pytest.mark.parametrize("field", ("dataset_id", "source", "schema_version"))
def test_admission_revalidates_nested_dependency_fields(api, valid, field) -> None:
    proposal = api.ResearchProposal(**valid)
    object.__setattr__(proposal.data_dependencies[0], field, ProviderAgent())
    with pytest.raises(TypeError, match=field):
        api.validate_research_proposal(proposal)


def test_proposal_subclasses_cannot_extend_the_admission_contract(api, valid) -> None:
    class AgentProposal(api.ResearchProposal):
        pass

    with pytest.raises(TypeError, match="ResearchProposal"):
        AgentProposal(**valid)


def test_dependency_subclasses_cannot_embed_provider_state(api) -> None:
    class AgentDependency(api.DataDependency):
        pass

    with pytest.raises(TypeError, match="DataDependency"):
        AgentDependency(dataset_id="snapshot", source="provider", schema_version="v1")


def test_identical_dependency_repetition_is_rejected(api, valid) -> None:
    valid["data_dependencies"] = (valid["data_dependencies"][0],) * 2
    with pytest.raises(ValueError, match="data_dependencies"):
        api.ResearchProposal(**valid)


def test_exception_messages_do_not_echo_rejected_content(api, valid) -> None:
    rejected_value = " private-provider-payload "
    valid["experiment_id"] = rejected_value
    with pytest.raises(ValueError) as caught:
        api.ResearchProposal(**valid)
    assert rejected_value.strip() not in str(caught.value)
    assert "experiment_id" in str(caught.value)


@pytest.mark.parametrize("field", SEQUENCE_FIELDS + ("data_dependencies",))
def test_nonempty_mutable_lists_cannot_bypass_immutability_checks(api, valid, field) -> None:
    valid[field] = list(valid[field])
    with pytest.raises(TypeError, match=field):
        api.ResearchProposal(**valid)
