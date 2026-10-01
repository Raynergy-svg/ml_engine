"""Real store/signature fixtures; no service credentials or mocked storage."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


from src.axiom2.contracts.research_proposal import DataDependency, ResearchProposal
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
COMMIT = "a" * 40


def proposal(experiment_id="exp-1"):
    return ResearchProposal(
        experiment_id=experiment_id,
        hypothesis="preregistered synthetic hypothesis",
        universe_id="pit-universe",
        feature_families=("price_return",),
        label_id="rank-five-bars",
        holding_horizon=5,
        benchmark_id="market",
        cost_model_id="costs-v1",
        model_id="ranker",
        baseline_ids=("momentum",),
        primary_metrics=("net_excess_return",),
        promotion_rule_id="rule-v1",
        code_commit=COMMIT,
        data_dependencies=(DataDependency(dataset_id="snapshot-1", source="fixture", schema_version="v1"),),
    )


@dataclass
class Context:
    store: EvidenceStore
    signers: dict
    registry: object
    operator: object
    evaluator: object


def build_context(root: Path, signers=None):
    import importlib.util

    assert importlib.util.find_spec("src.evidence.equity_research.experiment_registry") is not None, (
        "Task 5 registry is missing"
    )
    from src.evidence.equity_research.experiment_registry import ExperimentRegistry
    from src.evidence.equity_research.holdout import HoldoutAuthority

    if signers is None:
        signers = {
            role: Ed25519Signer.generate()
            for role in (
                AuthorityRole.LOCAL_IMPORTER,
                AuthorityRole.OPERATOR,
                AuthorityRole.INDEPENDENT_VERIFIER,
                AuthorityRole.PRODUCER,
            )
        }
    trust, authorities = TrustStore(), AuthorityRegistry()
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
        authorities.register(actor_id=role.value, role=role, key_ids=(signer.key_id,))
    store = EvidenceStore(root, trust_store=trust, authorities=authorities, trusted_clock=lambda: NOW)
    return Context(
        store,
        signers,
        ExperimentRegistry(store, signer=signers[AuthorityRole.LOCAL_IMPORTER], actor_id="local_importer"),
        HoldoutAuthority(store, signer=signers[AuthorityRole.OPERATOR], actor_id="operator"),
        HoldoutAuthority(store, signer=signers[AuthorityRole.INDEPENDENT_VERIFIER], actor_id="independent_verifier"),
    )
