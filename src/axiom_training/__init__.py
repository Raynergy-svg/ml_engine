"""Training support for AXIOM's action-policy layer.

This package is deliberately separate from ``src.agent_runtime``: the runtime
owns safe action execution, while this package owns datasets, model training,
and replay reports for choosing which action to propose next.
"""

from src.axiom_training.action_policy import (
    ACTION_POLICY_EXAMPLES_PATH,
    ACTION_POLICY_MODEL_PATH,
    ACTION_POLICY_REPORT_PATH,
    ACTION_REGISTRY,
    append_cycle_examples,
    build_inference_rows,
    build_examples_from_cycle,
    features_from_training_row,
    read_policy_status,
)
from src.axiom_training.episode_ledger import (
    EPISODES_PATH,
    append_episode_from_cycle,
    backfill_episodes_from_cycles,
    closure_summary,
    episode_from_cycle,
    read_episodes,
)
from src.axiom_training.outcome_linker import (
    OUTCOME_LINKS_PATH,
    link_delayed_outcomes,
    outcome_link_summary,
    read_outcome_links,
)
from src.axiom_training.policy_advisor import (
    POLICY_ADVICE_PATH,
    advise_and_append,
    backfill_policy_advice,
    build_policy_advice,
)
from src.axiom_training.promotion_criteria import (
    PromotionThresholds,
    evaluate_promotion,
    promotion_verdict_to_dict,
)
from src.axiom_training.runtime_health import (
    evaluate_runtime_health,
    runtime_health_to_dict,
)
from src.axiom_training.system_health import (
    axiom_system_health_to_dict,
    evaluate_axiom_system_health,
)
from src.axiom_training.risk_gym import (
    AxiomRiskGym,
    RiskPolicy,
    RiskPolicyParams,
    RiskStep,
    load_default_risk_steps,
    split_steps_three_way,
)
from src.axiom_training.risk_runtime import (
    RuntimeRiskDecision,
    evaluate_runtime_risk,
    promote_candidate,
    read_promotion_status,
    scale_target_units,
    validate_candidate,
)
from src.axiom_training.online_risk_learning import (
    OnlineRiskLearner,
    read_online_learning_status,
)

__all__ = [
    "ACTION_POLICY_EXAMPLES_PATH",
    "ACTION_POLICY_MODEL_PATH",
    "ACTION_POLICY_REPORT_PATH",
    "ACTION_REGISTRY",
    "EPISODES_PATH",
    "OUTCOME_LINKS_PATH",
    "POLICY_ADVICE_PATH",
    "PromotionThresholds",
    "advise_and_append",
    "append_cycle_examples",
    "append_episode_from_cycle",
    "backfill_episodes_from_cycles",
    "backfill_policy_advice",
    "build_inference_rows",
    "build_policy_advice",
    "build_examples_from_cycle",
    "closure_summary",
    "episode_from_cycle",
    "evaluate_promotion",
    "evaluate_runtime_health",
    "evaluate_axiom_system_health",
    "features_from_training_row",
    "link_delayed_outcomes",
    "outcome_link_summary",
    "promotion_verdict_to_dict",
    "read_policy_status",
    "read_episodes",
    "read_outcome_links",
    "runtime_health_to_dict",
    "axiom_system_health_to_dict",
    "AxiomRiskGym",
    "RiskPolicy",
    "RiskPolicyParams",
    "RiskStep",
    "load_default_risk_steps",
    "split_steps_three_way",
    "RuntimeRiskDecision",
    "evaluate_runtime_risk",
    "promote_candidate",
    "read_promotion_status",
    "scale_target_units",
    "validate_candidate",
    "OnlineRiskLearner",
    "read_online_learning_status",
]
