package policy.model_promotion

import rego.v1

# Rule: a PR that touches model-promotion code must carry fresh gate/eval
# evidence in the same PR — either a changed file under
# trained_data/backtests/ (ship-gate / backtest eval output) or a checklist
# file under .ci/policy/evidence/model-promotion/. Formalizes the Hard NO
# "nothing promotes to champion without passing the ship gate" (CLAUDE.md,
# .claude/rules/improvement.md "Hard Ship Gate").
#
# Input contract: see .ci/policy/README.md. Notably:
#   input.changed_files                            -- list[string]
#   input.gate_evidence.ship_gate_files_changed     -- list[string]
#   input.gate_evidence.eval_report_files_changed   -- list[string]
#   input.review_evidence.checklist_file_changed    -- bool (reused signal;
#     only true when the changed checklist path also matches
#     model-promotion evidence, see generator script)

promotion_prefixes := [
	"src/brain_loop/promotion.py",
	"src/training/promotion_policy.py",
	"src/equity/ship_gate.py",
	"src/scanner/automation/staged_deployer.py",
	"src/evaluation/soak_orchestrator.py",
]

touched_promotion_files(files) := [f |
	some f in files
	some p in promotion_prefixes
	startswith(f, p)
]

touches_promotion_surface(files) if count(touched_promotion_files(files)) > 0

has_gate_evidence if count(object.get(input, ["gate_evidence", "ship_gate_files_changed"], [])) > 0

has_gate_evidence if count(object.get(input, ["gate_evidence", "eval_report_files_changed"], [])) > 0

has_gate_evidence if {
	some f in input.changed_files
	startswith(f, ".ci/policy/evidence/model-promotion/")
	f != ".ci/policy/evidence/model-promotion/TEMPLATE.md"
}

deny contains msg if {
	touches_promotion_surface(input.changed_files)
	not has_gate_evidence
	msg := sprintf(
		"model-promotion path(s) changed without gate/eval evidence in the same PR: %s. Include an updated trained_data/backtests/ ship-gate or eval report, or add a checklist file under .ci/policy/evidence/model-promotion/ (copy the TEMPLATE.md).",
		[concat(", ", touched_promotion_files(input.changed_files))],
	)
}
