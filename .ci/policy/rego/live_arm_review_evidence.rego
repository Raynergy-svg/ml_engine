package policy.live_arm

import rego.v1

# Rule: a PR that touches a live/arm/unhalt surface, or whose diff contains a
# risky live-arm pattern, must carry explicit review evidence: either a
# checklist file under .ci/policy/evidence/live-arm/ added/changed in the same
# PR, or the "policy:live-arm-reviewed" label. Formalizes the Hard NOs in
# CLAUDE.md ("oanda_environment stays practice", "respect halted:true") and
# docs/ENGINEERING_BRAIN.md II.4 (LiveGate is operator-token-gated).
#
# Input contract: see .ci/policy/README.md. Notably:
#   input.changed_files             -- list[string]
#   input.diff_signals.risky_live_patterns_found -- list[string] (may be empty)
#   input.review_evidence.checklist_file_changed -- bool
#   input.review_evidence.labels    -- list[string]

live_arm_prefixes := [
	"src/equity/live_gate.py",
	"src/crypto/crypto_live_gate.py",
	"src/crypto/crypto_carry_live_gate.py",
	"src/equity/track_b_live_gate.py",
	"src/agent_runtime/policy.py",
	"src/scanner/automation/state_engine.py",
]

required_label := "policy:live-arm-reviewed"

touched_live_arm_files(files) := [f |
	some f in files
	some p in live_arm_prefixes
	startswith(f, p)
]

touches_live_arm_surface(files) if count(touched_live_arm_files(files)) > 0

has_risky_diff_signal if count(object.get(input, ["diff_signals", "risky_live_patterns_found"], [])) > 0

triggered if touches_live_arm_surface(input.changed_files)

triggered if has_risky_diff_signal

has_review_evidence if object.get(input, ["review_evidence", "checklist_file_changed"], false) == true

has_review_evidence if required_label in object.get(input, ["review_evidence", "labels"], [])

deny contains msg if {
	triggered
	not has_review_evidence
	msg := sprintf(
		"live/arm surface touched (files=%s, risky_diff_patterns=%s) without review evidence. Add a checklist file under .ci/policy/evidence/live-arm/ (copy the TEMPLATE.md) or apply the '%s' label in the same PR.",
		[
			concat(", ", touched_live_arm_files(input.changed_files)),
			concat(", ", object.get(input, ["diff_signals", "risky_live_patterns_found"], [])),
			required_label,
		],
	)
}
