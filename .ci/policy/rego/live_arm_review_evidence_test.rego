package policy.live_arm

import rego.v1

test_denies_live_gate_change_without_evidence if {
	r := deny with input as {
		"changed_files": ["src/equity/live_gate.py"],
		"review_evidence": {"checklist_file_changed": false, "labels": []},
	}
	count(r) == 1
}

test_allows_live_gate_change_with_checklist_file if {
	r := deny with input as {
		"changed_files": [
			"src/equity/live_gate.py",
			".ci/policy/evidence/live-arm/2026-07-07-arm-review.md",
		],
		"review_evidence": {"checklist_file_changed": true, "labels": []},
	}
	count(r) == 0
}

test_allows_live_gate_change_with_label if {
	r := deny with input as {
		"changed_files": ["src/agent_runtime/policy.py"],
		"review_evidence": {
			"checklist_file_changed": false,
			"labels": ["policy:live-arm-reviewed"],
		},
	}
	count(r) == 0
}

test_denies_risky_diff_pattern_without_evidence if {
	r := deny with input as {
		"changed_files": ["src/scanner/config.py"],
		"diff_signals": {"risky_live_patterns_found": ["oanda_environment=\"live\""]},
		"review_evidence": {"checklist_file_changed": false, "labels": []},
	}
	count(r) == 1
}

test_allows_risky_diff_pattern_with_evidence if {
	r := deny with input as {
		"changed_files": ["src/scanner/config.py"],
		"diff_signals": {"risky_live_patterns_found": ["oanda_environment=\"live\""]},
		"review_evidence": {"checklist_file_changed": true, "labels": []},
	}
	count(r) == 0
}

test_allows_unrelated_change if {
	r := deny with input as {
		"changed_files": ["docs/strategy.md"],
		"diff_signals": {"risky_live_patterns_found": []},
		"review_evidence": {"checklist_file_changed": false, "labels": []},
	}
	count(r) == 0
}
