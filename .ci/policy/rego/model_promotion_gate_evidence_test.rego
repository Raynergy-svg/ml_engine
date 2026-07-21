package policy.model_promotion

import rego.v1

test_denies_promotion_change_without_evidence if {
	r := deny with input as {
		"changed_files": ["src/brain_loop/promotion.py"],
		"gate_evidence": {"ship_gate_files_changed": [], "eval_report_files_changed": []},
	}
	count(r) == 1
}

test_allows_promotion_change_with_ship_gate_evidence if {
	r := deny with input as {
		"changed_files": [
			"src/brain_loop/promotion.py",
			"trained_data/backtests/SHIP_GATE.json",
		],
		"gate_evidence": {
			"ship_gate_files_changed": ["trained_data/backtests/SHIP_GATE.json"],
			"eval_report_files_changed": [],
		},
	}
	count(r) == 0
}

test_allows_promotion_change_with_eval_report_evidence if {
	r := deny with input as {
		"changed_files": [
			"src/training/promotion_policy.py",
			"trained_data/backtests/eval_report_2026_07_07.json",
		],
		"gate_evidence": {
			"ship_gate_files_changed": [],
			"eval_report_files_changed": ["trained_data/backtests/eval_report_2026_07_07.json"],
		},
	}
	count(r) == 0
}

test_allows_promotion_change_with_checklist_file if {
	r := deny with input as {
		"changed_files": [
			"src/equity/ship_gate.py",
			".ci/policy/evidence/model-promotion/2026-07-07-promotion-review.md",
		],
		"gate_evidence": {"ship_gate_files_changed": [], "eval_report_files_changed": []},
	}
	count(r) == 0
}

test_template_file_alone_does_not_count_as_evidence if {
	r := deny with input as {
		"changed_files": [
			"src/equity/ship_gate.py",
			".ci/policy/evidence/model-promotion/TEMPLATE.md",
		],
		"gate_evidence": {"ship_gate_files_changed": [], "eval_report_files_changed": []},
	}
	count(r) == 1
}

test_allows_unrelated_change if {
	r := deny with input as {
		"changed_files": ["docs/strategy.md"],
		"gate_evidence": {"ship_gate_files_changed": [], "eval_report_files_changed": []},
	}
	count(r) == 0
}
