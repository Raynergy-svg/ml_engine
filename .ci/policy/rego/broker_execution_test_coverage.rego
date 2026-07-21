package policy.broker_execution

import rego.v1

# Rule: a PR that touches broker/execution code must also touch a test file
# in the same PR. Formalizes the repo's existing "never ship execution-path
# changes without tests" expectation (CLAUDE.md "Code quality non-negotiables",
# .claude/rules/improvement.md "Test Coverage Gates").
#
# Input contract: {"changed_files": ["path/a.py", ...], ...}
# (see .ci/policy/README.md for the full schema)

protected_prefixes := [
	"src/brokers/",
	"src/scanner/execution.py",
	"src/equity/executors.py",
	"src/equity/order_lifecycle.py",
	"src/equity/kill_switch.py",
]

protected_files(files) := [f |
	some f in files
	some p in protected_prefixes
	startswith(f, p)
]

touches_protected(files) if count(protected_files(files)) > 0

has_test_change(files) if {
	some f in files
	startswith(f, "tests/")
	base := split(f, "/")[count(split(f, "/")) - 1]
	startswith(base, "test_")
	endswith(base, ".py")
}

has_test_change(files) if {
	some f in files
	startswith(f, "tests/e2e/")
}

deny contains msg if {
	files := input.changed_files
	touches_protected(files)
	not has_test_change(files)
	msg := sprintf(
		"broker/execution path(s) changed without a corresponding tests/ change in the same PR: %s. Add or update an integration/E2E test alongside this change.",
		[concat(", ", protected_files(files))],
	)
}
