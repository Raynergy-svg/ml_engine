package policy.broker_execution

import rego.v1

test_denies_broker_change_without_tests if {
	r := deny with input as {"changed_files": ["src/brokers/oanda.py"]}
	count(r) == 1
}

test_allows_broker_change_with_test_file if {
	r := deny with input as {"changed_files": [
		"src/brokers/oanda.py",
		"tests/test_brokers_oanda.py",
	]}
	count(r) == 0
}

test_allows_broker_change_with_e2e_test if {
	r := deny with input as {"changed_files": [
		"src/scanner/execution.py",
		"tests/e2e/test_execution_flow.py",
	]}
	count(r) == 0
}

test_allows_unrelated_change if {
	r := deny with input as {"changed_files": ["docs/strategy.md"]}
	count(r) == 0
}

test_denies_execution_manager_change_alone if {
	r := deny with input as {"changed_files": ["src/scanner/execution.py"]}
	count(r) == 1
}

test_ignores_docstring_containing_test_word if {
	# A file merely containing the substring "test_" but not under tests/
	# must NOT satisfy the coverage requirement.
	r := deny with input as {"changed_files": [
		"src/brokers/oanda.py",
		"src/brokers/test_helpers_not_a_real_test.py",
	]}
	count(r) == 1
}

test_ignores_non_test_file_under_tests_dir if {
	# A file under tests/ that merely CONTAINS "test_" in its path (a fixture,
	# a notes file) but isn't itself a test_*.py must NOT satisfy the
	# coverage requirement -- e.g. tests/fixtures/test_data.json.
	r := deny with input as {"changed_files": [
		"src/brokers/oanda.py",
		"tests/fixtures/test_data.json",
	]}
	count(r) == 1
}
