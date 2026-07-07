#!/usr/bin/env bash
# Runs conftest against every allow/deny fixture in .ci/policy/testdata/ and
# asserts each behaves as its filename claims: *_allow.json must PASS
# (conftest exit 0), *_deny.json must FAIL (conftest exit 1). This is the
# acceptance-level check that the policies actually enforce their rules
# rather than being vacuously true — a bug here would mean a rule that
# passes conftest verify's unit tests but never fires in practice.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${SCRIPT_DIR}/rego"
TESTDATA_DIR="${SCRIPT_DIR}/testdata"

fail=0

for fixture in "${TESTDATA_DIR}"/*_allow.json; do
    [ -e "${fixture}" ] || continue
    if conftest test --all-namespaces --policy "${POLICY_DIR}" "${fixture}" > /tmp/conftest_out.$$ 2>&1; then
        echo "PASS (expected allow): ${fixture}"
    else
        echo "FAIL (expected allow, but policy denied it): ${fixture}"
        cat /tmp/conftest_out.$$
        fail=1
    fi
    rm -f /tmp/conftest_out.$$
done

for fixture in "${TESTDATA_DIR}"/*_deny.json; do
    [ -e "${fixture}" ] || continue
    if conftest test --all-namespaces --policy "${POLICY_DIR}" "${fixture}" > /tmp/conftest_out.$$ 2>&1; then
        echo "FAIL (expected deny, but policy allowed it): ${fixture}"
        cat /tmp/conftest_out.$$
        fail=1
    else
        echo "PASS (expected deny): ${fixture}"
    fi
    rm -f /tmp/conftest_out.$$
done

if [ "${fail}" -ne 0 ]; then
    echo "::error::One or more policy fixtures did not behave as expected — see FAIL lines above."
    exit 1
fi

echo "All policy fixtures behaved as expected."
