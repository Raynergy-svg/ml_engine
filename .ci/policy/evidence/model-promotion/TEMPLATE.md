# Model promotion review evidence

Copy this file to `.ci/policy/evidence/model-promotion/<yyyy-mm-dd>-<short-slug>.md` and commit
it in the same PR as the model-promotion code change. This file itself is the evidence — its
presence (other than this TEMPLATE.md) satisfies `policy.model_promotion` in
`.ci/policy/rego/model_promotion_gate_evidence.rego`. Prefer attaching a real
`trained_data/backtests/` ship-gate or eval report artifact instead when one exists; use this
checklist only when no such artifact is being changed in the PR.

- **PR:** <link>
- **Files touched:** <list the promotion files this PR changes>
- **Ship-gate result referenced (if any):** <path + train/val gap + PASS/FAIL>
- **What changed and why:**
- **Reviewer:** <name>
- **Reviewer sign-off notes:**
