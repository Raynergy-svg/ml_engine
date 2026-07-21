# Live/arm review evidence

Copy this file to `.ci/policy/evidence/live-arm/<yyyy-mm-dd>-<short-slug>.md` and commit it in
the same PR as the live/arm surface change. This file itself is the evidence — its presence
(other than this TEMPLATE.md) satisfies `policy.live_arm` in `.ci/policy/rego/live_arm_review_evidence.rego`.

- **PR:** <link>
- **Files touched:** <list the live/arm files this PR changes>
- **What changed and why:**
- **Confirms `oanda_environment` remains `"practice"`:** yes / no (explain if no)
- **Confirms `halted` semantics are unchanged or explicitly intended:** yes / no (explain if no)
- **Reviewer:** <name>
- **Reviewer sign-off notes:**
