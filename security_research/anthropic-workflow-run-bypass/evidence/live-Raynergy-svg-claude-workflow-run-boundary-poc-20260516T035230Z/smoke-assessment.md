# Owner smoke assessment

Result: mechanics validated, not bounty-grade.

## What the smoke test proved

- Parent PR workflow ran from PR #1 and failed intentionally.
- Child workflow automatically started from `workflow_run`.
- Child workflow had elevated token permissions:
  - `contents: write`
  - `issues: write`
  - `pull-requests: write`
  - `actions: read`
- Claude Code Action auto-detected `agent` mode for event `workflow_run`.
- The action verified only `Raynergy-svg` as a human actor; there is no log line
  showing an entity-style `checkWritePermissions()` gate.
- Claude read parent CI log/artifact marker `BENIGN_WORKFLOW_RUN_BOUNDARY_POC`.
- Claude committed `workflow-run-poc-result.md` to branch
  `claude-workflow-run-poc-owner-smoke-pr-20260516T035028Z-25952035483`.
- GitHub Actions commented on PR #1 with the PoC branch.

## Why this is not submission-ready

- PR #1 was opened by `Raynergy-svg`, the repository owner.
- The result file was created because the child workflow's trusted prompt told
  Claude to create it if the parent marker was present. This proves data flow,
  not independent prompt-injection takeover.
- Anthropic can reasonably classify this smoke result as trusted-owner workflow
  automation plus documented `workflow_run` risk.

## Required bounty-grade evidence

Submit only after a second test where:

- The PR author is not a write/admin collaborator.
- The child workflow still runs automatically from `workflow_run`.
- Parent-controlled logs/artifacts reach Claude.
- Claude performs a privileged action, or the action's design clearly permits
  that privileged action, without a maintainer tag/manual invocation.

