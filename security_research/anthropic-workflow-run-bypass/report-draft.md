# Draft report: Claude Code Action workflow_run automation bypasses entity actor boundary

## Summary

Claude Code Action treats `workflow_run` as an automation context. In automation
mode, the action does not call the same `checkWritePermissions()` gate that
protects issue/PR/comment entity triggers. A child `workflow_run` workflow can
run with secrets and write-capable tokens while ingesting PR-controlled logs or
artifacts from a parent workflow.

If reproduced from a non-write or fork PR actor, this bypasses the boundary
described in the prior triage response: the actor did not directly trigger an
entity event that passed the write/admin check, but still influenced a privileged
Claude Code Action run.

## Affected component

- `anthropics/claude-code-action@v1`
- Agent/automation mode triggered by `workflow_run`

## Relevant source evidence

- `src/github/context.ts`: `workflow_run` is included in automation events.
- `src/entrypoints/run.ts`: `checkWritePermissions()` is called only for entity
  contexts.
- `src/modes/agent/index.ts`: automation mode checks human-vs-bot actor, not
  repository write/admin permission.
- `examples/ci-failure-auto-fix.yml`: official example passes CI logs from a
  failed parent run into Claude and checks out the parent `head_branch` with
  elevated permissions.

## Reproduction

Repository: `<throwaway repo URL>`
Parent workflow run: `<URL>`
Child workflow run: `<URL>`
PR: `<URL>`
PoC branch: `<URL>`

1. Install the two attached workflows:
   - `01-parent-pr-ci.yml`
   - `02-child-workflow-run-claude.yml`
2. Set repository variable `CLAUDE_WORKFLOW_RUN_POC_ENABLED=true`.
3. Set fake marker secret `POC_FAKE_MARKER=FAKE-H1-POC-MARKER-12345`.
4. Open a PR that triggers the parent `pull_request` workflow.
5. The parent workflow fails and emits benign PR-controlled log/artifact content
   containing `BENIGN_WORKFLOW_RUN_BOUNDARY_POC`.
6. The child `workflow_run` workflow runs with write permissions, checks out the
   trusted base branch, downloads parent logs/artifacts, and invokes Claude Code
   Action.
7. Claude reads the parent log/artifact and creates
   `workflow-run-poc-result.md` on the PoC branch.

## Expected result

Claude Code Action should not process PR-controlled parent workflow data with
write-capable credentials unless the original PR actor satisfies the same
write/admin boundary enforced for entity triggers, or the action should clearly
detect and reject unsafe `workflow_run` PR handoffs.

## Actual result

`<Fill after live test>`

Owner-account smoke test on 2026-05-16 validated the mechanics but is not
submission evidence. PR #1 was opened by the repository owner, so it does not
prove bypass of the write/admin actor boundary. The smoke test confirmed:

- The parent `pull_request` workflow failed with
  `BENIGN_WORKFLOW_RUN_BOUNDARY_POC` in logs/artifact.
- The child `workflow_run` workflow started automatically with write-capable
  permissions.
- Claude Code Action ran in `agent` mode for event `workflow_run`.
- Claude committed `workflow-run-poc-result.md` to a PoC branch.

This section must be replaced with non-write PR evidence before submission.

## Impact

If confirmed for non-write or fork PR actors:

- A low-trust actor can influence a privileged Claude Code Action run through
  parent CI logs/artifacts.
- The privileged child can push commits or comment on PRs using repository
  write permissions.
- If maintainers expose secrets or sensitive files to the child job, Claude can
  be induced to read fake-marker-equivalent data from the workspace.

Tentative severity:

- Medium if the strongest proof is unauthorized write-capable action influence.
- High only if broad default impact or real secret-equivalent access is proven
  without explicit dangerous configuration.

## Why this differs from generic workflow_run risk

The issue is not merely that GitHub `workflow_run` can be dangerous. The
action-specific boundary issue is that Claude Code Action enforces write/admin
permission for entity contexts but not for automation contexts, while its own
CI auto-fix example demonstrates exactly the risky pattern of feeding parent CI
logs into Claude in a privileged child workflow.

## Suggested remediation

- For `workflow_run` payloads associated with pull requests, resolve the original
  PR actor and require the same write/admin permission check used for entity
  triggers before running Claude.
- Treat `workflow_run` data from failed parent workflows as untrusted.
- Restore `.claude/` and `.mcp.json` from the trusted base when checking out PR
  heads in automation mode, or warn/block when the child workflow checks out
  `workflow_run.head_branch`.
- Update examples to require explicit same-repo trusted actor validation, not
  only `pull_requests[0]`.
