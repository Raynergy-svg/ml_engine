# Source findings

Target source reviewed locally at `/tmp/claude-code-action-bounty`.

## Boundary cited by Anthropic

Anthropic closed the earlier report as Informative because their stated controls
are:

- Default entity triggers require a write/admin repository actor.
- External PR prompt-injection and workspace-file disclosure are documented.
- In non-write-user mode, the token is not written into `.git/config`.

## Action-specific facts to validate

- `src/github/context.ts` classifies `workflow_run` as an automation event, not
  an entity event.
- `src/entrypoints/run.ts` calls `checkWritePermissions()` only when
  `isEntityContext(context)` is true.
- `src/modes/agent/index.ts` calls `checkHumanActor()` but not
  `checkWritePermissions()`.
- `examples/ci-failure-auto-fix.yml` uses `workflow_run`, downloads failed CI
  logs, checks out `workflow_run.head_branch`, creates a branch, and invokes
  Claude Code Action with write-capable permissions.

## GitHub platform facts

GitHub's official `workflow_run` documentation says a workflow triggered by
`workflow_run` can access secrets and write tokens even when the triggering
workflow cannot. The same section warns that running untrusted code or using
untrusted data from the triggering workflow can create security risks.

Reference:
https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run

## Current interpretation

This is reportable only if the PoC proves one of the following:

- A non-write actor can cause a privileged Claude Code Action execution through
  `workflow_run`.
- Fork PRs are not actually excluded despite example comments implying they are.
- PR-controlled logs/artifacts/config influence Claude in automation mode where
  PR config restoration and write-permission checks are not applied.

This is not reportable if it only demonstrates the generic and documented
GitHub `workflow_run` trust-boundary hazard.

