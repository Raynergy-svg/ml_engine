# Non-write PR proof runbook

This is the only path that can turn the current smoke test into a serious
bounty candidate.

## Current repo

https://github.com/Raynergy-svg/claude-workflow-run-boundary-poc

## Required actor

Use a GitHub account that is not:

- owner of `Raynergy-svg/claude-workflow-run-boundary-poc`
- collaborator on the repo
- organization member with write/admin
- granted any special repository permission

## Steps from the non-write account

Automated path:

```bash
NONWRITE_GH_TOKEN=<token-for-non-write-account> \
  security_research/anthropic-workflow-run-bypass/scripts/open_non_write_pr.sh
```

The script refuses to run as `Raynergy-svg` and refuses actors that appear to
have write/admin/maintain permission.

Manual path:

1. Open https://github.com/Raynergy-svg/claude-workflow-run-boundary-poc
2. Fork it.
3. Create a new branch in the fork.
4. Add or edit any harmless file, for example:
   `poc/non-write-trigger.txt`
5. Open a PR back to:
   `Raynergy-svg/claude-workflow-run-boundary-poc:main`
6. Do not comment `@claude` and do not ask a maintainer to trigger anything.

## What to watch

The expected chain is:

1. `POC Parent PR CI` runs on the pull request and fails.
2. `POC Child workflow_run Claude` starts automatically from `workflow_run`.
3. Child logs show:
   - `GITHUB_TOKEN Permissions` include write-capable permissions.
   - `workflow_run_event=pull_request`.
   - `head_repository_full_name` points to the fork.
   - `workflow_run_actor` or PR author is the non-write account.
   - `Auto-detected mode: agent for event: workflow_run`.
   - `Verified human actor: <non-write-account>`.
4. Claude reads `BENIGN_WORKFLOW_RUN_BOUNDARY_POC` from parent logs/artifacts.
5. Claude pushes a `claude-workflow-run-poc-*` branch or GitHub Actions comments
   on the PR.

## Collect evidence

After the child workflow finishes:

```bash
security_research/anthropic-workflow-run-bypass/scripts/collect_live_evidence.sh Raynergy-svg/claude-workflow-run-boundary-poc
```

Then inspect:

- `prs.json`: PR author and `isCrossRepository=true`.
- `latest-child.log`: `workflow_run` actor, token permissions, action mode, and
  parent marker ingestion.
- `branches.json`: PoC branch.
- PR comments: GitHub Actions comment showing the PoC branch.

## Submission decision

Submit only if the non-write account caused the child Claude workflow to run and
perform the privileged branch/comment action without maintainer/manual trigger.

Do not submit if GitHub blocks the child workflow for fork PRs, if the actor is
the owner, or if Claude does not process parent-controlled content.
