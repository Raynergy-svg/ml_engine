# Anthropic Claude Code Action workflow_run boundary PoC kit

This folder contains a safe reproduction kit for testing whether `workflow_run`
automation mode can bypass the boundary Anthropic stated for Claude Code Action:
entity events get a write/admin actor check, while automation events are handled
separately.

The kit is designed for a throwaway personal GitHub repository only. It uses fake
markers, a required opt-in repository variable, and harmless branch/comment
operations. Do not run this against third-party repositories or real secrets.

## What this tests

Primary hypothesis:

- A low-trust pull request can control CI logs or artifacts in a parent
  `pull_request` workflow.
- A child `workflow_run` workflow can run with `contents: write`, `pull-requests:
  write`, secrets, and `id-token: write`.
- Claude Code Action parses `workflow_run` as an automation context, so its
  entity-only `checkWritePermissions()` gate is not reached.
- If Claude follows PR-controlled log content and writes/comments with elevated
  permissions, that is stronger than the prior symlink report because it targets
  Anthropic's stated boundary directly.

Kill condition:

- If only trusted write-access users can trigger the effect, or if the result is
  just generic `workflow_run` misuse without an action-specific boundary bypass,
  do not submit a bounty report.

## Files

- `workflows/01-parent-pr-ci.yml`: low-privilege parent CI that emits benign,
  PR-controlled log/artifact content.
- `workflows/02-child-workflow-run-claude.yml`: gated child workflow using
  Claude Code Action in `workflow_run` agent mode.
- `payloads/ci-log-prompt-injection.txt`: marker-only prompt-injection payload
  used by the parent workflow.
- `report-draft.md`: HackerOne-ready draft with placeholders for live evidence.
- `evidence/source-findings.md`: local source findings and current interpretation.
- `scripts/validate_kit.py`: dependency-free safety and source-evidence checker.
- `scripts/setup_live_poc_repo.sh`: creates a gated throwaway GitHub repo once
  `gh` and `ANTHROPIC_API_KEY` are available.
- `scripts/open_non_write_pr.sh`: opens the proof PR using a separate non-write
  GitHub token or account.
- `scripts/collect_live_evidence.sh`: captures PR/run/log metadata into
  `evidence/live-*` after a test run.

## Safe setup in a throwaway repo

1. Create a new private or public test repository that you own.
2. Add repository variable `CLAUDE_WORKFLOW_RUN_POC_ENABLED` with value `true`.
3. Add repository secret `ANTHROPIC_API_KEY` for Claude Code Action.
4. Optionally add fake marker secret `POC_FAKE_MARKER` with a value like
   `FAKE-H1-POC-MARKER-12345`. Do not use real credentials.
5. Copy both files from `workflows/` into `.github/workflows/`.
6. Open a test PR that intentionally fails the parent workflow and emits the
   benign payload.

Collect evidence from:

- The parent run: actor, event, PR origin, failed log lines, artifact contents.
- The child run: `workflow_run` payload summary, permissions, actor, and whether
  Claude created the harmless PoC file/comment from parent logs/artifacts.
- The resulting branch/comment proving elevated write capability.

The child workflow checks out the trusted base branch by default so fork PRs can
be tested without failing at checkout. To test the official CI auto-fix pattern
more literally, change the checkout ref to `github.event.workflow_run.head_branch`
inside the throwaway repository after the log/artifact-only case is understood.

## Local validation

Run:

```bash
python3 security_research/anthropic-workflow-run-bypass/scripts/validate_kit.py
```

The script checks that the workflows are gated, use only fake markers, and that
the local Claude Code Action source still matches the evidence assumptions when
`/tmp/claude-code-action-bounty` is present.

## Live harness

Prerequisites:

- `gh auth login`
- `export ANTHROPIC_API_KEY=...`

Create the throwaway repo:

```bash
REPO_NAME=claude-workflow-run-boundary-poc \
  security_research/anthropic-workflow-run-bypass/scripts/setup_live_poc_repo.sh
```

Optional owner-account smoke test:

```bash
CREATE_OWNER_SMOKE_PR=1 \
  security_research/anthropic-workflow-run-bypass/scripts/setup_live_poc_repo.sh
```

The owner smoke test validates mechanics only. It is not bounty-grade proof.

Collect evidence:

```bash
security_research/anthropic-workflow-run-bypass/scripts/collect_live_evidence.sh OWNER/REPO
```
