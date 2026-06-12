#!/usr/bin/env bash
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="${REPO_NAME:-claude-workflow-run-boundary-poc}"
VISIBILITY="${VISIBILITY:-public}"
FAKE_MARKER="${POC_FAKE_MARKER:-FAKE-H1-POC-MARKER-12345}"
CREATE_OWNER_SMOKE_PR="${CREATE_OWNER_SMOKE_PR:-0}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login" >&2
  exit 1
fi

HAS_ANTHROPIC_API_KEY=0
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  HAS_ANTHROPIC_API_KEY=1
fi

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "VISIBILITY must be public or private." >&2
  exit 1
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/claude-workflow-run-poc.XXXXXX")"
cleanup() {
  echo "Working copy left at: $tmpdir"
}
trap cleanup EXIT

echo "Creating throwaway $VISIBILITY repo: $REPO_NAME"
gh repo create "$REPO_NAME" "--$VISIBILITY" --description "Throwaway Claude Code Action workflow_run boundary PoC" --clone=false

repo_url="$(gh repo view "$REPO_NAME" --json url --jq .url)"
owner_repo="$(gh repo view "$REPO_NAME" --json nameWithOwner --jq .nameWithOwner)"

git -C "$tmpdir" init -b main
git -C "$tmpdir" config user.name "whitehat-poc"
git -C "$tmpdir" config user.email "whitehat-poc@example.invalid"
git -C "$tmpdir" remote add origin "https://github.com/${owner_repo}.git"

mkdir -p "$tmpdir/.github/workflows" "$tmpdir/poc"
cp "$KIT_DIR/workflows/01-parent-pr-ci.yml" "$tmpdir/.github/workflows/01-parent-pr-ci.yml"
cp "$KIT_DIR/workflows/02-child-workflow-run-claude.yml" "$tmpdir/.github/workflows/02-child-workflow-run-claude.yml"
cp "$KIT_DIR/payloads/ci-log-prompt-injection.txt" "$tmpdir/poc/ci-log-prompt-injection.txt"

cat > "$tmpdir/README.md" <<'EOF'
# Claude workflow_run boundary PoC

Throwaway repository for testing whether PR-controlled parent CI logs/artifacts
can influence a privileged Claude Code Action child workflow_run.

This repo must not contain real secrets. The PoC uses fake markers only.
EOF

git -C "$tmpdir" add .
git -C "$tmpdir" commit -m "Add workflow_run boundary PoC"
git -C "$tmpdir" push -u origin main

echo "Setting repo variable and fake-marker secret"
gh variable set CLAUDE_WORKFLOW_RUN_POC_ENABLED --repo "$owner_repo" --body true
printf '%s' "$FAKE_MARKER" | gh secret set POC_FAKE_MARKER --repo "$owner_repo"

if [[ "$HAS_ANTHROPIC_API_KEY" == "1" ]]; then
  printf '%s' "$ANTHROPIC_API_KEY" | gh secret set ANTHROPIC_API_KEY --repo "$owner_repo"
else
  echo "ANTHROPIC_API_KEY was not set locally, so the repo secret was not created."
  echo "Set it before the live proof run:"
  echo "  gh secret set ANTHROPIC_API_KEY --repo $owner_repo"
fi

cat <<EOF

Repository ready: $repo_url

Important:
- A PR opened by this same owner account is only a mechanics smoke test.
- A bounty-grade proof needs a PR from an account that does NOT have write/admin
  on $owner_repo.

For the real test:
1. From a second GitHub account, fork $repo_url
2. Create any branch with a trivial file change.
3. Open a PR back to $owner_repo:main.
4. Let "POC Parent PR CI" fail.
5. Watch for "POC Child workflow_run Claude".

Monitor with:
  security_research/anthropic-workflow-run-bypass/scripts/collect_live_evidence.sh $owner_repo
EOF

if [[ "$CREATE_OWNER_SMOKE_PR" == "1" ]]; then
  echo
  echo "Creating owner-account smoke PR. This is NOT bounty-grade proof."
  git -C "$tmpdir" switch -c owner-smoke-pr
  date -u +"owner smoke trigger %Y-%m-%dT%H:%M:%SZ" > "$tmpdir/poc/owner-smoke-trigger.txt"
  git -C "$tmpdir" add poc/owner-smoke-trigger.txt
  git -C "$tmpdir" commit -m "Trigger owner smoke PR"
  git -C "$tmpdir" push -u origin owner-smoke-pr
  gh pr create \
    --repo "$owner_repo" \
    --base main \
    --head owner-smoke-pr \
    --title "Owner smoke PR for workflow_run PoC" \
    --body "Owner-account smoke test only. Not bounty-grade proof because actor owns the repository."
fi
