#!/usr/bin/env bash
set -euo pipefail

OWNER_REPO="${1:-}"
if [[ -z "$OWNER_REPO" ]]; then
  echo "Usage: $0 OWNER/REPO" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not authenticated. Run: gh auth login" >&2
  exit 1
fi

out_dir="security_research/anthropic-workflow-run-bypass/evidence/live-${OWNER_REPO//\//-}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out_dir"

echo "Collecting evidence into $out_dir"

gh repo view "$OWNER_REPO" --json nameWithOwner,url,visibility,isPrivate,defaultBranchRef \
  > "$out_dir/repo.json"

gh pr list --repo "$OWNER_REPO" --state all --limit 20 \
  --json number,title,state,author,headRepositoryOwner,headRefName,baseRefName,url,isCrossRepository \
  > "$out_dir/prs.json"

gh run list --repo "$OWNER_REPO" --limit 30 \
  --json databaseId,workflowName,event,status,conclusion,createdAt,updatedAt,headBranch,headSha,url,displayTitle \
  > "$out_dir/runs.json"

python3 - "$out_dir/runs.json" "$out_dir/run-summary.md" <<'PY'
import json
import sys

runs = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("# Run summary\n\n")
    for run in runs:
        f.write(
            f"- {run['workflowName']} | {run['event']} | {run['status']}/"
            f"{run.get('conclusion')} | branch={run.get('headBranch')} | "
            f"[run]({run['url']}) | id={run['databaseId']}\n"
        )
PY

latest_child="$(
  gh run list \
    --repo "$OWNER_REPO" \
    --workflow "POC Child workflow_run Claude" \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId // empty'
)"

latest_parent="$(
  gh run list \
    --repo "$OWNER_REPO" \
    --workflow "POC Parent PR CI" \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId // empty'
)"

if [[ -n "$latest_parent" ]]; then
  gh run view "$latest_parent" --repo "$OWNER_REPO" --json url,event,headBranch,headSha,status,conclusion,createdAt,updatedAt \
    > "$out_dir/latest-parent-run.json"
  gh run view "$latest_parent" --repo "$OWNER_REPO" --log-failed \
    > "$out_dir/latest-parent-failed.log" || true
fi

if [[ -n "$latest_child" ]]; then
  gh run view "$latest_child" --repo "$OWNER_REPO" --json url,event,headBranch,headSha,status,conclusion,createdAt,updatedAt \
    > "$out_dir/latest-child-run.json"
  gh run view "$latest_child" --repo "$OWNER_REPO" --log \
    > "$out_dir/latest-child.log" || true
fi

gh api "repos/$OWNER_REPO/branches" --paginate > "$out_dir/branches.json"

cat > "$out_dir/triage-checklist.md" <<'EOF'
# Triage checklist

Report only if all of these are true:

- The triggering PR author is not a write/admin collaborator.
- The child workflow is `workflow_run`, not a maintainer tag/comment/manual run.
- Parent CI logs/artifacts contain `BENIGN_WORKFLOW_RUN_BOUNDARY_POC`.
- Child Claude run reads that parent-controlled content.
- Child run writes a branch or PR comment using privileged workflow permissions.

Do not report if:

- The only successful run was opened by the repo owner/admin.
- The child workflow did not run.
- The child ran but Claude did not process parent-controlled content.
- The result is only generic GitHub `workflow_run` risk with no action-specific bypass.
EOF

echo "Evidence written to $out_dir"

