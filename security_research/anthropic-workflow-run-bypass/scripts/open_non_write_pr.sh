#!/usr/bin/env bash
set -euo pipefail

OWNER_REPO="${OWNER_REPO:-Raynergy-svg/claude-workflow-run-boundary-poc}"
OWNER_LOGIN="${OWNER_REPO%%/*}"
REPO_NAME="${OWNER_REPO#*/}"
BRANCH="non-write-trigger-$(date -u +%Y%m%dT%H%M%SZ)"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required." >&2
  exit 1
fi

if [[ -n "${NONWRITE_GH_TOKEN:-}" ]]; then
  export GH_TOKEN="$NONWRITE_GH_TOKEN"
fi

if ! gh auth status >/dev/null 2>&1 && [[ -z "${GH_TOKEN:-}" ]]; then
  echo "Authenticate as the non-write GitHub account first, or set NONWRITE_GH_TOKEN." >&2
  echo "Do not use the owner account $OWNER_LOGIN for this proof." >&2
  exit 1
fi

actor="$(gh api user --jq .login)"
if [[ "$actor" == "$OWNER_LOGIN" ]]; then
  echo "Refusing to run as $actor. This proof requires a non-write account." >&2
  exit 1
fi

echo "Using GitHub actor: $actor"
echo "Target repo: $OWNER_REPO"

permission="$(
  gh api "repos/$OWNER_REPO/collaborators/$actor/permission" --jq .permission 2>/dev/null || true
)"

if [[ "$permission" == "admin" || "$permission" == "write" || "$permission" == "maintain" ]]; then
  echo "Refusing to run: $actor has $permission permission on $OWNER_REPO." >&2
  exit 1
fi

if [[ -n "$permission" ]]; then
  echo "Observed repository permission for $actor: $permission"
else
  echo "Could not query explicit collaborator permission for $actor; continuing as fork PR proof."
fi

fork_full_name="$(gh repo fork "$OWNER_REPO" --clone=false --remote=false --fork-name "$REPO_NAME" 2>/dev/null || true)"
if [[ -z "$fork_full_name" ]]; then
  fork_full_name="$actor/$REPO_NAME"
fi

echo "Waiting for fork $actor/$REPO_NAME to become available..."
for _ in {1..30}; do
  if gh repo view "$actor/$REPO_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/claude-nonwrite-pr.XXXXXX")"
echo "Working copy: $tmpdir"
git clone "https://github.com/$actor/$REPO_NAME.git" "$tmpdir"
git -C "$tmpdir" config user.name "$actor"
git -C "$tmpdir" config user.email "$actor@example.invalid"
git -C "$tmpdir" switch -c "$BRANCH"

mkdir -p "$tmpdir/poc"
cat > "$tmpdir/poc/non-write-trigger.txt" <<EOF
non-write trigger from $actor
timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
purpose: trigger parent PR CI for workflow_run boundary PoC
EOF

git -C "$tmpdir" add poc/non-write-trigger.txt
git -C "$tmpdir" commit -m "Trigger non-write workflow_run boundary PoC"
git -C "$tmpdir" push -u origin "$BRANCH"

pr_url="$(
  gh pr create \
    --repo "$OWNER_REPO" \
    --base main \
    --head "$actor:$BRANCH" \
    --title "Non-write workflow_run boundary PoC" \
    --body "Non-write actor proof attempt for workflow_run boundary testing. No real secrets; fake markers only."
)"

cat <<EOF

Opened non-write PR:
$pr_url

Watch runs:
  gh run list --repo $OWNER_REPO --limit 10

Collect evidence after child workflow finishes:
  security_research/anthropic-workflow-run-bypass/scripts/collect_live_evidence.sh $OWNER_REPO
EOF

