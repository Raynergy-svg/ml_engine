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
