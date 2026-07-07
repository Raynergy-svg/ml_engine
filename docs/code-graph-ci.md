# Code Graph CI — codebase-memory-mcp

`.github/workflows/code-graph.yml` runs [codebase-memory-mcp](https://github.com/DeusData/codebase-memory-mcp)
(the operator's fork, [Raynergy-svg/codebase-memory-mcp](https://github.com/Raynergy-svg/codebase-memory-mcp),
is an unmodified fork of the same upstream) as a **CI/dev-plane-only** job: it
indexes this repo's source into a structural code graph and, on pull requests,
posts a "what changed + what to retest" impact summary.

## What this is NOT

- **Not part of the runtime.** Nothing here executes on the live trading host.
  There is no code path from this workflow into `src/scanner/execution.py`,
  `state.json`, halt handling, or any trade/arm path.
- **Not a required check.** The job is fail-soft end-to-end (every step is
  `continue-on-error`, plus a job-level `continue-on-error: true` as a second
  guard) and must never be added to branch-protection required checks. A
  broken tool install, a bad index, or a failed impact query all degrade to a
  warning comment/summary — they never block a merge.
- **Not `install`.** The workflow never runs `codebase-memory-mcp install`
  (which registers MCP server entries into agent config files like
  `.claude/.mcp.json`). It only calls one-shot `cli <tool>` subcommands
  against a pinned, checksum-verified binary. It writes to exactly two
  places, both scoped to the ephemeral CI workspace: `$CBM_CACHE_DIR`
  (the tool's own SQLite store) and `.codebase-memory/graph.db.zst` (the
  team-shared snapshot artifact).

## How it works

1. **Sparse checkout.** The checkout excludes `trained_data/`, `market_data/`,
   `logs/`, `notebooks/`, `legacy_quarantine/`, `bug_reports/`, `tmp/` —
   this repo tracks hundreds of MB of model binaries (`.pkl`, `.keras`,
   `.h5`) and price data (`.csv`, `.parquet`) that codebase-memory-mcp does
   **not** skip by extension even in `fast` mode (confirmed by local
   verification — see "Why the exclude list" below). Indexing the full repo
   took multiple minutes just parsing quarantined `.pkl` files; indexing the
   sparse code-only tree takes ~8 seconds.
2. **Pinned, checksum-verified install.** Downloads
   `codebase-memory-mcp-linux-amd64.tar.gz` from a pinned release tag
   (`CBM_VERSION` in the workflow) and verifies its SHA-256 against a
   hardcoded checksum before running it. A checksum mismatch or download
   failure fails soft with a clear reason in the job summary — it never runs
   an unverified binary.
3. **Index.** `cli index_repository` with `mode:"fast"` (skips
   similarity/semantic edges — not needed for impact analysis) and
   `persistence:true` (writes `.codebase-memory/graph.db.zst`). Critically,
   `repo_path` is set to the checkout **root** (`github.workspace`), not a
   subdirectory — `detect_changes` matches git-diff paths (repo-root-relative)
   against the graph's stored file paths, and if `repo_path` were a
   subdirectory the prefixes would never match and impact analysis would
   silently return zero impacted symbols. Verified locally both ways.
4. **Impact analysis (PRs only).** `cli detect_changes` with `base_branch`
   set to the PR's base ref, default `scope` (returns both `changed_files`
   and `impacted_symbols` — passing an unrecognized `scope` value silently
   disables symbol mapping, so the workflow never overrides it).
5. **PR comment.** A single sticky comment (found/updated via an HTML marker,
   not reposted on every push) summarizing changed-file count, impacted
   symbol count, and a sample of impacted symbols.
6. **Artifact.** `.codebase-memory/graph.db.zst` is uploaded as a workflow
   artifact (14-day retention) — **not** committed back to the repo. This
   keeps the job read-only with respect to the repository (no `contents:
   write` permission needed).

## Running it locally (off the live host)

Do this on your dev machine, never on the live trading host:

```bash
CBM_VERSION=v0.8.1
curl -fsSL -o cbm.tar.gz \
  "https://github.com/DeusData/codebase-memory-mcp/releases/download/${CBM_VERSION}/codebase-memory-mcp-$(uname -s | tr A-Z a-z)-$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz"
tar xzf cbm.tar.gz codebase-memory-mcp
chmod +x codebase-memory-mcp

# Keep the index workspace-local — don't pollute ~/.cache
export CBM_CACHE_DIR="$PWD/.cbm-cache"

# Point repo_path at the repo ROOT (not a subdirectory — see "How it works" #3)
./codebase-memory-mcp cli --json index_repository \
  '{"repo_path":"'"$PWD"'","mode":"fast","persistence":true}'

# Ask what a local diff would impact
./codebase-memory-mcp cli --json detect_changes \
  '{"project":"<project-name-from-index-output>","base_branch":"main"}'
```

Note: CI always diffs against `origin/<base-branch>` specifically (not a bare
local branch name — see `code-graph.yml`'s `BASE_REF`/`BASE` handling), since
a fresh Actions checkout has no local `main` branch, only the fetched
`origin/main` ref. If your local `main` isn't synced with `origin/main`, your
local `detect_changes` output can differ from CI's.

Do **not** run `codebase-memory-mcp install` in this repo unless you
specifically want it registered as an MCP server for your own agent config —
that command writes to `~/.claude/.mcp.json` / project `.mcp.json` and is
outside the scope of this CI integration.

### Consuming the shared snapshot

Download the `codebase-memory-graph-<sha>` artifact from a recent
`code-graph` workflow run (Actions tab → the run → Artifacts), extract it to
`.codebase-memory/graph.db.zst` in your local checkout, then run
`index_repository` locally — the tool bootstraps from the artifact instead of
a full reindex.

## Fail-soft behavior / what "blocked" looks like

If `DeusData/codebase-memory-mcp` renames its release asset, moves to a new
major version with a different CLI shape, or the pinned tag is deleted, the
**Install** step fails cleanly: `install_ok=false` with a `reason` string,
surfaced in both the job summary and (on PRs) the sticky comment. No indexing
or impact-analysis step runs — but the summary/comment steps still run (they
use `if: always()`) specifically so the `reason` is visible. The job exits 0
either way. To recover:

1. Check the current release: `gh release view --repo DeusData/codebase-memory-mcp latest`
2. Update `CBM_VERSION` and `CBM_SHA256_LINUX_AMD64` in
   `.github/workflows/code-graph.yml` (checksum is in that release's
   `checksums.txt` asset).
3. Re-run the workflow via `workflow_dispatch`.

## Why the exclude list (verification notes)

Confirmed locally (isolated git-worktree copy, never the live checkout):
`mode:"full"` on this repo's untouched tree spent 300k+ ms per file parsing
individual quarantined `.pkl` files under `trained_data/models/*/_quarantine/`
(2,590 files queued) — neither `.pkl`, `.csv`, `.keras`, `.h5`, nor `.jsonl`
are in the tool's `ALWAYS_IGNORED_SUFFIXES` or `FAST_IGNORED_SUFFIXES` lists,
so `mode:"fast"` alone does not skip them. The sparse-checkout exclude list
above is what actually keeps this job fast — extend it if new large
binary/data directories get added to the repo root.
