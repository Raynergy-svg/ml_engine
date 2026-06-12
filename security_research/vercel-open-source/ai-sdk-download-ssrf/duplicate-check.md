# Duplicate / Prior Art Check

Date: 2026-05-17

Verdict: do not submit as-is.

## Public Matches

- GitHub issue: `vercel/ai#13510` — "Security: SSRF bypass via DNS resolution in validateDownloadUrl"
  - Created: 2026-03-17
  - State: open
  - Same root cause: `validateDownloadUrl()` checks hostname strings without DNS resolution.
  - Same `169.254.169.254.nip.io` technique.
  - URL: https://github.com/vercel/ai/issues/13510

- Public gist by the same reporter:
  - Created: 2026-03-17
  - Title: "SSRF via DNS Resolution Bypass in validateDownloadUrl"
  - Same affected functions: `download()` and `downloadBlob()`.
  - URL: https://gist.github.com/Ochk0/01004526ef830113ab21da84628a7c7b

- Related PRs:
  - `vercel/ai#13512` — "fix(provider-utils): prevent SSRF bypass via DNS rebinding"; closed unmerged on 2026-05-03.
  - `vercel/ai#13718` — "fix(provider-utils): prevent SSRF bypass via DNS resolution in validateDownloadUrl"; closed unmerged on 2026-03-22.
  - `vercel/ai#14132` — "fix(provider-utils): harden SSRF protection with missing RFC-defined private IP ranges"; open, but appears focused on literal missing ranges, not the full DNS-resolution bypass.

## HackerOne Visibility

Indexed `site:hackerone.com/reports` searches for `validateDownloadUrl`, `@ai-sdk/provider-utils`, `169.254.169.254.nip.io`, and `Vercel AI SDK SSRF` did not return a public HackerOne report.

That does not rule out a private HackerOne duplicate. The public GitHub issue alone is enough to make this high duplicate/signal risk.

## Recommendation

Do not submit this root cause as a new bounty report. Only revisit if we find a materially different variant that bypasses a shipped fix or demonstrates impact outside the already public `validateDownloadUrl()` DNS-resolution issue.
