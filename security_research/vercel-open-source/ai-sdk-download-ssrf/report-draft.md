# SSRF protection bypass in AI SDK download URL validation via DNS-resolving hostnames

## Summary

`@ai-sdk/provider-utils` exposes `validateDownloadUrl()` and uses it in server-side download helpers such as `downloadBlob()`. The `ai` package also exposes `createDownload()`, which calls the same validation path before fetching URLs.

The validator blocks literal private/internal IP addresses such as `127.0.0.1`, `10.0.0.1`, and `169.254.169.254`, but it does not resolve DNS before allowing the request. A hostname such as `127.0.0.1.nip.io` or `169.254.169.254.nip.io` passes validation even though DNS resolves it to a blocked address. The subsequent fetch reaches the internal address.

## Affected Versions

Verified on current npm releases:

- `ai@6.0.184`
- `@ai-sdk/provider-utils@4.0.27`

Affected source:

- `packages/provider-utils/src/validate-download-url.ts`
- `packages/ai/src/util/download/download.ts`
- `packages/provider-utils/src/download-blob.ts`

## Security Impact

Applications that accept user-controlled file/media URLs and rely on AI SDK's default download utilities for SSRF protection can be made to send server-side requests to loopback, private RFC1918 ranges, or link-local metadata addresses by using DNS names that resolve to those addresses.

This bypasses the package's explicit internal-address blocking and can expose internal services, local admin panels, or cloud metadata endpoints depending on the deployment network.

No Vercel-owned infrastructure was tested. The PoC only starts a local HTTP server bound to `127.0.0.1` and proves that the AI SDK download helper reaches it through a DNS alias.

## Reproduction

1. Create a fresh directory and install the tested versions:

```bash
npm init -y
npm install ai@6.0.184 @ai-sdk/provider-utils@4.0.27
```

2. Run the attached `poc.mjs`.

Expected result:

- `validateDownloadUrl("http://127.0.0.1:<port>/...")` is blocked.
- `validateDownloadUrl("http://127.0.0.1.nip.io:<port>/...")` is allowed.
- `ai.createDownload()` fetches the loopback-only local server and prints `AI_SDK_SSRF_LOOPBACK_MARKER`.
- `@ai-sdk/provider-utils.downloadBlob()` does the same.
- `169.254.169.254.nip.io` resolves to `169.254.169.254` and is allowed by validation, showing the same bypass applies to link-local metadata-style addresses without contacting that address.

## Root Cause

`validateDownloadUrl()` checks `parsed.hostname` directly. It rejects only literal localhost/private IP hostnames and `.local`/`.localhost` suffixes. It does not resolve hostnames to IP addresses and validate the resolved address before connecting.

The redirect check repeats the same string-based validation on `response.url`, so it does not address DNS-based aliases or DNS rebinding.

## Suggested Fix

Resolve the target hostname before network fetches and reject any resolved address in loopback, link-local, private RFC1918, unique-local IPv6, unspecified, multicast, or otherwise non-global ranges.

To avoid DNS rebinding between validation and connect, pin the request to the validated address through a custom dispatcher/lookup function or equivalent fetch transport. Repeat the same resolved-IP validation for every redirect target.

## Severity

Suggested severity: Medium.

Rationale: network-adjacent SSRF from user-controlled media/file URLs against a core security validation function in a Tier 1 Vercel OSS project. Impact depends on how the downstream app accepts URLs and its deployment network, but the bypass is default, unauthenticated from the application user's perspective, and affects distributable npm code.
