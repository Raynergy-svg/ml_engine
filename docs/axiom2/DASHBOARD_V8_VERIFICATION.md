# Axiom 2.0 dashboard version 8 checkpoint

## Scope and lineage

Approved version 8 Library image: `libfile_0cadf82c9a1c81918f2aaef279260568`, PNG 1536×1024. Materialized in this task executor and visually inspected before implementation. New isolated checkout `task-4/axiom-dashboard-v8`, branch `codex/axiom-2-dashboard-v8`, base `6125da6f97cd59f5bb6cc4537f1482fda2416db9`. Remote recovery base rechecked unchanged before publication. No original/recovery checkout writes, resets, merges, deployments or broker actions.

Implemented overview, Markets, Evidence, Portfolio, Execution and System navigation, responsive desktop/iPhone layout, source inspection, evidence package selection, bounded refresh/reconnect, skip/back navigation and sign-out. Existing terminal is preserved at `/legacy`; no legacy implementation changed. Login wording now identifies Axiom and uses orange emphasis.

GET `/api/axiom2/overview` is a versioned read-only projection of the actual committed evidence cache and fixed kernel boundary manifests. It reports exact byte digest, relative source, local receipt and file modification time separately from unknown source time. Cached states/classification/authenticity remain unverified and never imply capital eligibility. No store construction, signing, holdout consumption, promotion mutation or order transport was added. Existing Next session gate and loopback API proxy remain in place.

**Absent live producer:** this base has no authenticated market/account observation producer bound to the new surface. Quotes, allocation, account value, cash and buying power are unavailable; settled cash and source freshness remain unknown. No static-design sample price, allocation or experiment is presented as real. Execution and capital are hard-disabled. Genuine holdout, live shadow/reconciliation and operational qualification remain unverified. This is implemented and locally verified software, not live readiness.

## Verification

Existing Python interpreter: `repo-suite.Gi7lHb/venv/bin/python`. Python tests used bytecode disabled, plugin autoload disabled, warnings as errors, pytest cache disabled.

- `pytest dashboard/server/test_axiom2_projection.py -q -W error -p no:cacheprovider`: **12 passed**. Empty/no-write, equity-vs-legacy filtering, cache-not-capital, malformed files, nonfinite boundary, symlink escape, no-store and GET-only checks.
- Full approved scoped gate `pytest tests/axiom2 tests/evidence tests/test_evidence_contracts.py tests/test_evidence_store.py tests/test_equity_research_evidence_slice.py tests/test_equity_research_evidence_worker_no_authority.py dashboard/server/test_axiom2_projection.py -q -W error -p no:cacheprovider`: **1330 passed in 42.31s**.
- `tsc --noEmit`: PASS. Production `npm run build`: PASS, Next 16.2.9, including `/`, `/legacy`, login and authenticated API.
- ESLint on all added/changed web modules: PASS. Full web lint remains **29 errors / 4 warnings**, identical to an independently extracted base tree. Existing errors are in CandleChart (23), CommandPalette (2), EquityCurve (2), PriceTiles (2); no rule suppression or unrelated repair was introduced.
- Additional legacy backend gate: **59 passed / 3 failed**. The three failures in `test_data_sources_ledger.py` were reproduced unchanged on the untouched base (5 passed / 3 failed). Existing crypto imports fail under warnings-as-errors because requests reports unsupported urllib3/chardet/charset_normalizer combinations. Dependencies were not reconfigured.
- Actual TypeScript parser verification `dashboard/verification/v8/parser-check.cjs`: valid projection plus **11 hostile payload regressions** PASS, including array-valued status/name, missing provenance, null rows/boundaries, object prerequisites, enabled capital, fake settled zero and unknown schema.
- Headless Chromium browser suite `dashboard/verification/v8/browser-check.cjs`: PASS at desktop **1536×1024** and iPhone-sized **390×844**. Tested authenticated GET, unauthenticated 401, POST405, loading/empty/stale/error/reconnect, nested malformed payloads, evidence selection, disabled orders, all navigation, skip/back history, independent tabs, logout error/retry and signed-session logout. **Zero page errors**, no horizontal mobile overflow. Controlled fixtures are test presentation data only. Local HTTP harness installs a valid signed test token; production Secure-cookie code is unchanged.
- Separate unconfigured-auth process: API503 and page307 login redirect PASS.
- Research isolation probe: PASS; unchanged bundle SHA256 `1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`, `research_kernel_complete=false`, execution disabled.
- Offline read declaration probe: PASS, zero broker actions. Operational execution boundary probe remains **BLOCKED**, no authenticated operational verifier; this is the expected conservative result.

Temporary browser/format tooling lives outside the repo; no paid or runtime dependencies were added. Installed system Chrome initially failed to launch in this executor; temporary headless Chromium completed the checks. All local verification services are stopped after testing.

## Independent review

Three separate read-only agents reviewed the final implementation:

1. `review_integrity`: cache/source truth, holdout/capital boundary and hostile schema. Cleared final scope; independently reran 12 projection tests.
2. `review_access`: access/privacy, fixed paths, polling cleanup/multi-tab and no financial mutation. Cleared final scope; independently reran 12 projection tests.
3. `review_ui`: inspected approved image and final desktop/iPhone screenshots; reviewed responsive/accessibility/navigation/error semantics. Cleared final scope.

Findings repaired with regressions: boundary object/nonfinite JSON handling; complete nested client validation including array enum coercion; honest unknown-session wording; skip link preserving view/history; accessible sign-out restored. Author's browser execution is distinct from the reviewers' static/UI inspection and their backend reruns.

## Screenshots

Final rendered screenshots were visually inspected and saved to Library:

- Desktop overview: `libfile_f921846c86ec819198d334516c3c0f88`, underlying file `file_0000000090f881f5800ac6e510fded19`.
- iPhone overview: `libfile_d1c13117a694819194efcecdd843ae10`, underlying file `file_0000000057a081f59fd8fdaf86e9967e`.

No live observations, credential changes, provider calls, holdout use, deployment or full-live-ready claim is part of this checkpoint.
