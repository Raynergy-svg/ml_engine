# Improvement Rules

Meta-rules governing how Buddy learns and evolves.

## Learning Triggers
- Every closed trade triggers learning extraction (analyze outcome vs prediction)
- Every losing trade > $100 triggers deep analysis (LLM-assisted if enabled)
- Every 10 scan cycles triggers learnings audit (consolidation check)

## Promotion Criteria
- A pattern observed 3+ times in learnings.md gets promoted to rules/trading.md
- Promoted rules include the date, source count, and specific actionable directive
- Source learnings are marked [PROMOTED] after extraction

## Consolidation
- When learnings.md exceeds 30 entries: group by category, archive old entries
- When rules/trading.md exceeds 50 lines: split by domain (entry rules vs risk rules)
- When config_adjustments.json exceeds 100 entries: archive entries older than 30 days

## Code Quality Gates (promoted 2026-03-18, from 4 robustness observations)
- ALWAYS run code review specialist on new subsystems BEFORE first production use
- ALWAYS validate JSON parsing with try/except and graceful defaults (never crash on corrupted files)
- ALWAYS use file locking (fcntl) when multiple processes may write to shared JSON files
- ALWAYS verify state.json claims against source-of-truth files before acting on them
- ALWAYS use lazy imports in package __init__.py when submodules have heavy dependencies

## JSON Safety Gates (promoted 2026-03-23, from 31 observations)
- ALWAYS wrap JSON file reads in try/except with graceful fallback to empty dict/list
- ALWAYS validate JSON structure after parsing (check expected keys exist before access)
- ALWAYS write JSON atomically: write to .tmp file first, then os.rename() to final path
- NEVER trust JSON file contents without schema validation in production paths
- ALWAYS use json.dumps with indent=2 and sort_keys=True for human-readable persistence

## Retry & Robustness Gates (promoted 2026-03-23, from 27 observations)
- ALWAYS implement exponential backoff for OANDA API calls (base 1s, max 30s, jitter)
- ALWAYS set explicit timeouts on all HTTP requests (connect=5s, read=30s)
- ALWAYS catch specific exceptions (requests.Timeout, ConnectionError) not bare except
- NEVER retry on 4xx client errors — only on 5xx, timeout, and connection failures
- ALWAYS log retry attempts with attempt number, delay, and error context

## State Persistence Gates (promoted 2026-03-23, from 8 observations)
- ALWAYS flush state to disk before shutdown (save_state() in every module with mutable state)
- ALWAYS validate state file freshness on load (check timestamp, warn if stale > 1 hour)
- NEVER assume in-memory state survives process restart — always persist critical state
- ALWAYS include version field in persisted state files for forward compatibility

## Test Coverage Gates (promoted 2026-03-23, from 8 observations)
- ALWAYS write unit tests for new calculation/logic functions before merging
- ALWAYS test edge cases: zero values, negative values, None inputs, empty collections
- ALWAYS use mock-based testing for external API dependencies (OANDA, news feeds)
- NEVER ship a new subsystem without at least 5 unit tests covering core paths

## Config Validation Gates (promoted 2026-03-23, from 6 observations)
- ALWAYS validate config values at load time (range checks, type checks, required fields)
- ALWAYS provide sensible defaults for optional config fields via dataclass defaults
- NEVER silently ignore unknown config keys — log a warning for typo detection
- ALWAYS ensure profile-specific overrides don't violate safety invariants (min SL, max risk)

## Silent Exception Prevention (promoted 2026-03-23, from 4 observations)
- NEVER use bare except: or except Exception: pass — always log the error
- ALWAYS re-raise or return error status after logging — callers must know something failed
- ALWAYS include context in error logs: function name, input parameters, stack trace
- NEVER swallow errors in financial calculation paths — surface them as trade rejections

## Live Wiring Verification Gates (promoted 2026-03-24, from 6 observations in single audit)
- ALWAYS run a live scan smoke test after wiring a new module — verify the module's log output appears in production output, not just in unit tests
- ALWAYS add new config feature flags as dataclass fields FIRST, then profile dict entries, then consumer getattr() calls — missing any one of these three = dead feature silently skipped by apply_profile()
- ALWAYS verify both the write side AND read side of any feedback/telemetry system — if record_X() exists without a corresponding get_X() consumer in the live loop, the module is write-only dead code
- NEVER trust "passes: true" on wiring phases without verifying call sites exist in production files — unit tests mock boundaries and will pass even when the production call is missing
- ALWAYS check that methods defined for integration are actually CALLED, not just defined — grep for the method name outside its own class to confirm at least one live call site exists
- ALWAYS test new feature flag propagation end-to-end: set flag in profile dict → verify it reaches the dataclass field → verify consumer reads True → verify module activates (log line appears)

## Anti-Patterns
- Never create new .claude/ files without justification — edit existing ones
- Never let learnings accumulate without triage (apply / capture / dismiss)
- Never evolve config silently — log every adjustment with reason
- Never guess at stale state — read state.json, ask if unclear
