# Incident Record — why the hard rules exist

> Relocated from CLAUDE.md (2026-06-09). These are the institutional-memory stories behind
> the rules in `.claude/rules/honesty.md`, `.claude/rules/improvement.md`, and the No-Mock
> rule. Kept out of the always-loaded path; read when you want the "why".

## f070d39 — the orchestrator-routing lie (2026-04-30)

Shipped commit f070d39 claiming "orchestrator routes per-cycle diagnostics through the
meta-pipeline". Unit tests on `Orchestrator._maybe_route_to_meta` passed (mocked
dependencies). I told the operator the routing was wired and live. Reality:
`grep "Orchestrator(" src/tui/` returns nothing — the TUI never instantiates Orchestrator,
so the routing was dead code in the runtime path. Unit-test pass ≠ integration. The
operator's "are you sure??" forced re-verification; only then did the gap surface.

→ Rule: **integration grep before "wired"** (honesty.md #3).

## The $3,527 config-adjustment dead-write (cost over 14 trades)

Pending config-adjustment keys did not match `ScannerConfig` dataclass field names, so
`setattr()` created orphan attributes no code read. Both the write side and read side
existed, but the persistence layer didn't connect them. Confirmed orphan keys:
`min_confidence_threshold`→`min_confidence`, `atr_sl_multiplier_low_regime`→`atr_sl_multiplier`.

→ Rule: **validate config keys against `ScannerConfig` field names before writing**
(improvement.md "Config Adjustment Consumer Verification").

## The No-Mock catastrophe (2026-05-01)

38 passing `test_meta_manager.py` tests hid a production wiring gap: `StagedDeployer` was
constructed without `config_adjuster=ConfigAdjuster()` for two weeks. 11 ChangePackages
walked shadow→canary→live with **zero actual config mutation**. Mocks made the integration
gap invisible. Only an end-to-end log audit surfaced it.

→ Rule: **NO MOCK CODE** (CLAUDE.md code-quality + improvement.md "No-Mock Rule").

## gates.py:1902 mis-diagnosis (2026-05-11)

Shipped a calibrated-threshold fix calling it "the load-bearing fix" for an 88% SHORT bias.
But `modular_inference.py` was already reading and applying the calibrated thresholds
correctly — my fix closed a real gap in a *secondary* path but did NOT address the bias. I
called it load-bearing without grepping for parallel implementations of the same logic.

→ Rule: **pre-commit causal-claim discipline + calibrated confidence tags** (honesty.md).

## C1 train↔inference scaler skew (2026-05-08)

`transformer_direction.meta.pkl` had `scaler.var_ = 1.0` exactly across all 50 features —
the literal fingerprint of a scaler fitted on already-scaled data (double-fit). The
celebrated "70% M15 holdout" was the all-SHORT fallback predictor's accuracy on a
SHORT-heavy slice; the trained transformer never actually evaluated. Six simultaneous
contract violations broke every transformer prediction since C1.A landed.

→ Rule: **Train↔Inference Contract Gates** (improvement.md); detail in `docs/strategy.md`.
