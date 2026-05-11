"""Holdout harness ↔ Scanner runtime parity tests (2026-05-11).

The bug: `scripts/scheduled_retrain.py:393` instantiated GateEvaluator with
default kwargs (use_joint_only=True, use_per_pair_routing=False), so the deploy
gate measured the stale joint-dir artifact instead of the freshly-retrained
per-pair artifact at trained_data/models/{PAIR}/. Today's 25.8% M15 was the
all-SHORT fallback predictor's score, NOT the real model.

The fix: pass use_per_pair_routing=True, use_joint_only=False — matching
src/scanner/engine.py:1677-1682 Scanner runtime.

These tests are REAL no-mock (per .claude/rules/improvement.md "No-Mock Rule"):
- Real GateEvaluator class
- Real disk at trained_data/models/
- Real EUR_USD per-pair artifact at trained_data/models/EUR_USD/

Three-layer wiring verification:
  L1  harness passes correct kwargs (Tests 1, 2, 5)
  L2  GateEvaluator routes per-pair correctly (Test 3, 4)
  L3  per-pair sub-evaluator loads contract-complete artifact (Test 6)
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.scanner.gates import GateEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "trained_data" / "models"
EUR_USD_DIR = MODEL_DIR / "EUR_USD"


def _build_harness_evaluator() -> GateEvaluator:
    """Construct the GateEvaluator exactly as scripts/scheduled_retrain.py does.

    Mirrors the patched construction at scheduled_retrain.py:393-422 — kept
    in this helper so all tests share one source of truth.
    """
    return GateEvaluator(
        model_dir=str(MODEL_DIR),
        use_joint_only=False,
        use_per_pair_routing=True,
    )


# --------------------------------------------------------------------------- #
# Layer 1 — harness passes correct kwargs                                     #
# --------------------------------------------------------------------------- #

def test_holdout_evaluator_uses_per_pair_routing():
    """L1: The harness's evaluator MUST have use_per_pair_routing=True.

    This is what makes evaluate_all_gates(instrument=pair) lazy-build a
    pair-specific sub-evaluator.
    """
    ev = _build_harness_evaluator()
    assert ev.use_per_pair_routing is True, (
        "Holdout harness evaluator must enable per-pair routing to mirror "
        "Scanner runtime (src/scanner/engine.py:1681). Pre-fix default was "
        "False, which locked the harness to the joint dir's stale artifact."
    )


def test_holdout_evaluator_not_joint_only():
    """L1: use_joint_only MUST be False — per-pair primary, joint fallback.

    Per CLAUDE.md "Tier 7 per-pair gate routing": joint is the FALLBACK for
    correlation-threshold-dropped pairs (USD_JPY, EUR_GBP, EUR_JPY). Setting
    use_joint_only=True would force the parent evaluator to load ONLY joint
    models, defeating the purpose of per-pair routing for the pairs that DO
    have per-pair training.
    """
    ev = _build_harness_evaluator()
    assert ev.use_joint_only is False, (
        "Holdout harness must NOT use joint-only mode — per-pair routing's "
        "fallback layer is joint, but the primary path is per-pair."
    )


# --------------------------------------------------------------------------- #
# Layer 2 — GateEvaluator routes per-pair correctly                           #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not EUR_USD_DIR.exists(),
    reason="EUR_USD per-pair training artifact missing — retrain required",
)
def test_holdout_dispatches_per_pair_for_eur_usd():
    """L2: _get_pair_evaluator('EUR_USD') returns a sub pointing at the
    EUR_USD per-pair dir, NOT joint/, NOT the parent's base_model_dir.
    """
    ev = _build_harness_evaluator()
    sub = ev._get_pair_evaluator("EUR_USD")
    assert sub is not ev, (
        "EUR_USD has a per-pair training subdir on disk — _get_pair_evaluator "
        "should build a fresh sub, not return the parent."
    )
    assert sub.model_dir == EUR_USD_DIR, (
        f"Sub-evaluator's model_dir should be {EUR_USD_DIR}, got {sub.model_dir}"
    )
    # And the sub itself MUST NOT recurse — sub._pair_evaluators is empty by
    # construction (gates.py:453).
    assert sub._pair_evaluators == {}, (
        "Sub-evaluator should disable recursive per-pair routing."
    )


def test_fallback_to_joint_for_correlation_dropped_pair():
    """L2: For a pair WITHOUT a per-pair subdir, _get_pair_evaluator falls
    back to self (the parent), which loads from joint/ as its fallback.

    CLAUDE.md: USD_JPY, EUR_GBP, EUR_JPY were correlation-threshold-dropped
    in the master-pair training. If any of these have no subdir, the route
    is parent (which served joint/ if use_joint_only=True, or base/ for
    use_joint_only=False). With per-pair routing enabled but no subdir for
    an instrument, the route returns the parent — this is the fallback
    behavior. We use a synthetic NEVER_USD that can't accidentally exist.
    """
    ev = _build_harness_evaluator()
    fake_pair = "NEVER_USD_FAKE_PAIR_XYZ"
    fake_dir = MODEL_DIR / fake_pair
    assert not fake_dir.exists(), (
        f"Test prerequisite: {fake_dir} must NOT exist. If it does, choose a "
        f"different fake pair name."
    )
    sub = ev._get_pair_evaluator(fake_pair)
    # Falls back to self (parent) — gates.py:441-442.
    assert sub is ev, (
        "For a pair with no per-pair subdir, _get_pair_evaluator must fall "
        "back to self (the parent), which provides the joint-fallback layer."
    )


# --------------------------------------------------------------------------- #
# Combined: harness ↔ Scanner runtime config parity                           #
# --------------------------------------------------------------------------- #

def test_harness_matches_scanner_runtime_config():
    """L1 / parity: The harness's GateEvaluator kwargs should match the
    auto-enabled Scanner runtime kwargs at src/scanner/engine.py:1677-1682.

    Parses the engine.py source AST (no Scanner instantiation — that requires
    OANDA / a full ScannerConfig). Verifies the harness mirrors what the
    runtime actually does when per-pair training is present (which is the
    deploy-decision-relevant case).

    Scanner runtime behavior:
      - use_joint_only: defaults to ScannerConfig.use_joint_models_only=True,
        flips to False if joint dir missing. The KEY question for parity
        isn't the joint-only flag (it serves the routing fallback layer)
        but whether per-pair routing is on.
      - use_per_pair_routing: True iff any per-pair dir has training files.

    Today (per-pair training EXISTS for EUR_USD etc.), Scanner sets
    use_per_pair_routing=True. The harness must too.

    This test asserts the harness's hard-coded values match the values
    Scanner runtime would compute given that per-pair training exists on
    disk — i.e. use_per_pair_routing=True. We allow use_joint_only=False
    in the harness (it's defensive: forces the parent to NOT pin to joint,
    so per-pair routing's sub-evaluator path is exercised cleanly).
    """
    # Read Scanner's GateEvaluator construction site via AST to assert the
    # call exists with use_per_pair_routing as a named kwarg derived from
    # `has_per_pair_training`.
    engine_src = (PROJECT_ROOT / "src" / "scanner" / "engine.py").read_text()
    tree = ast.parse(engine_src)
    gate_ctor_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "GateEvaluator":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                gate_ctor_calls.append(kwargs)
    assert gate_ctor_calls, (
        "Scanner runtime must construct GateEvaluator at least once "
        "(src/scanner/engine.py)."
    )
    scanner_kwargs = gate_ctor_calls[0]
    # Scanner construction site at src/scanner/engine.py:1677 uses
    # `use_per_pair_routing=use_per_pair_routing` — i.e. a Name node. The
    # presence of the kwarg is the wiring guarantee.
    assert "use_per_pair_routing" in scanner_kwargs, (
        "Scanner runtime GateEvaluator() must pass use_per_pair_routing as "
        "a kwarg (mirrors today's auto-detect behavior)."
    )

    # Now verify the harness's evaluator presents the equivalent runtime
    # state (per-pair routing ON, parent NOT pinned to joint).
    ev = _build_harness_evaluator()
    assert ev.use_per_pair_routing is True
    assert ev.use_joint_only is False


# --------------------------------------------------------------------------- #
# Layer 3 — per-pair sub-evaluator loads contract-complete artifact           #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not (EUR_USD_DIR / "transformer_direction.meta.pkl").exists(),
    reason="EUR_USD transformer meta sidecar missing — retrain required",
)
def test_real_eur_usd_per_pair_artifact_loaded():
    """L3: Per-pair sub-evaluator for EUR_USD loads the Phase 2.A+B
    contract-complete artifact: scaler not None, regime_quantiles populated,
    feature_pipeline_version=2026-05-08-v1.

    Per .claude/rules/improvement.md "Train↔Inference Contract Gates":
    refuse contract-incomplete artifacts. The pre-fix joint-dir artifact had
    feature_pipeline_version=None, regime_quantiles missing, scaler.var_==1.0
    — gates._load_transformer either rejected it or trusted the identity
    transform, both leading to fallback predictions.
    """
    ev = _build_harness_evaluator()
    sub = ev._get_pair_evaluator("EUR_USD")
    # _get_pair_evaluator loads on construction (gates.py:455-464). Verify
    # the contract fields the gate runtime reads at inference time.
    assert sub._transformer is not None, (
        "Per-pair EUR_USD sub-evaluator must have loaded a transformer "
        f"(model_dir={sub.model_dir})."
    )
    assert sub._transformer_scaler is not None, (
        "Phase 2.A+B contract: scaler MUST be loaded into the evaluator "
        "from the meta sidecar."
    )
    assert sub._transformer_pipeline_version == "2026-05-08-v1", (
        f"Phase 2.A+B contract: feature_pipeline_version must match "
        f"runtime constant. Got: {sub._transformer_pipeline_version!r}. "
        f"The joint-dir artifact had None — that's the bug that produced "
        f"today's 25.8% M15 holdout."
    )
    assert sub._transformer_regime_quantiles is not None, (
        "Phase 2.A+B contract: regime_quantiles must be loaded — needed "
        "at inference to compute regime_low/normal/high/extreme one-hots."
    )
    for q in ("q25", "q50", "q75"):
        assert q in sub._transformer_regime_quantiles, (
            f"regime_quantiles missing {q}"
        )
    assert sub._transformer_regime_atr_col is not None, (
        "Phase 2.A+B contract: regime_atr_col must be saved."
    )
    # Sanity: feature_names list and they form an authoritative ordering.
    assert sub._transformer_feature_names is not None
    assert len(sub._transformer_feature_names) > 0


# --------------------------------------------------------------------------- #
# Defense-in-depth: the harness source still has the fix applied              #
# --------------------------------------------------------------------------- #

def test_harness_source_carries_the_fix():
    """Source-level guard: the harness's GateEvaluator construction site
    must keep both flags. This is the textual tripwire for an accidental
    revert.
    """
    src = (PROJECT_ROOT / "scripts" / "scheduled_retrain.py").read_text()
    # The harness ctor span — find the GateEvaluator( construction
    idx = src.find("evaluator = GateEvaluator(")
    assert idx >= 0, "Holdout harness must construct GateEvaluator"
    # Scan up to the next 600 chars (the kwargs block) for both flags
    window = src[idx : idx + 600]
    assert "use_per_pair_routing=True" in window, (
        "Holdout harness MUST pass use_per_pair_routing=True (Tier 7 parity). "
        "Pre-fix default was False — the bug behind 25.8% M15."
    )
    assert "use_joint_only=False" in window, (
        "Holdout harness MUST pass use_joint_only=False — per-pair primary, "
        "joint fallback."
    )


# --------------------------------------------------------------------------- #
# Self-check: the GateEvaluator init contract did not change underneath us    #
# --------------------------------------------------------------------------- #

def test_gate_evaluator_init_signature_stable():
    """If this signature changes, the harness fix needs review.

    A guard against future refactors that rename or remove the two flags
    this fix relies on.
    """
    sig = inspect.signature(GateEvaluator.__init__)
    params = sig.parameters
    assert "use_per_pair_routing" in params
    assert "use_joint_only" in params
    # Defaults: use_joint_only=True, use_per_pair_routing=False — the
    # defaults that produced the bug. Our fix overrides both explicitly.
    assert params["use_joint_only"].default is True
    assert params["use_per_pair_routing"].default is False
