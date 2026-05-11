"""
Bug A regression tests for the MC-dropout unscaled-input bug (2026-05-11).

Background
----------
``ModularEnsembleInference._predict_with_uncertainty`` previously called
``self.tcn.model(features, training=True)`` directly on RAW, UNSCALED
feature matrices. The trained EUR_USD M15 transformer expects inputs
through the fitted ``StandardScaler`` saved in
``trained_data/models/EUR_USD/transformer_direction.meta.pkl``; feeding
raw features (e.g. ``stoch_k mean≈51``) ~91× the training-time
magnitude produces out-of-distribution predictions that depress
``prob_long``. Live feature replay on a real EUR_USD bar reproduced:

  - PATH A (live MC path, raw unscaled + regime zero-filled): prob_long=0.170
  - PATH C (correctly scaled per Phase 2.B contract):          prob_long=0.681

Same model, same bar, opposite direction. This is the load-bearing
cause of the 88% SHORT bias observed in the live brain feed.

Fix
---
``ModularEnsembleInference._apply_scaler_and_regime`` (new helper) and
the updated ``_predict_with_uncertainty(df=...)`` rebuild the
contract-compliant scaled feature matrix BEFORE the keras call —
mirroring ``src/scanner/gates.py::_build_transformer_inference_matrix``
+ the trainer's own ``predict()`` scaling step
(``src/training/trainers/transformer_trainer.py:3023``).

No-Mock Rule
------------
Per ``.claude/rules/improvement.md`` "No-Mock Rule", these tests use
real ``ModularEnsembleInference``, real ``StandardScaler``, real
``DataFrame`` inputs, and real disk artifacts where available. No
``unittest.mock``, no ``MagicMock``, no test doubles.

The integration test loads our own meta sidecar via ``pickle.load``
(trusted training output — same way ``transformer_trainer.py:3225``
saves it). This is intentional and required.
"""

from __future__ import annotations

import pickle  # noqa: S403 — reads our own training output, see module docstring
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Real TensorFlow / numpy stack required — skip module if either missing,
# don't mock around it.
tf = pytest.importorskip("tensorflow")
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.core.modular_inference import (  # noqa: E402
    InferenceConfig,
    ModularEnsembleInference,
)


# --- Test fixtures ------------------------------------------------------

# Realistic EUR_USD-style feature list (subset of the 50 the live model uses).
# The point is: includes both "regular" and "regime_*" cols so we exercise
# both branches of _apply_scaler_and_regime.
_FEATURE_NAMES = [
    "returns_2",
    "returns_5",
    "atr_pct_20",
    "volatility_20",
    "stoch_k",
    "macd_norm",
    "volume_zscore",
    "regime_low",
    "regime_normal",
    "regime_high",
    "regime_extreme",
]

# Quantiles that span our synthetic atr_pct_20 distribution below.
_REGIME_QUANTILES = {"q25": 0.0010, "q50": 0.0020, "q75": 0.0035}


def _eur_usd_scale_df(n_rows: int = 80, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataFrame at realistic EUR_USD M15 magnitudes.

    `stoch_k` deliberately on its raw 0–100 scale to reproduce the OOD
    condition the live bug produces (training-time stoch_k_norm is the
    z-scored version; raw stoch_k is what `_extract_tcn_features` zero-
    fills when names don't match).
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "returns_2":    rng.normal(0.0, 0.0008, n_rows),
        "returns_5":    rng.normal(0.0, 0.0015, n_rows),
        "atr_pct_20":   np.linspace(0.0005, 0.005, n_rows),  # spans regimes
        "volatility_20": rng.uniform(0.0001, 0.001, n_rows),
        "stoch_k":      rng.uniform(20.0, 80.0, n_rows),     # raw 0-100!
        "macd_norm":    rng.normal(0.0, 1.0, n_rows),
        "volume_zscore": rng.normal(0.0, 1.0, n_rows),
    })


def _fitted_scaler_on_typical_distribution() -> StandardScaler:
    """Real fitted scaler with realistic (non-identity) per-column stats.

    Simulates a scaler the trainer would have saved: mean/std reflect
    the training-time distribution of each column (no var_=1.0 across
    all features — that's the double-fit fingerprint the Phase 2 audit
    promoted as a tripwire).
    """
    rng = np.random.default_rng(42)
    # 5000-row "training" sample with the same column ordering used in tests.
    train = np.column_stack([
        rng.normal(0.0, 0.0008, 5000),     # returns_2
        rng.normal(0.0, 0.0015, 5000),     # returns_5
        rng.uniform(0.0005, 0.005, 5000),  # atr_pct_20
        rng.uniform(0.0001, 0.001, 5000),  # volatility_20
        rng.uniform(20.0, 80.0, 5000),     # stoch_k  (raw 0-100)
        rng.normal(0.0, 1.0, 5000),        # macd_norm
        rng.normal(0.0, 1.0, 5000),        # volume_zscore
        rng.choice([0.0, 1.0], 5000, p=[0.75, 0.25]),  # regime_low
        rng.choice([0.0, 1.0], 5000, p=[0.50, 0.50]),  # regime_normal
        rng.choice([0.0, 1.0], 5000, p=[0.75, 0.25]),  # regime_high
        rng.choice([0.0, 1.0], 5000, p=[0.90, 0.10]),  # regime_extreme
    ])
    sc = StandardScaler().fit(train)
    return sc


class _TinyKerasDirectionStub:
    """Tiny REAL keras model that mimics the trainer's interface.

    Not a mock — a genuine `tf.keras.Model` so the production call
    ``self.tcn.model(features, training=True)`` runs through real
    TF code paths. Trained to be sensitive to its input magnitude
    so we can demonstrate the unscaled-vs-scaled gap.
    """

    def __init__(self, seq_len: int, n_features: int):
        inputs = tf.keras.Input(shape=(seq_len, n_features))
        # Dense path to a single sigmoid — sigmoid saturates for extreme
        # inputs, which is exactly the failure mode the live bug exhibits.
        x = tf.keras.layers.Flatten()(inputs)
        # Force a fixed, non-zero kernel so the test is deterministic.
        x = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.Constant(0.01),
            bias_initializer=tf.keras.initializers.Constant(0.0),
        )(x)
        self.model = tf.keras.Model(inputs, x)
        self.input_shape = self.model.input_shape


class _StubTransformerDirectionTrainer:
    """Real container with the same attr surface a real trainer exposes.

    Holds the keras model + the saved-contract attributes
    (``scaler``, ``feature_names``, ``selected_indices``,
    ``_regime_quantiles``, ``_regime_atr_col``). Not a mock — every
    attribute is a real object the production code reads.
    """

    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.feature_names = list(_FEATURE_NAMES)
        self.scaler = _fitted_scaler_on_typical_distribution()
        self.selected_indices = list(range(len(self.feature_names)))
        self._regime_quantiles = dict(_REGIME_QUANTILES)
        self._regime_atr_col = "atr_pct_20"
        # Build the inner keras model after scaler so we can size it.
        wrapper = _TinyKerasDirectionStub(seq_len=seq_len, n_features=len(self.feature_names))
        self.model = wrapper.model


def _make_ensemble_with_stub_tcn(seq_len: int = 60) -> ModularEnsembleInference:
    """Construct a real ModularEnsembleInference and inject the stub trainer.

    The MC-dropout path reads `self.tcn`, `self.config.mc_dropout_samples`
    and `self._direction_threshold`. We populate those directly — no
    mocks; the methods under test are real and exercised end-to-end.
    """
    cfg = InferenceConfig()
    cfg.mc_dropout_samples = 5  # small for fast tests
    ens = ModularEnsembleInference(config=cfg)
    ens.tcn = _StubTransformerDirectionTrainer(seq_len=seq_len)
    ens._direction_threshold = 0.5
    return ens


# --- Tests --------------------------------------------------------------


def test_unscaled_input_produces_extreme_predictions():
    """REGRESSION WITNESS: feeding raw unscaled features matches the live bug.

    Documents WHY we need the fix: when the legacy MC path fed raw
    features (stoch_k on 20-80 scale, atr on ~0.001 scale, regimes
    silently zero-filled) into the keras model, the dense+sigmoid head
    saturated — exactly the live failure mode. We bypass the fixed
    helper and call the same keras op the old code did so the witness
    is independent of the helper's correctness.
    """
    ens = _make_ensemble_with_stub_tcn(seq_len=60)
    df = _eur_usd_scale_df(n_rows=80, seed=1)

    # Build the "old" unscaled matrix exactly as _extract_tcn_features
    # used to: feature_names order, regime cols zero-filled.
    cols = []
    for name in _FEATURE_NAMES:
        if name in df.columns:
            cols.append(df[name].values.astype(np.float32))
        else:
            # regime_* — what the old path silently did.
            cols.append(np.zeros(len(df), dtype=np.float32))
    raw = np.column_stack(cols)[-60:].reshape(1, 60, -1).astype(np.float32)

    # Run the same forward op the legacy MC loop ran.
    pred = ens.tcn.model(raw, training=False).numpy().flatten()[-1]
    # With dense kernel=0.01 and large raw magnitudes (stoch_k 20-80
    # summed over 60 timesteps × 11 features), the sigmoid is in the
    # saturated regime — far from 0.5. This is the bug.
    assert pred > 0.99 or pred < 0.01, (
        f"Expected saturated prediction from unscaled input, got {pred:.4f}. "
        f"If this test no longer reflects the bug, the keras stub or feature "
        f"magnitudes have changed."
    )


def test_scaled_input_produces_balanced_predictions():
    """The contract-aware path produces a prediction in the sensible range.

    With scaler.transform(), per-column inputs have mean≈0 std≈1, so the
    summed dense activation is much smaller in magnitude and the sigmoid
    stays in its responsive region.
    """
    ens = _make_ensemble_with_stub_tcn(seq_len=60)
    df = _eur_usd_scale_df(n_rows=80, seed=1)

    X_scaled = ens._apply_scaler_and_regime(
        df=df,
        feature_names=_FEATURE_NAMES,
        scaler=ens.tcn.scaler,
        regime_quantiles=_REGIME_QUANTILES,
        regime_atr_col="atr_pct_20",
        selected_indices=None,
    )
    assert X_scaled is not None
    assert X_scaled.shape == (80, len(_FEATURE_NAMES))

    seq_in = X_scaled[-60:].reshape(1, 60, -1).astype(np.float32)
    pred = float(ens.tcn.model(seq_in, training=False).numpy().flatten()[-1])
    assert 0.30 < pred < 0.70, (
        f"Expected balanced sigmoid output from scaled input, got {pred:.4f}. "
        f"If the scaler is identity (var_=1.0 across columns) this would mean "
        f"the Phase 2 double-scaler regression has resurfaced."
    )


def test_helper_function_matches_gates_path():
    """The helper produces the same scaled output as the gates.py path.

    Mirrors `gates._build_transformer_inference_matrix` + the implicit
    `scaler.transform()` that `evaluate_transformer` applies at
    `gates.py:1939`. We build the expected matrix by hand using the
    same algorithm and assert exact equality.
    """
    ens = _make_ensemble_with_stub_tcn(seq_len=60)
    df = _eur_usd_scale_df(n_rows=80, seed=2)

    X_helper = ens._apply_scaler_and_regime(
        df=df,
        feature_names=_FEATURE_NAMES,
        scaler=ens.tcn.scaler,
        regime_quantiles=_REGIME_QUANTILES,
        regime_atr_col="atr_pct_20",
        selected_indices=None,
    )
    assert X_helper is not None

    # === Replicate gates.py:1977-2082 logic manually ===
    regime_cols = ("regime_low", "regime_normal", "regime_high", "regime_extreme")
    regime_class_idx = {name: i for i, name in enumerate(regime_cols)}
    cols = []
    for name in _FEATURE_NAMES:
        if name in df.columns:
            cols.append(df[name].values.astype(np.float32))
        elif name in regime_class_idx:
            atr = df["atr_pct_20"].values.astype(np.float32)
            q = _REGIME_QUANTILES
            cls = np.where(
                atr <= q["q25"], 0,
                np.where(atr <= q["q50"], 1,
                         np.where(atr <= q["q75"], 2, 3)),
            )
            cls = np.where(np.isfinite(atr), cls, 1)
            target = regime_class_idx[name]
            cols.append((cls == target).astype(np.float32))
    X_manual_raw = np.column_stack(cols)
    X_manual_scaled = ens.tcn.scaler.transform(X_manual_raw).astype(np.float32)

    np.testing.assert_allclose(X_helper, X_manual_scaled, rtol=1e-6, atol=1e-6)


def test_missing_regime_quantiles_falls_back_safely():
    """When regime_quantiles=None but regime cols required, helper returns None.

    Documents the contract behavior: NO silent zero-fill. The caller
    (_predict_with_uncertainty) treats None as "contract incomplete,
    fall back to the trainer.predict() path" — so the live bot keeps
    making predictions, but not via the OOD MC path.
    """
    ens = _make_ensemble_with_stub_tcn(seq_len=60)
    df = _eur_usd_scale_df(n_rows=80, seed=3)

    # Regime cols are in feature_names but quantiles are missing.
    out = ens._apply_scaler_and_regime(
        df=df,
        feature_names=_FEATURE_NAMES,
        scaler=ens.tcn.scaler,
        regime_quantiles=None,  # the gap
        regime_atr_col="atr_pct_20",
        selected_indices=None,
    )
    assert out is None, (
        "Helper must refuse when regime_quantiles is None and feature_names "
        "contains regime cols — silent zero-fill is forbidden by the no-fill "
        "rule in .claude/rules/improvement.md 'Train↔Inference Contract Gates'."
    )

    # Also: helper must refuse with scaler=None (pre-Phase-2.A artifacts).
    out2 = ens._apply_scaler_and_regime(
        df=df,
        feature_names=_FEATURE_NAMES,
        scaler=None,
        regime_quantiles=_REGIME_QUANTILES,
        regime_atr_col="atr_pct_20",
        selected_indices=None,
    )
    assert out2 is None

    # And: feature_names with no regime cols + no scaler still refuses.
    out3 = ens._apply_scaler_and_regime(
        df=df,
        feature_names=["returns_2", "returns_5"],
        scaler=None,
        regime_quantiles=None,
        regime_atr_col=None,
        selected_indices=None,
    )
    assert out3 is None


def test_predict_with_uncertainty_uses_scaled_path_when_df_provided():
    """End-to-end: passing df makes _predict_with_uncertainty produce balanced predictions.

    This is the integration assertion: the consumer
    (_predict_with_uncertainty) actually invokes the new helper when df
    is provided AND the contract is loaded, so the live MC-dropout
    output is in the sensible range. Mirrors the runtime call shape at
    `modular_inference.py` (caller in predict_modular at line 4021).
    """
    ens = _make_ensemble_with_stub_tcn(seq_len=60)
    df = _eur_usd_scale_df(n_rows=80, seed=4)

    # The legacy raw features the caller still passes positionally.
    cols = []
    for name in _FEATURE_NAMES:
        if name in df.columns:
            cols.append(df[name].values.astype(np.float32))
        else:
            cols.append(np.zeros(len(df), dtype=np.float32))
    raw = np.column_stack(cols).astype(np.float32)

    mean_p, std_p, direction = ens._predict_with_uncertainty(
        features=raw, n_samples=5, df=df,
    )
    # With the contract-aware path, mean_p stays in the sensible band.
    assert 0.30 < mean_p < 0.70, (
        f"Expected MC-dropout mean in (0.30, 0.70) with contract-aware path; "
        f"got {mean_p:.4f}. If this fails, the helper wasn't called."
    )
    # std_p should be non-negative finite (dropout produces variance).
    assert std_p >= 0.0 and np.isfinite(std_p)
    # direction should be a valid label (0 or 1, not None — model returned).
    assert direction in (0, 1)


@pytest.mark.integration
def test_real_eur_usd_holdout_prediction_after_fix():
    """Real EUR_USD artifact on disk + a synthetic feature row.

    Loads the actual trained model + meta sidecar from
    ``trained_data/models/EUR_USD/``. If the current artifact is pre-
    Phase-2.A (scaler=None, regime_quantiles=None — as it is at the
    time of writing this fix, 2026-05-11), the helper MUST refuse —
    that's the correct behavior; the test asserts the refusal. After
    Team 1 restores a contract-complete model, this same test will
    exercise the post-fix path.

    Note: pickle.load on our OWN training output (trusted source).
    """
    model_dir = Path("trained_data/models/EUR_USD")
    meta_path = model_dir / "transformer_direction.meta.pkl"
    if not meta_path.exists():
        pytest.skip(f"{meta_path} not present; cannot run integration check")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)  # noqa: S301 — see module docstring

    scaler = meta.get("scaler")
    feature_names = meta.get("feature_names")
    regime_quantiles = meta.get("regime_quantiles")
    regime_atr_col = meta.get("regime_atr_col")
    selected_indices = meta.get("selected_indices")

    if not feature_names:
        pytest.skip("meta has no feature_names; cannot run integration check")

    # Build a synthetic 80-row df with the columns the model expects.
    # We do not need to load the keras model for this assertion — the
    # helper's output (or None refusal) is the contract under test.
    rng = np.random.default_rng(99)
    df_cols = {}
    for fname in feature_names:
        if fname.startswith("regime_"):
            continue  # produced by the helper from atr_pct_20
        df_cols[fname] = rng.normal(0.0, 1.0, 80).astype(np.float32)
    # atr_pct_20 must span quantiles for regime one-hots to be non-trivial.
    df_cols["atr_pct_20"] = np.linspace(0.0001, 0.01, 80, dtype=np.float32)
    df = pd.DataFrame(df_cols)

    # Build the ensemble + inject what the meta provides (without loading
    # the keras model — we're testing the helper, not the keras forward).
    ens = ModularEnsembleInference(config=InferenceConfig())

    class _Bare:
        pass

    bare = _Bare()
    bare.scaler = scaler
    bare.feature_names = feature_names
    bare.selected_indices = selected_indices
    bare._regime_quantiles = regime_quantiles
    bare._regime_atr_col = regime_atr_col
    bare.model = None  # helper consults model.input_shape only if not None
    ens.tcn = bare

    out = ens._apply_scaler_and_regime(
        df=df,
        feature_names=feature_names,
        scaler=scaler,
        regime_quantiles=regime_quantiles,
        regime_atr_col=regime_atr_col,
        selected_indices=selected_indices,
    )

    needs_regime = any(c.startswith("regime_") for c in feature_names)
    if scaler is None or (needs_regime and regime_quantiles is None):
        # Pre-Phase-2.A artifact: helper MUST refuse (no silent fill).
        assert out is None, (
            "Contract gap (scaler=None or regime_quantiles=None) must "
            "force refusal; helper returned a matrix instead. This is "
            "the failure mode the fix is meant to prevent."
        )
    else:
        # Post-restore artifact: helper produces a real scaled matrix.
        assert out is not None
        assert out.shape == (80, len(feature_names))
        # Scaled values: each column roughly mean≈0 std≈1 over the bar
        # range; not strict because our synthetic df is small.
        assert np.all(np.isfinite(out))
