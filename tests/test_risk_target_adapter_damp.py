"""Risk-target forward-vol adapter + sizing damp (audit gaps G1+G2).

NO MOCKS — real ScannerConfig, real DynamicPositionSizer, real
RiskTargetTrainer training a real LightGBM artifact onto tmp_path disk,
and a real EvidenceStore with a real Ed25519-signed champion pointer.

Covers the mission's required behaviors:
- elevated forward vol -> size strictly decreases, bounded by the floor;
- normal forward vol -> size unchanged;
- adapter missing/refusing -> multiplier exactly 1.0 (fail-neutral);
- multiplier can never exceed 1.0 (property-style sweep);
- pre-fix incumbent (meta missing learnability fields) -> refused;
- the drawdown head is structurally unreachable from the adapter;
- champion pointer resolution with artifact-hash verification.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.position_sizing import (
    REGIME_HIGH,
    create_regime_aware_position_sizer,
)
from src.risk.risk_target_adapter import (
    MIN_DAMP_FLOOR,
    RiskTargetAdapter,
    damp_multiplier_from_quantiles,
)
from src.scanner.config import ScannerConfig
from src.training.labels import compute_forward_realized_volatility
from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    RISK_TARGET_FEATURE_PIPELINE_VERSION,
    compute_risk_target_features,
)
from src.training.trainers.risk_target_trainer import RiskTargetTrainer

PAIR = "EUR_USD"
_ANNUALIZATION = float(np.sqrt(252.0))


# ---------------------------------------------------------------------------
# Real-data scaffolding
# ---------------------------------------------------------------------------

def _make_daily_df(n: int = 420, seed: int = 7, end: datetime | None = None) -> pd.DataFrame:
    """Synthetic but realistic daily OHLCV random walk ending 'end' (UTC)."""
    rng = np.random.default_rng(seed)
    end = end or datetime.now(timezone.utc)
    dates = pd.bdate_range(end=pd.Timestamp(end).normalize(), periods=n, tz="UTC")
    # Volatility varies over time so the model has a learnable signal.
    vol = 0.004 + 0.004 * (1.0 + np.sin(np.linspace(0.0, 9.0, n))) / 2.0
    rets = rng.normal(0.0, vol)
    close = 1.10 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0.0, vol)) + vol
    high = close * (1.0 + spread)
    low = close * (1.0 - spread)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(50_000, 150_000, size=n)
    return pd.DataFrame(
        {
            "date": dates.astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _train_real_trainer(df: pd.DataFrame) -> RiskTargetTrainer:
    """Train a real (tiny) dual-head LightGBM RiskTargetTrainer on df."""
    feat = compute_risk_target_features(df.copy())
    y_vol, _ = compute_forward_realized_volatility(
        feat, 20, annualization_factor=_ANNUALIZATION
    )
    valid = feat[FEATURE_COLUMNS].notna().all(axis=1).values & np.isfinite(y_vol)
    rows = feat.loc[valid].copy()
    y_vol = y_vol[valid]
    rows["pair"] = pd.Categorical([PAIR] * len(rows))
    rows["day_of_week"] = rows["day_of_week"].astype("category")
    feature_names = list(FEATURE_COLUMNS) + ["pair"]
    X = rows[feature_names]
    y_dd = (y_vol > np.median(y_vol)).astype(np.int64)

    split = int(len(X) * 0.7)
    trainer = RiskTargetTrainer()
    trainer.train(
        X.iloc[:split], y_vol[:split], X.iloc[split:], y_vol[split:],
        feature_names=feature_names,
        instrument="pooled_fx_pairs",
        y_train_drawdown=y_dd[:split],
        y_val_drawdown=y_dd[split:],
        n_estimators=40,
        early_stopping_rounds=10,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        random_state=42,
    )
    return trainer


@pytest.fixture(scope="module")
def daily_df() -> pd.DataFrame:
    return _make_daily_df()


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, daily_df: pd.DataFrame):
    """One real trained artifact shared by the module (train once = fast)."""
    base_dir = tmp_path_factory.mktemp("risk_target_models")
    trainer = _train_real_trainer(daily_df)
    base = base_dir / "risk_target_model"
    trainer.save(str(base))
    return {"trainer": trainer, "base": base}


def _copy_artifact(trained: dict, dest_dir: Path) -> Path:
    """Fresh copy so each test can edit meta without cross-test bleed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = dest_dir / "risk_target_model"
    src = trained["base"]
    base.with_suffix(".pkl").write_bytes(src.with_suffix(".pkl").read_bytes())
    base.with_suffix(".meta.json").write_text(src.with_suffix(".meta.json").read_text())
    return base


def _edit_meta(base: Path, mutate) -> None:
    meta_path = base.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    mutate(meta)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _set_meta_quantiles(base: Path, q75: float, q90: float) -> None:
    _edit_meta(base, lambda m: m.__setitem__(
        "forward_vol_reference_quantiles", {"q75": q75, "q90": q90}
    ))


def _adapter(base: Path, tmp_path: Path, **kwargs) -> RiskTargetAdapter:
    kwargs.setdefault("evidence_root", tmp_path / "no_evidence")
    kwargs.setdefault("signing_dir", tmp_path / "no_signing")
    kwargs.setdefault("daily_data_dir", tmp_path / "no_data")
    return RiskTargetAdapter(model_base_path=base, **kwargs)


# ---------------------------------------------------------------------------
# Adapter: prediction contract
# ---------------------------------------------------------------------------

class TestAdapterPrediction:
    def test_incumbent_with_learnability_fields_predicts_float(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path)
        value = adapter.predict_forward_vol(PAIR, daily_df)
        assert isinstance(value, float)
        assert 0.0 < value < 5.0  # annualized FX vol, within the trainer's clip bounds

    def test_prefix_incumbent_missing_learnability_fields_is_refused(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)

        def strip(meta):
            meta["metrics"].pop("drawdown_learnable", None)
            meta["metrics"].pop("val_brier_drawdown_baseline", None)

        _edit_meta(base, strip)
        adapter = _adapter(base, tmp_path)
        assert adapter.predict_forward_vol(PAIR, daily_df) is None
        assert "predates_learnability_gate" in adapter.last_refusal_reason

    def test_feature_pipeline_version_mismatch_is_refused(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        _edit_meta(base, lambda m: m.__setitem__(
            "risk_target_feature_pipeline_version", "1999-01-01-v0"
        ))
        adapter = _adapter(base, tmp_path)
        assert adapter.predict_forward_vol(PAIR, daily_df) is None
        assert "version_mismatch" in adapter.last_refusal_reason

    def test_missing_artifact_is_refused(self, tmp_path, daily_df):
        adapter = _adapter(tmp_path / "nothing_here", tmp_path)
        assert adapter.predict_forward_vol(PAIR, daily_df) is None
        assert "no_incumbent_artifact" in adapter.last_refusal_reason

    def test_insufficient_history_refused_never_zero_filled(self, trained, tmp_path):
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path)
        short_df = _make_daily_df(n=40)  # < 60-bar warm-up -> NaN features
        assert adapter.predict_forward_vol(PAIR, short_df) is None
        assert "no_zero_fill" in adapter.last_refusal_reason

    def test_stale_history_is_refused(self, trained, tmp_path):
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path)
        stale_df = _make_daily_df(end=datetime.now(timezone.utc) - timedelta(days=60))
        assert adapter.predict_forward_vol(PAIR, stale_df) is None
        assert "daily_history_stale" in adapter.last_refusal_reason

    def test_unseen_pair_is_refused(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path)
        assert adapter.predict_forward_vol("GBP_JPY", daily_df) is None
        assert "pair_not_in_training_categories" in adapter.last_refusal_reason

    def test_empirical_reference_quantiles_from_history(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)  # meta has no quantiles
        adapter = _adapter(base, tmp_path)
        quantiles = adapter.get_reference_quantiles(PAIR, daily_df)
        assert quantiles is not None
        assert 0.0 < quantiles["q75"] < quantiles["q90"]


# ---------------------------------------------------------------------------
# Adapter: champion pointer path (real signed EvidenceStore, no mocks)
# ---------------------------------------------------------------------------

def _build_champion_store(tmp_path: Path, vol_model) -> tuple:
    from src.evidence.contracts import (
        ArtifactRef,
        AuthorityRole,
        ChampionPointer,
        DispositionEvent,
        DispositionState,
        EvidencePackage,
        ImportCheck,
        SafetyAssertion,
    )
    from src.evidence.hashing import sha256_bytes
    from src.evidence.importer import build_import_verdict
    from src.evidence.signing import Ed25519Signer, TrustStore
    from src.evidence.store import EvidenceStore
    from src.evidence.transition_policy import AuthorityRegistry

    now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
    signers = {role: Ed25519Signer.generate() for role in (
        AuthorityRole.PRODUCER, AuthorityRole.LOCAL_IMPORTER,
        AuthorityRole.INDEPENDENT_VERIFIER, AuthorityRole.OPERATOR,
        AuthorityRole.PROMOTION_SERVICE,
    )}
    actors = {
        AuthorityRole.PRODUCER: "worker-1",
        AuthorityRole.LOCAL_IMPORTER: "local-importer",
        AuthorityRole.INDEPENDENT_VERIFIER: "independent-verifier",
        AuthorityRole.OPERATOR: "operator-primary",
        AuthorityRole.PROMOTION_SERVICE: "local-promotion-service",
    }
    trust = TrustStore()
    authorities = AuthorityRegistry()
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=now - timedelta(days=1)))
        authorities.register(actor_id=actors[role], role=role, key_ids=(signer.key_id,))

    artifact_bytes = pickle.dumps(vol_model, protocol=pickle.DEFAULT_PROTOCOL)
    artifact = ArtifactRef(
        artifact_id="forward_volatility-model",
        relative_path="artifacts/model.pkl",
        digest=sha256_bytes(artifact_bytes),
        size_bytes=len(artifact_bytes),
        media_type="application/octet-stream",
    )
    package = EvidencePackage(
        package_id="risk-target-package-v1",
        lane_id="risk_target",
        producer_id=actors[AuthorityRole.PRODUCER],
        created_at=now,
        job_manifest_digest="1" * 64,
        dataset_manifest_digests=("2" * 64,),
        evaluation_report_digests=("3" * 64,),
        artifacts=(artifact,),
        safety_assertions=(SafetyAssertion(assertion_id="remote-no-authority", passed=True),),
        checksums={artifact.relative_path: artifact.digest},
    )
    store = EvidenceStore(
        tmp_path / "evidence", trust_store=trust, authorities=authorities,
        trusted_clock=lambda: now + timedelta(minutes=1),
    )
    envelope = signers[AuthorityRole.PRODUCER].sign(package, created_at=now)
    package_digest = store.write_package(envelope, {artifact.relative_path: artifact_bytes})

    verdict = build_import_verdict(
        verdict_id="accepted-risk-target-package-v1",
        package_digest=package_digest,
        importer_id=actors[AuthorityRole.LOCAL_IMPORTER],
        created_at=now,
        checks=(ImportCheck(check_id="local-policy", passed=True),),
    )
    verdict_digest = store.write_verdict(
        signers[AuthorityRole.LOCAL_IMPORTER].sign(verdict, created_at=now)
    )

    steps = [
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, AuthorityRole.INDEPENDENT_VERIFIER),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.QUARANTINED, DispositionState.SHADOW, AuthorityRole.OPERATOR),
        (DispositionState.SHADOW, DispositionState.OPERATOR_APPROVED, AuthorityRole.OPERATOR),
        (DispositionState.OPERATOR_APPROVED, DispositionState.CHAMPION, AuthorityRole.PROMOTION_SERVICE),
    ]
    head = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        metadata: dict = {}
        if to_state == DispositionState.QUARANTINED:
            metadata = {"verdict_digest": verdict_digest}
        if to_state == DispositionState.CHAMPION:
            metadata = {
                "artifact_digest": artifact.digest,
                "artifact_id": artifact.artifact_id,
                "lane_id": package.lane_id,
            }
        signer = signers[role]
        event = DispositionEvent(
            package_digest=package_digest,
            sequence=sequence,
            previous_event_digest=head,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=actors[role],
            signer_key_id=signer.key_id,
            occurred_at=now + timedelta(seconds=sequence),
            reason=f"advance to {to_state.value}",
            metadata=metadata,
        )
        state = store.append_disposition(
            signer.sign(event, created_at=event.occurred_at),
            expected_head_digest=head,
        )
        head = state.head_event_digest

    promotion = signers[AuthorityRole.PROMOTION_SERVICE]
    pointer = ChampionPointer(
        pointer_id="risk-target-champion-v1",
        lane_id=package.lane_id,
        package_digest=package_digest,
        artifact_id=artifact.artifact_id,
        artifact_digest=artifact.digest,
        disposition_head_digest=head,
        actor_id=actors[AuthorityRole.PROMOTION_SERVICE],
        signer_key_id=promotion.key_id,
        created_at=now + timedelta(minutes=1),
    )
    store.write_champion_pointer(
        promotion.sign(pointer, created_at=pointer.created_at),
        expected_pointer_digest=None,
    )
    return store, artifact.digest


class TestChampionPointerPath:
    def test_champion_pointer_resolves_and_predicts(self, trained, tmp_path, daily_df):
        store, _digest = _build_champion_store(tmp_path, trained["trainer"].vol_model)
        adapter = _adapter(tmp_path / "no_incumbent", tmp_path, store=store)
        value = adapter.predict_forward_vol(PAIR, daily_df)
        assert isinstance(value, float) and value > 0.0
        assert adapter._loaded is not None
        assert adapter._loaded.source.startswith("champion:risk_target:")

    def test_champion_takes_priority_over_incumbent(self, trained, tmp_path, daily_df):
        store, _ = _build_champion_store(tmp_path, trained["trainer"].vol_model)
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path, store=store)
        assert adapter.predict_forward_vol(PAIR, daily_df) is not None
        assert adapter._loaded.source.startswith("champion:")

    def test_no_champion_pointer_falls_back_to_incumbent(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        adapter = _adapter(base, tmp_path)  # evidence dirs don't exist
        assert adapter.predict_forward_vol(PAIR, daily_df) is not None
        assert adapter._loaded.source.startswith("incumbent:")

    def test_drawdown_lanes_are_never_eligible(self, trained, tmp_path):
        adapter = _adapter(tmp_path / "x", tmp_path,
                           champion_lanes=("risk_target_drawdown", "drawdown_state"))
        assert adapter.champion_lanes == ()


# ---------------------------------------------------------------------------
# Damp math: bounded, monotone, never above 1.0 (property-style sweep)
# ---------------------------------------------------------------------------

class TestDampMultiplierBounds:
    def test_sweep_never_exceeds_one_and_never_undershoots_floor(self):
        preds = [0.0, 1e-6, 1e-4, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0,
                 float("nan"), float("inf"), -0.3]
        q75s = [-1.0, 0.0, 1e-6, 0.03, 0.08, 0.5, 5.0, float("nan")]
        q90s = [-2.0, 0.0, 2e-6, 0.05, 0.10, 0.4, 10.0, float("inf")]
        floors = [-1.0, 0.0, 0.1, 0.25, 0.3, 0.5, 0.9, 1.0, 2.0]
        for pred in preds:
            for q75 in q75s:
                for q90 in q90s:
                    for floor in floors:
                        m = damp_multiplier_from_quantiles(pred, q75, q90, floor)
                        assert m <= 1.0, (pred, q75, q90, floor, m)
                        assert m >= MIN_DAMP_FLOOR, (pred, q75, q90, floor, m)

    def test_monotone_non_increasing_in_predicted_vol(self):
        prev = 1.0
        for pred in np.linspace(0.01, 0.5, 200):
            m = damp_multiplier_from_quantiles(float(pred), 0.08, 0.15, 0.4)
            assert m <= prev + 1e-12
            prev = m

    def test_at_or_below_q75_is_exactly_one(self):
        assert damp_multiplier_from_quantiles(0.05, 0.08, 0.15, 0.4) == 1.0
        assert damp_multiplier_from_quantiles(0.08, 0.08, 0.15, 0.4) == 1.0

    def test_far_above_q90_is_exactly_floor(self):
        assert damp_multiplier_from_quantiles(1.0, 0.08, 0.15, 0.4) == 0.4

    def test_degenerate_quantiles_are_neutral(self):
        assert damp_multiplier_from_quantiles(0.2, 0.15, 0.15, 0.4) == 1.0
        assert damp_multiplier_from_quantiles(0.2, 0.15, 0.10, 0.4) == 1.0

    def test_floor_below_hard_minimum_is_clamped_up(self):
        m = damp_multiplier_from_quantiles(1.0, 0.08, 0.15, 0.05)
        assert m == MIN_DAMP_FLOOR


# ---------------------------------------------------------------------------
# Sizing integration: real DynamicPositionSizer + real ScannerConfig
# ---------------------------------------------------------------------------

def _sized_units(adapter, scanner_config) -> tuple:
    sizer = create_regime_aware_position_sizer()
    sizer.risk_target_adapter = adapter
    sizer.scanner_config = scanner_config
    result = sizer.calculate_regime_scaled_position_size(
        account_equity=100_000,
        stop_loss_pips=20,
        instrument=PAIR,
        volatility_regime=REGIME_HIGH,
        meta_confidence=0.9,
        raw_confidence=0.9,
    )
    return result


class TestSizingDampIntegration:
    def _write_history_csv(self, adapter_data_dir: Path, df: pd.DataFrame) -> None:
        adapter_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(adapter_data_dir / f"{PAIR}_D.csv", index=False)

    def test_elevated_vol_strictly_decreases_size_bounded_by_floor(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        # Reference quantiles far below any plausible prediction -> full damp.
        _set_meta_quantiles(base, q75=1e-6, q90=2e-6)
        data_dir = tmp_path / "factor"
        self._write_history_csv(data_dir, daily_df)
        adapter = _adapter(base, tmp_path, daily_data_dir=data_dir)
        cfg = ScannerConfig()

        damped = _sized_units(adapter, cfg)
        neutral = _sized_units(
            _adapter(tmp_path / "missing", tmp_path), cfg
        )
        assert neutral.risk_target_damp_applied == 1.0
        assert damped.risk_target_damp_applied == cfg.risk_target_damp_floor == 0.5
        assert damped.units < neutral.units  # strictly decreases
        assert damped.units >= int(cfg.risk_target_damp_floor * neutral.units) - 1

    def test_normal_vol_leaves_size_unchanged(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        # Reference quantiles far above any plausible prediction -> neutral.
        _set_meta_quantiles(base, q75=5.0, q90=6.0)
        data_dir = tmp_path / "factor"
        self._write_history_csv(data_dir, daily_df)
        adapter = _adapter(base, tmp_path, daily_data_dir=data_dir)
        cfg = ScannerConfig()

        result = _sized_units(adapter, cfg)
        neutral = _sized_units(_adapter(tmp_path / "missing", tmp_path), cfg)
        assert result.risk_target_damp_applied == 1.0
        assert result.units == neutral.units

    def test_adapter_missing_or_refusing_is_exactly_neutral(self, tmp_path):
        adapter = _adapter(tmp_path / "definitely_missing", tmp_path)
        cfg = ScannerConfig()
        result = _sized_units(adapter, cfg)
        assert result.is_valid
        assert result.risk_target_damp_applied == 1.0
        assert "adapter_refused" in result.risk_target_damp_reason

    def test_disabled_flag_short_circuits_damp(self, trained, tmp_path, daily_df):
        base = _copy_artifact(trained, tmp_path)
        _set_meta_quantiles(base, q75=1e-6, q90=2e-6)  # would fully damp
        data_dir = tmp_path / "factor"
        self._write_history_csv(data_dir, daily_df)
        adapter = _adapter(base, tmp_path, daily_data_dir=data_dir)
        cfg = ScannerConfig(enable_risk_target_damp=False)
        result = _sized_units(adapter, cfg)
        assert result.risk_target_damp_applied == 1.0
        assert result.risk_target_damp_reason == "disabled_by_config"

    def test_damp_multiplier_never_exceeds_one_through_sizer(self, trained, tmp_path, daily_df):
        """Even with adversarial meta quantiles the applied multiplier is
        bounded to [0.25, 1.0] and units never exceed the neutral size."""
        cfg = ScannerConfig()
        neutral = _sized_units(_adapter(tmp_path / "missing", tmp_path), cfg)
        for q75, q90 in [(1e-6, 2e-6), (-5.0, -1.0), (0.0, 1e-9), (5.0, 6.0)]:
            case_dir = tmp_path / f"case_{abs(hash((q75, q90)))}"
            base = _copy_artifact(trained, case_dir)
            _set_meta_quantiles(base, q75=q75, q90=q90)
            data_dir = case_dir / "factor"
            self._write_history_csv(data_dir, daily_df)
            adapter = _adapter(base, case_dir, daily_data_dir=data_dir)
            result = _sized_units(adapter, cfg)
            assert 0.25 <= result.risk_target_damp_applied <= 1.0
            assert result.units <= neutral.units


# ---------------------------------------------------------------------------
# The drawdown head is structurally unreachable (static + API surface)
# ---------------------------------------------------------------------------

class TestNoDrawdownPath:
    ADAPTER_SOURCE = Path("src/risk/risk_target_adapter.py")

    def test_adapter_public_api_exposes_no_drawdown_surface(self, tmp_path):
        adapter = _adapter(tmp_path / "x", tmp_path)
        public = [n for n in dir(adapter) if not n.startswith("_")]
        assert not [n for n in public if "drawdown" in n.lower()]

    @staticmethod
    def _code_identifiers(path: Path) -> set:
        """All attribute/name identifiers referenced by ACTUAL CODE in the
        module (AST walk) — comments and docstrings don't count, so the
        guard can't be tripped by prose or dodged by clever formatting."""
        import ast

        tree = ast.parse(path.read_text())
        idents: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                idents.add(node.attr)
            elif isinstance(node, ast.Name):
                idents.add(node.id)
        return idents

    def test_adapter_code_never_invokes_the_drawdown_model(self):
        idents = self._code_identifiers(self.ADAPTER_SOURCE)
        # No classifier inference surface of any kind, and no touch of the
        # failed head. RiskTargetTrainer.predict (which runs BOTH heads) is
        # never referenced; only the vol-only _predict_vol transform is.
        forbidden = {
            "predict",
            "predict_proba",
            "drawdown_model",
            "predicted_drawdown_stressed_prob",
        }
        assert not (idents & forbidden), idents & forbidden
        assert "_predict_vol" in idents

    def test_position_sizing_code_has_no_drawdown_reference(self):
        idents = self._code_identifiers(Path("src/risk/position_sizing.py"))
        forbidden = {
            "predict_proba",
            "drawdown_model",
            "predicted_drawdown_stressed_prob",
        }
        assert not (idents & forbidden), idents & forbidden


# ---------------------------------------------------------------------------
# REAL on-disk state (read-only): today's incumbent MUST be refused, and the
# real evidence store has no champion yet, so the whole chain returns None.
# ---------------------------------------------------------------------------

REAL_INCUMBENT_BASE = Path("trained_data/risk_targets/models/risk_target_model")
REAL_EVIDENCE_ROOT = Path("trained_data/evidence")
REAL_SIGNING_DIR = Path("trained_data/axiom/signing")


class TestRealDiskState:
    """READ-ONLY assertions against the repo's real artifacts. These pin the
    mission's known facts: the Path-A incumbent predates the drawdown
    learnability gate (its meta lacks ``drawdown_learnable`` +
    ``val_brier_drawdown_baseline``) and no champion has been promoted
    (the only evidence package is quarantined), so the adapter must refuse
    end-to-end and sizing must stay exactly neutral."""

    def _real_meta(self) -> dict:
        meta_path = REAL_INCUMBENT_BASE.with_suffix(".meta.json")
        if not meta_path.exists():
            pytest.skip("real incumbent artifact not present in this checkout")
        return json.loads(meta_path.read_text())

    def test_real_incumbent_meta_lacks_learnability_fields(self):
        meta = self._real_meta()
        metrics = meta.get("metrics") or {}
        for field in ("drawdown_learnable", "val_brier_drawdown_baseline"):
            assert field not in metrics and field not in meta, (
                f"real incumbent now declares {field!r} — it was retrained "
                "post-fix; update the adapter fallback expectations"
            )
        # The refusal must be due to the missing learnability fields ONLY —
        # the artifact's feature-pipeline version matches the runtime's.
        assert (
            meta.get("risk_target_feature_pipeline_version")
            == RISK_TARGET_FEATURE_PIPELINE_VERSION
        )

    def test_real_incumbent_is_refused_by_adapter(self, daily_df):
        self._real_meta()  # skip guard
        adapter = RiskTargetAdapter(
            model_base_path=REAL_INCUMBENT_BASE,
            evidence_root=Path("nonexistent_evidence"),
            signing_dir=Path("nonexistent_signing"),
            daily_data_dir=Path("nonexistent_data"),
        )
        assert adapter.predict_forward_vol(PAIR, daily_df) is None
        assert "predates_learnability_gate" in adapter.last_refusal_reason

    def test_real_evidence_layout_has_no_champion_so_chain_refuses(self, daily_df):
        self._real_meta()  # skip guard
        if any((REAL_EVIDENCE_ROOT / "champions").glob("*.json")):
            pytest.skip(
                "a champion pointer now exists — the fallback-only assertion "
                "no longer describes the real state"
            )
        # Default-path adapter over the REAL store layout: champion pointer
        # resolution finds nothing, falls back to the real incumbent, which
        # is refused -> None. This is the CURRENT production behavior.
        adapter = RiskTargetAdapter()
        assert adapter.predict_forward_vol(PAIR, daily_df) is None
        assert "predates_learnability_gate" in adapter.last_refusal_reason
        # And sizing over that refusing chain is exactly fail-neutral.
        multiplier, reason = adapter.compute_damp_multiplier(PAIR, floor=0.5)
        assert multiplier == 1.0
        assert "adapter_refused" in reason


# ---------------------------------------------------------------------------
# Config contract: fields exist, profiles carry no orphan keys
# ---------------------------------------------------------------------------

class TestConfigContract:
    def test_dataclass_fields_exist_with_safe_defaults(self):
        cfg = ScannerConfig()
        assert cfg.enable_risk_target_damp is True
        assert 0.25 <= cfg.risk_target_damp_floor <= 1.0

    def test_profile_keys_match_dataclass_field_names(self):
        import dataclasses
        from src.scanner.config import SCAN_PROFILES

        names = {f.name for f in dataclasses.fields(ScannerConfig)}
        for profile in ("balanced", "conservative", "aggressive", "smart"):
            overrides = SCAN_PROFILES[profile]
            assert overrides["enable_risk_target_damp"] is True
            assert overrides["risk_target_damp_floor"] == 0.5
            for key in ("enable_risk_target_damp", "risk_target_damp_floor"):
                assert key in names, f"orphan profile key {key} in {profile}"

    def test_out_of_range_floor_is_clamped_at_load(self):
        assert ScannerConfig(risk_target_damp_floor=0.05).risk_target_damp_floor == 0.5
        assert ScannerConfig(risk_target_damp_floor=1.5).risk_target_damp_floor == 0.5
