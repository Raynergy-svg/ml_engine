"""RiskTargetAdapter — production consumption seam for the risk-target
forward-volatility head (closes audit gaps G1+G2, 2026-07-30).

The 2026-07-30 Phase I/J audit (docs/architecture/audit/phase_I_J1_J7_models.md
§2) found both trained risk-target heads had ZERO production consumers. This
adapter makes the forward-vol head consumable by sizing code as a
RISK-DECREASING damp input only.

Contract (all fail-closed, per .claude/rules/improvement.md
"Train↔Inference Contract Gates"):

- Model resolution order:
    1. Evidence champion pointer (``EvidenceStore.load_champion_pointer``)
       from ``trained_data/evidence/`` — the artifact hash is verified
       against the pointer before any bytes are unpickled.
    2. Fallback to the ungoverned Path-A incumbent
       ``trained_data/risk_targets/models/risk_target_model.pkl`` ONLY if
       its meta.json declares the drawdown-learnability fields
       (``drawdown_learnable`` + ``val_brier_drawdown_baseline``). Their
       absence marks the known pre-fix artifact (audit §2.2) — refused.
- Features are computed by importing the SAME functions training uses
  (``src.training.risk_target_features.compute_risk_target_features``,
  ``FEATURE_COLUMNS``, ``RISK_TARGET_FEATURE_PIPELINE_VERSION``) — never
  reimplemented. Any missing/NaN required feature → refuse (return None),
  NEVER zero-fill.
- THE DRAWDOWN HEAD IS NEVER EXPOSED OR USED. It failed its pre-registered
  bar (OOS Brier 0.232 vs 0.143 base-rate baseline — see
  docs/prereg-risk-target-vol-drawdown-2026-07-08.md). This module never
  calls ``RiskTargetTrainer.predict`` (which would run both heads) and
  never touches ``drawdown_model`` / ``predict_proba`` — only the
  vol-model-only ``RiskTargetTrainer._predict_vol`` transform. Enforced by
  a static test in tests/test_risk_target_adapter_damp.py.
- The damp multiplier is bounded to [floor, 1.0] with floor >= 0.25 —
  strictly risk-decreasing; it can never scale a position UP.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Training-side contract — imported, never reimplemented (improvement.md).
from src.training.risk_target_features import (
    FEATURE_COLUMNS,
    RISK_TARGET_FEATURE_PIPELINE_VERSION,
    compute_risk_target_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_BASE_PATH = Path("trained_data/risk_targets/models/risk_target_model")
DEFAULT_EVIDENCE_ROOT = Path("trained_data/evidence")
DEFAULT_SIGNING_DIR = Path("trained_data/axiom/signing")
DEFAULT_DAILY_DATA_DIR = Path("market_data/factor")

# Champion lanes to try, in order. "risk_target" is the pointer lane the
# wiring mission names; "risk_target_vol" is the lane_id the evidence
# evaluator stamps on the vol head (src/evidence/risk_target/evaluation.py).
# A lane containing "drawdown" is NEVER eligible (hard guard below).
DEFAULT_CHAMPION_LANES: Tuple[str, ...] = ("risk_target", "risk_target_vol")

# Absolute lower bound on the damp floor. A configured floor below this is
# clamped UP — mirrors the headless-learning "no risk multipliers above
# 1.0" rule on the other side of the interval.
MIN_DAMP_FLOOR = 0.25

# The two learnability fields whose PRESENCE distinguishes a post-fix
# artifact from the known-bad pre-fix incumbent (audit §2.2).
_LEARNABILITY_FIELDS = ("drawdown_learnable", "val_brier_drawdown_baseline")

# Meta key (top-level or under "metrics") that a future gated retrain may
# use to persist the training-history forward-vol distribution.
_META_QUANTILES_KEY = "forward_vol_reference_quantiles"

# Minimum number of valid trailing rows required to compute an honest
# empirical reference distribution from the model's own predictions.
_MIN_REFERENCE_ROWS = 250

_REQUIRED_OHLCV_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def damp_multiplier_from_quantiles(
    predicted_vol: float,
    q75: float,
    q90: float,
    floor: float,
) -> float:
    """Pure damp curve: 1.0 at/below q75, linear down to ``floor`` at q90+.

    Always returns a value in [max(MIN_DAMP_FLOOR, floor_clamped), 1.0].
    Degenerate inputs (non-finite, non-positive prediction, q90 <= q75)
    return exactly 1.0 (fail-neutral, never fail-up).
    """
    try:
        floor = float(floor)
        predicted_vol = float(predicted_vol)
        q75 = float(q75)
        q90 = float(q90)
    except (TypeError, ValueError):
        return 1.0
    if not all(math.isfinite(v) for v in (predicted_vol, q75, q90, floor)):
        return 1.0
    floor = min(1.0, max(MIN_DAMP_FLOOR, floor))
    if predicted_vol <= 0.0 or q90 <= q75:
        return 1.0
    if predicted_vol <= q75:
        return 1.0
    fraction = min(1.0, (predicted_vol - q75) / (q90 - q75))
    multiplier = 1.0 - fraction * (1.0 - floor)
    # Belt-and-braces: the damp can NEVER exceed 1.0 and never undershoot floor.
    return max(floor, min(1.0, multiplier))


class _LoadedVolModel:
    """Per-process cache entry: the vol head + its inference contract."""

    def __init__(
        self,
        vol_model: Any,
        feature_names: list,
        source: str,
        trained_at: Optional[datetime],
        meta_quantiles: Optional[Dict[str, float]],
    ) -> None:
        self.vol_model = vol_model
        self.feature_names = list(feature_names)
        self.source = source
        self.trained_at = trained_at
        self.meta_quantiles = meta_quantiles


class RiskTargetAdapter:
    """Loads the risk-target forward-vol head and serves bounded,
    fail-closed forward-volatility estimates for sizing damping.

    Never raises out of its public methods — every failure path returns
    ``None`` (predictions) or ``(1.0, reason)`` (damp), with one WARNING
    per distinct reason per process.
    """

    def __init__(
        self,
        *,
        model_base_path: Path = DEFAULT_MODEL_BASE_PATH,
        evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
        signing_dir: Path = DEFAULT_SIGNING_DIR,
        daily_data_dir: Path = DEFAULT_DAILY_DATA_DIR,
        store: Any = None,
        champion_lanes: Tuple[str, ...] = DEFAULT_CHAMPION_LANES,
        max_artifact_age_days: float = 45.0,
        max_history_age_days: float = 10.0,
    ) -> None:
        self.model_base_path = Path(model_base_path)
        self.evidence_root = Path(evidence_root)
        self.signing_dir = Path(signing_dir)
        self.daily_data_dir = Path(daily_data_dir)
        self._injected_store = store
        # HARD GUARD: the drawdown head must never be consumable through
        # this adapter — a lane/artifact whose id mentions "drawdown" is
        # ineligible no matter how the adapter is configured.
        self.champion_lanes = tuple(
            lane for lane in champion_lanes if "drawdown" not in lane.lower()
        )
        self.max_artifact_age_days = float(max_artifact_age_days)
        self.max_history_age_days = float(max_history_age_days)

        self._loaded: Optional[_LoadedVolModel] = None
        self._load_attempted = False
        self._reference_quantiles_cache: Dict[str, Optional[Dict[str, float]]] = {}
        self._warned_reasons: set = set()
        self.last_refusal_reason: Optional[str] = None

    # -- logging ----------------------------------------------------------
    def _refuse(self, reason: str) -> None:
        """Record a refusal; WARN once per distinct reason per process."""
        self.last_refusal_reason = reason
        if reason not in self._warned_reasons:
            self._warned_reasons.add(reason)
            logger.warning("RiskTargetAdapter refused: %s", reason)
        else:
            logger.debug("RiskTargetAdapter refused (repeat): %s", reason)

    # -- model resolution -------------------------------------------------
    def _get_model(self) -> Optional[_LoadedVolModel]:
        if self._loaded is not None:
            return self._loaded
        if self._load_attempted:
            return None  # per-process cache of the failed load too
        self._load_attempted = True
        loaded = self._load_from_champion()
        if loaded is None:
            loaded = self._load_incumbent()
        if loaded is not None:
            self._check_artifact_staleness(loaded)
            logger.info(
                "RiskTargetAdapter loaded forward-vol head from %s (trained_at=%s)",
                loaded.source, loaded.trained_at,
            )
        self._loaded = loaded
        return loaded

    def _build_store(self) -> Any:
        """Best-effort read-only EvidenceStore over trained_data/evidence.

        Returns None when the store or persisted signing identities are
        absent/unusable — never raises, never creates key material.
        """
        if self._injected_store is not None:
            return self._injected_store
        identities_path = self.signing_dir / "identities.json"
        if not (self.evidence_root / "champions").is_dir() or not identities_path.exists():
            return None
        try:
            from src.evidence.signing import Ed25519Signer, TrustStore
            from src.evidence.store import EvidenceStore
            from src.evidence.transition_policy import AuthorityRegistry
            from src.evidence.contracts import AuthorityRole

            metadata = json.loads(identities_path.read_text())
            trust = TrustStore()
            authorities = AuthorityRegistry()
            for role_value, entry in (metadata.get("roles") or {}).items():
                try:
                    role = AuthorityRole(role_value)
                except ValueError:
                    continue  # forward-compat: skip roles this build doesn't know
                key_path = self.signing_dir / f"{role_value}.key"
                if not key_path.exists():
                    continue
                signer = Ed25519Signer.from_private_bytes(key_path.read_bytes())
                if signer.key_id != entry.get("key_id"):
                    logger.warning(
                        "RiskTargetAdapter: key_id mismatch for role %s — skipping",
                        role_value,
                    )
                    continue
                valid_from = datetime.fromisoformat(entry["valid_from"])
                trust.add(signer.trusted_key(valid_from=valid_from))
                authorities.register(
                    actor_id=entry["actor_id"], role=role, key_ids=(signer.key_id,),
                )
            return EvidenceStore(
                self.evidence_root, trust_store=trust, authorities=authorities,
            )
        except Exception as exc:  # noqa: BLE001 — any store failure = no champion path
            logger.debug("RiskTargetAdapter: evidence store unavailable: %s", exc)
            return None

    def _load_from_champion(self) -> Optional[_LoadedVolModel]:
        store = self._build_store()
        if store is None:
            return None
        for lane_id in self.champion_lanes:
            try:
                pointer = store.load_champion_pointer(lane_id=lane_id)
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001 — corrupt store = fall back
                logger.warning(
                    "RiskTargetAdapter: champion pointer load failed for lane %s: %s",
                    lane_id, exc,
                )
                continue
            # NEVER consume a drawdown artifact, whatever the pointer says.
            if "drawdown" in (pointer.artifact_id or "").lower():
                self._refuse(f"champion_artifact_is_drawdown_head:{pointer.artifact_id}")
                continue
            try:
                package = store.load_package(pointer.package_digest)
                refs = {a.artifact_id: a for a in package.artifacts}
                ref = refs.get(pointer.artifact_id)
                if ref is None or ref.digest != pointer.artifact_digest:
                    self._refuse(
                        f"champion_artifact_hash_mismatch:lane={lane_id}"
                    )
                    continue
                data = store.load_package_file(
                    pointer.package_digest, ref.relative_path
                )
                from src.evidence.hashing import sha256_bytes

                if sha256_bytes(data) != pointer.artifact_digest:
                    self._refuse(f"champion_bytes_digest_mismatch:lane={lane_id}")
                    continue
                # Unpickle ONLY after signature + digest verification
                # (security note, src/evidence/risk_target/evaluation.py).
                vol_model = pickle.loads(data)
            except Exception as exc:  # noqa: BLE001 — verified-load failure = fall back
                logger.warning(
                    "RiskTargetAdapter: champion artifact load failed (lane %s): %s",
                    lane_id, exc,
                )
                continue
            trained_at = getattr(package, "created_at", None)
            return _LoadedVolModel(
                vol_model=vol_model,
                # The evidence evaluator's fixed contract:
                # FEATURE_COLUMNS + ["pair"] (evaluation.py assemble_frame).
                feature_names=list(FEATURE_COLUMNS) + ["pair"],
                source=f"champion:{lane_id}:{pointer.package_digest[:12]}",
                trained_at=trained_at,
                meta_quantiles=None,
            )
        return None

    def _load_incumbent(self) -> Optional[_LoadedVolModel]:
        base = self.model_base_path
        meta_path = base.with_suffix(".meta.json")
        pkl_path = base.with_suffix(".pkl")
        if not meta_path.exists() or not pkl_path.exists():
            self._refuse(f"no_incumbent_artifact_at:{base}")
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            self._refuse(f"incumbent_meta_unreadable:{exc}")
            return None

        metrics = meta.get("metrics") or {}
        missing = [f for f in _LEARNABILITY_FIELDS if f not in metrics and f not in meta]
        if missing:
            # Pre-fix artifact (audit §2.2): produced before the drawdown
            # learnability gate existed. Its provenance is unaudited — refuse.
            self._refuse(
                "incumbent_predates_learnability_gate:missing="
                + ",".join(missing)
            )
            return None

        artifact_version = meta.get("risk_target_feature_pipeline_version")
        if artifact_version != RISK_TARGET_FEATURE_PIPELINE_VERSION:
            self._refuse(
                f"feature_pipeline_version_mismatch:artifact={artifact_version}"
                f"!=runtime={RISK_TARGET_FEATURE_PIPELINE_VERSION}"
            )
            return None

        try:
            from src.training.trainers.risk_target_trainer import RiskTargetTrainer

            trainer = RiskTargetTrainer()
            trainer.load(str(base))
        except Exception as exc:  # noqa: BLE001 — any load failure is fail-closed
            self._refuse(f"incumbent_load_failed:{exc}")
            return None

        feature_names = trainer.feature_names or (list(FEATURE_COLUMNS) + ["pair"])
        trained_at = self._incumbent_trained_at(meta, pkl_path)
        return _LoadedVolModel(
            vol_model=trainer.vol_model,
            feature_names=feature_names,
            source=f"incumbent:{base}",
            trained_at=trained_at,
            meta_quantiles=self._quantiles_from_meta(meta),
        )

    @staticmethod
    def _incumbent_trained_at(meta: Dict[str, Any], pkl_path: Path) -> Optional[datetime]:
        stamp = meta.get("trained_at")
        if isinstance(stamp, str):
            try:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(pkl_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None

    @staticmethod
    def _quantiles_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, float]]:
        for container in (meta, meta.get("metrics") or {}):
            raw = container.get(_META_QUANTILES_KEY)
            if isinstance(raw, dict) and "q75" in raw and "q90" in raw:
                try:
                    q75, q90 = float(raw["q75"]), float(raw["q90"])
                except (TypeError, ValueError):
                    return None
                if math.isfinite(q75) and math.isfinite(q90) and q90 > q75:
                    return {"q75": q75, "q90": q90}
        return None

    def _check_artifact_staleness(self, loaded: _LoadedVolModel) -> None:
        if loaded.trained_at is None:
            return
        age_days = (datetime.now(timezone.utc) - loaded.trained_at).total_seconds() / 86400.0
        if age_days > self.max_artifact_age_days:
            # WARNING + still usable (bounded-age policy is advisory).
            logger.warning(
                "RiskTargetAdapter: artifact is stale (%.1f days > %.1f) — "
                "still usable, schedule a retrain (source=%s)",
                age_days, self.max_artifact_age_days, loaded.source,
            )

    # -- feature construction ---------------------------------------------
    def _load_daily_history(self, pair: str) -> Optional[pd.DataFrame]:
        path = self.daily_data_dir / f"{pair}_D.csv"
        if not path.exists():
            self._refuse(f"no_daily_history_csv:{path}")
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 — unreadable data = refuse
            self._refuse(f"daily_history_unreadable:{path}:{exc}")
            return None
        return df

    def _feature_row(
        self, pair: str, daily_history_df: Optional[pd.DataFrame], loaded: _LoadedVolModel
    ) -> Optional[pd.DataFrame]:
        df = daily_history_df if daily_history_df is not None else self._load_daily_history(pair)
        if df is None:
            return None
        missing_cols = _REQUIRED_OHLCV_COLUMNS - set(df.columns)
        if missing_cols:
            self._refuse(f"daily_history_missing_columns:{sorted(missing_cols)}")
            return None

        try:
            feat = compute_risk_target_features(
                df.sort_values("date").reset_index(drop=True)
            )
        except (ValueError, TypeError) as exc:
            self._refuse(f"feature_computation_failed:{exc}")
            return None

        last = feat.iloc[[-1]].copy()
        # NEVER zero-fill: any NaN required feature is a refusal
        # (.claude/rules/improvement.md hard rule).
        nan_cols = [c for c in FEATURE_COLUMNS if bool(last[c].isna().iloc[0])]
        if nan_cols:
            self._refuse(f"missing_features_refuse_no_zero_fill:{nan_cols}")
            return None

        # History freshness: a forward-vol estimate computed from stale bars
        # would be a fabricated "current" risk number — refuse.
        asof = pd.to_datetime(last["date"].iloc[0], utc=True, errors="coerce")
        if pd.isna(asof):
            self._refuse("unparseable_asof_date")
            return None
        age_days = (pd.Timestamp.now(tz="UTC") - asof).total_seconds() / 86400.0
        if age_days > self.max_history_age_days:
            self._refuse(
                f"daily_history_stale:{age_days:.1f}d>"
                f"{self.max_history_age_days:.1f}d:pair={pair}"
            )
            return None

        last["pair"] = pd.Categorical([pair])
        last["day_of_week"] = last["day_of_week"].astype("category")

        if not all(c in last.columns for c in loaded.feature_names):
            self._refuse("feature_columns_missing_vs_artifact_contract")
            return None
        if not self._pair_in_training_categories(pair, loaded):
            self._refuse(f"pair_not_in_training_categories:{pair}")
            return None
        return last[loaded.feature_names]

    @staticmethod
    def _pair_in_training_categories(pair: str, loaded: _LoadedVolModel) -> bool:
        """A pair the model never saw would be treated as a MISSING
        categorical by LightGBM — the zero-fill failure mode in disguise.
        Refuse when the booster's stored pandas categories prove the pair
        is unknown; pass when the contract can't be introspected."""
        try:
            booster = getattr(loaded.vol_model, "booster_", None)
            pandas_categorical = getattr(booster, "pandas_categorical", None)
            if not pandas_categorical:
                return True  # not trained with pandas categoricals — can't check
            string_lists = [
                cats for cats in pandas_categorical
                if cats and all(isinstance(c, str) for c in cats)
            ]
            if not string_lists:
                return True
            return any(pair in cats for cats in string_lists)
        except Exception:  # noqa: BLE001 — introspection must never block
            return True

    def _predict_vol_only(self, loaded: _LoadedVolModel, X: pd.DataFrame) -> np.ndarray:
        """Vol-head-only prediction, reusing the trainer's own exp/clip
        transform (``RiskTargetTrainer._predict_vol``). The drawdown model
        is NEVER attached to this shim, so no code path can invoke it."""
        from src.training.trainers.risk_target_trainer import RiskTargetTrainer

        shim = RiskTargetTrainer()
        shim.vol_model = loaded.vol_model
        # drawdown_model deliberately stays None — the failed head is
        # structurally unreachable from this adapter.
        return shim._predict_vol(X)

    # -- public API -------------------------------------------------------
    def predict_forward_vol(
        self, pair: str, daily_history_df: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """Annualized forward-vol estimate for ``pair``, or None (refusal).

        Never raises. ``daily_history_df`` (date/open/high/low/close/volume,
        ascending) overrides the default on-disk daily panel.
        """
        try:
            loaded = self._get_model()
            if loaded is None:
                return None
            X = self._feature_row(pair, daily_history_df, loaded)
            if X is None:
                return None
            value = float(self._predict_vol_only(loaded, X)[0])
            if not math.isfinite(value) or value <= 0.0:
                self._refuse(f"non_finite_prediction:{value}")
                return None
            self.last_refusal_reason = None
            return value
        except Exception as exc:  # noqa: BLE001 — adapter must never crash sizing
            self._refuse(f"unexpected_adapter_error:{type(exc).__name__}:{exc}")
            return None

    def get_reference_quantiles(
        self, pair: str, daily_history_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, float]]:
        """{"q75", "q90"} reference distribution for "elevated" detection.

        Order: (1) quantiles persisted in the artifact meta, if any;
        (2) empirical quantiles of the model's own predictions over the
        pair's full valid trailing history (honest, per-pair, trailing-only
        features — >= _MIN_REFERENCE_ROWS rows required); else (3) None
        (refuse — the damp stays neutral).
        """
        try:
            loaded = self._get_model()
            if loaded is None:
                return None
            if loaded.meta_quantiles is not None:
                return dict(loaded.meta_quantiles)
            cache_key = f"{loaded.source}:{pair}"
            if cache_key in self._reference_quantiles_cache:
                cached = self._reference_quantiles_cache[cache_key]
                return dict(cached) if cached else None

            df = daily_history_df if daily_history_df is not None else self._load_daily_history(pair)
            result: Optional[Dict[str, float]] = None
            if df is not None and not (_REQUIRED_OHLCV_COLUMNS - set(df.columns)):
                feat = compute_risk_target_features(
                    df.sort_values("date").reset_index(drop=True)
                )
                valid = feat[FEATURE_COLUMNS].notna().all(axis=1)
                rows = feat.loc[valid].copy()
                if len(rows) >= _MIN_REFERENCE_ROWS and self._pair_in_training_categories(pair, loaded):
                    rows["pair"] = pd.Categorical([pair] * len(rows))
                    rows["day_of_week"] = rows["day_of_week"].astype("category")
                    if all(c in rows.columns for c in loaded.feature_names):
                        preds = self._predict_vol_only(loaded, rows[loaded.feature_names])
                        q75, q90 = (float(q) for q in np.quantile(preds, [0.75, 0.90]))
                        if math.isfinite(q75) and math.isfinite(q90) and q90 > q75:
                            result = {"q75": q75, "q90": q90}
            if result is None:
                self._refuse(f"no_reference_distribution:pair={pair}")
            self._reference_quantiles_cache[cache_key] = result
            return dict(result) if result else None
        except Exception as exc:  # noqa: BLE001 — adapter must never crash sizing
            self._refuse(f"reference_quantiles_error:{type(exc).__name__}:{exc}")
            return None

    def compute_damp_multiplier(
        self,
        pair: str,
        floor: float,
        daily_history_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, str]:
        """Bounded risk-decreasing multiplier in [max(0.25, floor), 1.0].

        (1.0, reason) whenever the adapter refuses — fail-neutral, never
        fail-up, never raises.
        """
        try:
            predicted = self.predict_forward_vol(pair, daily_history_df)
            if predicted is None:
                return 1.0, f"adapter_refused:{self.last_refusal_reason}"
            quantiles = self.get_reference_quantiles(pair, daily_history_df)
            if quantiles is None:
                return 1.0, f"no_reference:{self.last_refusal_reason}"
            multiplier = damp_multiplier_from_quantiles(
                predicted, quantiles["q75"], quantiles["q90"], floor
            )
            reason = (
                f"forward_vol={predicted:.4f} vs q75={quantiles['q75']:.4f}/"
                f"q90={quantiles['q90']:.4f} -> x{multiplier:.3f}"
            )
            return multiplier, reason
        except Exception as exc:  # noqa: BLE001 — sizing must never crash on this
            return 1.0, f"damp_error:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Per-process singleton (production seam used by DynamicPositionSizer)
# ---------------------------------------------------------------------------
_ADAPTER_SINGLETON: Optional[RiskTargetAdapter] = None


def get_risk_target_adapter() -> RiskTargetAdapter:
    """Module-level cached adapter (model loads at most once per process)."""
    global _ADAPTER_SINGLETON
    if _ADAPTER_SINGLETON is None:
        _ADAPTER_SINGLETON = RiskTargetAdapter()
    return _ADAPTER_SINGLETON


def reset_risk_target_adapter() -> None:
    """Test/ops hook: drop the cached adapter so the next call reloads."""
    global _ADAPTER_SINGLETON
    _ADAPTER_SINGLETON = None


__all__ = [
    "RiskTargetAdapter",
    "damp_multiplier_from_quantiles",
    "get_risk_target_adapter",
    "reset_risk_target_adapter",
    "MIN_DAMP_FLOOR",
    "DEFAULT_CHAMPION_LANES",
]
