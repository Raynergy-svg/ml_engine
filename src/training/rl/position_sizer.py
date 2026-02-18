"""RL-based position sizing using PPO with lazy optional dependencies."""

from __future__ import annotations

import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from src.training.rl import runtime
from src.training.rl.environments.trading_env import POSITION_LEVELS, TradingEnv

if TYPE_CHECKING:
    from src.training.rl.config import RLPositionSizingConfig

logger = logging.getLogger(__name__)

_import_start = time.perf_counter()

# Model save paths.
RL_MODEL_PATH = Path("trained_data/models/rl_position_sizer.zip")
RL_SCALER_PATH = Path("trained_data/models/rl_scaler.pkl")


def _ensure_gym_imported():
    """Compatibility wrapper for legacy imports."""
    ok = runtime._ensure_gym_imported()
    global GYM_AVAILABLE
    GYM_AVAILABLE = runtime.GYM_AVAILABLE
    return ok


def _ensure_sb3_imported():
    """Compatibility wrapper for legacy imports."""
    ok = runtime._ensure_sb3_imported()
    global SB3_AVAILABLE
    SB3_AVAILABLE = runtime.SB3_AVAILABLE
    return ok


# Backward-compatible dependency flags.
GYM_AVAILABLE = runtime.GYM_AVAILABLE
SB3_AVAILABLE = runtime.SB3_AVAILABLE


@dataclass
class RLConfig:
    """Configuration for RL position sizing."""

    # Environment.
    sequence_length: int = 60
    max_position_pct: float = 0.10
    min_position_pct: float = 0.01

    # Reward shaping.
    sharpe_weight: float = 0.1
    drawdown_penalty: float = 0.5
    win_rate_bonus: float = 0.2

    # Training (optimized for CPU training).
    total_timesteps: int = 50_000  # Reduced from 100k - excessive data reuse
    learning_rate: float = 3e-4
    n_steps: int = 256  # Reduced from 2048 for faster first rollout on CPU
    batch_size: int = 32  # Reduced from 64 for lower memory usage
    n_epochs: int = 5  # Reduced from 10 for faster training cycles
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Risk limits.
    max_drawdown_pct: float = 0.10
    daily_loss_limit_pct: float = 0.03

    @classmethod
    def from_position_sizing_config(
        cls,
        cfg: "RLPositionSizingConfig",
    ) -> "RLConfig":
        """Map RLIntegrationConfig.position_sizing schema to runtime PPO config."""
        return cls(
            sequence_length=int(cfg.sequence_length),
            max_position_pct=float(cfg.max_position_pct),
            min_position_pct=float(cfg.min_position_pct),
            sharpe_weight=float(cfg.sharpe_weight),
            drawdown_penalty=float(cfg.drawdown_penalty),
            win_rate_bonus=float(cfg.win_rate_bonus),
            total_timesteps=int(cfg.timesteps),
            learning_rate=float(cfg.learning_rate),
            n_steps=int(cfg.n_steps),
            batch_size=int(cfg.batch_size),
            n_epochs=int(cfg.n_epochs),
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
            max_drawdown_pct=float(cfg.max_drawdown_pct),
            daily_loss_limit_pct=float(cfg.daily_loss_limit_pct),
        )


class RLProgressCallback:
    """Callback wrapper that prints periodic PPO training progress."""

    def __new__(cls, *args, **kwargs):  # noqa: D401
        if runtime.BaseCallback is None:
            return None
        return super().__new__(cls)

    def __init__(
        self,
        total_timesteps: int,
        print_freq: int = 50_000,
        console_print=None,
    ):
        if runtime.BaseCallback is None:
            return
        self._total = total_timesteps
        self._print_freq = print_freq
        self._console_print = console_print or print
        self._last_printed = 0
        self._start_time = None
        self._first_rollout_done = False
        self._inner = self._make_inner()

    def _make_inner(self):
        outer = self

        class _Inner(runtime.BaseCallback):
            def __init__(self_inner):
                super().__init__(verbose=0)

            def _on_training_start(self_inner):
                import sys as _sys
                import time as _t

                outer._start_time = _t.perf_counter()
                outer._console_print(
                    f"        ▶ Collecting first rollout..."
                )
                _sys.stdout.flush()

            def _on_step(self_inner) -> bool:
                import sys as _sys
                import time as _t

                step = self_inner.num_timesteps
                elapsed = _t.perf_counter() - outer._start_time if outer._start_time else 0

                # Print once after first rollout iteration starts
                if not outer._first_rollout_done and step > 0:
                    outer._first_rollout_done = True
                    outer._console_print(
                        f"        ✓ Training started ({elapsed:.1f}s to first step)"
                    )
                    outer._last_printed = step
                    _sys.stdout.flush()
                    return True

                # Regular progress updates
                if step - outer._last_printed >= outer._print_freq:
                    outer._last_printed = step
                    pct = 100 * step / outer._total
                    rate = step / elapsed if elapsed > 0 else 0
                    eta = (outer._total - step) / rate if rate > 0 else 0
                    outer._console_print(
                        f"        ⏳ {step:>6,}/{outer._total:,} ({pct:4.1f}%) "
                        f"| {rate:,.0f} sps | ETA {eta:.0f}s"
                    )
                    _sys.stdout.flush()
                return True

            def _on_training_end(self_inner):
                import sys as _sys
                import time as _t

                elapsed = _t.perf_counter() - outer._start_time if outer._start_time else 0
                outer._console_print(f"        ✓ PPO training finished in {elapsed:.1f}s")
                _sys.stdout.flush()

        return _Inner()

    def unwrap(self):
        return self._inner


class RLPositionSizer:
    """PPO-based position size optimizer."""

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.model = None
        self.scaler = None
        self._is_trained = False
        self._feature_dim_warning_emitted = False

    def _infer_expected_feature_dim(self) -> Optional[int]:
        """Infer expected raw feature dimension for inference."""
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            try:
                return int(self.scaler.n_features_in_)
            except Exception:  # noqa: BLE001
                pass

        # TradingEnv observation is: features + 2 predictions + 4 account stats.
        if self.model is not None and hasattr(self.model, "observation_space"):
            obs_shape = getattr(self.model.observation_space, "shape", None)
            if obs_shape and len(obs_shape) == 1:
                obs_dim = int(obs_shape[0])
                inferred = obs_dim - 6
                if inferred > 0:
                    return inferred

        return None

    def _align_feature_vector(self, features: np.ndarray) -> np.ndarray:
        """Align feature vector to expected dimension via truncation/padding."""
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        expected_dim = self._infer_expected_feature_dim()
        if expected_dim is None or vec.shape[0] == expected_dim:
            return vec

        got_dim = vec.shape[0]
        if got_dim > expected_dim:
            aligned = vec[:expected_dim]
            action = "truncated"
        else:
            aligned = np.pad(vec, (0, expected_dim - got_dim), mode="constant")
            action = "padded"

        # Avoid noisy logs on repeated inference calls.
        if not self._feature_dim_warning_emitted:
            logger.warning(
                "RL sizer feature mismatch: got %d, expected %d; %s input for compatibility.",
                got_dim,
                expected_dim,
                action,
            )
            self._feature_dim_warning_emitted = True

        return aligned

    @staticmethod
    def _normalize_ensemble_prediction(ensemble_prediction: np.ndarray) -> np.ndarray:
        """Normalize ensemble prediction vector to exactly 2 values."""
        pred = np.asarray(ensemble_prediction, dtype=np.float32).reshape(-1)
        if pred.shape[0] >= 2:
            return pred[:2]
        if pred.shape[0] == 1:
            return np.array([pred[0], 0.5], dtype=np.float32)
        return np.array([0.5, 0.5], dtype=np.float32)

    def train(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        *,
        eval_freq: int = 10000,
        verbose: int = 0,
        callback: Any = None,
        use_subprocess: bool = True,  # Use subprocess to avoid TF/PyTorch conflict
    ) -> Dict[str, Any]:
        """Train the PPO position sizing policy.

        Args:
            use_subprocess: If True, train in isolated subprocess to avoid
                           TensorFlow/PyTorch Metal conflicts on macOS.
        """
        if use_subprocess:
            return self._train_subprocess(features, ensemble_predictions, prices)

        return self._train_inprocess(
            features, ensemble_predictions, prices,
            eval_freq=eval_freq, verbose=verbose, callback=callback
        )

    def _train_subprocess(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """Train PPO in isolated subprocess (avoids TF/PyTorch conflict)."""
        import json
        import subprocess
        import tempfile

        print("  [RL] Using subprocess training (isolates PyTorch from TensorFlow)", flush=True)

        # Create temp directory for data exchange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_path = tmpdir / "train_data.pkl"
            output_dir = Path("trained_data/models")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save training data
            print("  [1/3] Preparing training data...", flush=True)
            with open(data_path, "wb") as f:
                pickle.dump({
                    "features": features,
                    "ensemble_predictions": ensemble_predictions,
                    "prices": prices,
                }, f)
            print(f"        ✓ Saved {len(features):,} samples to temp file", flush=True)

            # Config as JSON
            config_dict = {
                "sequence_length": self.config.sequence_length,
                "total_timesteps": self.config.total_timesteps,
                "learning_rate": self.config.learning_rate,
                "n_steps": self.config.n_steps,
                "batch_size": self.config.batch_size,
                "n_epochs": self.config.n_epochs,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "drawdown_penalty": self.config.drawdown_penalty,
            }

            # Run subprocess
            print("  [2/3] Launching subprocess trainer...", flush=True)
            script_path = Path(__file__).parent / "subprocess_trainer.py"

            cmd = [
                sys.executable,
                str(script_path),
                "--data", str(data_path),
                "--output", str(output_dir),
                "--config", json.dumps(config_dict),
            ]

            t0 = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=False,  # Stream output to console
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Subprocess training failed with code {result.returncode}")

            print(f"  [3/3] ✓ Subprocess training complete ({time.perf_counter() - t0:.1f}s)", flush=True)

            # Load stats
            stats_path = output_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
            else:
                stats = {"total_timesteps": self.config.total_timesteps}

            # Load model and scaler
            self.load()
            self._is_trained = True

            return stats

    def _train_inprocess(
        self,
        features: np.ndarray,
        ensemble_predictions: np.ndarray,
        prices: np.ndarray,
        *,
        eval_freq: int = 10000,
        verbose: int = 0,
        callback: Any = None,
    ) -> Dict[str, Any]:
        """Train PPO in-process (may hang on macOS with TensorFlow Metal)."""
        _ensure_sb3_imported()
        if not runtime.SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required for RL training. "
                "Install with: pip install stable-baselines3"
            )

        # Pre-flight check: ensure PyTorch CPU mode works (prevents TF/Metal hang)
        print("  [0/4] PyTorch pre-flight check...", flush=True)
        try:
            import torch
            # Force CPU and test basic tensor ops
            torch.set_default_device("cpu")
            _test = torch.zeros(1, device="cpu")
            del _test
            print(f"        ✓ PyTorch {torch.__version__} OK (device=cpu)", flush=True)
        except Exception as e:
            raise RuntimeError(f"PyTorch CPU initialization failed: {e}") from e

        from sklearn.preprocessing import StandardScaler

        t0 = time.perf_counter()

        # Step 1: Fit StandardScaler
        print("  [1/4] Scaling features...", flush=True)
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        print(f"  [1/4] ✓ Features scaled ({features.shape[1]} dims, {time.perf_counter() - t0:.1f}s)", flush=True)

        # Step 2: Create TradingEnv
        print("  [2/4] Creating trading environment...", flush=True)
        env = TradingEnv(
            features=features_scaled,
            ensemble_predictions=ensemble_predictions,
            prices=prices,
            config=self.config,
        )

        # Step 3: Wrap in DummyVecEnv
        vec_env = runtime.DummyVecEnv([lambda: env])
        print(f"  [2/4] ✓ Environment ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Step 4: Create PPO model
        print("  [3/4] Building PPO policy network...", flush=True)
        print(f"        Calling PPO() constructor with device='cpu'...", flush=True)
        sys.stdout.flush()

        ppo_start = time.perf_counter()

        self.model = runtime.PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            verbose=1,
            device="cpu",
        )
        print(f"  [3/4] ✓ PPO model ready (n_steps={self.config.n_steps}, {time.perf_counter() - ppo_start:.1f}s)", flush=True)
        sys.stdout.flush()

        # Step 5: Create eval environment
        print("        Creating eval environment...", flush=True)
        eval_env = runtime.DummyVecEnv(
            [
                lambda: TradingEnv(
                    features=features_scaled,
                    ensemble_predictions=ensemble_predictions,
                    prices=prices,
                    config=self.config,
                )
            ]
        )
        print("        ✓ Eval environment ready", flush=True)

        # Step 6: Setup callbacks
        print("        Setting up callbacks...", flush=True)
        stop_callback = runtime.StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=5,
            verbose=0,
        )

        # Increase eval_freq for faster training (evaluate less often)
        effective_eval_freq = max(eval_freq, self.config.total_timesteps // 5)

        eval_callback = runtime.EvalCallback(
            eval_env,
            best_model_save_path=str(RL_MODEL_PATH.parent),
            log_path=str(Path("trained_data/logs/rl")),
            eval_freq=effective_eval_freq,
            callback_after_eval=stop_callback,
            deterministic=True,
            render=False,
            verbose=0,
        )

        callbacks = [eval_callback]
        if callback is not None:
            callbacks.append(callback)

        # Progress callback with frequent updates
        progress_cb = RLProgressCallback(
            total_timesteps=self.config.total_timesteps,
            print_freq=max(2_500, self.config.total_timesteps // 20),  # More frequent updates
        )
        if progress_cb is not None:
            callbacks.append(progress_cb.unwrap())
        print(f"        ✓ Callbacks ready ({len(callbacks)} total)", flush=True)
        sys.stdout.flush()

        # Step 7: Begin training
        print(f"  [4/4] Starting PPO training ({self.config.total_timesteps:,} timesteps)...", flush=True)
        print(f"        First rollout ({self.config.n_steps} steps) may take 10-30s on CPU", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()

        learn_start = time.perf_counter()
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            progress_bar=False,  # Disable tqdm, use callback-based progress
        )
        print(f"  [4/4] ✓ Training complete ({time.perf_counter() - learn_start:.1f}s)", flush=True)

        self._is_trained = True
        self.save()

        stats = {
            "total_timesteps": self.config.total_timesteps,
            "final_equity": env.equity,
            "max_drawdown": (
                (env.max_equity - env.equity) / env.max_equity
                if env.max_equity > 0
                else 0
            ),
            "total_trades": len(env.trade_history),
            "win_rate": sum(1 for t in env.trade_history if t.get("pnl", 0) > 0)
            / max(len(env.trade_history), 1),
        }
        return stats

    def get_position_size(
        self,
        features: np.ndarray,
        ensemble_prediction: np.ndarray,
        account_equity: float = 10000.0,
    ) -> float:
        """Return position size in dollars."""
        if not self._is_trained or self.model is None:
            logger.warning("RL model not trained, using fallback 1%% position size")
            return account_equity * 0.01

        features_aligned = self._align_feature_vector(features)
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features_aligned.reshape(1, -1)).flatten()
            except Exception as e:  # noqa: BLE001
                logger.warning("RL scaler transform failed, using unscaled features: %s", e)
                features_scaled = features_aligned
        else:
            features_scaled = features_aligned

        ensemble_pred = self._normalize_ensemble_prediction(ensemble_prediction)

        obs = np.concatenate(
            [
                features_scaled,
                ensemble_pred,
                [0.0, 0.0, 0.0, 0.5],  # Neutral account placeholder.
            ]
        ).astype(np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        position_pct = POSITION_LEVELS[int(action)]
        return account_equity * position_pct

    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Save model and scaler."""
        model_path = model_path or RL_MODEL_PATH
        scaler_path = scaler_path or RL_SCALER_PATH

        model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(str(model_path))
            logger.info("RL model saved to %s", model_path)

        if self.scaler is not None:
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("RL scaler saved to %s", scaler_path)

    def load(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None) -> bool:
        """Load model and scaler."""
        model_path = model_path or RL_MODEL_PATH
        scaler_path = scaler_path or RL_SCALER_PATH

        _ensure_sb3_imported()
        if not runtime.SB3_AVAILABLE:
            logger.debug("stable-baselines3 not installed, RL position sizing disabled")
            return False

        try:
            if model_path.exists():
                import sys

                if "tensorflow" in sys.modules:
                    self._load_via_subprocess(model_path)
                else:
                    self.model = runtime.PPO.load(str(model_path), device="cpu")
                    self._is_trained = True
                    logger.info("RL model loaded from %s", model_path)

            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("RL scaler loaded from %s", scaler_path)

            return self._is_trained

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load RL model: %s", e)
            return False

    def _load_via_subprocess(self, model_path: Path) -> None:
        """Load PPO model in a child process to avoid TF/PyTorch deadlocks."""
        import subprocess
        import sys
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            script = (
                "import sys\n"
                "try:\n"
                "    import cloudpickle\n"
                "    from stable_baselines3 import PPO\n"
                f'    model = PPO.load("{model_path}", device="cpu")\n'
                f'    with open("{tmp_path}", "wb") as f:\n'
                "        cloudpickle.dump(model, f)\n"
                "    sys.exit(0)\n"
                "except Exception as e:\n"
                '    print(f"RL subprocess load failed: {e}", file=sys.stderr)\n'
                "    sys.exit(1)\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path.cwd()),
            )

            if result.returncode == 0 and Path(tmp_path).exists():
                import cloudpickle

                with open(tmp_path, "rb") as f:
                    self.model = cloudpickle.load(f)
                self._is_trained = True
                logger.info("RL model loaded from %s (via subprocess)", model_path)
            else:
                err = result.stderr.strip() if result.stderr else "unknown error"
                logger.warning("RL subprocess load failed: %s", err)
        except subprocess.TimeoutExpired:
            logger.warning("RL subprocess load timed out (30s)")
        except Exception as e:  # noqa: BLE001
            logger.warning("RL subprocess load failed: %s", e)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Whether RL sizing dependencies are available."""
        _ensure_sb3_imported()
        _ensure_gym_imported()
        return bool(runtime.SB3_AVAILABLE and runtime.GYM_AVAILABLE)


def train_rl_position_sizer(
    features: np.ndarray,
    ensemble_predictions: np.ndarray,
    prices: np.ndarray,
    config: Optional[RLConfig] = None,
    verbose: int = 0,
) -> RLPositionSizer:
    """Convenience function to train and return a sizer."""
    sizer = RLPositionSizer(config)
    sizer.train(features, ensemble_predictions, prices, verbose=verbose)
    return sizer


def get_position_sizer() -> RLPositionSizer:
    """Load a position sizer from disk (if available)."""
    sizer = RLPositionSizer()
    sizer.load()
    return sizer


_import_time = time.perf_counter() - _import_start
if _import_time > 5.0:
    logger.info("position_sizer module imported in %.1fs", _import_time)
