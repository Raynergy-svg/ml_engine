"""Diffusion-based market simulation for stress testing and data augmentation.

US-009: Latent diffusion model trained on daily factor returns.
Generates synthetic paths given a seed and regime context.
Validates via KS test + autocorrelation matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional scipy for KS test
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    stats = None  # type: ignore[assignment]


@dataclass
class DiffusionConfig:
    """Configuration for diffusion market model."""

    n_factors: int = 5
    latent_dim: int = 16
    n_diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    random_seed: int = 42


class DiffusionMarketModel:
    """Latent diffusion model for synthetic market path generation.

    Architecture (numpy-only, no heavy DL dependencies):
      - Encoder: linear projection from factor space to latent space
      - Diffusion: Gaussian noise schedule in latent space (DDPM-style)
      - Decoder: linear projection back to factor space

    The model is trained on daily factor returns and can generate
    plausible synthetic paths for stress testing or training data
    augmentation.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.cfg = config or DiffusionConfig()
        self._trained = False

        # Noise schedule (beta_t)
        self.betas = np.linspace(
            self.cfg.beta_start, self.cfg.beta_end, self.cfg.n_diffusion_steps
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate(
            ([1.0], self.alphas_cumprod[:-1])
        )

        # Model weights (initialized on first fit)
        self._W_encode: Optional[np.ndarray] = None
        self._b_encode: Optional[np.ndarray] = None
        self._W_decode: Optional[np.ndarray] = None
        self._b_decode: Optional[np.ndarray] = None

        # Training stats for standardization
        self._factor_mean: Optional[np.ndarray] = None
        self._factor_std: Optional[np.ndarray] = None

        np.random.seed(self.cfg.random_seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "DiffusionMarketModel":
        """Train the diffusion model on factor returns.

        Args:
            X: Array of shape (n_samples, n_factors) with daily factor returns.

        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.cfg.n_factors:
            raise ValueError(
                f"Expected {self.cfg.n_factors} factors, got {X.shape[1]}"
            )

        n_samples = X.shape[0]

        # Standardize factors
        self._factor_mean = np.mean(X, axis=0)
        self._factor_std = np.std(X, axis=0) + 1e-8
        X_norm = (X - self._factor_mean) / self._factor_std

        # Xavier init for encoder / decoder
        limit_enc = np.sqrt(6.0 / (self.cfg.n_factors + self.cfg.latent_dim))
        self._W_encode = np.random.uniform(
            -limit_enc, limit_enc, (self.cfg.n_factors, self.cfg.latent_dim)
        )
        self._b_encode = np.zeros(self.cfg.latent_dim)

        limit_dec = np.sqrt(6.0 / (self.cfg.latent_dim + self.cfg.n_factors))
        self._W_decode = np.random.uniform(
            -limit_dec, limit_dec, (self.cfg.latent_dim, self.cfg.n_factors)
        )
        self._b_decode = np.zeros(self.cfg.n_factors)

        # Simple SGD on reconstruction + diffusion denoising
        lr = self.cfg.learning_rate
        bs = min(self.cfg.batch_size, n_samples)

        for epoch in range(self.cfg.epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, bs):
                batch_idx = idx[i : i + bs]
                batch = X_norm[batch_idx]

                # Forward: encode
                z = np.tanh(batch @ self._W_encode + self._b_encode)

                # Diffusion step: add noise at random timestep
                t = np.random.randint(0, self.cfg.n_diffusion_steps)
                noise = np.random.randn(*z.shape)
                alpha_t = self.alphas_cumprod[t]
                z_noisy = np.sqrt(alpha_t) * z + np.sqrt(1 - alpha_t) * noise

                # Decode
                x_recon = z_noisy @ self._W_decode + self._b_decode

                # Loss: reconstruction + latent consistency
                recon_loss = np.mean((x_recon - batch) ** 2)
                latent_loss = np.mean((z_noisy - z) ** 2)
                loss = recon_loss + 0.1 * latent_loss
                epoch_loss += loss * len(batch_idx)

                # Backprop (manual numpy)
                grad_recon = 2 * (x_recon - batch) / len(batch_idx)
                grad_W_dec = z_noisy.T @ grad_recon
                grad_b_dec = np.sum(grad_recon, axis=0)

                grad_z = grad_recon @ self._W_decode.T + 0.2 * (z_noisy - z) / len(batch_idx)
                grad_tanh = grad_z * (1 - z ** 2)
                grad_W_enc = batch.T @ grad_tanh
                grad_b_enc = np.sum(grad_tanh, axis=0)

                # Update
                self._W_decode -= lr * grad_W_dec
                self._b_decode -= lr * grad_b_dec
                self._W_encode -= lr * grad_W_enc
                self._b_encode -= lr * grad_b_enc

            if epoch % 20 == 0:
                logger.debug(
                    "Diffusion epoch %d loss %.4f", epoch, epoch_loss / n_samples
                )

        self._trained = True
        logger.info("DiffusionMarketModel trained on %d samples", n_samples)
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        n_paths: int,
        path_length: int,
        seed: int = 42,
        regime_context: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Generate N synthetic market paths.

        Args:
            n_paths: Number of paths to generate.
            path_length: Number of timesteps per path.
            seed: Random seed for reproducibility.
            regime_context: Optional dict with keys like
                {"regime": "HIGH", "volatility_multiplier": 1.5}.

        Returns:
            Array of shape (n_paths, path_length, n_factors).
        """
        if not self._trained:
            raise RuntimeError("Model must be fitted before generate() is called.")
        if self._factor_mean is None or self._factor_std is None:
            raise RuntimeError("Model stats are missing.")

        np.random.seed(seed)

        vol_mult = 1.0
        if regime_context:
            vol_mult = float(regime_context.get("volatility_multiplier", 1.0))

        # Start from random latent noise
        z = np.random.randn(n_paths, self.cfg.latent_dim)

        # Reverse diffusion (simplified DDPM)
        for t in reversed(range(self.cfg.n_diffusion_steps)):
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod_prev[t] if t > 0 else 1.0
            beta_t = self.betas[t]

            # Predict x0 from noise (simplified: just denoise toward data manifold)
            noise = np.random.randn(n_paths, self.cfg.latent_dim) * vol_mult
            z = (z - beta_t * noise) / np.sqrt(1 - beta_t)

            if t > 0:
                sigma_t = np.sqrt(beta_t)
                z += sigma_t * np.random.randn(n_paths, self.cfg.latent_dim)

        # Decode to factor space
        z_activated = np.tanh(z)
        paths = z_activated @ self._W_decode + self._b_decode

        # Denormalize
        paths = paths * self._factor_std + self._factor_mean

        # Expand to path_length via temporal correlation (AR-1)
        full_paths = np.zeros((n_paths, path_length, self.cfg.n_factors))
        for i in range(n_paths):
            full_paths[i, 0] = paths[i]
            for t in range(1, path_length):
                ar_coef = 0.3  # moderate autocorrelation
                full_paths[i, t] = (
                    ar_coef * full_paths[i, t - 1]
                    + (1 - ar_coef) * paths[i]
                    + np.random.randn(self.cfg.n_factors) * vol_mult * 0.1
                )

        return full_paths

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(
        self,
        real_returns: np.ndarray,
        synthetic_paths: np.ndarray,
    ) -> Dict[str, Any]:
        """Validate synthetic paths against real returns.

        Returns:
            Dict with keys:
                - ks_statistic: float
                - ks_pvalue: float
                - autocorr_real: List[float]
                - autocorr_synthetic: List[float]
                - autocorr_mse: float
                - passed: bool (True if KS p > 0.05 and autocorr MSE < 0.1)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for validation. Install it.")

        if real_returns.ndim == 2:
            real_returns = real_returns.reshape(1, *real_returns.shape)

        n_real = real_returns.shape[0] * real_returns.shape[1]
        n_synth = synthetic_paths.shape[0] * synthetic_paths.shape[1]

        # Flatten across paths and time for marginal distribution test
        real_flat = real_returns.reshape(-1, real_returns.shape[-1])
        synth_flat = synthetic_paths.reshape(-1, synthetic_paths.shape[-1])

        # KS test per factor
        ks_stats = []
        ks_pvalues = []
        for f in range(min(real_flat.shape[1], synth_flat.shape[1])):
            stat, pval = stats.ks_2samp(real_flat[:, f], synth_flat[:, f])
            ks_stats.append(float(stat))
            ks_pvalues.append(float(pval))

        avg_ks_stat = float(np.mean(ks_stats))
        avg_ks_pval = float(np.mean(ks_pvalues))

        # Autocorrelation (lag-1) per factor, averaged across paths
        def _avg_autocorr(paths: np.ndarray) -> List[float]:
            n_p, length, n_f = paths.shape
            if length < 2:
                return [0.0] * n_f
            ac = []
            for f in range(n_f):
                corr = 0.0
                for p in range(n_p):
                    x = paths[p, :, f]
                    x_centered = x - np.mean(x)
                    denom = np.sum(x_centered ** 2)
                    if denom > 1e-8:
                        corr += np.sum(x_centered[1:] * x_centered[:-1]) / denom
                    else:
                        corr += 0.0
                ac.append(float(corr / n_p))
            return ac

        ac_real = _avg_autocorr(real_returns)
        ac_synth = _avg_autocorr(synthetic_paths)
        ac_mse = float(np.mean([(a - b) ** 2 for a, b in zip(ac_real, ac_synth)]))

        passed = avg_ks_pval > 0.05 and ac_mse < 0.1

        return {
            "ks_statistic": avg_ks_stat,
            "ks_pvalue": avg_ks_pval,
            "autocorr_real": ac_real,
            "autocorr_synthetic": ac_synth,
            "autocorr_mse": ac_mse,
            "passed": passed,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model weights and stats to disk."""
        np.savez(
            path,
            W_encode=self._W_encode,
            b_encode=self._b_encode,
            W_decode=self._W_decode,
            b_decode=self._b_decode,
            factor_mean=self._factor_mean,
            factor_std=self._factor_std,
            betas=self.betas,
            alphas=self.alphas,
            alphas_cumprod=self.alphas_cumprod,
            alphas_cumprod_prev=self.alphas_cumprod_prev,
        )
        logger.info("DiffusionMarketModel saved to %s", path)

    def load(self, path: str) -> "DiffusionMarketModel":
        """Load model weights and stats from disk."""
        data = np.load(path)
        self._W_encode = data["W_encode"]
        self._b_encode = data["b_encode"]
        self._W_decode = data["W_decode"]
        self._b_decode = data["b_decode"]
        self._factor_mean = data["factor_mean"]
        self._factor_std = data["factor_std"]
        self.betas = data["betas"]
        self.alphas = data["alphas"]
        self.alphas_cumprod = data["alphas_cumprod"]
        self.alphas_cumprod_prev = data["alphas_cumprod_prev"]
        self._trained = True
        logger.info("DiffusionMarketModel loaded from %s", path)
        return self
