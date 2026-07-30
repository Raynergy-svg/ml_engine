"""Canonical multiple-testing-aware significance calculations."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.trial_budget import (
    BONFERRONI_ALPHA as AUTHORITATIVE_BONFERRONI_ALPHA,
    N_TRIALS as AUTHORITATIVE_N_TRIALS,
    ledger_summary,
)

MIN_BARS_FOR_SIGNIFICANCE = 10
BOOTSTRAP_BLOCK_SIZE = 21
BOOTSTRAP_N_REPS = 5000
BOOTSTRAP_SEED = 20260630
EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> Optional[dict[str, float]]:
    """Bailey/Lopez de Prado DSR in per-period Sharpe units."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    ret = returns.dropna()
    t = len(ret)
    if t < MIN_BARS_FOR_SIGNIFICANCE:
        return None
    std = float(ret.std(ddof=1))
    if std <= 1e-12:
        return None
    sr = float(ret.mean() / std)
    skew = float(ret.skew())
    kurt = float(ret.kurtosis()) + 3.0
    if pd.isna(skew) or pd.isna(kurt):
        return None
    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if variance_term <= 0:
        return None
    sr_std = math.sqrt(variance_term / (t - 1))
    if sr_std <= 1e-12:
        return None
    sr0 = sr_std * (
        (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    dsr = float(norm.cdf((sr - sr0) / sr_std))
    return {
        "sr_hat_per_period": sr,
        "sr0_benchmark_per_period": float(sr0),
        "sr_std_per_period": sr_std,
        "skew": skew,
        "kurtosis_pearson": kurt,
        "n_obs": t,
        "dsr": dsr,
    }


def circular_block_bootstrap_sharpe_pvalue(
    returns: pd.Series,
    *,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    n_reps: int = BOOTSTRAP_N_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> Optional[dict[str, float]]:
    """Deterministic one-sided circular-block bootstrap under Sharpe <= 0."""
    if block_size < 1 or n_reps < 1:
        raise ValueError("block_size and n_reps must be positive")
    ret = returns.dropna().to_numpy()
    t = len(ret)
    if t < MIN_BARS_FOR_SIGNIFICANCE:
        return None
    block = max(1, min(block_size, t))
    n_blocks = math.ceil(t / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, t, size=(n_reps, n_blocks))
    boot_sharpes = np.empty(n_reps, dtype=float)
    for i in range(n_reps):
        pieces = [ret[np.arange(s, s + block) % t] for s in starts[i]]
        sample = np.concatenate(pieces)[:t]
        sd = sample.std(ddof=1)
        boot_sharpes[i] = (sample.mean() / sd) if sd > 1e-12 else 0.0
    return {
        "p_oos_sharpe_le_zero": float(np.mean(boot_sharpes <= 0.0)),
        "block_size": block,
        "n_bootstrap_reps": n_reps,
        "seed": seed,
        "boot_sharpe_mean": float(np.mean(boot_sharpes)),
        "boot_sharpe_std": float(np.std(boot_sharpes, ddof=1)),
    }


def corrected_significance(
    returns: pd.Series,
    *,
    n_trials: int,
    family_alpha: float = 0.05,
) -> dict[str, Any]:
    """Binding DSR plus Bonferroni block-bootstrap decision.

    ``n_trials`` stays a free parameter — the arithmetic must remain callable at
    any budget, since demonstrating trial inflation means computing the same
    series at two budgets. What is NOT free is which budget a lane may declare
    for a verdict; that is bound by
    :class:`~src.research.gated_harness.preregistration.ResearchSpecification`
    against the campaign register. To keep any divergence visible in the
    artifact itself, the authoritative budget is always reported alongside the
    applied one, plus an explicit ``budget_is_authoritative`` flag.
    """
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must be between zero and one")
    dsr_block = deflated_sharpe_ratio(returns, n_trials)
    boot_block = circular_block_bootstrap_sharpe_pvalue(returns)
    alpha = family_alpha / n_trials
    passes = None
    if dsr_block is not None and boot_block is not None:
        passes = bool(
            dsr_block["dsr"] >= 0.95
            and boot_block["p_oos_sharpe_le_zero"] < alpha
        )
    return {
        "dsr": dsr_block["dsr"] if dsr_block else None,
        "bonferroni_alpha": alpha,
        "p_oos_bootstrap": boot_block["p_oos_sharpe_le_zero"] if boot_block else None,
        "passes_significance": passes,
        "n_trials": n_trials,
        "authoritative_n_trials": AUTHORITATIVE_N_TRIALS,
        "authoritative_bonferroni_alpha": AUTHORITATIVE_BONFERRONI_ALPHA,
        "budget_is_authoritative": int(n_trials) == AUTHORITATIVE_N_TRIALS,
        "budget_source": "src.research.trial_budget.TRIAL_LEDGER",
        "trial_ledger": ledger_summary()["allocations"],
        "detail": {"dsr": dsr_block, "bootstrap": boot_block},
        "insufficient_data": dsr_block is None or boot_block is None,
    }
