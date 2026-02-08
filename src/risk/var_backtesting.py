"""
Value at Risk (VaR) Backtesting Module.

Implements EWMA VaR calculation and statistical backtesting with Kupiec and
Christoffersen tests to validate risk model accuracy.

Key Components:
- EWMA VaR with configurable lambda (default 0.94)
- 95% confidence level with 5% target violation rate
- Kupiec test (unconditional coverage)
- Christoffersen test (independence of violations)

Usage:
    from src.risk.var_backtesting import compute_ewma_var, backtest_var, VaRConfig

    config = VaRConfig()  # 95% confidence, λ=0.94
    var_series = compute_ewma_var(returns, config)
    result = backtest_var(returns, var_series, config)

    print(f"Violation rate: {result.violation_rate:.2%}")
    print(f"Kupiec test: {'PASSED' if result.kupiec_passed else 'FAILED'}")

Reference: RiskMetrics Technical Document (J.P. Morgan, 1996)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VaRConfig:
    """Configuration for VaR calculation and backtesting.

    Attributes:
        lambda_param: EWMA decay factor (0.94 = industry standard)
        confidence_level: VaR confidence (0.95 = 95%)
        violation_target: Expected violation rate (1 - confidence_level)
        min_history: Minimum samples for EWMA warmup
        significance_level: Alpha for statistical tests
    """
    lambda_param: float = 0.94
    confidence_level: float = 0.95
    violation_target: float = 0.05
    min_history: int = 60
    significance_level: float = 0.05


@dataclass
class VaRBacktestResult:
    """Result of VaR backtesting.

    Attributes:
        var_series: Computed VaR series
        violations: Boolean array where True = VaR exceeded
        violation_rate: Observed violation rate
        violation_count: Number of violations
        total_observations: Total valid observations
        kupiec_stat: Kupiec test statistic (LR POF)
        kupiec_pvalue: Kupiec test p-value
        kupiec_passed: Whether Kupiec test passed
        christoffersen_stat: Christoffersen independence test statistic
        christoffersen_pvalue: Christoffersen test p-value
        christoffersen_passed: Whether independence test passed
        combined_passed: Both tests passed
    """
    var_series: np.ndarray
    violations: np.ndarray
    violation_rate: float
    violation_count: int
    total_observations: int
    kupiec_stat: float
    kupiec_pvalue: float
    kupiec_passed: bool
    christoffersen_stat: float
    christoffersen_pvalue: float
    christoffersen_passed: bool
    combined_passed: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'violation_rate': self.violation_rate,
            'violation_count': self.violation_count,
            'total_observations': self.total_observations,
            'kupiec_stat': self.kupiec_stat,
            'kupiec_pvalue': self.kupiec_pvalue,
            'kupiec_passed': self.kupiec_passed,
            'christoffersen_stat': self.christoffersen_stat,
            'christoffersen_pvalue': self.christoffersen_pvalue,
            'christoffersen_passed': self.christoffersen_passed,
            'combined_passed': self.combined_passed,
        }


def compute_ewma_var(
    returns: np.ndarray,
    config: Optional[VaRConfig] = None,
) -> np.ndarray:
    """
    Compute EWMA (Exponentially Weighted Moving Average) VaR.

    EWMA volatility reacts faster to recent changes than simple moving average,
    making it more responsive to volatility clustering common in FX markets.

    Formula:
        σ²_t = λ * σ²_{t-1} + (1 - λ) * r²_{t-1}
        VaR_t = -z_α * σ_t

    Args:
        returns: Array of returns (log or simple)
        config: VaR configuration (uses defaults if None)

    Returns:
        VaR series (positive values represent potential loss)
    """
    if config is None:
        config = VaRConfig()

    returns = np.asarray(returns).flatten()
    n = len(returns)

    if n < config.min_history:
        logger.warning(
            f"Insufficient data for EWMA VaR: {n} samples < {config.min_history} required"
        )
        return np.full(n, np.nan)

    # Z-score for confidence level
    z_score = stats.norm.ppf(1 - config.confidence_level)

    # Initialize variance with sample variance of warmup period
    warmup_var = np.var(returns[:config.min_history])

    # EWMA variance calculation
    ewma_var = np.zeros(n)
    ewma_var[0] = warmup_var

    lambda_param = config.lambda_param
    one_minus_lambda = 1 - lambda_param

    for t in range(1, n):
        # EWMA variance: λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        ewma_var[t] = lambda_param * ewma_var[t-1] + one_minus_lambda * returns[t-1]**2

    # VaR = -z_α * σ (positive value = potential loss)
    ewma_vol = np.sqrt(ewma_var)
    var_series = -z_score * ewma_vol

    # Set warmup period to NaN
    var_series[:config.min_history] = np.nan

    return var_series


def _kupiec_test(
    violations: np.ndarray,
    expected_rate: float,
    significance_level: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Kupiec's Proportion of Failures (POF) test.

    Tests if the observed violation rate equals the expected rate under H0.
    Uses likelihood ratio test statistic which follows χ²(1) under H0.

    LR_POF = -2 * ln[(1-p)^(n-x) * p^x / (1-π)^(n-x) * π^x]

    where:
        p = expected violation rate (1 - confidence)
        π = observed violation rate
        n = total observations
        x = number of violations

    Args:
        violations: Boolean array of VaR violations
        expected_rate: Expected violation rate (e.g., 0.05 for 95% VaR)
        significance_level: Test significance level

    Returns:
        (test_statistic, p_value, passed)
    """
    n = len(violations)
    x = np.sum(violations)

    if n == 0:
        return 0.0, 1.0, True

    observed_rate = x / n
    p = expected_rate

    # Handle edge cases
    if x == 0:
        # No violations - check if this is statistically unlikely
        lr_stat = -2 * n * np.log(1 - p)
    elif x == n:
        # All violations - definitely fails
        lr_stat = -2 * n * np.log(p)
    else:
        # General case: LR = -2 * [log(null) - log(alternative)]
        # Null: p (expected rate), Alternative: π (observed rate)
        try:
            log_null = (n - x) * np.log(1 - p) + x * np.log(p)
            log_alt = (n - x) * np.log(1 - observed_rate) + x * np.log(observed_rate)
            lr_stat = -2 * (log_null - log_alt)
        except (ValueError, RuntimeWarning):
            # Numerical issues - treat as failure
            lr_stat = 100.0

    # Test statistic follows χ²(1)
    pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
    passed = pvalue > significance_level

    return float(lr_stat), float(pvalue), passed


def _christoffersen_test(
    violations: np.ndarray,
    significance_level: float = 0.05,
) -> Tuple[float, float, bool]:
    """
    Christoffersen's Independence Test.

    Tests if violations are independently distributed (no clustering).
    VaR violations should be random, not clustered together.

    Computes transition probabilities between violation states:
        n_ij = count of transitions from state i to state j
        π_ij = P(state j | state i)

    LR_ind = -2 * ln[L(π) / L(π_01, π_11)]

    Args:
        violations: Boolean array of VaR violations
        significance_level: Test significance level

    Returns:
        (test_statistic, p_value, passed)
    """
    n = len(violations)

    if n < 2:
        return 0.0, 1.0, True

    violations = violations.astype(int)

    # Count transitions
    n_00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))  # 0 -> 0
    n_01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))  # 0 -> 1
    n_10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))  # 1 -> 0
    n_11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))  # 1 -> 1

    # Handle edge cases
    n_0 = n_00 + n_01  # Total starting from 0
    n_1 = n_10 + n_11  # Total starting from 1

    if n_0 == 0 or n_1 == 0:
        # Not enough transitions to test independence
        return 0.0, 1.0, True

    # Transition probabilities
    pi_01 = n_01 / n_0 if n_0 > 0 else 0
    pi_11 = n_11 / n_1 if n_1 > 0 else 0

    # Overall violation probability under independence
    pi = (n_01 + n_11) / (n - 1)

    # Handle degenerate cases
    if pi == 0 or pi == 1:
        return 0.0, 1.0, True

    try:
        # Log-likelihood under independence (π_01 = π_11 = π)
        log_L0 = (n_00 + n_10) * np.log(1 - pi) + (n_01 + n_11) * np.log(pi)

        # Log-likelihood under alternative (different π_01, π_11)
        log_L1 = 0
        if n_00 > 0:
            log_L1 += n_00 * np.log(1 - pi_01)
        if n_01 > 0:
            log_L1 += n_01 * np.log(pi_01)
        if n_10 > 0:
            log_L1 += n_10 * np.log(1 - pi_11)
        if n_11 > 0:
            log_L1 += n_11 * np.log(pi_11)

        lr_stat = -2 * (log_L0 - log_L1)
    except (ValueError, RuntimeWarning):
        lr_stat = 0.0

    # Ensure non-negative (numerical issues can cause small negative values)
    lr_stat = max(0.0, lr_stat)

    # Test statistic follows χ²(1)
    pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
    passed = pvalue > significance_level

    return float(lr_stat), float(pvalue), passed


def backtest_var(
    returns: np.ndarray,
    var_series: np.ndarray,
    config: Optional[VaRConfig] = None,
) -> VaRBacktestResult:
    """
    Backtest VaR model using Kupiec and Christoffersen tests.

    Validates that:
    1. Violation rate is close to expected (Kupiec)
    2. Violations are not clustered (Christoffersen)

    Args:
        returns: Array of actual returns
        var_series: Computed VaR series (from compute_ewma_var)
        config: VaR configuration

    Returns:
        VaRBacktestResult with test statistics and pass/fail status
    """
    if config is None:
        config = VaRConfig()

    returns = np.asarray(returns).flatten()
    var_series = np.asarray(var_series).flatten()

    # Align arrays
    n = min(len(returns), len(var_series))
    returns = returns[:n]
    var_series = var_series[:n]

    # VaR is typically computed for t using info up to t-1
    # Compare return_t with VaR_t (VaR computed at t-1 for t)
    # We need to shift: violation if return < -VaR (loss exceeds VaR)
    # Since VaR is positive, violation when return < -VaR
    violations = returns < -var_series

    # Exclude NaN periods (warmup)
    valid_mask = ~np.isnan(var_series)
    valid_violations = violations[valid_mask]
    returns[valid_mask]

    total_obs = len(valid_violations)
    violation_count = int(np.sum(valid_violations))
    violation_rate = violation_count / total_obs if total_obs > 0 else 0.0

    # Run statistical tests
    kupiec_stat, kupiec_pvalue, kupiec_passed = _kupiec_test(
        valid_violations,
        config.violation_target,
        config.significance_level,
    )

    christoffersen_stat, christoffersen_pvalue, christoffersen_passed = _christoffersen_test(
        valid_violations,
        config.significance_level,
    )

    combined_passed = kupiec_passed and christoffersen_passed

    logger.info(
        f"VaR Backtest: {violation_count}/{total_obs} violations "
        f"({violation_rate:.2%}, target {config.violation_target:.2%})"
    )
    logger.info(
        f"  Kupiec: LR={kupiec_stat:.4f}, p={kupiec_pvalue:.4f} "
        f"({'PASS' if kupiec_passed else 'FAIL'})"
    )
    logger.info(
        f"  Christoffersen: LR={christoffersen_stat:.4f}, p={christoffersen_pvalue:.4f} "
        f"({'PASS' if christoffersen_passed else 'FAIL'})"
    )

    return VaRBacktestResult(
        var_series=var_series,
        violations=violations,
        violation_rate=violation_rate,
        violation_count=violation_count,
        total_observations=total_obs,
        kupiec_stat=kupiec_stat,
        kupiec_pvalue=kupiec_pvalue,
        kupiec_passed=kupiec_passed,
        christoffersen_stat=christoffersen_stat,
        christoffersen_pvalue=christoffersen_pvalue,
        christoffersen_passed=christoffersen_passed,
        combined_passed=combined_passed,
    )


def compute_parametric_var(
    returns: np.ndarray,
    window: int = 60,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """
    Compute simple parametric VaR using rolling window.

    Alternative to EWMA for comparison. Uses rolling mean/std.

    Args:
        returns: Return series
        window: Rolling window size
        confidence_level: VaR confidence level

    Returns:
        VaR series
    """
    import pandas as pd

    returns = pd.Series(returns)
    z_score = stats.norm.ppf(1 - confidence_level)

    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    var_series = -(rolling_mean + z_score * rolling_std)

    return var_series.values


def compute_historical_var(
    returns: np.ndarray,
    window: int = 250,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """
    Compute historical simulation VaR.

    Non-parametric method using actual return distribution.

    Args:
        returns: Return series
        window: Historical window size
        confidence_level: VaR confidence level

    Returns:
        VaR series
    """
    n = len(returns)
    var_series = np.full(n, np.nan)
    percentile = (1 - confidence_level) * 100

    for t in range(window, n):
        historical_returns = returns[t-window:t]
        var_series[t] = -np.percentile(historical_returns, percentile)

    return var_series


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("VaR Backtesting Module")
    print("=" * 60)

    # Generate synthetic returns with volatility clustering
    np.random.seed(42)
    n = 1000

    # GARCH-like returns
    returns = np.zeros(n)
    vol = 0.01
    for t in range(1, n):
        vol = 0.94 * vol + 0.06 * returns[t-1]**2
        returns[t] = np.sqrt(max(0.0001, vol)) * np.random.randn()

    print(f"\nGenerated {n} synthetic returns")
    print(f"Mean: {np.mean(returns):.6f}, Std: {np.std(returns):.6f}")

    # Compute VaR
    config = VaRConfig()
    var_series = compute_ewma_var(returns, config)

    print(f"\nEWMA VaR (λ={config.lambda_param}, {config.confidence_level*100:.0f}%)")
    print(f"Mean VaR: {np.nanmean(var_series):.6f}")

    # Backtest
    result = backtest_var(returns, var_series, config)

    print("\nBacktest Results:")
    print(f"  Violations: {result.violation_count}/{result.total_observations}")
    print(f"  Rate: {result.violation_rate:.2%} (target: {config.violation_target:.2%})")
    print(f"  Kupiec: {'PASSED' if result.kupiec_passed else 'FAILED'} (p={result.kupiec_pvalue:.4f})")
    print(f"  Independence: {'PASSED' if result.christoffersen_passed else 'FAILED'} (p={result.christoffersen_pvalue:.4f})")
    print(f"  Overall: {'PASSED' if result.combined_passed else 'FAILED'}")

    print("\n✓ VaR backtesting module ready")
