# Phase 4: Production Hardening - Complete Documentation

## Overview

Phase 4 implements comprehensive production-hardening features for the FX Trading Bot, including:

1. **Trading Guardrails** - Risk management and safety controls
2. **Advanced Backtesting** - Monte Carlo simulation and transaction cost modeling
3. **Monitoring & Alerting** - System health, model drift, and performance tracking

All features are fully implemented and tested.

---

## 1. Trading Guardrails ✅

### Implementation

**File:** `src/risk/fx_guardrails.py`  
**Tests:** `tests/test_fx_guardrails.py`

### Features

#### Session Time Management
- Trading only during configured hours (default: 08:00 - 11:30 EST)
- Force-flat cutoff to close all positions before session end
- Session date tracking with timezone support

#### Risk Limits
- Daily loss stop (default: 5%)
- Maximum drawdown threshold (default: 10%)
- Maximum entries per day (default: 1)
- Maximum open positions (default: 1)

#### Spread Filtering
- Maximum spread limits per instrument
- Fallback spread values for missing data
- Slippage modeling

#### Confidence-Based Profit Stops
- Different profit targets per confidence band
- Low confidence: no target
- Medium confidence: 30% profit stop
- High confidence: 30% profit stop (configurable)

### Configuration

```yaml
# config/config_improved_H1.yaml
fx:
  enabled: true
  tier: 1
  timezone: America/New_York
  session:
    start: "08:00"
    end: "11:30"
    force_flat_at: "11:55"
    reset_at: "08:00"
  limits:
    max_open_positions: 1
    max_entries_per_day: 1
  risk:
    risk_per_trade_pct: 0.02
    daily_loss_stop_pct: 0.05
    atr_stop_mult: 1.5
    rr_take_profit: 2.0
  costs:
    max_spread_pips:
      EUR_USD: 1.0
      GBP_USD: 1.6
      USD_JPY: 1.2
    slippage_pips_min: 0.2
    slippage_pips_spread_mult: 0.25
  confidence:
    low_lt: 0.6
    high_gte: 0.75
```

### Usage

```python
from src.risk.fx_guardrails import (
    load_fx_policy,
    load_state,
    save_state,
    can_open_new_trade,
    check_daily_stops,
)

# Load policy from config
policy = load_fx_policy(config)

# Load or create daily state
state = load_state(config, policy)

# Check if can trade
can_trade, reason = can_open_new_trade(policy, state)
if not can_trade:
    print(f"Cannot trade: {reason}")

# Check daily stops
hit, reason, kind = check_daily_stops(
    policy,
    drawdown_pct=current_drawdown,
    realized_pct=realized_profit,
    confidence_band="medium",
)
if hit:
    print(f"Daily stop hit: {reason} ({kind})")
```

---

## 2. Advanced Backtesting ✅

### Implementation

**File:** `src/training/walkforward_validation.py`

### New Features

#### Monte Carlo Simulation

Runs N simulations to estimate confidence intervals for trading performance.

**Features:**
- Prediction uncertainty (adds noise to probabilities)
- Bootstrap sampling (simulates different market samples)
- Transaction cost modeling
- Confidence intervals for Sharpe ratio
- Probability of positive returns

**Function:** `monte_carlo_simulation()`

```python
from src.training.walkforward_validation import (
    monte_carlo_simulation,
    TransactionCosts,
)

# Define transaction costs
costs = TransactionCosts(
    spread_pips=1.0,        # Bid-ask spread
    slippage_pips=0.5,      # Expected slippage
    commission_pct=0.0,     # Commission percentage
    pip_value=10.0,         # Value of 1 pip (standard lot)
)

# Run Monte Carlo simulation
mc_result = monte_carlo_simulation(
    y_true=y_test,              # True labels
    y_pred=predictions,         # Model predictions
    prices=price_series,        # Price data
    n_simulations=1000,         # Number of MC runs
    costs=costs,                # Transaction costs
    confidence_noise=0.05,      # Std dev of prediction noise
    random_seed=42,             # For reproducibility
)

# Results
print(f"Mean Sharpe: {mc_result.mean_sharpe:.3f} ± {mc_result.std_sharpe:.3f}")
print(f"95% CI: [{mc_result.percentile_5:.3f}, {mc_result.percentile_95:.3f}]")
print(f"P(positive Sharpe): {mc_result.probability_positive_sharpe:.1%}")
print(f"P(positive return): {mc_result.probability_positive_return:.1%}")
```

#### Transaction Cost Modeling

**Function:** `apply_transaction_costs()`

Applies realistic trading costs to returns based on position changes.

**Costs Modeled:**
1. **Spread Cost**: Bid-ask spread paid on every trade
2. **Slippage Cost**: Additional cost from execution vs. quoted price
3. **Commission**: Percentage of notional (if applicable)

```python
from src.training.walkforward_validation import apply_transaction_costs

# Apply costs to returns
adjusted_returns = apply_transaction_costs(
    returns=strategy_returns,
    positions=position_series,  # 1=long, 0=flat, -1=short
    costs=costs,
)
```

#### Walk-Forward with Monte Carlo

**Function:** `run_monte_carlo_walkforward()`

Combines walk-forward validation with Monte Carlo simulation for each fold.

```python
from src.training.walkforward_validation import run_monte_carlo_walkforward

results = run_monte_carlo_walkforward(
    model_fn=lambda: create_model(),
    X=features,
    y=targets,
    prices=price_data,
    n_splits=5,                 # Walk-forward folds
    n_simulations=1000,         # MC simulations per fold
    costs=costs,
)

print(f"Mean Sharpe: {results['aggregate_sharpe_mean']:.3f}")
print(f"Probability profitable: {results['probability_profitable']:.1%}")
```

### Walk-Forward Validation (Existing)

**Modes:**
- **Expanding**: Training window grows with each fold
- **Rolling**: Fixed training window slides forward

**Features:**
- Purged K-fold (removes samples too close to test)
- Embargo gap (prevents leakage)
- Trading-specific metrics (Sharpe, Sortino, Calmar, win rate, profit factor)
- Time-series aware (no data leakage)

---

## 3. Monitoring & Alerting ✅

### Implementation

**Files:**
- `src/utils/monitoring.py` - Core monitoring system
- `scripts/monitoring_dashboard.py` - CLI dashboard
- `tests/test_monitoring.py` - Unit tests

### Features

#### Alert System

**Three Alert Levels:**
- **INFO**: Informational messages
- **WARNING**: Issues requiring attention
- **CRITICAL**: Urgent issues requiring immediate action

**Alert Categories:**
- `daily_loss`: Daily loss threshold exceeded
- `max_drawdown`: Maximum drawdown exceeded
- `consecutive_losses`: Too many consecutive losses
- `win_rate_degradation`: Win rate dropped significantly
- `model_drift`: Model predictions drifting
- `confidence_drop`: Model confidence decreased

#### Model Drift Detection

Tracks model health over time using:
- Prediction mean/std statistics
- Confidence level trends
- Feature distribution changes
- Performance degradation vs baseline

```python
from src.utils.monitoring import MonitoringSystem

monitor = MonitoringSystem(config)

# Check model drift
drift_metrics = monitor.check_model_drift(
    predictions=model_predictions,
    confidences=confidence_scores,
    feature_stats={'mean_rsi': 45.2, 'mean_atr': 0.0012},
)

if drift_metrics.alert_triggered:
    print(f"⚠️  Model drift detected: {drift_metrics.feature_drift_score:.3f}")
```

#### Performance Monitoring

Tracks daily trading performance:
- Total P&L
- Win/loss counts and win rate
- Consecutive losses
- Maximum drawdown
- Comparison to baseline metrics

```python
# Check daily performance
monitor.check_daily_performance(
    pnl=200.0,
    trades=trade_list,
    starting_balance=10000.0,
)

# Alerts will be generated if thresholds exceeded
critical_alerts = monitor.get_alerts(level=AlertLevel.CRITICAL)
```

#### Baseline Tracking

Establish performance baselines for comparison:

```python
from src.utils.monitoring import PerformanceMetrics

baseline = PerformanceMetrics(
    date='2025-01-01',
    total_pnl=1000.0,
    total_trades=50,
    winning_trades=30,
    losing_trades=20,
    win_rate=0.60,
    sharpe_ratio=1.5,
    max_drawdown=0.08,
    avg_trade_duration_hours=2.5,
    largest_win=200.0,
    largest_loss=-100.0,
)

monitor.save_baseline_metrics(baseline)
```

### CLI Commands

#### Show Dashboard

```bash
buddy monitor
```

**Output:**
- Alert summary (total, critical, warning, info)
- Recent alerts (last 5)
- Model health status
- Performance baseline
- Quick action suggestions

#### View Detailed Alerts

```bash
buddy monitor --monitor-alerts
```

Shows all alerts grouped by category with timestamps and details.

#### Check Model Drift

```bash
buddy monitor --monitor-drift
```

Shows recent model drift history with confidence and drift scores.

#### Generate Report

```bash
buddy monitor --monitor-report
```

Creates comprehensive JSON report with:
- All alerts
- Drift history
- Dashboard data
- Timestamp

### Monitoring Configuration

```yaml
# config/config_improved_H1.yaml
monitoring:
  enable_memory_monitoring: true
  alert_thresholds:
    daily_loss_threshold: 0.05          # 5% daily loss
    max_drawdown_threshold: 0.10        # 10% max drawdown
    win_rate_degradation: 0.10          # 10% win rate drop
    sharpe_ratio_min: 0.3               # Min acceptable Sharpe
    model_drift_threshold: 0.15         # 15% drift score
    confidence_drop_threshold: 0.10     # 10% confidence drop
    consecutive_losses_max: 5           # Max consecutive losses
```

### Dashboard Data Export

```python
# Get dashboard data for external visualization
data = monitor.get_dashboard_data()

# Returns:
# {
#     'timestamp': '2025-02-05T12:00:00',
#     'alerts': {'total': 3, 'critical': 1, 'warning': 2, 'info': 0},
#     'recent_drift': [...],
#     'baseline_metrics': {...}
# }
```

---

## TensorBoard Integration ✅

### Already Implemented

TensorBoard is fully integrated in `src/models/tensorflow_engine.py`:

**Features:**
- Training/validation loss curves
- Learning rate schedules
- Gradient statistics
- Trading-specific metrics
- Histogram tracking

**Configuration:**

```yaml
tensorboard:
  enabled: true
  log_dir: trained_data/tensorboard
  histogram_freq: 1
  write_graph: true
  write_images: false
  profile_batch: 0
  update_freq: epoch
```

**Launch TensorBoard:**

```bash
tensorboard --logdir trained_data/tensorboard
```

Then navigate to http://localhost:6006

---

## Testing

### Run All Phase 4 Tests

```bash
# Test guardrails
pytest tests/test_fx_guardrails.py -v

# Test monitoring
pytest tests/test_monitoring.py -v

# Test walk-forward (requires numpy)
python src/training/walkforward_validation.py
```

### Test Coverage

**Guardrails:**
- ✅ Session time validation
- ✅ Daily loss stops
- ✅ Drawdown limits
- ✅ Consecutive loss detection
- ✅ State persistence
- ✅ Timezone handling

**Monitoring:**
- ✅ Alert creation and filtering
- ✅ Performance monitoring
- ✅ Drift detection
- ✅ Baseline tracking
- ✅ Report generation
- ✅ Dashboard data

**Backtesting:**
- ✅ Walk-forward splits
- ✅ Purged K-fold
- ✅ Trading metrics
- ✅ Monte Carlo simulation
- ✅ Transaction costs

---

## Production Checklist

### Before Live Trading

- [ ] Review and adjust guardrail thresholds in config
- [ ] Establish baseline metrics from paper trading
- [ ] Test monitoring alerts with simulated scenarios
- [ ] Run walk-forward validation with transaction costs
- [ ] Verify Monte Carlo confidence intervals
- [ ] Set up TensorBoard monitoring
- [ ] Configure alert notification channels
- [ ] Test force-flat mechanism
- [ ] Verify session time restrictions
- [ ] Document emergency stop procedures

### Ongoing Monitoring

- [ ] Check `buddy monitor` daily
- [ ] Review drift metrics weekly
- [ ] Update baseline metrics monthly
- [ ] Run walk-forward validation on new data
- [ ] Monitor TensorBoard during retraining
- [ ] Review alert logs regularly
- [ ] Adjust thresholds based on experience

---

## Integration Example

```python
from src.utils.monitoring import MonitoringSystem
from src.risk.fx_guardrails import load_fx_policy, load_state
from src.training.walkforward_validation import monte_carlo_simulation

# Initialize monitoring
config = load_config('config/config_improved_H1.yaml')
monitor = MonitoringSystem(config)

# Load guardrails
policy = load_fx_policy(config)
state = load_state(config, policy)

# Before trading
can_trade, reason = can_open_new_trade(policy, state)
if not can_trade:
    monitor.add_alert(AlertLevel.WARNING, "trading_blocked", reason)

# After predictions
drift = monitor.check_model_drift(
    predictions=model_preds,
    confidences=model_conf,
)

# After trading session
monitor.check_daily_performance(
    pnl=daily_pnl,
    trades=todays_trades,
    starting_balance=start_balance,
)

# Check alerts
critical_alerts = monitor.get_alerts(level=AlertLevel.CRITICAL)
if critical_alerts:
    # Send notification
    for alert in critical_alerts:
        print(f"🔴 {alert.category}: {alert.message}")

# Generate daily report
report_path = create_monitoring_report(monitor)
print(f"Daily report: {report_path}")
```

---

## Future Enhancements (Optional)

### Grafana Integration
- Real-time dashboard with live metrics
- Custom panels for trading performance
- Alert visualization
- Historical trend analysis

### Notification Channels
- Email alerts for critical events
- Slack/Telegram bot integration
- SMS notifications for emergency stops
- Webhook support for custom integrations

### Advanced Analytics
- Anomaly detection using ML
- Regime change detection
- Correlation analysis across pairs
- Adaptive threshold adjustment

### Cloud Deployment
- Containerized monitoring service
- Cloud logging (CloudWatch, Stackdriver)
- Distributed tracing
- High-availability setup

---

## Summary

Phase 4 is **COMPLETE** ✅

All three components are fully implemented, tested, and documented:

1. ✅ **Guardrails**: Session management, risk limits, spread filtering
2. ✅ **Backtesting**: Monte Carlo, transaction costs, walk-forward
3. ✅ **Monitoring**: Alerts, drift detection, performance tracking

The system is production-ready with comprehensive safety controls and observability.

**Command Reference:**
```bash
buddy monitor                    # Dashboard
buddy monitor --monitor-alerts   # Detailed alerts
buddy monitor --monitor-drift    # Model drift
buddy monitor --monitor-report   # Full report
```

**Next Steps:**
- Review and adjust configuration thresholds
- Establish baseline metrics
- Optional: Set up Grafana and notification channels
