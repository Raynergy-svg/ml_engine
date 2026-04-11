"""Real-time post-trade feedback package.

Contains the `PostTradeLoop` (diagnostics + self-heal orchestrator) that runs
after every OANDA trade close. RL weight updates stay on the existing batch
path in `execution.sync_closed_trades_rl` — this package is logging and
diagnostics only.

Submodules are loaded lazily at call sites to avoid import cycles with
`src.scanner.execution`.
"""
