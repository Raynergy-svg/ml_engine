"""Order/position book features + persistence.

OANDA exposes per-instrument order book (pending orders) and position book
(open positions) via REST endpoints. Both update every 30 min and only carry
the last 24h — there is no historical backfill. This module:

  - ``features.extract_*`` : derive numerical features from a single bucket
    snapshot, suitable for both the ``order_flow`` agent's heuristics today
    and the direction transformer once enough history accumulates.
  - ``persister.snapshot`` : append a snapshot (raw buckets + derived
    features) to a per-pair JSONL so future retrains can use order-book
    features as inputs. Live-only constraint means we have to start logging
    today to have training data in N weeks.

The existing ``_evaluate_order_flow`` in ``src/scanner/agents/_team.py`` calls
``persister.snapshot`` after each successful fetch and uses ``features.*``
for richer contrarian/imbalance signals than the prior binary thresholds.
"""

from src.data.order_book.features import (
    extract_position_features,
    extract_order_features,
)
from src.data.order_book.persister import snapshot

__all__ = [
    "extract_position_features",
    "extract_order_features",
    "snapshot",
]
