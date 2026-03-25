"""
Automation subpackage for continuous scanning and maintenance.

Provides:
- ContinuousScanner: Run scans in a loop with configurable intervals
- IdleMaintenance: Background retraining and journal sync during idle periods
- PairModelSelector: Per-pair model selection with rolling accuracy tracking
"""

from src.scanner.automation.continuous import ContinuousScanner
from src.scanner.automation.maintenance import IdleMaintenance
from src.scanner.automation.pair_model_selector import PairModelSelector

__all__ = ["ContinuousScanner", "IdleMaintenance", "PairModelSelector"]
