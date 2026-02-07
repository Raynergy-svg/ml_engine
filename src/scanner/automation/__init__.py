"""
Automation subpackage for continuous scanning and maintenance.

Provides:
- ContinuousScanner: Run scans in a loop with configurable intervals
- IdleMaintenance: Background retraining and journal sync during idle periods
"""

from src.scanner.automation.continuous import ContinuousScanner
from src.scanner.automation.maintenance import IdleMaintenance

__all__ = ["ContinuousScanner", "IdleMaintenance"]
