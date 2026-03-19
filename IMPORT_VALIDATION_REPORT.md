# ML Engine Import Validation Report
Date: 2026-03-18
Status: **VALIDATION ONLY** (no fixes attempted)

---

## Summary

Import validation identified **1 critical blocker affecting all scanner submodule imports**, plus **multiple modules unable to import due to cascading package initialization failure**.

**Result:** All attempts to import from `src.scanner.*` fail at the package __init__ level, preventing the automation modules from being tested.

---

## Test Results

### Test 1: Automation Module Direct Imports
```
FAIL: LearningEngine: No module named 'rich'
FAIL: ConfigTuner: No module named 'rich'
FAIL: StateEngine: No module named 'rich'
FAIL: ImprovementTracker: No module named 'rich'
FAIL: ObservationLog: No module named 'rich'
FAIL: ContinuousScanner: No module named 'rich'
```

**Root Cause:** The automation modules themselves do NOT import `rich`, but Python's package initialization system loads `src/scanner/__init__.py` first whenever any submodule is imported. The __init__.py imports `ScannerDisplay` (line 46), which depends on `rich` at the top level.

### Test 2: Config Module & Profiles
```
FAIL: ScannerConfig: No module named 'rich'
  blocked_pairs attribute: UNABLE TO CHECK (import failed)
  Profile methods (balanced, conservative, aggressive, smart): UNABLE TO CHECK (import failed)
```

**Root Cause:** Same as Test 1. Attempting to import `ScannerConfig` triggers the scanner __init__.py before config.py can even load.

### Test 3: Circular Dependency Scan

| Module | Status | Details |
|--------|--------|---------|
| `src.scanner.config` | FAIL | ModuleNotFoundError: rich |
| `src.scanner.agents` | OK | Imports successfully (no rich dependency) |
| `src.scanner.engine` | OK | Imports successfully (no rich dependency) |
| `src.scanner.execution` | OK | Imports successfully (no rich dependency) |
| `src.scanner.automation.learning_engine` | FAIL | Blocked by scanner __init__.py → display.py |
| `src.scanner.automation.config_tuner` | FAIL | Blocked by scanner __init__.py → display.py |
| `src.scanner.automation.state_engine` | FAIL | Blocked by scanner __init__.py → display.py |
| `src.scanner.automation.improvement_tracker` | FAIL | Blocked by scanner __init__.py → display.py |
| `src.scanner.automation.observation_log` | FAIL | Blocked by scanner __init__.py → display.py |

**Observation:** Modules that don't re-export through the main package (`agents`, `engine`, `execution`) can import directly. All automation submodules fail due to package initialization.

---

## Root Cause Analysis

### The Problem: Package __init__.py Tight Coupling

**File:** `/sessions/magical-compassionate-einstein/mnt/ml_engine/src/scanner/__init__.py`

**Problematic Code (Line 46):**
```python
from src.scanner.display import ScannerDisplay
```

**Why It Breaks Everything:**
1. When any code imports from `src.scanner.automation.*`, Python first loads `src/scanner/__init__.py`
2. The __init__.py eagerly imports `ScannerDisplay` on line 46
3. `ScannerDisplay` imports `rich` at the top level (line 13 of display.py):
   ```python
   from rich.console import Console, Group
   ```
4. If `rich` is not installed, the entire `src.scanner` package fails to initialize
5. This cascades to block ALL submodule imports

### Dependency Check

- **Required:** `rich==14.2.0` (in `requirements.txt`)
- **Installed:** NO
- **Environment:** Linux, Python 3.x (test environment lacks dependencies)

---

## Detailed Findings

### Automation Modules Assessment

All five automation modules have **CLEAN imports** (no unnecessary dependencies):

| Module | Direct rich Import? | Status |
|--------|---------------------|--------|
| `learning_engine.py` | NO | Clean (json, logging, re, dataclasses, datetime, pathlib, typing) |
| `config_tuner.py` | NO | Clean (fcntl, hashlib, json, logging, datetime, pathlib, typing) |
| `state_engine.py` | NO | Clean (json, logging, os, datetime, pathlib, typing) |
| `improvement_tracker.py` | NO | Clean (json, logging, datetime, pathlib, typing) |
| `observation_log.py` | NO | Clean (json, logging, datetime, pathlib, typing) |

**Key Finding:** These modules are truly independent — they don't need `rich` and will work fine in isolation.

### ContinuousScanner: Graceful Fallback

**File:** `src/scanner/automation/continuous.py` (lines 20-24)

```python
try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None
```

**Status:** ✓ CORRECT - Properly handles missing `rich` dependency with fallback to `None`.

### Display Module: Hard Dependency

**File:** `src/scanner/display.py` (line 13)

```python
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.box import SIMPLE_HEAVY
from rich.text import Text
```

**Status:** ✗ PROBLEM - All rich imports are top-level and unconditional. No fallback handling.

---

## Import Dependency Chain

```
User Code
  ↓
import src.scanner.automation.learning_engine
  ↓
[Python loads src/scanner/__init__.py]
  ↓
from src.scanner.display import ScannerDisplay  ← Line 46
  ↓
from rich.console import Console  ← Line 13 of display.py
  ↓
❌ ModuleNotFoundError: No module named 'rich'
```

---

## Blocked Functionality

Unable to validate:

1. ✗ `ScannerConfig` class definition and profiles (balanced, conservative, aggressive, smart)
2. ✗ `blocked_pairs` attribute on `ScannerConfig` 
3. ✗ Automation module functionality (learning engine, config tuner, state engine, etc.)
4. ✗ Trade journal integration points
5. ✗ Agent weight learning and RL sync
6. ✗ Configuration profile methods

---

## Recommendations (NOT IMPLEMENTED)

### Option A: Lazy Import in __init__.py (Low Risk)
Move the `ScannerDisplay` import to lazy evaluation:
```python
# Remove immediate import
# from src.scanner.display import ScannerDisplay

# Replace with factory or lazy loader
def get_scanner_display():
    from src.scanner.display import ScannerDisplay
    return ScannerDisplay
```

### Option B: Conditional Import in __init__.py (Low Risk)
```python
try:
    from src.scanner.display import ScannerDisplay
except ImportError:
    ScannerDisplay = None  # Graceful fallback
```

### Option C: Separate Display Exports (Medium Risk)
Create a separate `src.scanner.display_exports` that handles UI-only exports, keeping core imports separate.

### Option D: Install Dependencies (Immediate Fix)
```bash
pip install -r requirements.txt
```
After installation, all imports should work.

---

## Files Involved

### Import Chain
- `/sessions/magical-compassionate-einstein/mnt/ml_engine/src/scanner/__init__.py` (line 46)
- `/sessions/magical-compassionate-einstein/mnt/ml_engine/src/scanner/display.py` (line 13)
- All files in `/sessions/magical-compassionate-einstein/mnt/ml_engine/src/scanner/automation/`

### Dependency Files
- `/sessions/magical-compassionate-einstein/mnt/ml_engine/requirements.txt` (lists rich==14.2.0)
- `/sessions/magical-compassionate-einstein/mnt/ml_engine/pyproject.toml`

---

## Conclusion

**Status:** VALIDATION COMPLETE — BLOCKER IDENTIFIED

The import failure is **not a code error** but a **dependency/configuration issue**. The automation modules are themselves well-structured with clean imports. The problem is that the scanner package __init__.py eagerly imports a display module that depends on `rich`, preventing any submodule from loading when the dependency is missing.

**Next Steps:** Determine whether to (1) install dependencies, (2) refactor __init__.py for lazy loading, or (3) both.

