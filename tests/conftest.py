import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure repo root is importable for tests that use top-level modules.
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
