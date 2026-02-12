"""BuddyScanner shim – wraps src.scanner.Scanner for CLI compatibility."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, List, Optional

from src.scanner import Scanner, ScannerConfig, ScanResult, PairAnalysis, ScannerDisplay


class BuddyScanner:
    """Thin wrapper around src.scanner.Scanner with the interface expected by cli/buddy_scanning.py."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        account_equity: Optional[float] = None,
        use_rl_sizer: bool = True,
    ):
        cfg = ScannerConfig()
        if config_path:
            from pathlib import Path
            cfg.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        if account_equity is not None:
            cfg.account_equity = account_equity
        cfg.non_interactive = True  # CLI never prompts
        self._scanner = Scanner(config=cfg)
        self._display = ScannerDisplay()

    # ------------------------------------------------------------------
    def scan(
        self,
        pairs: Optional[List[str]] = None,
        granularity: str = "H1",
        top_n: int = 5,
        verbose: bool = True,
        prompt_train: bool = True,
        diversified: bool = False,
        force: bool = False,
        **kwargs: Any,
    ) -> List[PairAnalysis]:
        """Run scan and return a list of PairAnalysis objects."""
        result: ScanResult = self._scanner.scan(pairs=pairs)

        # Sort by overall score and take top_n
        analyses = sorted(result.analyses, key=lambda a: a.overall_score, reverse=True)
        if top_n:
            analyses = analyses[:top_n]

        # Display results using the rich display
        if verbose:
            display_result = ScanResult(
                analyses=analyses,
                model_type=result.model_type,
                granularity=granularity,
            )
            self._display.show_result(display_result)

        return analyses


@contextlib.contextmanager
def suppress_logging(level: int = logging.WARNING):
    """Context manager that temporarily raises the root log level."""
    root = logging.getLogger()
    prev = root.level
    root.setLevel(level)
    try:
        yield
    finally:
        root.setLevel(prev)
