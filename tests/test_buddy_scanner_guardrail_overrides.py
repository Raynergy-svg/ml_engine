from __future__ import annotations

from buddy_scanner import BuddyScanner
from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis, ScanResult


class _FakeGateEvaluator:
    def __init__(self, min_volatility_regime: int) -> None:
        self.min_volatility_regime = min_volatility_regime


class _FakeScannerCore:
    def __init__(self) -> None:
        self.config = ScannerConfig()
        self._gate_evaluator = _FakeGateEvaluator(self.config.min_volatility_regime)
        self.observed_session_filter = None
        self.observed_window = None

    def scan(self, pairs=None):  # noqa: D401 - test double
        self.observed_session_filter = self.config.enable_session_filter
        self.observed_window = (self.config.session_start_utc, self.config.session_end_utc)
        return ScanResult(
            analyses=[
                PairAnalysis(
                    pair="EUR_USD",
                    direction="LONG",
                    confidence=0.7,
                    gates_passed=True,
                )
            ],
            model_type="ensemble",
            granularity=self.config.granularity,
        )


def _make_scanner() -> tuple[BuddyScanner, _FakeScannerCore]:
    scanner = BuddyScanner.__new__(BuddyScanner)
    core = _FakeScannerCore()
    scanner._scanner = core
    return scanner, core


def test_buddy_scanner_scan_applies_and_restores_session_overrides():
    scanner, core = _make_scanner()
    original_filter = core.config.enable_session_filter
    original_window = (core.config.session_start_utc, core.config.session_end_utc)

    result = scanner.scan(
        pairs=["EUR_USD"],
        granularity="H1",
        top_n=1,
        verbose=False,
        prompt_train=False,
        session_filter_enabled_override=False,
        session_window_utc_override=(2, 22),
    )

    assert len(result) == 1
    assert core.observed_session_filter is False
    assert core.observed_window == (2, 22)
    assert core.config.enable_session_filter == original_filter
    assert (core.config.session_start_utc, core.config.session_end_utc) == original_window


def test_buddy_scanner_scan_force_still_wins_over_enable_override():
    scanner, core = _make_scanner()

    scanner.scan(
        pairs=["EUR_USD"],
        granularity="H1",
        top_n=1,
        verbose=False,
        prompt_train=False,
        force=True,
        session_filter_enabled_override=True,
        session_window_utc_override=(8, 21),
    )

    assert core.observed_session_filter is False
