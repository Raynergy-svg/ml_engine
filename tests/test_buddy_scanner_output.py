from rich.console import Console

from buddy_scanner import BuddyScanner
from src.scanner.display import ScannerDisplay
from src.scanner.results import PairAnalysis


def _scanner_with_recording_console() -> tuple[BuddyScanner, Console]:
    console = Console(record=True, force_terminal=False, width=120)
    scanner = BuddyScanner.__new__(BuddyScanner)
    scanner._display = ScannerDisplay(console=console)
    return scanner, console


def test_clean_output_labels_session_block_as_session():
    scanner, console = _scanner_with_recording_console()

    scanner._render_clean_output(
        [
            PairAnalysis(
                pair="EUR_USD",
                direction="HOLD",
                confidence=0.0,
                gates_passed=False,
                error="Outside trading session (4:00 UTC, active: 8-21 UTC)",
            )
        ],
        model_type="ensemble",
        granularity="M5",
    )

    out = console.export_text()
    assert "BUDDY PLANNER SCAN" in out
    assert "[SESSION]" in out
    assert "[ERROR]" not in out


def test_clean_output_keeps_real_failures_as_error():
    scanner, console = _scanner_with_recording_console()

    scanner._render_clean_output(
        [
            PairAnalysis(
                pair="EUR_USD",
                direction="HOLD",
                confidence=0.0,
                gates_passed=False,
                error="No data (OANDA offline, no CSV)",
            )
        ],
        model_type="ensemble",
        granularity="M5",
    )

    out = console.export_text()
    assert "[ERROR]" in out


def test_rich_display_uses_planner_palette_header():
    console = Console(record=True, force_terminal=False, width=120)
    display = ScannerDisplay(console=console)
    display._model_type = "ensemble"
    display._granularity = "M5"
    display._current_analyses = [
        PairAnalysis(pair="EUR_USD", direction="LONG", confidence=0.81, gates_passed=True)
    ]

    console.print(display.generate_header())
    console.print(display.generate_table())

    out = console.export_text()
    assert "BUDDY PLANNER SCANNER" in out
    assert "EUR/USD" in out
