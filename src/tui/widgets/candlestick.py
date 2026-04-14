"""
BUDDY TUI -- ASCII Candlestick Chart Widget
Renders OHLC candle data as colorized ASCII art for the cyberpunk terminal.

Usage:
    chart = CandlestickChart(height=15, max_candles=40, id="price-chart")
    chart.set_data(candles=[(1.0880, 1.0895, 1.0870, 1.0890), ...])
    chart.set_levels(sl=1.0860, tp=1.0920, current=1.0890)
"""
from __future__ import annotations

from typing import Optional

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


# ── Theme constants (Cyberpunk Command Bridge palette) ─────────────
COLOR_VOID = "#0d0d18"
COLOR_AXIS = "#2a2a4a"
COLOR_AXIS_LABEL = "#6666aa"
COLOR_BULLISH = "#00ff41"
COLOR_BEARISH = "#ff1744"
COLOR_SL = "#ff1744"
COLOR_TP = "#00ff41"
COLOR_CURRENT = "#00ffcc"
COLOR_WICK = "#6666aa"

# ── Glyph constants ───────────────────────────────────────────────
GLYPH_BODY_BULL = "\u2588"  # Full block
GLYPH_BODY_BEAR = "\u2593"  # Dark shade
GLYPH_WICK = "\u2502"       # Box drawing light vertical
GLYPH_AXIS_TICK = "\u2524"  # Box drawing light vertical + left
GLYPH_AXIS_LINE = "\u2502"  # Vertical axis line
GLYPH_SL_DASH = "\u2500"    # Horizontal dash for SL/TP lines
GLYPH_EMPTY = " "


class CandlestickChart(Static):
    """Full-size ASCII candlestick chart with Y axis, SL/TP lines, and color.

    Parameters
    ----------
    chart_height : int
        Number of rows for the price axis (default 15).
    max_candles : int
        Maximum candles to display (default 40). Data is right-trimmed.
    """

    # Reactives to trigger re-render on data change
    _candle_version = reactive(0)

    def __init__(
        self,
        chart_height: int = 15,
        max_candles: int = 40,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._chart_height = max(5, chart_height)
        self._max_candles = max(1, max_candles)
        self._candles: list[tuple[float, float, float, float]] = []
        self._sl: Optional[float] = None
        self._tp: Optional[float] = None
        self._current_price: Optional[float] = None

    # ── Public API ─────────────────────────────────────────────────

    def set_data(
        self,
        candles: list[tuple[float, float, float, float]],
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        current: Optional[float] = None,
    ) -> None:
        """Replace candle data and optionally set SL/TP/current price.

        Parameters
        ----------
        candles : list of (open, high, low, close)
        sl : optional stop-loss price
        tp : optional take-profit price
        current : optional current market price
        """
        self._candles = list(candles[-self._max_candles:])
        self._sl = sl
        self._tp = tp
        self._current_price = current
        self._candle_version += 1

    def set_levels(
        self,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        current: Optional[float] = None,
    ) -> None:
        """Update SL/TP/current without replacing candle data."""
        self._sl = sl
        self._tp = tp
        self._current_price = current
        self._candle_version += 1

    # ── Render ─────────────────────────────────────────────────────

    def render(self) -> Text:
        if not self._candles:
            return self._render_empty()

        candles = self._candles
        height = self._chart_height
        num_candles = len(candles)

        # Compute global price range
        all_prices = []
        for o, h, l, c in candles:
            all_prices.extend([h, l])
        if self._sl is not None:
            all_prices.append(self._sl)
        if self._tp is not None:
            all_prices.append(self._tp)
        if self._current_price is not None:
            all_prices.append(self._current_price)

        price_max = max(all_prices)
        price_min = min(all_prices)
        price_range = price_max - price_min

        # Edge case: all prices identical -- expand range by 0.1%
        if price_range == 0:
            nudge = max(abs(price_max) * 0.001, 0.00010)
            price_max += nudge
            price_min -= nudge
            price_range = price_max - price_min

        # Add 5% padding top/bottom for visual breathing room
        padding = price_range * 0.05
        price_max += padding
        price_min -= padding
        price_range = price_max - price_min

        # Map price to row index (0 = top row = highest price)
        def price_to_row(price: float) -> int:
            if price_range == 0:
                return height // 2
            frac = (price_max - price) / price_range
            return max(0, min(height - 1, int(frac * (height - 1) + 0.5)))

        # Determine Y axis label width from price formatting
        label_decimals = self._auto_decimals(price_min, price_max)
        sample_label = f"{price_max:.{label_decimals}f}"
        label_width = len(sample_label)

        # Pre-compute SL/TP/current row positions
        sl_row = price_to_row(self._sl) if self._sl is not None else None
        tp_row = price_to_row(self._tp) if self._tp is not None else None
        cur_row = price_to_row(self._current_price) if self._current_price is not None else None

        # Pre-compute candle column data
        col_data = []
        for o, h, l, c in candles:
            bullish = c >= o
            body_top = max(o, c)
            body_bot = min(o, c)
            row_high = price_to_row(h)
            row_low = price_to_row(l)
            row_body_top = price_to_row(body_top)
            row_body_bot = price_to_row(body_bot)
            # Ensure body occupies at least 1 row (doji)
            if row_body_top == row_body_bot:
                pass  # single row body is fine
            col_data.append((bullish, row_high, row_low, row_body_top, row_body_bot))

        # Build output line by line
        text = Text()

        for row in range(height):
            # Price on Y axis: show label at top, bottom, and every ~4 rows
            if row == 0 or row == height - 1 or row % max(1, height // 5) == 0:
                price_at_row = price_max - (row / max(1, height - 1)) * price_range
                label = f"{price_at_row:.{label_decimals}f}"
                text.append(f"{label:>{label_width}}", style=COLOR_AXIS_LABEL)
                text.append(f" {GLYPH_AXIS_TICK}", style=COLOR_AXIS)
            else:
                text.append(" " * label_width, style=COLOR_AXIS_LABEL)
                text.append(f" {GLYPH_AXIS_LINE}", style=COLOR_AXIS)

            # Candle columns
            for ci, (bullish, row_high, row_low, row_body_top, row_body_bot) in enumerate(col_data):
                color = COLOR_BULLISH if bullish else COLOR_BEARISH

                if row_body_top <= row <= row_body_bot:
                    # Body region
                    glyph = GLYPH_BODY_BULL if bullish else GLYPH_BODY_BEAR
                    text.append(glyph, style=color)
                elif row_high <= row <= row_low:
                    # Wick region (above or below body)
                    text.append(GLYPH_WICK, style=COLOR_WICK)
                else:
                    # Empty space -- check for SL/TP/current line
                    marker = self._level_char(row, sl_row, tp_row, cur_row)
                    if marker:
                        text.append(marker[0], style=marker[1])
                    else:
                        text.append(GLYPH_EMPTY)

            # SL/TP/current annotations at end of row
            annotations = self._row_annotation(row, sl_row, tp_row, cur_row)
            if annotations:
                text.append(annotations[0], style=annotations[1])

            text.append("\n")

        return text

    # ── Helpers ─────────────────────────────────────────────────────

    def _render_empty(self) -> Text:
        t = Text()
        t.append("  No candle data", style=COLOR_AXIS_LABEL)
        return t

    @staticmethod
    def _auto_decimals(lo: float, hi: float) -> int:
        """Pick decimal places for Y axis labels based on price magnitude."""
        span = abs(hi - lo)
        if span == 0:
            span = abs(hi) if hi != 0 else 1.0
        if span < 0.001:
            return 5
        if span < 0.01:
            return 5
        if span < 0.1:
            return 4
        if span < 1.0:
            return 4
        if span < 10:
            return 3
        if span < 100:
            return 2
        return 1

    @staticmethod
    def _level_char(
        row: int,
        sl_row: Optional[int],
        tp_row: Optional[int],
        cur_row: Optional[int],
    ) -> Optional[tuple[str, str]]:
        """Return (char, style) if this row intersects an SL/TP/current line."""
        if cur_row is not None and row == cur_row:
            return (GLYPH_SL_DASH, COLOR_CURRENT)
        if sl_row is not None and row == sl_row:
            return (GLYPH_SL_DASH, COLOR_SL)
        if tp_row is not None and row == tp_row:
            return (GLYPH_SL_DASH, COLOR_TP)
        return None

    @staticmethod
    def _row_annotation(
        row: int,
        sl_row: Optional[int],
        tp_row: Optional[int],
        cur_row: Optional[int],
    ) -> Optional[tuple[str, str]]:
        """Return annotation text + style for the right margin."""
        if sl_row is not None and row == sl_row:
            return (" SL", COLOR_SL)
        if tp_row is not None and row == tp_row:
            return (" TP", COLOR_TP)
        if cur_row is not None and row == cur_row:
            return (" NOW", COLOR_CURRENT)
        return None


class MiniCandlestick(Static):
    """Compact 3-row candlestick for inline/table use.

    Shows the last N candles in a 3-row sparkline-style strip:
        Row 0 (top):    wick/body if in upper third of range
        Row 1 (mid):    wick/body if in middle third
        Row 2 (bottom): wick/body if in lower third

    Usage:
        mini = MiniCandlestick(max_candles=12, id="mini-eurusd")
        mini.set_data([(1.08, 1.09, 1.07, 1.085), ...])
    """

    _candle_version = reactive(0)

    def __init__(self, max_candles: int = 12, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_candles = max(1, max_candles)
        self._candles: list[tuple[float, float, float, float]] = []

    def set_data(self, candles: list[tuple[float, float, float, float]]) -> None:
        """Replace candle data and trigger re-render."""
        self._candles = list(candles[-self._max_candles:])
        self._candle_version += 1

    def render(self) -> Text:
        if not self._candles:
            t = Text()
            t.append("---", style=COLOR_AXIS_LABEL)
            return t

        candles = self._candles
        height = 3

        # Global range
        all_highs = [h for _, h, _, _ in candles]
        all_lows = [l for _, _, l, _ in candles]
        price_max = max(all_highs)
        price_min = min(all_lows)
        price_range = price_max - price_min

        if price_range == 0:
            nudge = max(abs(price_max) * 0.001, 0.00010)
            price_max += nudge
            price_min -= nudge
            price_range = price_max - price_min

        def price_to_row(price: float) -> int:
            frac = (price_max - price) / price_range
            return max(0, min(height - 1, int(frac * (height - 1) + 0.5)))

        # Pre-compute columns
        col_data = []
        for o, h, l, c in candles:
            bullish = c >= o
            body_top = max(o, c)
            body_bot = min(o, c)
            col_data.append((
                bullish,
                price_to_row(h),
                price_to_row(l),
                price_to_row(body_top),
                price_to_row(body_bot),
            ))

        text = Text()
        for row in range(height):
            for bullish, row_high, row_low, row_body_top, row_body_bot in col_data:
                color = COLOR_BULLISH if bullish else COLOR_BEARISH
                if row_body_top <= row <= row_body_bot:
                    glyph = GLYPH_BODY_BULL if bullish else GLYPH_BODY_BEAR
                    text.append(glyph, style=color)
                elif row_high <= row <= row_low:
                    text.append(GLYPH_WICK, style=COLOR_WICK)
                else:
                    text.append(GLYPH_EMPTY)
            if row < height - 1:
                text.append("\n")

        return text
