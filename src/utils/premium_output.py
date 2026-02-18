"""
Premium Output Module - Borderless CLI output formatter.

This module provides a clean, borderless CLI output system using Rich.
It replaces Panel() usage and ASCII art borders with a minimalist,
typography-driven design.

Design Philosophy:
- NO ASCII Borders: Eliminate all box-drawing characters except ─ for dividers
- Minimalist Hierarchy: Use whitespace, indentation, and text styling
- Typography-First: Headers use uppercase bold with color + underline
- Status Glyphs: Unicode symbols (✓, ✗, ⚙, ⚠) for status indication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, ClassVar

# Optional Rich import with fallback
try:
    from rich.console import Console
    from rich.text import Text
    from rich.table import Table
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Text = None
    Table = None
    Style = None


@dataclass
class PremiumConfig:
    """
    Style constants for premium output formatting.
    
    Attributes:
        HEADER_COLOR: Color for main section headers
        HEADER_STYLE: Rich style string for headers
        KEY_COLOR: Color for dictionary keys/labels
        VALUE_COLOR: Default color for values
        SUCCESS_COLOR: Color for success states
        WARNING_COLOR: Color for warning states
        ERROR_COLOR: Color for error states
        DIVIDER_CHAR: Character used for divider lines
        DIVIDER_WIDTH: Default width of divider lines
        KEY_WIDTH: Default width for key column alignment
        KEY_VALUE_GAP: Spaces between key and value
        INDENT: String used for one level of indentation
    """
    # Color Palette
    HEADER_COLOR: str = "cyan"
    HEADER_STYLE: str = "bold"
    KEY_COLOR: str = "cyan"
    VALUE_COLOR: str = "white"
    SUCCESS_COLOR: str = "green"
    WARNING_COLOR: str = "yellow"
    ERROR_COLOR: str = "red"
    
    # Layout Constants
    DIVIDER_CHAR: str = "─"
    DIVIDER_WIDTH: int = 50
    KEY_WIDTH: int = 15
    KEY_VALUE_GAP: int = 4
    INDENT: str = "  "  # 2-space indent
    
    # Value color mapping by semantic type
    VALUE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "success": "bold green",
        "good": "green",
        "warning": "yellow",
        "error": "red",
        "dim": "dim",
        "accent": "magenta",
    })
    
    # Typography
    UNDERLINE_HEADERS: bool = True


class StatusGlyphs:
    """
    Unicode status indicators for consistent visual language.
    
    Provides a set of Unicode symbols for status indication without
    using box-drawing characters.
    """
    # Primary status indicators
    SUCCESS: ClassVar[str] = "✓"
    ERROR: ClassVar[str] = "✗"
    WARNING: ClassVar[str] = "⚠"
    PROCESSING: ClassVar[str] = "⚙"
    PENDING: ClassVar[str] = "○"
    
    # Secondary indicators
    BEST: ClassVar[str] = "★"
    GOOD: ClassVar[str] = "●"
    NEUTRAL: ClassVar[str] = "─"
    INFO: ClassVar[str] = "ℹ"
    STAR: ClassVar[str] = "★"
    
    # Directional indicators
    ARROW_UP: ClassVar[str] = "↑"
    ARROW_DOWN: ClassVar[str] = "↓"
    ARROW_RIGHT: ClassVar[str] = "→"
    
    # Step/section indicators
    STEP: ClassVar[str] = "⚙"
    CHEVRON: ClassVar[str] = "›"
    BULLET: ClassVar[str] = "•"
    
    @classmethod
    def get_glyph(cls, status: str) -> str:
        """
        Get glyph for a given status type.
        
        Args:
            status: Status type string (success, error, warning, etc.)
            
        Returns:
            Unicode glyph string
        """
        glyph_map = {
            "success": cls.SUCCESS,
            "error": cls.ERROR,
            "warning": cls.WARNING,
            "processing": cls.PROCESSING,
            "pending": cls.PENDING,
            "best": cls.BEST,
            "good": cls.GOOD,
            "neutral": cls.NEUTRAL,
            "info": cls.INFO,
            "step": cls.STEP,
        }
        return glyph_map.get(status.lower(), cls.NEUTRAL)


class PremiumConsole:
    """
    Premium CLI output formatter with borderless, typography-driven design.
    
    Wraps rich.Console to provide a clean, modern output aesthetic without
    ASCII borders or box-drawing characters (except ─ for dividers).
    
    Example:
        >>> console = PremiumConsole()
        >>> console.section_header("OANDA API")
        >>> console.key_value("Instrument", "NZD_USD")
        >>> console.key_value("Granularity", "H1")
        
        Output:
            OANDA API
            ──────────────────────────────────────
              Instrument:    NZD_USD
              Granularity:   H1
    """
    
    def __init__(self, console: Optional["Console"] = None, config: PremiumConfig = None):
        """
        Initialize premium console.
        
        Args:
            console: Rich Console instance (creates new if None)
            config: PremiumConfig instance (uses defaults if None)
        """
        self._config = config if config is not None else PremiumConfig()
        
        if console is not None:
            self._console = console
            self._rich_available = RICH_AVAILABLE
        elif RICH_AVAILABLE:
            self._console = Console()
            self._rich_available = True
        else:
            self._console = None
            self._rich_available = False
    
    # =========================================================================
    # HEADER METHODS
    # =========================================================================
    
    def section_header(
        self,
        title: str,
        subtitle: str = None,
        color: str = None,
        underline: bool = None
    ) -> None:
        """
        Print a main section header with underline divider.
        
        Output:
            SECTION TITLE
            ──────────────────────────────────────
            Optional subtitle text here
        
        Args:
            title: Section title (auto-uppercased)
            subtitle: Optional subtitle text
            color: Header color (default: config.HEADER_COLOR)
            underline: Whether to show underline divider
        """
        header_color = color if color else self._config.HEADER_COLOR
        show_underline = underline if underline is not None else self._config.UNDERLINE_HEADERS
        upper_title = title.upper()
        
        if self._rich_available:
            # Print header with style
            style = f"bold {header_color}"
            self._console.print(upper_title, style=style)
            
            # Print underline
            if show_underline:
                divider_text = self._config.DIVIDER_CHAR * self._config.DIVIDER_WIDTH
                self._console.print(divider_text, style="dim")
            
            # Print subtitle if provided
            if subtitle:
                self._console.print(subtitle, style="dim")
        else:
            # Plain text fallback
            print(upper_title)
            if show_underline:
                print(self._config.DIVIDER_CHAR * self._config.DIVIDER_WIDTH)
            if subtitle:
                print(subtitle)
    
    def step_progress(
        self,
        step: int,
        total: int,
        title: str,
        color: str = None
    ) -> None:
        """
        Print step progress header (e.g., "⚙ STEP 1/4: NEURAL NETWORK").
        
        Output:
            ⚙ STEP 1/4: NEURAL NETWORK
        
        Args:
            step: Current step number
            total: Total number of steps
            title: Step title (auto-uppercased)
            color: Optional color override for the title
        """
        glyph = StatusGlyphs.STEP
        upper_title = title.upper()
        text = f"{glyph} STEP {step}/{total}: {upper_title}"
        
        if self._rich_available:
            style = f"bold {color}" if color else "bold"
            self._console.print(text, style=style)
        else:
            print(text)
    
    # =========================================================================
    # STATUS METHODS
    # =========================================================================
    
    def status_line(
        self,
        key: str,
        value: str,
        status: str = "success",
        indent: bool = True,
        value_style: str = None
    ) -> None:
        """
        Print a status line with aligned key-value pair and status glyph.
        
        Output:
            ✓ Architecture:   Transformer (Directional)
            ⚠ Warning:        Overfitting detected (Gap: 20.6%)
        
        Args:
            key: Label/key text
            value: Value text
            status: Status type (success, error, warning, processing, neutral)
            indent: Whether to indent the line
            value_style: Optional style override for value
        """
        glyph = StatusGlyphs.get_glyph(status)
        indent_str = self._config.INDENT if indent else ""
        
        # Calculate padding for alignment
        key_display = f"{key}:"
        key_width = self._config.KEY_WIDTH + len(self._config.INDENT)
        padded_key = f"{key_display:<{key_width}}"
        
        # Get status color for glyph
        status_color = self._config.VALUE_COLORS.get(status, self._config.VALUE_COLOR)
        
        if self._rich_available:
            # Build styled text
            text = Text()
            text.append(indent_str)
            text.append(f"{glyph} ", style=status_color)
            text.append(f"{padded_key}", style=self._config.KEY_COLOR)
            text.append(f"{' ' * self._config.KEY_VALUE_GAP}")
            if value_style:
                text.append(value, style=value_style)
            else:
                text.append(value, style=self._config.VALUE_COLOR)
            self._console.print(text)
        else:
            print(f"{indent_str}{glyph} {padded_key}{' ' * self._config.KEY_VALUE_GAP}{value}")
    
    def status_block(
        self,
        items: List[Dict[str, Any]],
        indent: bool = True
    ) -> None:
        """
        Print multiple status lines as a block.
        
        Output:
            ✓ Architecture:   Transformer (Directional)
            ✓ Status:         Training complete (Best Acc: 52.4%)
            ⚠ Warning:        Overfitting detected (Gap: 20.6%)
        
        Args:
            items: List of dicts with keys: key, value, status, value_style
            indent: Whether to indent all items
        """
        for item in items:
            self.status_line(
                key=item.get("key", ""),
                value=item.get("value", ""),
                status=item.get("status", "neutral"),
                indent=indent,
                value_style=item.get("value_style")
            )
    
    # =========================================================================
    # TABLE METHODS
    # =========================================================================
    
    def key_value(
        self,
        key: str,
        value: str,
        key_width: int = None,
        indent: bool = True
    ) -> None:
        """
        Print aligned key-value pair.
        
        Output:
            Instrument:    NZD_USD
            Granularity:   H1
        
        Args:
            key: Label/key text
            value: Value text
            key_width: Width for key column (default: config.KEY_WIDTH)
            indent: Whether to indent the line
        """
        width = key_width if key_width is not None else self._config.KEY_WIDTH
        indent_str = self._config.INDENT if indent else ""
        
        key_display = f"{key}:"
        padded_key = f"{key_display:<{width}}"
        
        if self._rich_available:
            text = Text()
            text.append(indent_str)
            text.append(padded_key, style=self._config.KEY_COLOR)
            text.append(f"{' ' * self._config.KEY_VALUE_GAP}")
            text.append(value, style=self._config.VALUE_COLOR)
            self._console.print(text)
        else:
            print(f"{indent_str}{padded_key}{' ' * self._config.KEY_VALUE_GAP}{value}")
    
    def key_value_table(
        self,
        data: Dict[str, str],
        key_width: int = None,
        indent: bool = True,
        title: str = None
    ) -> None:
        """
        Print multiple aligned key-value pairs.
        
        Output:
            Instrument:    NZD_USD
            Granularity:   H1
            Candles:       18,000
        
        Args:
            data: Dictionary of key-value pairs
            key_width: Width for key column (default: config.KEY_WIDTH)
            indent: Whether to indent all items
            title: Optional title above the table
        """
        if title:
            if self._rich_available:
                self._console.print(f"{self._config.INDENT}{title}", style="bold")
            else:
                print(f"{self._config.INDENT}{title}")
        
        for key, value in data.items():
            self.key_value(key, str(value), key_width, indent)
    
    def data_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        column_widths: List[int] = None,
        title: str = None
    ) -> None:
        """
        Print a borderless data table with aligned columns.
        
        Output:
            PERFORMANCE METRICS
            ────────────────────────────────────────────
            Component         Metric               Value
            Transformer       Validation Acc       52.4%
            Gradient Boost    Momentum MAE         0.0248
            Random Forest     Drawdown MAE        18.3 bps
        
        Args:
            headers: Column header texts
            rows: List of row data lists
            column_widths: Optional column widths (auto-calculated if None)
            title: Optional table title
        """
        # Print title if provided
        if title:
            self.section_header(title)
        
        # Calculate column widths if not provided
        if column_widths is None:
            column_widths = []
            for i, header in enumerate(headers):
                max_width = len(header)
                for row in rows:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                column_widths.append(max_width + 2)  # Add padding
        
        # Print headers
        if self._rich_available:
            header_text = Text()
            for i, header in enumerate(headers):
                width = column_widths[i] if i < len(column_widths) else len(header) + 2
                header_text.append(f"{header:<{width}}", style="bold cyan")
            self._console.print(header_text)
        else:
            header_line = ""
            for i, header in enumerate(headers):
                width = column_widths[i] if i < len(column_widths) else len(header) + 2
                header_line += f"{header:<{width}}"
            print(header_line)
        
        # Print rows
        for row in rows:
            if self._rich_available:
                row_text = Text()
                for i, cell in enumerate(row):
                    width = column_widths[i] if i < len(column_widths) else len(str(cell)) + 2
                    row_text.append(f"{str(cell):<{width}}", style="white")
                self._console.print(row_text)
            else:
                row_line = ""
                for i, cell in enumerate(row):
                    width = column_widths[i] if i < len(column_widths) else len(str(cell)) + 2
                    row_line += f"{str(cell):<{width}}"
                print(row_line)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def divider(
        self,
        width: int = None,
        style: str = "dim"
    ) -> None:
        """
        Print a horizontal divider line.
        
        Output:
            ──────────────────────────────────────
        
        Args:
            width: Divider width (default: config.DIVIDER_WIDTH)
            style: Rich style string
        """
        div_width = width if width is not None else self._config.DIVIDER_WIDTH
        divider_text = self._config.DIVIDER_CHAR * div_width
        
        if self._rich_available:
            self._console.print(divider_text, style=style)
        else:
            print(divider_text)
    
    def blank(self, count: int = 1) -> None:
        """
        Print blank line(s) for spacing.
        
        Args:
            count: Number of blank lines to print
        """
        for _ in range(count):
            if self._rich_available:
                self._console.print("")
            else:
                print("")
    
    def indent(self, level: int = 1) -> str:
        """
        Return indent string for given level.
        
        Args:
            level: Indentation level (1 = 2 spaces)
            
        Returns:
            Indent string
        """
        return self._config.INDENT * level
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def success(self, message: str, indent: bool = False) -> None:
        """
        Print success message with ✓ glyph.
        
        Args:
            message: Message text
            indent: Whether to indent the message
        """
        indent_str = self._config.INDENT if indent else ""
        text = f"{indent_str}{StatusGlyphs.SUCCESS} {message}"
        
        if self._rich_available:
            self._console.print(text, style="bold green")
        else:
            print(text)
    
    def warning(self, message: str, indent: bool = False) -> None:
        """
        Print warning message with ⚠ glyph.
        
        Args:
            message: Message text
            indent: Whether to indent the message
        """
        indent_str = self._config.INDENT if indent else ""
        text = f"{indent_str}{StatusGlyphs.WARNING} {message}"
        
        if self._rich_available:
            self._console.print(text, style="bold yellow")
        else:
            print(text)
    
    def error(self, message: str, indent: bool = False) -> None:
        """
        Print error message with ✗ glyph.
        
        Args:
            message: Message text
            indent: Whether to indent the message
        """
        indent_str = self._config.INDENT if indent else ""
        text = f"{indent_str}{StatusGlyphs.ERROR} {message}"
        
        if self._rich_available:
            self._console.print(text, style="bold red")
        else:
            print(text)
    
    def info(self, message: str, indent: bool = False) -> None:
        """
        Print info message with ℹ glyph.
        
        Args:
            message: Message text
            indent: Whether to indent the message
        """
        indent_str = self._config.INDENT if indent else ""
        text = f"{indent_str}{StatusGlyphs.INFO} {message}"
        
        if self._rich_available:
            self._console.print(text, style="cyan")
        else:
            print(text)
    
    # =========================================================================
    # FORMATTING HELPERS
    # =========================================================================
    
    def format_metric(
        self,
        value: float,
        metric_type: str = "default",
        thresholds: Dict[str, float] = None
    ) -> str:
        """
        Format a metric value with appropriate color based on thresholds.
        
        Args:
            value: Metric value
            metric_type: Type of metric (accuracy, loss, mae, r2)
            thresholds: Optional custom thresholds
            
        Returns:
            Formatted string (plain text, styling handled by caller)
        """
        # Default thresholds by metric type
        default_thresholds = {
            "accuracy": {"excellent": 0.70, "good": 0.55, "fair": 0.50},
            "loss": {"excellent": 0.3, "good": 0.5, "fair": 0.7},
            "mae": {"excellent": 0.05, "good": 0.1, "fair": 0.15},
            "r2": {"excellent": 0.8, "good": 0.6, "fair": 0.4},
        }
        
        thresh = thresholds or default_thresholds.get(metric_type, {})
        
        if metric_type == "accuracy":
            if value >= thresh.get("excellent", 0.70):
                return f"[bold green]{value:.4f}[/bold green]"
            elif value >= thresh.get("good", 0.55):
                return f"[green]{value:.4f}[/green]"
            elif value >= thresh.get("fair", 0.50):
                return f"[yellow]{value:.4f}[/yellow]"
            else:
                return f"[red]{value:.4f}[/red]"
        elif metric_type == "loss":
            if value <= thresh.get("excellent", 0.3):
                return f"[bold green]{value:.4f}[/bold green]"
            elif value <= thresh.get("good", 0.5):
                return f"[green]{value:.4f}[/green]"
            elif value <= thresh.get("fair", 0.7):
                return f"[yellow]{value:.4f}[/yellow]"
            else:
                return f"[red]{value:.4f}[/red]"
        else:
            return f"{value:.4f}"
    
    def format_duration(self, seconds: float) -> str:
        """
        Format duration in human-readable form.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., '2m 30s')
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def format_number(self, value: int, format_type: str = "default") -> str:
        """
        Format number with appropriate separators.
        
        Args:
            value: Number to format
            format_type: Type of formatting (default, compact)
            
        Returns:
            Formatted number string
        """
        if format_type == "compact" and value >= 1000000:
            return f"{value / 1000000:.1f}M"
        elif format_type == "compact" and value >= 1000:
            return f"{value / 1000:.1f}K"
        else:
            return f"{value:,}"


# Factory function for convenience
def create_premium_console(console: Optional["Console"] = None) -> PremiumConsole:
    """
    Factory function to create a PremiumConsole instance.
    
    Args:
        console: Optional Rich Console instance
        
    Returns:
        Configured PremiumConsole instance
    """
    return PremiumConsole(console=console)
