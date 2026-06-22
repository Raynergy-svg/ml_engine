"""Instrument model for broker integrations.

Defines the Instrument dataclass with asset class-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


AssetClass = Literal["FX", "FUTURES", "EQUITY"]


@dataclass(frozen=True)
class Instrument:
    """Trading instrument definition.

    Supports FX, FUTURES, and EQUITY asset classes with appropriate fields.

    Attributes:
        symbol: Display symbol (e.g., "EUR_USD", "ES").
        broker_symbol: Broker-specific symbol format (e.g., "EUR/USD" on some brokers).
        asset_class: "FX" or "FUTURES".
        tick_size: Minimum price increment (FUTURES only; optional for FX).
        tick_value: Dollar value of 1 tick movement (FUTURES only; optional for FX).
        multiplier: Contract multiplier (FUTURES only; optional for FX).
        pip_value: Dollar value of 1 pip move (FX only; optional for FUTURES).
        price_precision: Number of decimal places for prices (e.g., 5 for EUR_USD).
        margin_requirement: Margin per lot/contract.
        exchange: Exchange name (e.g., "CME", "OANDA").
        currency: Account currency for this instrument (e.g., "USD").
        contract_month: Contract month code for FUTURES (e.g., "202406" = June 2024).

    Validation:
        - For FX instruments: pip_value should be set, tick_size/tick_value/multiplier can be None.
        - For FUTURES instruments: tick_size, tick_value, multiplier should be set, pip_value can be None.
    """
    symbol: str
    broker_symbol: str
    asset_class: AssetClass
    price_precision: int
    margin_requirement: float
    exchange: str
    currency: str
    tick_size: Optional[float] = None
    tick_value: Optional[float] = None
    multiplier: Optional[float] = None
    pip_value: Optional[float] = None
    contract_month: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate instrument configuration."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if not self.broker_symbol:
            raise ValueError("broker_symbol cannot be empty")
        if self.asset_class not in ("FX", "FUTURES", "EQUITY"):
            raise ValueError(f"Invalid asset_class: {self.asset_class}")
        if self.price_precision < 0:
            raise ValueError("price_precision cannot be negative")
        if self.margin_requirement < 0:
            raise ValueError("margin_requirement cannot be negative")
        if not self.exchange:
            raise ValueError("exchange cannot be empty")
        if not self.currency:
            raise ValueError("currency cannot be empty")

        # Validate asset-class specific fields
        if self.asset_class == "FX":
            # FX instruments should have pip_value
            if self.pip_value is None:
                raise ValueError(
                    f"FX instrument {self.symbol} must define pip_value"
                )
            if self.pip_value <= 0:
                raise ValueError(
                    f"FX pip_value must be positive, got {self.pip_value}"
                )
        elif self.asset_class == "EQUITY":
            # EQUITY instruments are whole-share US listings routed via SMART;
            # no FX pip_value, no FUTURES tick triple needed. Their symbol IS
            # the trading ticker (e.g. "AAPL"); the broker_symbol mirrors it
            # so existing pipelines that key off broker_symbol still work.
            pass
        elif self.asset_class == "FUTURES":
            # FUTURES instruments should have tick_size, tick_value, multiplier
            if self.tick_size is None:
                raise ValueError(
                    f"FUTURES instrument {self.symbol} must define tick_size"
                )
            if self.tick_value is None:
                raise ValueError(
                    f"FUTURES instrument {self.symbol} must define tick_value"
                )
            if self.multiplier is None:
                raise ValueError(
                    f"FUTURES instrument {self.symbol} must define multiplier"
                )
            if self.tick_size <= 0:
                raise ValueError(
                    f"FUTURES tick_size must be positive, got {self.tick_size}"
                )
            if self.tick_value <= 0:
                raise ValueError(
                    f"FUTURES tick_value must be positive, got {self.tick_value}"
                )
            if self.multiplier <= 0:
                raise ValueError(
                    f"FUTURES multiplier must be positive, got {self.multiplier}"
                )

        # Optional fields validation
        if self.tick_size is not None and self.tick_size <= 0:
            raise ValueError(
                f"tick_size must be positive, got {self.tick_size}"
            )
        if self.tick_value is not None and self.tick_value <= 0:
            raise ValueError(
                f"tick_value must be positive, got {self.tick_value}"
            )
        if self.multiplier is not None and self.multiplier <= 0:
            raise ValueError(
                f"multiplier must be positive, got {self.multiplier}"
            )
        if self.pip_value is not None and self.pip_value <= 0:
            raise ValueError(
                f"pip_value must be positive, got {self.pip_value}"
            )

    @classmethod
    def fx(
        cls,
        symbol: str,
        broker_symbol: str,
        pip_value: float,
        price_precision: int = 5,
        margin_requirement: float = 1.0,
        exchange: str = "OANDA",
        currency: str = "USD",
    ) -> Instrument:
        """Factory method for creating FX instruments.

        Args:
            symbol: Display symbol (e.g., "EUR_USD").
            broker_symbol: Broker-specific format (e.g., "EUR/USD").
            pip_value: Dollar value of 1 pip move.
            price_precision: Decimal places (default 5).
            margin_requirement: Margin per lot (default 1%).
            exchange: Exchange name (default "OANDA").
            currency: Account currency (default "USD").

        Returns:
            Instrument with asset_class="FX".
        """
        return cls(
            symbol=symbol,
            broker_symbol=broker_symbol,
            asset_class="FX",
            tick_size=None,
            tick_value=None,
            multiplier=None,
            pip_value=pip_value,
            price_precision=price_precision,
            margin_requirement=margin_requirement,
            exchange=exchange,
            currency=currency,
        )

    @classmethod
    def equity(
        cls,
        symbol: str,
        broker_symbol: Optional[str] = None,
        price_precision: int = 2,
        margin_requirement: float = 100.0,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Instrument:
        """Factory method for creating EQUITY (single-stock) instruments.

        Args:
            symbol: Display + trading ticker (e.g., "AAPL").
            broker_symbol: Broker-specific symbol (defaults to ``symbol``).
            price_precision: Decimal places (default 2 for US equities).
            margin_requirement: Reg-T cash default = 100% (no leverage).
            exchange: IBKR routing exchange (default "SMART").
            currency: Account currency (default "USD").

        Returns:
            Instrument with asset_class="EQUITY".
        """
        return cls(
            symbol=symbol,
            broker_symbol=broker_symbol if broker_symbol else symbol,
            asset_class="EQUITY",
            tick_size=None,
            tick_value=None,
            multiplier=None,
            pip_value=None,
            price_precision=price_precision,
            margin_requirement=margin_requirement,
            exchange=exchange,
            currency=currency,
        )

    @classmethod
    def futures(
        cls,
        symbol: str,
        broker_symbol: str,
        tick_size: float,
        tick_value: float,
        multiplier: float,
        price_precision: int = 2,
        margin_requirement: float = 10.0,
        exchange: str = "CME",
        currency: str = "USD",
        contract_month: Optional[str] = None,
    ) -> Instrument:
        """Factory method for creating FUTURES instruments.

        Args:
            symbol: Display symbol (e.g., "ES").
            broker_symbol: Broker-specific format (e.g., "ESZ23").
            tick_size: Minimum price increment.
            tick_value: Dollar value per tick.
            multiplier: Contract multiplier.
            price_precision: Decimal places (default 2).
            margin_requirement: Margin per contract (default $10).
            exchange: Exchange name (default "CME").
            currency: Account currency (default "USD").
            contract_month: Optional contract month code.

        Returns:
            Instrument with asset_class="FUTURES".
        """
        return cls(
            symbol=symbol,
            broker_symbol=broker_symbol,
            asset_class="FUTURES",
            tick_size=tick_size,
            tick_value=tick_value,
            multiplier=multiplier,
            pip_value=None,
            price_precision=price_precision,
            margin_requirement=margin_requirement,
            exchange=exchange,
            currency=currency,
            contract_month=contract_month,
        )
