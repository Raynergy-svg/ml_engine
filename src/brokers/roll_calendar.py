"""Contract roll calendar for futures trading.

Manages contract expiration and roll dates for CME quarterly futures (ES, NQ)
and NYMEX monthly futures (CL, GC).

Contract Specifications:
- ES/NQ (CME Quarterly): March(H), June(M), September(U), December(Z)
  Expiration: 3rd Friday of contract month
  Roll Window: 8 business days before expiration

- CL (NYMEX Crude Oil Monthly): Expires ~20 days before delivery month
  Roll Window: 3 business days before expiration

- GC (NYMEX Gold Monthly): Expires ~27 days before delivery month
  Roll Window: 3 business days before expiration
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

logger = logging.getLogger(__name__)


# Contract month codes (CME standard)
CONTRACT_MONTH_CODES = {
    1: "F",   # January
    2: "G",   # February
    3: "H",   # March
    4: "J",   # April
    5: "K",   # May
    6: "M",   # June
    7: "N",   # July
    8: "Q",   # August
    9: "U",   # September
    10: "V",  # October
    11: "X",  # November
    12: "Z",  # December
}

# Quarterly contract months for ES and NQ
QUARTERLY_MONTHS = [3, 6, 9, 12]  # March, June, September, December


@dataclass
class RollSchedule:
    """Contract roll schedule definition."""
    symbol: str
    schedule_type: Literal["quarterly", "monthly"]
    contract_months: list[int]  # List of months (1-12) when contracts expire
    roll_days_before: int  # Days before expiration to start rolling
    notes: str = ""


# Roll schedules for supported contracts
ROLL_SCHEDULES = {
    "ES": RollSchedule(
        symbol="ES",
        schedule_type="quarterly",
        contract_months=QUARTERLY_MONTHS,
        roll_days_before=8,
        notes="CME E-mini S&P 500, 3rd Friday of March/June/September/December",
    ),
    "NQ": RollSchedule(
        symbol="NQ",
        schedule_type="quarterly",
        contract_months=QUARTERLY_MONTHS,
        roll_days_before=8,
        notes="CME E-mini Nasdaq-100, 3rd Friday of March/June/September/December",
    ),
    "CL": RollSchedule(
        symbol="CL",
        schedule_type="monthly",
        contract_months=list(range(1, 13)),  # All months
        roll_days_before=3,
        notes="NYMEX Crude Oil, expires ~20 days before delivery month",
    ),
    "GC": RollSchedule(
        symbol="GC",
        schedule_type="monthly",
        contract_months=list(range(1, 13)),  # All months
        roll_days_before=3,
        notes="NYMEX Gold, expires ~27 days before delivery month",
    ),
}


def is_business_day(d: date) -> bool:
    """Check if date is a business day (Monday-Friday).

    Args:
        d: Date to check

    Returns:
        True if date is Monday-Friday, False if weekend
    """
    return d.weekday() < 5


def count_business_days(start: date, end: date) -> int:
    """Count business days between two dates (inclusive of start, exclusive of end).

    Args:
        start: Start date (inclusive)
        end: End date (exclusive)

    Returns:
        Number of business days
    """
    count = 0
    current = start
    while current < end:
        if is_business_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def get_third_friday(year: int, month: int) -> date:
    """Get the 3rd Friday of a given month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        Date of 3rd Friday
    """
    # Find first Friday of the month
    first_day = date(year, month, 1)
    days_until_friday = (4 - first_day.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7

    first_friday = first_day + timedelta(days=days_until_friday)

    # Third Friday is 2 weeks later
    third_friday = first_friday + timedelta(weeks=2)

    return third_friday


def get_monthly_expiry_date(year: int, month: int, contract_type: str) -> date:
    """Get expiration date for monthly contracts (CL, GC).

    CL (Crude Oil): Expires the day before the delivery month begins (~20th of prior month)
    GC (Gold): Expires the day before the delivery month begins (~27th of prior month)

    More precisely:
    - CL delivery month N expires on the 20th of month N-1
    - GC delivery month N expires on the 27th of month N-1

    Args:
        year: Year
        month: Delivery month (1-12)
        contract_type: "CL" or "GC"

    Returns:
        Date of contract expiration
    """
    # Work backwards from delivery month to get prior month
    if month == 1:
        prior_month_year = year - 1
        prior_month = 12
    else:
        prior_month_year = year
        prior_month = month - 1

    last_day_of_prior = calendar.monthrange(prior_month_year, prior_month)[1]

    # CL expires 20th, GC expires 27th of prior month
    # Clamp to last day if the month is shorter
    if contract_type == "CL":
        expiry_day = min(20, last_day_of_prior)
    elif contract_type == "GC":
        expiry_day = min(27, last_day_of_prior)
    else:
        expiry_day = min(20, last_day_of_prior)

    return date(prior_month_year, prior_month, expiry_day)


class RollCalendar:
    """Manages contract roll dates and active contracts for futures.

    Supports both quarterly (ES, NQ) and monthly (CL, GC) contracts.
    Provides methods to:
    - Get the active contract for a given date
    - Calculate days until next roll
    - Determine if within roll window
    """

    def __init__(self):
        """Initialize roll calendar with supported contracts."""
        self.schedules = ROLL_SCHEDULES
        logger.info("RollCalendar initialized with %d contract(s)", len(self.schedules))

    def _get_expiration_date(self, symbol: str, contract_month_year: str) -> date:
        """Get expiration date for a specific contract.

        Args:
            symbol: Contract symbol (ES, NQ, CL, GC)
            contract_month_year: Contract month in YYYYMM format

        Returns:
            Expiration date

        Raises:
            ValueError: If symbol not supported or format invalid
        """
        if symbol not in self.schedules:
            raise ValueError(f"Unsupported symbol: {symbol}")

        try:
            year = int(contract_month_year[:4])
            month = int(contract_month_year[4:6])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid contract month format: {contract_month_year}")

        if month < 1 or month > 12:
            raise ValueError(f"Invalid month in contract: {contract_month_year}")

        schedule = self.schedules[symbol]

        if schedule.schedule_type == "quarterly":
            # For quarterly contracts, expiration is 3rd Friday of the month
            return get_third_friday(year, month)
        else:
            # For monthly contracts (CL, GC)
            return get_monthly_expiry_date(year, month, symbol)

    def _get_next_contract_month(self, symbol: str, current_month: int) -> int:
        """Get the next contract month after the current one.

        Args:
            symbol: Contract symbol
            current_month: Current month (1-12)

        Returns:
            Next contract month (1-12)
        """
        schedule = self.schedules[symbol]
        contract_months = schedule.contract_months

        # Find the next month in the cycle
        for next_month in contract_months:
            if next_month > current_month:
                return next_month

        # If no month found after current, wrap to first month of next year
        return contract_months[0]

    def get_active_contract(self, symbol: str, check_date: date | None = None) -> str:
        """Get the active contract for a given date.

        The active contract is the nearest contract that hasn't yet expired.
        A contract remains active until after its expiration date (we treat
        expiry as EOD - the day is still tradeable but rolling should begin).

        Args:
            symbol: Contract symbol (ES, NQ, CL, GC)
            check_date: Date to check (default: today)

        Returns:
            Active contract in YYYYMM format (e.g., '202406')

        Raises:
            ValueError: If symbol not supported
        """
        if symbol not in self.schedules:
            raise ValueError(f"Unsupported symbol: {symbol}")

        if check_date is None:
            check_date = date.today()

        schedule = self.schedules[symbol]
        year = check_date.year
        month = check_date.month

        # Try to find an active contract in the current year starting from current month
        for contract_month in sorted(schedule.contract_months):
            if contract_month >= month:
                contract_str = f"{year:04d}{contract_month:02d}"
                try:
                    expiry = self._get_expiration_date(symbol, contract_str)
                    # Contract is active on or before its expiration date
                    if check_date <= expiry:
                        return contract_str
                except ValueError:
                    continue

        # If no contract found in current year, use first contract of next year
        next_year = year + 1
        next_month = schedule.contract_months[0]
        return f"{next_year:04d}{next_month:02d}"

    def get_roll_date(self, symbol: str, contract_month_year: str) -> date:
        """Get the roll date (start of roll window) for a contract.

        Args:
            symbol: Contract symbol
            contract_month_year: Contract month in YYYYMM format

        Returns:
            First day of roll window

        Raises:
            ValueError: If symbol not supported
        """
        if symbol not in self.schedules:
            raise ValueError(f"Unsupported symbol: {symbol}")

        schedule = self.schedules[symbol]
        expiry = self._get_expiration_date(symbol, contract_month_year)

        # For quarterly contracts: count back 8 business days
        # For monthly contracts: count back 3 business days
        days_to_count = schedule.roll_days_before

        # Work backwards from expiration date
        current = expiry - timedelta(days=1)
        business_days_counted = 0

        while business_days_counted < days_to_count:
            if is_business_day(current):
                business_days_counted += 1
            if business_days_counted < days_to_count:
                current -= timedelta(days=1)

        return current

    def days_until_roll(self, symbol: str, check_date: date | None = None) -> int:
        """Calculate days until the next roll date.

        Returns the number of calendar days from check_date until the roll
        date for the active contract. If within the roll window or past
        the expiration, returns 0 or negative value.

        Args:
            symbol: Contract symbol
            check_date: Date to check (default: today)

        Returns:
            Days until roll (can be negative if past roll date)
        """
        if check_date is None:
            check_date = date.today()

        active_contract = self.get_active_contract(symbol, check_date)
        roll_date = self.get_roll_date(symbol, active_contract)

        delta = (roll_date - check_date).days
        return delta

    def should_roll(self, symbol: str, check_date: date | None = None) -> bool:
        """Determine if we should roll to the next contract.

        Returns True if within the roll window (days_until_roll <= 2).

        Args:
            symbol: Contract symbol
            check_date: Date to check (default: today)

        Returns:
            True if within roll window, False otherwise
        """
        if check_date is None:
            check_date = date.today()

        days_left = self.days_until_roll(symbol, check_date)
        return days_left <= 2

    def get_contract_details(self, symbol: str) -> RollSchedule:
        """Get contract schedule details.

        Args:
            symbol: Contract symbol

        Returns:
            RollSchedule dataclass with contract specifications

        Raises:
            ValueError: If symbol not supported
        """
        if symbol not in self.schedules:
            raise ValueError(f"Unsupported symbol: {symbol}")

        return self.schedules[symbol]
