"""Test suite for Contract Roll Calendar module.

Tests cover:
1. ES quarterly March -> June roll logic
2. CL monthly expiration and roll window
3. Boundary dates (exactly on roll day)
4. days_until_roll calculation accuracy
5. should_roll True/False conditions
"""

import pytest
from datetime import date, timedelta

from src.brokers.roll_calendar import (
    RollCalendar,
    get_third_friday,
    is_business_day,
    count_business_days,
    get_monthly_expiry_date,
)


class TestBusinessDayFunctions:
    """Test business day helper functions."""

    def test_is_business_day_weekday(self):
        """Monday-Friday should be business days."""
        # Monday, March 3, 2025
        assert is_business_day(date(2025, 3, 3)) is True
        # Tuesday, March 4, 2025
        assert is_business_day(date(2025, 3, 4)) is True
        # Friday, March 7, 2025
        assert is_business_day(date(2025, 3, 7)) is True

    def test_is_business_day_weekend(self):
        """Saturday-Sunday should not be business days."""
        # Saturday, March 1, 2025
        assert is_business_day(date(2025, 3, 1)) is False
        # Sunday, March 2, 2025
        assert is_business_day(date(2025, 3, 2)) is False

    def test_count_business_days(self):
        """Count business days between two dates."""
        # March 3-7, 2025: Mon-Fri = 5 business days
        start = date(2025, 3, 3)
        end = date(2025, 3, 8)
        assert count_business_days(start, end) == 5

        # March 3-10, 2025: Mon-Fri + Sat + Sun = 5 business days
        start = date(2025, 3, 3)
        end = date(2025, 3, 11)
        assert count_business_days(start, end) == 6  # Mon-Sat (Sat is day after Fri)

    def test_count_business_days_same_day(self):
        """Same day should have 0 business days."""
        d = date(2025, 3, 3)
        assert count_business_days(d, d) == 0


class TestThirdFriday:
    """Test 3rd Friday calculation."""

    def test_third_friday_march_2025(self):
        """3rd Friday of March 2025 is March 21."""
        friday = get_third_friday(2025, 3)
        assert friday == date(2025, 3, 21)
        assert friday.weekday() == 4  # Friday is 4

    def test_third_friday_june_2025(self):
        """3rd Friday of June 2025 is June 20."""
        friday = get_third_friday(2025, 6)
        assert friday == date(2025, 6, 20)
        assert friday.weekday() == 4

    def test_third_friday_september_2025(self):
        """3rd Friday of September 2025 is September 19."""
        friday = get_third_friday(2025, 9)
        assert friday == date(2025, 9, 19)
        assert friday.weekday() == 4

    def test_third_friday_december_2025(self):
        """3rd Friday of December 2025 is December 19."""
        friday = get_third_friday(2025, 12)
        assert friday == date(2025, 12, 19)
        assert friday.weekday() == 4


class TestMonthlyExpiryDate:
    """Test monthly contract expiration dates."""

    def test_cl_expiry_date(self):
        """CL expires ~20 days before delivery month."""
        # May delivery = expires ~April 20
        expiry = get_monthly_expiry_date(2025, 5, "CL")
        assert expiry == date(2025, 4, 20)

    def test_gc_expiry_date(self):
        """GC expires ~27 days before delivery month."""
        # June delivery = expires ~May 27
        expiry = get_monthly_expiry_date(2025, 6, "GC")
        assert expiry == date(2025, 5, 27)

    def test_cl_expiry_january_wraps_year(self):
        """CL January delivery wraps to December of prior year."""
        expiry = get_monthly_expiry_date(2025, 1, "CL")
        assert expiry == date(2024, 12, 20)

    def test_cl_expiry_day_clamped_to_month_length(self):
        """If day would exceed month length, clamp to last day."""
        # February only has 28/29 days, so 20 is valid
        expiry = get_monthly_expiry_date(2025, 3, "CL")
        assert expiry == date(2025, 2, 20)


class TestRollCalendarBasic:
    """Test basic RollCalendar functionality."""

    def test_initialization(self):
        """RollCalendar should initialize with supported contracts."""
        cal = RollCalendar()
        assert "ES" in cal.schedules
        assert "NQ" in cal.schedules
        assert "CL" in cal.schedules
        assert "GC" in cal.schedules

    def test_unsupported_symbol_raises_error(self):
        """Unsupported symbol should raise ValueError."""
        cal = RollCalendar()
        with pytest.raises(ValueError, match="Unsupported symbol"):
            cal.get_active_contract("INVALID")

    def test_invalid_contract_format_raises_error(self):
        """Invalid contract format should raise ValueError."""
        cal = RollCalendar()
        with pytest.raises(ValueError, match="Invalid contract month format"):
            cal.days_until_roll("ES")
            # Force invalid format in internal call
            cal._get_expiration_date("ES", "2025")


class TestESQuarterlyContracts:
    """Test ES quarterly contract logic."""

    def test_es_march_contract_expiry(self):
        """ES March 2025 (H25) expires 3rd Friday of March."""
        cal = RollCalendar()
        expiry = cal._get_expiration_date("ES", "202503")
        # 3rd Friday of March 2025 is March 21
        assert expiry == date(2025, 3, 21)

    def test_es_june_contract_expiry(self):
        """ES June 2025 (M25) expires 3rd Friday of June."""
        cal = RollCalendar()
        expiry = cal._get_expiration_date("ES", "202506")
        # 3rd Friday of June 2025 is June 20
        assert expiry == date(2025, 6, 20)

    def test_es_active_contract_before_march_expiry(self):
        """Before March 21, active contract is March 2025."""
        cal = RollCalendar()
        # March 10, 2025 - before expiry
        active = cal.get_active_contract("ES", date(2025, 3, 10))
        assert active == "202503"

    def test_es_active_contract_after_march_expiry(self):
        """After March 21, active contract is June 2025."""
        cal = RollCalendar()
        # March 24, 2025 - after expiry
        active = cal.get_active_contract("ES", date(2025, 3, 24))
        assert active == "202506"

    def test_es_active_contract_between_quarters(self):
        """April 15, 2025 should still be June contract."""
        cal = RollCalendar()
        active = cal.get_active_contract("ES", date(2025, 4, 15))
        assert active == "202506"


class TestESRollWindow:
    """Test ES roll window (8 business days before expiry)."""

    def test_es_march_roll_date(self):
        """ES March 2025 roll starts 8 business days before March 21."""
        cal = RollCalendar()
        roll_date = cal.get_roll_date("ES", "202503")

        # March 21 is 3rd Friday
        # Count back 8 business days:
        # Mar 20 (Thu) - 1
        # Mar 19 (Wed) - 2
        # Mar 18 (Tue) - 3
        # Mar 17 (Mon) - 4
        # Mar 14 (Fri) - 5
        # Mar 13 (Thu) - 6
        # Mar 12 (Wed) - 7
        # Mar 11 (Tue) - 8
        # So roll date is March 11, 2025
        assert roll_date == date(2025, 3, 11)

    def test_es_days_until_roll_before_window(self):
        """Days until roll should be positive before roll window."""
        cal = RollCalendar()
        # March 3, 2025 - before roll window
        days = cal.days_until_roll("ES", date(2025, 3, 3))
        assert days > 0
        # Should be about 8 days to roll date of March 11
        assert days == 8

    def test_es_days_until_roll_in_window(self):
        """Days until roll should be small within roll window."""
        cal = RollCalendar()
        # March 11, 2025 - first day of roll window
        days = cal.days_until_roll("ES", date(2025, 3, 11))
        assert days == 0

    def test_es_days_until_roll_one_day_before(self):
        """One day before roll window."""
        cal = RollCalendar()
        # March 10, 2025
        days = cal.days_until_roll("ES", date(2025, 3, 10))
        assert days == 1

    def test_es_should_roll_before_window(self):
        """should_roll False when > 2 days until roll."""
        cal = RollCalendar()
        # March 3, 2025
        should = cal.should_roll("ES", date(2025, 3, 3))
        assert should is False

    def test_es_should_roll_in_window(self):
        """should_roll True when <= 2 days until roll."""
        cal = RollCalendar()
        # March 11, 2025 - roll day
        should = cal.should_roll("ES", date(2025, 3, 11))
        assert should is True

    def test_es_should_roll_two_days_before(self):
        """should_roll True when exactly 2 days until roll."""
        cal = RollCalendar()
        # March 9, 2025
        days = cal.days_until_roll("ES", date(2025, 3, 9))
        assert days == 2
        should = cal.should_roll("ES", date(2025, 3, 9))
        assert should is True


class TestCLMonthlyContracts:
    """Test CL monthly contract logic."""

    def test_cl_active_contract_january(self):
        """January 10, 2025 should have active CL contract (Feb delivery)."""
        cal = RollCalendar()
        active = cal.get_active_contract("CL", date(2025, 1, 10))
        # January 10 is after Feb delivery contract expires (Jan 20)
        # So we move to March delivery contract (expires Feb 20)
        # But wait - let me reconsider: Jan 10 is before Feb 20 expiry
        # So Feb contract should be active if Jan 10 < Feb 20
        # Actually, CL Feb (delivery in Feb) expires Jan 20
        # So on Jan 10, we're still in January delivery (expires Dec 20)
        # Let's check: on Jan 10, active should be Jan delivery (expires Dec 20, 2024)
        # Since Jan 10, 2025 > Dec 20, 2024, we need Feb delivery (expires Jan 20, 2025)
        # Since Jan 10 < Jan 20, Feb delivery is active
        assert active == "202502"

    def test_cl_roll_window_three_days(self):
        """CL rolls 3 business days before expiry."""
        cal = RollCalendar()
        roll_date = cal.get_roll_date("CL", "202502")

        # CL Feb 2025 (delivery month Feb) expires Jan 20, 2025 (20th of prior month)
        # Roll 3 business days before:
        # Jan 20 (Mon) - expiry
        # Count back 3 business days:
        # Jan 17 (Fri) - 1
        # Jan 16 (Thu) - 2
        # Jan 15 (Wed) - 3
        # Roll date = Jan 15, 2025
        expiry = cal._get_expiration_date("CL", "202502")
        assert expiry == date(2025, 1, 20)
        assert roll_date == date(2025, 1, 15)

    def test_cl_days_until_roll_calculation(self):
        """days_until_roll should return accurate count for CL."""
        cal = RollCalendar()
        # December 17, 2024 is the roll date for Feb 2025 CL
        # If checking from December 16, 2024, should be 1 day
        days = cal.days_until_roll("CL", date(2024, 12, 16))
        assert days == 1

    def test_cl_should_roll_true_in_window(self):
        """CL should_roll True within 3-day window."""
        cal = RollCalendar()
        # December 17, 2024 - roll day
        should = cal.should_roll("CL", date(2024, 12, 17))
        assert should is True

    def test_cl_should_roll_false_before_window(self):
        """CL should_roll False > 2 days before roll."""
        cal = RollCalendar()
        # December 14, 2024 - before roll window
        should = cal.should_roll("CL", date(2024, 12, 14))
        assert should is False


class TestBoundaryConditions:
    """Test boundary and edge case conditions."""

    def test_exactly_on_roll_day(self):
        """Behavior exactly on roll day."""
        cal = RollCalendar()
        # March 11, 2025 is roll day for ES March
        days = cal.days_until_roll("ES", date(2025, 3, 11))
        assert days == 0
        assert cal.should_roll("ES", date(2025, 3, 11)) is True

    def test_exactly_on_expiry_day(self):
        """Behavior on contract expiration day."""
        cal = RollCalendar()
        # March 21, 2025 is expiry for ES March
        # The contract is still active on expiry day (tradeable until EOD)
        # So on March 21, we still use March contract
        active = cal.get_active_contract("ES", date(2025, 3, 21))
        assert active == "202503"

    def test_day_after_expiry(self):
        """Day after expiration, should be on new contract."""
        cal = RollCalendar()
        # March 22, 2025 - day after ES March expiry
        active = cal.get_active_contract("ES", date(2025, 3, 22))
        assert active == "202506"

    def test_contract_month_year_wraparound(self):
        """Contract logic wraps correctly at year boundary."""
        cal = RollCalendar()
        # December 19, 2025 is 3rd Friday, so expiry is Dec 19
        # December 20, 2025 is AFTER expiry, so should have wrapped
        # to next year's first contract (March 2026)
        active = cal.get_active_contract("ES", date(2025, 12, 20))
        # Dec 20 > Dec 19 (expiry), so wrap to 2026 March
        assert active == "202603"

        # December 10, 2025 is before expiry
        active = cal.get_active_contract("ES", date(2025, 12, 10))
        assert active == "202512"


class TestDefaultDate:
    """Test default date behavior (using today)."""

    def test_active_contract_default_date(self):
        """get_active_contract with None should use today."""
        cal = RollCalendar()
        active = cal.get_active_contract("ES", None)
        # Should not raise and should return valid format
        assert isinstance(active, str)
        assert len(active) == 6
        assert active[0:4].isdigit()
        assert active[4:6].isdigit()

    def test_days_until_roll_default_date(self):
        """days_until_roll with None should use today."""
        cal = RollCalendar()
        days = cal.days_until_roll("ES", None)
        # Should not raise and should return integer
        assert isinstance(days, int)

    def test_should_roll_default_date(self):
        """should_roll with None should use today."""
        cal = RollCalendar()
        result = cal.should_roll("ES", None)
        # Should not raise and should return boolean
        assert isinstance(result, bool)


class TestContractDetails:
    """Test getting contract schedule details."""

    def test_get_es_details(self):
        """Get ES contract details."""
        cal = RollCalendar()
        details = cal.get_contract_details("ES")
        assert details.symbol == "ES"
        assert details.schedule_type == "quarterly"
        assert details.contract_months == [3, 6, 9, 12]
        assert details.roll_days_before == 8

    def test_get_cl_details(self):
        """Get CL contract details."""
        cal = RollCalendar()
        details = cal.get_contract_details("CL")
        assert details.symbol == "CL"
        assert details.schedule_type == "monthly"
        assert details.roll_days_before == 3

    def test_get_invalid_details_raises_error(self):
        """Invalid symbol raises error when getting details."""
        cal = RollCalendar()
        with pytest.raises(ValueError, match="Unsupported symbol"):
            cal.get_contract_details("INVALID")


# Integration test: ES March -> June transition
class TestESMarchToJuneTransition:
    """Test the complete ES March -> June contract roll sequence."""

    def test_march_to_june_sequence(self):
        """Verify ES rolls from March to June correctly."""
        cal = RollCalendar()

        # Phase 1: Before roll window (March 1)
        assert cal.get_active_contract("ES", date(2025, 3, 1)) == "202503"
        assert cal.should_roll("ES", date(2025, 3, 1)) is False
        assert cal.days_until_roll("ES", date(2025, 3, 1)) == 10

        # Phase 2: In roll window (March 11)
        assert cal.get_active_contract("ES", date(2025, 3, 11)) == "202503"
        assert cal.should_roll("ES", date(2025, 3, 11)) is True
        assert cal.days_until_roll("ES", date(2025, 3, 11)) == 0

        # Phase 3: After expiry (March 24)
        assert cal.get_active_contract("ES", date(2025, 3, 24)) == "202506"
        assert cal.should_roll("ES", date(2025, 3, 24)) is False

        # Phase 4: Well into June contract (April 15)
        assert cal.get_active_contract("ES", date(2025, 4, 15)) == "202506"
        assert cal.should_roll("ES", date(2025, 4, 15)) is False
