"""Unit tests for futures margin validation module."""

import pytest
from src.risk.margin import MarginChecker
from src.brokers.instrument import Instrument


class TestMarginChecker:
    """Test MarginChecker class."""

    def test_init_default_config(self):
        """Test default MarginChecker configuration."""
        checker = MarginChecker()
        assert checker.max_margin_pct == 0.80
        assert len(checker.positions_margin) == 0

    def test_init_custom_max_margin(self):
        """Test MarginChecker with custom max_margin_pct."""
        checker = MarginChecker(max_margin_pct=0.60)
        assert checker.max_margin_pct == 0.60

    def test_init_invalid_max_margin_zero(self):
        """Test MarginChecker rejects max_margin_pct of 0."""
        with pytest.raises(ValueError):
            MarginChecker(max_margin_pct=0.0)

    def test_init_invalid_max_margin_negative(self):
        """Test MarginChecker rejects negative max_margin_pct."""
        with pytest.raises(ValueError):
            MarginChecker(max_margin_pct=-0.1)

    def test_init_invalid_max_margin_over_1(self):
        """Test MarginChecker rejects max_margin_pct > 1."""
        with pytest.raises(ValueError):
            MarginChecker(max_margin_pct=1.5)

    def test_track_margin_single_position(self):
        """Test tracking margin for a single position."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        checker.track_margin(position_id="es_long_1", instrument=instrument, contracts=2)
        assert checker.positions_margin["es_long_1"] == 1000.0

    def test_track_margin_multiple_positions(self):
        """Test tracking margin across multiple positions."""
        checker = MarginChecker()
        es = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        nq = Instrument.futures(
            symbol="NQ",
            broker_symbol="NQH24",
            tick_size=0.25,
            tick_value=20.0,
            multiplier=100,
            margin_requirement=500.0,
        )
        cl = Instrument.futures(
            symbol="CL",
            broker_symbol="CLH24",
            tick_size=0.01,
            tick_value=10.0,
            multiplier=1000,
            margin_requirement=1000.0,
        )

        checker.track_margin("es_long", es, 1)
        checker.track_margin("nq_short", nq, 2)
        checker.track_margin("cl_long", cl, 1)

        assert checker.positions_margin["es_long"] == 500.0
        assert checker.positions_margin["nq_short"] == 1000.0
        assert checker.positions_margin["cl_long"] == 1000.0
        assert sum(checker.positions_margin.values()) == 2500.0

    def test_track_margin_zero_contracts(self):
        """Test tracking margin with zero contracts."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        checker.track_margin("es_zero", instrument, 0)
        assert checker.positions_margin["es_zero"] == 0.0

    def test_track_margin_empty_position_id(self):
        """Test tracking margin rejects empty position_id."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        with pytest.raises(ValueError):
            checker.track_margin("", instrument, 1)

    def test_track_margin_negative_contracts(self):
        """Test tracking margin rejects negative contracts."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        with pytest.raises(ValueError):
            checker.track_margin("es_neg", instrument, -1)

    def test_release_margin_existing_position(self):
        """Test releasing margin for an existing position."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        checker.track_margin("es_long", instrument, 2)
        assert "es_long" in checker.positions_margin

        released = checker.release_margin("es_long")
        assert released == 1000.0
        assert "es_long" not in checker.positions_margin

    def test_release_margin_nonexistent_position(self):
        """Test releasing margin for a non-existent position returns 0."""
        checker = MarginChecker()
        released = checker.release_margin("nonexistent")
        assert released == 0.0

    def test_check_margin_passes(self):
        """Test margin check passes when within limits."""
        checker = MarginChecker(max_margin_pct=0.80)
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000
        # New position: 100 contracts @ $500 = $50,000
        # Total: $50,000 < $80,000 ✓
        result = checker.check_margin(instrument, 100, 100000.0)
        assert result is True

    def test_check_margin_fails(self):
        """Test margin check fails when exceeding limits."""
        checker = MarginChecker(max_margin_pct=0.80)
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000
        # New position: 200 contracts @ $500 = $100,000
        # Total: $100,000 >= $80,000 ✗
        result = checker.check_margin(instrument, 200, 100000.0)
        assert result is False

    def test_check_margin_exactly_at_limit(self):
        """Test margin check with position exactly at limit."""
        checker = MarginChecker(max_margin_pct=0.80)
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000
        # New position: 160 contracts @ $500 = $80,000
        # Total: $80,000 = $80,000 (technically >= so fails)
        result = checker.check_margin(instrument, 160, 100000.0)
        # Should be False because projected_margin < allowed_margin requires strictly less than
        assert result is False

    def test_check_margin_just_below_limit(self):
        """Test margin check with position just below limit."""
        checker = MarginChecker(max_margin_pct=0.80)
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000
        # New position: 159 contracts @ $500 = $79,500
        # Total: $79,500 < $80,000 ✓
        result = checker.check_margin(instrument, 159, 100000.0)
        assert result is True

    def test_check_margin_with_existing_positions(self):
        """Test margin check accounts for existing positions."""
        checker = MarginChecker(max_margin_pct=0.80)
        es = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000
        # Existing position: 50 contracts @ $500 = $25,000
        checker.track_margin("es_long", es, 50)

        # New position: 110 contracts @ $500 = $55,000
        # Total would be: $25,000 + $55,000 = $80,000 (exactly at limit, should fail)
        result = checker.check_margin(es, 110, 100000.0)
        assert result is False

    def test_check_margin_accumulation(self):
        """Test margin check with multiple positions accumulating."""
        checker = MarginChecker(max_margin_pct=0.80)
        es = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        nq = Instrument.futures(
            symbol="NQ",
            broker_symbol="NQH24",
            tick_size=0.25,
            tick_value=20.0,
            multiplier=100,
            margin_requirement=500.0,
        )
        # Account NAV: $100,000
        # Max allowed: $80,000

        # Add first position: 50 ES contracts = $25,000
        assert checker.check_margin(es, 50, 100000.0) is True
        checker.track_margin("es_long", es, 50)

        # Add second position: 50 NQ contracts = $25,000
        # Total: $50,000 < $80,000 ✓
        assert checker.check_margin(nq, 50, 100000.0) is True
        checker.track_margin("nq_short", nq, 50)

        # Try to add third position: 110 ES contracts = $55,000
        # Total would be: $50,000 + $55,000 = $105,000 >= $80,000 ✗
        assert checker.check_margin(es, 110, 100000.0) is False

    def test_check_margin_zero_nav(self):
        """Test margin check rejects zero NAV."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        with pytest.raises(ValueError):
            checker.check_margin(instrument, 10, 0.0)

    def test_check_margin_negative_nav(self):
        """Test margin check rejects negative NAV."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        with pytest.raises(ValueError):
            checker.check_margin(instrument, 10, -100000.0)

    def test_get_margin_utilization_empty(self):
        """Test margin utilization with no positions."""
        checker = MarginChecker()
        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 0.0
        assert util["num_positions"] == 0
        assert util["avg_position_margin"] == 0.0

    def test_get_margin_utilization_single_position(self):
        """Test margin utilization with single position."""
        checker = MarginChecker()
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        checker.track_margin("es_long", instrument, 10)
        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 5000.0
        assert util["num_positions"] == 1
        assert util["avg_position_margin"] == 5000.0

    def test_get_margin_utilization_multiple_positions(self):
        """Test margin utilization with multiple positions."""
        checker = MarginChecker()
        es = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )
        nq = Instrument.futures(
            symbol="NQ",
            broker_symbol="NQH24",
            tick_size=0.25,
            tick_value=20.0,
            multiplier=100,
            margin_requirement=500.0,
        )
        checker.track_margin("es_long", es, 10)
        checker.track_margin("nq_short", nq, 5)

        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 5000.0 + 2500.0
        assert util["num_positions"] == 2
        assert util["avg_position_margin"] == 3750.0

    def test_check_margin_different_margin_requirements(self):
        """Test margin check with different margin requirements per instrument."""
        checker = MarginChecker(max_margin_pct=0.80)
        es = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,  # $500 per ES contract
        )
        cl = Instrument.futures(
            symbol="CL",
            broker_symbol="CLH24",
            tick_size=0.01,
            tick_value=10.0,
            multiplier=1000,
            margin_requirement=1000.0,  # $1000 per CL contract
        )

        # Account NAV: $100,000, max: $80,000
        # ES: 100 contracts = $50,000
        checker.track_margin("es_long", es, 100)

        # CL: 20 contracts = $20,000
        # Total: $70,000 < $80,000 ✓
        result = checker.check_margin(cl, 20, 100000.0)
        assert result is True

        # CL: 25 contracts = $25,000
        # Total would be: $75,000 < $80,000 ✓
        result = checker.check_margin(cl, 25, 100000.0)
        assert result is True

        # CL: 31 contracts = $31,000
        # Total would be: $81,000 >= $80,000 ✗
        result = checker.check_margin(cl, 31, 100000.0)
        assert result is False

    def test_margin_tracking_and_release_cycle(self):
        """Test complete lifecycle: track, check, and release margin."""
        checker = MarginChecker(max_margin_pct=0.80)
        instrument = Instrument.futures(
            symbol="ES",
            broker_symbol="ESH24",
            tick_size=0.25,
            tick_value=12.5,
            multiplier=50,
            margin_requirement=500.0,
        )

        # Initial check: pass
        # Max allowed: $100,000 * 0.80 = $80,000
        # New position: 100 contracts @ $500 = $50,000
        assert checker.check_margin(instrument, 100, 100000.0) is True

        # Track position 1
        checker.track_margin("pos_1", instrument, 100)
        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 50000.0

        # Add position 2
        # Existing: $50,000
        # New: 50 contracts @ $500 = $25,000
        # Total: $75,000 < $80,000 ✓
        assert checker.check_margin(instrument, 50, 100000.0) is True
        checker.track_margin("pos_2", instrument, 50)
        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 75000.0

        # Cannot add more (would exceed)
        # Existing: $75,000
        # New: 10 contracts @ $500 = $5,000
        # Total would be: $80,000 >= $80,000 ✗
        assert checker.check_margin(instrument, 10, 100000.0) is False

        # Release position 1
        checker.release_margin("pos_1")
        util = checker.get_margin_utilization()
        assert util["total_margin_used"] == 25000.0

        # Now can add more
        # Existing: $25,000
        # New: 100 contracts @ $500 = $50,000
        # Total: $75,000 < $80,000 ✓
        assert checker.check_margin(instrument, 100, 100000.0) is True
