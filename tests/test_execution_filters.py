"""Unit tests for FilterChain (Phase 50: US-310).

Tests filter registration, execution order, short-circuiting, size multiplier
compounding, and exception handling.
"""

import pytest
from src.scanner.execution_filters import FilterChain, FilterResult


class TestFilterChainBasics:
    """Basic FilterChain functionality."""

    def test_empty_chain_passes(self) -> None:
        """Empty filter chain passes everything."""
        chain = FilterChain()
        result = chain.apply({})
        assert result.passed is True
        assert result.action == "PASS"
        assert result.size_multiplier == 1.0

    def test_register_filter(self) -> None:
        """Can register a single filter."""

        def dummy_filter(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="dummy pass")

        chain = FilterChain()
        chain.register("dummy", dummy_filter)
        assert "dummy" in chain.get_filter_names()
        assert len(chain.get_filter_names()) == 1

    def test_get_filter_names(self) -> None:
        """get_filter_names() returns registered names in order."""

        def f1(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="f1")

        def f2(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="f2")

        chain = FilterChain()
        chain.register("first", f1)
        chain.register("second", f2)
        assert chain.get_filter_names() == ["first", "second"]

    def test_clear_filters(self) -> None:
        """clear() removes all registered filters."""

        def f(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="f")

        chain = FilterChain()
        chain.register("f", f)
        assert len(chain.get_filter_names()) == 1
        chain.clear()
        assert len(chain.get_filter_names()) == 0


class TestFilterChainExecution:
    """Filter execution and decision logic."""

    def test_single_pass_filter(self) -> None:
        """Single PASS filter allows trade."""

        def pass_filter(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="all good")

        chain = FilterChain()
        chain.register("pass", pass_filter)
        result = chain.apply({})
        assert result.passed is True
        assert result.action == "PASS"

    def test_block_filter_short_circuits(self) -> None:
        """BLOCK filter short-circuits the chain."""

        def pass1(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="pass1")

        def block(ctx: dict) -> FilterResult:
            return FilterResult(passed=False, action="BLOCK", reason="blocked")

        def pass2(ctx: dict) -> FilterResult:
            # Should not be called
            raise AssertionError("pass2 should not run after BLOCK")

        chain = FilterChain()
        chain.register("pass1", pass1)
        chain.register("block", block)
        chain.register("pass2", pass2)

        result = chain.apply({})
        assert result.passed is False
        assert result.action == "BLOCK"
        assert result.filter_name == "block"

    def test_filters_run_in_order(self) -> None:
        """Filters execute in registration order."""
        execution_order = []

        def f1(ctx: dict) -> FilterResult:
            execution_order.append("f1")
            return FilterResult(passed=True, action="PASS", reason="f1")

        def f2(ctx: dict) -> FilterResult:
            execution_order.append("f2")
            return FilterResult(passed=True, action="PASS", reason="f2")

        def f3(ctx: dict) -> FilterResult:
            execution_order.append("f3")
            return FilterResult(passed=True, action="PASS", reason="f3")

        chain = FilterChain()
        chain.register("f1", f1)
        chain.register("f2", f2)
        chain.register("f3", f3)
        chain.apply({})

        assert execution_order == ["f1", "f2", "f3"]


class TestSizeMultiplier:
    """Size multiplier compounding."""

    def test_single_reduce_size(self) -> None:
        """Single REDUCE_SIZE filter multiplies size."""

        def reduce(ctx: dict) -> FilterResult:
            return FilterResult(
                passed=True, action="REDUCE_SIZE", reason="reduce", size_multiplier=0.8
            )

        chain = FilterChain()
        chain.register("reduce", reduce)
        result = chain.apply({})

        assert result.passed is True
        assert result.action == "REDUCE_SIZE"
        assert result.size_multiplier == pytest.approx(0.8)

    def test_multipliers_compound(self) -> None:
        """Multiple REDUCE_SIZE filters multiply together."""

        def reduce1(ctx: dict) -> FilterResult:
            return FilterResult(
                passed=True, action="REDUCE_SIZE", reason="reduce1", size_multiplier=0.8
            )

        def reduce2(ctx: dict) -> FilterResult:
            return FilterResult(
                passed=True, action="REDUCE_SIZE", reason="reduce2", size_multiplier=0.75
            )

        chain = FilterChain()
        chain.register("reduce1", reduce1)
        chain.register("reduce2", reduce2)
        result = chain.apply({})

        # 0.8 * 0.75 = 0.6
        assert result.size_multiplier == pytest.approx(0.6)
        assert result.action == "REDUCE_SIZE"

    def test_full_size_filter_no_multiply(self) -> None:
        """Filter with size_multiplier=1.0 doesn't reduce."""

        def full(ctx: dict) -> FilterResult:
            return FilterResult(
                passed=True, action="PASS", reason="full", size_multiplier=1.0
            )

        chain = FilterChain()
        chain.register("full", full)
        result = chain.apply({})

        assert result.size_multiplier == 1.0
        assert result.action == "PASS"

    def test_mixed_reduce_and_pass(self) -> None:
        """Mix of PASS and REDUCE_SIZE filters."""

        def pass_f(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="pass")

        def reduce_f(ctx: dict) -> FilterResult:
            return FilterResult(
                passed=True, action="REDUCE_SIZE", reason="reduce", size_multiplier=0.9
            )

        chain = FilterChain()
        chain.register("pass", pass_f)
        chain.register("reduce", reduce_f)
        result = chain.apply({})

        assert result.passed is True
        assert result.action == "REDUCE_SIZE"
        assert result.size_multiplier == pytest.approx(0.9)


class TestExceptionHandling:
    """Exception handling and fail-open behavior."""

    def test_exception_in_filter_is_pass(self) -> None:
        """Exception in a filter is treated as PASS (fail-open)."""

        def failing_filter(ctx: dict) -> FilterResult:
            raise ValueError("Filter bug")

        def normal_filter(ctx: dict) -> FilterResult:
            return FilterResult(passed=True, action="PASS", reason="ok")

        chain = FilterChain()
        chain.register("failing", failing_filter)
        chain.register("normal", normal_filter)

        result = chain.apply({})
        # Should pass because failing filter is treated as PASS
        assert result.passed is True

    def test_exception_in_first_filter(self) -> None:
        """Exception in first filter doesn't stop chain."""

        def failing(ctx: dict) -> FilterResult:
            raise RuntimeError("First filter fails")

        def blocking(ctx: dict) -> FilterResult:
            return FilterResult(passed=False, action="BLOCK", reason="blocked")

        chain = FilterChain()
        chain.register("failing", failing)
        chain.register("blocking", blocking)

        result = chain.apply({})
        # Blocking filter should still run and block
        assert result.passed is False
        assert result.action == "BLOCK"

    def test_exception_after_block(self) -> None:
        """If BLOCK comes first, exception in later filter is never reached."""

        def blocking(ctx: dict) -> FilterResult:
            return FilterResult(passed=False, action="BLOCK", reason="blocked early")

        def would_fail(ctx: dict) -> FilterResult:
            raise RuntimeError("Should not be called")

        chain = FilterChain()
        chain.register("blocking", blocking)
        chain.register("would_fail", would_fail)

        result = chain.apply({})
        assert result.passed is False


class TestFilterResultValidation:
    """FilterResult validation."""

    def test_invalid_action_raises(self) -> None:
        """FilterResult rejects invalid action."""
        with pytest.raises(ValueError, match="Invalid action"):
            FilterResult(passed=True, action="INVALID", reason="test")

    def test_valid_actions(self) -> None:
        """All valid actions accepted."""
        for action in ["PASS", "BLOCK", "REDUCE_SIZE"]:
            result = FilterResult(passed=True, action=action, reason="test")
            assert result.action == action


class TestContextPassthrough:
    """Filter receives context dict."""

    def test_filter_can_read_context(self) -> None:
        """Filter can access and read context."""

        def context_check(ctx: dict) -> FilterResult:
            pair = ctx.get("pair", "UNKNOWN")
            price = ctx.get("price", 0.0)
            if price > 1.5:
                return FilterResult(passed=False, action="BLOCK", reason="price too high")
            return FilterResult(passed=True, action="PASS", reason=f"{pair} ok")

        chain = FilterChain()
        chain.register("context_check", context_check)

        # Test case 1: Normal context
        result = chain.apply({"pair": "EUR_USD", "price": 1.2})
        assert result.passed is True

        # Test case 2: High price
        result = chain.apply({"pair": "EUR_USD", "price": 1.6})
        assert result.passed is False

    def test_filter_tolerates_missing_keys(self) -> None:
        """Filter should handle missing keys gracefully."""

        def safe_filter(ctx: dict) -> FilterResult:
            # Use .get() with defaults
            atr = ctx.get("atr", 0.0)
            if atr == 0:
                return FilterResult(passed=True, action="PASS", reason="no ATR, pass")
            return FilterResult(passed=True, action="PASS", reason="has ATR")

        chain = FilterChain()
        chain.register("safe", safe_filter)

        result = chain.apply({})  # Empty context
        assert result.passed is True


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_realistic_filter_chain(self) -> None:
        """Realistic chain: spread check, calendar check, fitness check."""

        def spread_filter(ctx: dict) -> FilterResult:
            spread = ctx.get("spread_pips", 0.0)
            if spread > 2.0:
                return FilterResult(
                    passed=False, action="BLOCK", reason="spread > 2 pips"
                )
            if spread > 1.5:
                return FilterResult(
                    passed=True,
                    action="REDUCE_SIZE",
                    reason="spread elevated",
                    size_multiplier=0.8,
                )
            return FilterResult(passed=True, action="PASS", reason="spread ok")

        def calendar_filter(ctx: dict) -> FilterResult:
            if ctx.get("in_blackout"):
                return FilterResult(
                    passed=False, action="BLOCK", reason="in economic blackout"
                )
            return FilterResult(passed=True, action="PASS", reason="calendar ok")

        def fitness_filter(ctx: dict) -> FilterResult:
            fitness = ctx.get("pair_fitness", 1.0)
            if fitness < 0.5:
                return FilterResult(
                    passed=True,
                    action="REDUCE_SIZE",
                    reason="low fitness",
                    size_multiplier=0.75,
                )
            return FilterResult(passed=True, action="PASS", reason="fitness ok")

        chain = FilterChain()
        chain.register("spread", spread_filter)
        chain.register("calendar", calendar_filter)
        chain.register("fitness", fitness_filter)

        # Test case 1: Clean trade
        result = chain.apply({
            "spread_pips": 0.5,
            "in_blackout": False,
            "pair_fitness": 0.9,
        })
        assert result.passed is True
        assert result.action == "PASS"
        assert result.size_multiplier == 1.0

        # Test case 2: Elevated spread + low fitness
        result = chain.apply({
            "spread_pips": 1.6,
            "in_blackout": False,
            "pair_fitness": 0.4,
        })
        assert result.passed is True
        assert result.action == "REDUCE_SIZE"
        # 0.8 (spread) * 0.75 (fitness) = 0.6
        assert result.size_multiplier == pytest.approx(0.6)

        # Test case 3: Blocked by calendar
        result = chain.apply({
            "spread_pips": 0.5,
            "in_blackout": True,
            "pair_fitness": 0.9,
        })
        assert result.passed is False
        assert result.action == "BLOCK"
