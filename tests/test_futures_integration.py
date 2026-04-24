"""Futures integration smoke tests.

Covers:
  - InstrumentRegistry: futures symbol lookup, asset-class filtering
  - RollCalendar: active contract, roll date, should_roll
  - futures_position_sizing: calculate_futures_contracts() math + edge cases
  - OandaBroker: constructor overloads (legacy + credential-based + env)
  - IBKRBroker: instantiation without live connection, connect() error path
  - factory.create_broker(): routing to correct client class
  - Routing: FUTURES vs FX instruments dispatched via registry

All tests run fully offline — no live broker required.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ─── Registry & Instrument ────────────────────────────────────────────────────

from src.brokers.instrument import Instrument
from src.brokers.registry import (
    InstrumentRegistry,
    get_default_registry,
    get_registry,
    reset_registry,
)
from src.brokers.roll_calendar import RollCalendar


# ─── Futures Position Sizing ──────────────────────────────────────────────────

from src.risk.futures_position_sizing import (
    MAX_CONTRACTS_PER_TRADE,
    MAX_MARGIN_UTILIZATION,
    calculate_futures_contracts,
    calculate_futures_contracts_from_symbol,
    dollar_risk_per_contract,
    pnl_per_point,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_global_registry():
    """Ensure the global registry singleton is reset between tests."""
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def registry() -> InstrumentRegistry:
    return get_default_registry()


@pytest.fixture
def es() -> Instrument:
    """E-mini S&P 500 — the reference futures instrument for most tests."""
    return get_default_registry().get("ES")


@pytest.fixture
def eur_usd() -> Instrument:
    """EUR/USD FX instrument."""
    return get_default_registry().get("EUR_USD")


@pytest.fixture
def roll_calendar() -> RollCalendar:
    return RollCalendar()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. InstrumentRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstrumentRegistry:

    def test_futures_symbols_present(self, registry):
        """Default registry must contain all six listed futures contracts."""
        for sym in ("ES", "NQ", "CL", "GC", "ZB", "6E"):
            assert registry.contains(sym), f"Missing futures symbol: {sym}"

    def test_fx_symbols_present(self, registry):
        """Default registry must contain major FX pairs."""
        for sym in ("EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"):
            assert registry.contains(sym), f"Missing FX symbol: {sym}"

    def test_is_futures_instrument(self, registry, es, eur_usd):
        assert es.asset_class == "FUTURES"
        assert eur_usd.asset_class == "FX"

    def test_get_by_asset_class_futures(self, registry):
        futures = registry.get_by_asset_class("FUTURES")
        syms = {i.symbol for i in futures}
        assert "ES" in syms
        assert "EUR_USD" not in syms

    def test_get_by_asset_class_fx(self, registry):
        fx = registry.get_by_asset_class("FX")
        syms = {i.symbol for i in fx}
        assert "EUR_USD" in syms
        assert "ES" not in syms

    def test_missing_symbol_raises_key_error(self, registry):
        with pytest.raises(KeyError):
            registry.get("DOES_NOT_EXIST")

    def test_es_tick_specs(self, es):
        """ES tick specs must match CME official values."""
        assert es.tick_size == 0.25
        assert es.tick_value == 12.50
        assert es.multiplier == 50.0
        assert es.exchange == "CME"

    def test_nq_tick_specs(self, registry):
        nq = registry.get("NQ")
        assert nq.tick_size == 0.25
        assert nq.tick_value == 5.00
        assert nq.multiplier == 20.0

    def test_cl_tick_specs(self, registry):
        cl = registry.get("CL")
        assert cl.tick_size == 0.01
        assert cl.tick_value == 10.00
        assert cl.exchange == "NYMEX"

    def test_singleton_reuse(self):
        """get_registry() must return the same object on repeated calls."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RollCalendar
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollCalendar:

    def test_active_contract_es_returns_yyyymm(self, roll_calendar):
        contract = roll_calendar.get_active_contract("ES")
        assert len(contract) == 6
        assert contract.isdigit()

    def test_active_contract_cl_monthly(self, roll_calendar):
        """CL is a monthly contract — active contract should not be empty."""
        contract = roll_calendar.get_active_contract("CL")
        assert len(contract) == 6

    def test_roll_date_is_before_expiry(self, roll_calendar):
        """Roll date must be strictly before expiry date."""
        contract = roll_calendar.get_active_contract("ES")
        roll_date = roll_calendar.get_roll_date("ES", contract)
        expiry = roll_calendar._get_expiration_date("ES", contract)
        assert roll_date < expiry

    def test_unsupported_symbol_raises(self, roll_calendar):
        with pytest.raises(ValueError):
            roll_calendar.get_active_contract("XXXYYY")

    def test_days_until_roll_int(self, roll_calendar):
        days = roll_calendar.days_until_roll("ES")
        assert isinstance(days, int)

    def test_should_roll_bool(self, roll_calendar):
        result = roll_calendar.should_roll("GC")
        assert isinstance(result, bool)

    def test_get_third_friday_is_friday(self):
        from src.brokers.roll_calendar import get_third_friday
        fri = get_third_friday(2025, 3)  # March 2025
        assert fri.weekday() == 4  # Friday=4

    def test_quarterly_months_march(self, roll_calendar):
        """A date in February must resolve to the March quarterly contract."""
        feb_date = date(2025, 2, 10)
        contract = roll_calendar.get_active_contract("ES", feb_date)
        # Contract year must be 2025 and month >= 3 (March, June, Sep, Dec)
        year = int(contract[:4])
        month = int(contract[4:])
        assert year == 2025
        assert month in (3, 6, 9, 12)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Futures Position Sizing — Core Math
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateFuturesContracts:
    """Unit tests for the position sizing formula.

    Reference calculation for ES (tick_size=0.25, tick_value=$12.50):
        $100,000 NAV, 2 % risk → dollar_risk = $2,000
        SL distance = 10.0 pts → ticks_at_risk = 10.0 / 0.25 = 40
        risk_per_contract = 40 * $12.50 = $500
        raw_contracts = $2,000 / $500 = 4 → 4 contracts
    """

    def test_es_reference_calculation(self, es):
        """Known-value test: 2 % of $100k NAV, 10-point ES stop → 4 contracts."""
        contracts = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,  # 10-point stop
            instrument=es,
        )
        assert contracts == 4

    def test_nq_known_value(self, registry):
        """NQ (tick_size=0.25, tick_value=$5): 2 % of $100k, 20-pt stop → 5 contracts.

        dollar_risk = 2000
        ticks = 20/0.25 = 80
        risk_per_contract = 80 * 5 = $400
        raw = 2000 / 400 = 5.0 → 5
        """
        nq = registry.get("NQ")
        contracts = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=15000.0,
            stop_price=14980.0,  # 20-point stop
            instrument=nq,
        )
        assert contracts == 5

    def test_max_contracts_cap_applied(self, es):
        """Very large NAV + small stop must be capped at MAX_CONTRACTS_PER_TRADE."""
        contracts = calculate_futures_contracts(
            nav=10_000_000,
            risk_pct=0.05,
            entry_price=4500.0,
            stop_price=4499.25,  # just 3 ticks
            instrument=es,
        )
        assert contracts == MAX_CONTRACTS_PER_TRADE

    def test_custom_max_contracts_respected(self, es):
        contracts = calculate_futures_contracts(
            nav=10_000_000,
            risk_pct=0.05,
            entry_price=4500.0,
            stop_price=4499.25,
            instrument=es,
            max_contracts=2,
        )
        assert contracts == 2

    def test_margin_cap_applied(self, es):
        """If current_margin_used exhaust budget, returns 0."""
        # 50 % NAV = $50k margin budget. ES margin_requirement = $10 per contract
        # (from registry: margin_requirement=10.0 means $10 — paper placeholder)
        # current_margin_used = $50k → budget = 0 → 0 contracts
        contracts = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,
            instrument=es,
            current_margin_used=50_000,  # fully consumed
        )
        assert contracts == 0

    def test_zero_sl_distance_returns_zero(self, es):
        """Entry == stop is degenerate — should return 0 without raising."""
        contracts = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4500.0,   # no distance
            instrument=es,
        )
        assert contracts == 0

    def test_zero_nav_returns_zero(self, es):
        contracts = calculate_futures_contracts(
            nav=0,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,
            instrument=es,
        )
        assert contracts == 0

    def test_tiny_nav_may_return_zero(self, es):
        """$1,000 NAV at 2 % risk → $20 risk → cannot afford even 1 contract."""
        contracts = calculate_futures_contracts(
            nav=1_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,  # 10-pt stop → $500/contract risk
            instrument=es,
        )
        assert contracts == 0  # $20 < $500 per contract

    def test_raises_for_fx_instrument(self, eur_usd):
        """Must reject non-FUTURES instruments with ValueError."""
        with pytest.raises(ValueError, match="FUTURES"):
            calculate_futures_contracts(
                nav=100_000,
                risk_pct=0.02,
                entry_price=1.10,
                stop_price=1.09,
                instrument=eur_usd,
            )

    def test_floor_rounds_down(self, es):
        """Sizing must floor (not round) the raw contract count."""
        # 2 % of $100k = $2000. ES 10-pt stop → $500/contract → raw = 4.0 exactly.
        # Tweak to get raw = 3.99 (make stop 1 tick wider → 10.25 pts):
        contracts = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4489.75,   # 10.25 pts → $512.50/contract → raw ≈ 3.90
            instrument=es,
        )
        assert contracts == 3  # floor(3.90) = 3

    def test_from_symbol_convenience_wrapper(self):
        """calculate_futures_contracts_from_symbol must match direct call."""
        direct = calculate_futures_contracts(
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,
            instrument=get_default_registry().get("ES"),
        )
        via_symbol = calculate_futures_contracts_from_symbol(
            symbol="ES",
            nav=100_000,
            risk_pct=0.02,
            entry_price=4500.0,
            stop_price=4490.0,
        )
        assert direct == via_symbol

    def test_from_symbol_unknown_raises(self):
        with pytest.raises(KeyError):
            calculate_futures_contracts_from_symbol(
                symbol="AAPL",  # not a futures instrument in registry
                nav=100_000,
                risk_pct=0.02,
                entry_price=150.0,
                stop_price=148.0,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Helper functions: dollar_risk_per_contract + pnl_per_point
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuturesHelpers:

    def test_dollar_risk_per_contract_es_10pt(self, es):
        """10-point ES stop → 40 ticks → $500."""
        risk = dollar_risk_per_contract(es, sl_distance=10.0)
        assert risk == pytest.approx(500.0)

    def test_dollar_risk_per_contract_cl_1dollar(self, registry):
        """$1 CL stop → 100 ticks → $1,000 (CL tick_size=0.01, tick_value=$10)."""
        cl = registry.get("CL")
        risk = dollar_risk_per_contract(cl, sl_distance=1.0)
        assert risk == pytest.approx(1000.0)

    def test_pnl_per_point_es(self, es):
        """ES: 1 pt = 4 ticks × $12.50 = $50."""
        assert pnl_per_point(es) == pytest.approx(50.0)

    def test_pnl_per_point_nq(self, registry):
        """NQ: 1 pt = 4 ticks × $5 = $20."""
        nq = registry.get("NQ")
        assert pnl_per_point(nq) == pytest.approx(20.0)

    def test_pnl_per_point_raises_for_fx(self, eur_usd):
        with pytest.raises(ValueError):
            pnl_per_point(eur_usd)

    def test_dollar_risk_raises_for_fx(self, eur_usd):
        with pytest.raises(ValueError):
            dollar_risk_per_contract(eur_usd, sl_distance=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. OandaBroker constructor overloads
# ═══════════════════════════════════════════════════════════════════════════════

class TestOandaBrokerConstructor:

    def test_from_explicit_client(self):
        """Passing a pre-built OandaPracticeClient must be accepted."""
        from src.brokers.oanda import OandaBroker
        from src.utils.oanda_practice import OandaPracticeClient, OandaPracticeConfig

        cfg = OandaPracticeConfig(api_token="tok", account_id="acc-123")
        raw_client = OandaPracticeClient(cfg)
        broker = OandaBroker(client=raw_client)
        assert broker._client is raw_client

    def test_from_account_id_and_api_key(self):
        """Passing account_id + api_key builds OandaPracticeClient internally."""
        from src.brokers.oanda import OandaBroker
        broker = OandaBroker(account_id="001-001-999", api_key="dummy-token-xyz")
        # Should have created an OandaPracticeClient internally
        from src.utils.oanda_practice import OandaPracticeClient
        assert isinstance(broker._client, OandaPracticeClient)
        assert broker._client._config.account_id == "001-001-999"
        assert broker._client._config.api_token == "dummy-token-xyz"

    def test_from_env_raises_on_missing(self, monkeypatch):
        """No client + no kwargs + no env vars → OSError."""
        from src.brokers.oanda import OandaBroker
        monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
        monkeypatch.delenv("OANDA_API_KEY", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
        with pytest.raises(OSError):
            OandaBroker()

    def test_connect_and_disconnect_are_noops(self):
        """connect()/disconnect() are REST no-ops — must not raise."""
        from src.brokers.oanda import OandaBroker
        from src.utils.oanda_practice import OandaPracticeClient, OandaPracticeConfig

        cfg = OandaPracticeConfig(api_token="tok", account_id="acc-123")
        broker = OandaBroker(client=OandaPracticeClient(cfg))
        broker.connect()
        broker.disconnect()  # no exception


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IBKRBroker — offline instantiation & connect error path
# ═══════════════════════════════════════════════════════════════════════════════

class TestIBKRBrokerOffline:

    def test_instantiation_without_connection(self):
        """IBKRBroker can be created without connecting."""
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=99)
        assert broker.host == "127.0.0.1"
        assert broker.port == 7497
        assert broker.client_id == 99
        assert not broker._connected

    def test_connect_raises_connection_error_when_gateway_down(self):
        """connect() must raise ConnectionError if TWS is not reachable."""
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker(host="127.0.0.1", port=19999, client_id=99)  # unbound port
        with pytest.raises((ConnectionError, ImportError)):
            # ConnectionError if ib_insync installed; ImportError if not
            broker.connect()

    def test_is_connected_false_before_connect(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        assert not broker.is_connected

    def test_disconnect_before_connect_is_safe(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        broker.disconnect()  # must not raise

    def test_get_nav_returns_zero_when_not_connected(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        assert broker.get_nav() == 0.0

    def test_get_open_positions_returns_empty_when_not_connected(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        assert broker.get_open_positions() == []

    def test_get_trades_returns_empty_when_not_connected(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        assert broker.get_trades() == []

    def test_get_account_summary_returns_zeros_when_not_connected(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        summary = broker.get_account_summary()
        assert summary.nav == 0.0
        assert summary.balance == 0.0

    def test_place_order_returns_rejected_when_not_connected(self, es):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        result = broker.place_order(
            instrument=es,
            direction="BUY",
            quantity=1,
            sl_price=4450.0,
            tp_price=4600.0,
            entry_price=4500.0,
        )
        assert result.status == "rejected"
        assert "Not connected" in (result.error_message or "")

    def test_close_trade_raises_when_not_connected(self):
        from src.brokers.ibkr import IBKRBroker
        broker = IBKRBroker()
        with pytest.raises(ConnectionError):
            broker.close_trade("12345")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. factory.create_broker() routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrokerFactory:

    def test_unsupported_type_raises(self):
        from src.brokers.factory import create_broker
        with pytest.raises(ValueError, match="Unsupported broker type"):
            create_broker(broker_type="interactive_brokers_pro")

    def test_oanda_missing_credentials_raises(self):
        from src.brokers.factory import create_broker
        with pytest.raises(ValueError, match="oanda_account_id"):
            create_broker(broker_type="oanda")  # no credentials

    def test_oanda_returns_oanda_broker(self):
        """create_broker('oanda') must return an OandaBroker instance."""
        from src.brokers.factory import create_broker
        from src.brokers.oanda import OandaBroker

        broker = create_broker(
            broker_type="oanda",
            oanda_account_id="001-001-test",
            oanda_api_key="test-token",
        )
        assert isinstance(broker, OandaBroker)

    def test_ibkr_returns_ibkr_broker_instance(self):
        """create_broker('ibkr') must attempt to connect; test the type before connect."""
        from src.brokers.factory import create_broker
        from src.brokers.ibkr import IBKRBroker

        # Patch IBKRBroker.connect so we don't need a live gateway
        with patch.object(IBKRBroker, "connect", return_value=None):
            broker = create_broker(
                broker_type="ibkr",
                ibkr_host="127.0.0.1",
                ibkr_port=7497,
                ibkr_client_id=1,
            )
        assert isinstance(broker, IBKRBroker)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Instrument routing by asset class (simulates execution dispatch logic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstrumentRouting:
    """Simulate the routing logic in execution._init_broker() / ExecutionManager.

    We don't need a live broker — we verify the registry correctly classifies
    instruments so the routing conditions work.
    """

    def test_fx_pairs_route_to_oanda(self, registry):
        """All default FX pairs should NOT be in the futures universe."""
        fx_syms = [i.symbol for i in registry.get_by_asset_class("FX")]
        fut_syms = {i.symbol for i in registry.get_by_asset_class("FUTURES")}
        for sym in fx_syms:
            assert sym not in fut_syms, f"{sym} found in both FX and FUTURES"

    def test_futures_route_to_ibkr(self, registry):
        """All futures instruments should NOT be in the FX universe."""
        fut_syms = [i.symbol for i in registry.get_by_asset_class("FUTURES")]
        fx_syms = {i.symbol for i in registry.get_by_asset_class("FX")}
        for sym in fut_syms:
            assert sym not in fx_syms, f"{sym} found in both FUTURES and FX"

    def test_es_dispatches_to_futures_broker(self, registry):
        """Registry lookup for ES must return a FUTURES instrument."""
        inst = registry.get("ES")
        assert inst.asset_class == "FUTURES"
        # Simulate the routing condition used in execution.py:
        target_broker = "ibkr" if inst.asset_class == "FUTURES" else "oanda"
        assert target_broker == "ibkr"

    def test_eur_usd_dispatches_to_oanda(self, registry):
        inst = registry.get("EUR_USD")
        assert inst.asset_class == "FX"
        target_broker = "ibkr" if inst.asset_class == "FUTURES" else "oanda"
        assert target_broker == "oanda"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Instrument dataclass validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestInstrumentValidation:

    def test_fx_requires_pip_value(self):
        with pytest.raises(ValueError, match="pip_value"):
            Instrument(
                symbol="EUR_USD",
                broker_symbol="EUR_USD",
                asset_class="FX",
                price_precision=5,
                margin_requirement=1.0,
                exchange="OANDA",
                currency="USD",
                pip_value=None,  # missing!
            )

    def test_futures_requires_tick_size(self):
        with pytest.raises(ValueError, match="tick_size"):
            Instrument(
                symbol="ES",
                broker_symbol="ES",
                asset_class="FUTURES",
                price_precision=2,
                margin_requirement=10.0,
                exchange="CME",
                currency="USD",
                tick_size=None,   # missing!
                tick_value=12.50,
                multiplier=50.0,
            )

    def test_invalid_asset_class_raises(self):
        with pytest.raises(ValueError, match="asset_class"):
            Instrument(
                symbol="XX",
                broker_symbol="XX",
                asset_class="CRYPTO",  # type: ignore
                price_precision=2,
                margin_requirement=1.0,
                exchange="CBOE",
                currency="USD",
            )

    def test_fx_factory_method(self):
        inst = Instrument.fx("EUR_USD", "EUR_USD", pip_value=0.0001)
        assert inst.asset_class == "FX"
        assert inst.pip_value == 0.0001

    def test_futures_factory_method(self):
        inst = Instrument.futures(
            "ES", "ES",
            tick_size=0.25, tick_value=12.50, multiplier=50.0
        )
        assert inst.asset_class == "FUTURES"
        assert inst.multiplier == 50.0
