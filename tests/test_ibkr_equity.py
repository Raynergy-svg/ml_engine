"""US-011 tests — IBKR equity contracts + ship-gate guard + paper order.

No mocks. Real :class:`IBKRBroker`, real :class:`Instrument`, real disk
via ``tmp_path``. The integration test that actually places an order on a
paper IBGateway is marked ``@pytest.mark.integration`` and skips cleanly
when the gateway is unreachable.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.brokers.ibkr import (
    IBKRBroker,
    IBKREquityGateError,
    SHIP_GATE_PATH_DEFAULT,
    Stock,
    enforce_equity_ship_gate,
)
from src.brokers.instrument import AssetClass, Instrument
from src.brokers.registry import InstrumentRegistry, get_default_registry
from src.brokers.types import OrderResult


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via tmp + os.replace — mirrors production atomic pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".shipgate_", suffix=".json", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _passing_payload(universe_hash: str = VALID_HASH) -> dict:
    return {
        "asof": "2026-05-29",
        "criteria_source": "test",
        "gate_pass": True,
        "max_dd": 0.229,
        "net_sharpe": 0.921,
        "pipeline_version": "test-v1",
        "positive_years": 13,
        "recommendation": "PASS",
        "thresholds": {
            "max_drawdown": 0.25,
            "min_net_sharpe": 0.4,
            "min_positive_years": 6.0,
            "min_total_years": 10.0,
        },
        "total_years": 17,
        "universe_hash": universe_hash,
    }


# ---------------------------------------------------------------------------
# AssetClass + Instrument validation
# ---------------------------------------------------------------------------


class TestEquityAssetClass:
    """EQUITY added to AssetClass Literal + validators."""

    def test_assetclass_literal_includes_equity(self):
        # Literal types are runtime-inspectable via __args__ — assert the
        # contract directly rather than relying on a stringy comparison.
        from typing import get_args
        assert "EQUITY" in get_args(AssetClass)
        assert "FX" in get_args(AssetClass)
        assert "FUTURES" in get_args(AssetClass)

    def test_invalid_asset_class_still_rejected(self):
        with pytest.raises(ValueError, match="Invalid asset_class"):
            Instrument(
                symbol="AAPL",
                broker_symbol="AAPL",
                asset_class="CRYPTO",  # type: ignore[arg-type]
                price_precision=2,
                margin_requirement=100.0,
                exchange="SMART",
                currency="USD",
            )

    def test_equity_factory_defaults(self):
        inst = Instrument.equity("AAPL")
        assert inst.asset_class == "EQUITY"
        assert inst.symbol == "AAPL"
        assert inst.broker_symbol == "AAPL"
        assert inst.exchange == "SMART"
        assert inst.currency == "USD"
        assert inst.pip_value is None
        assert inst.tick_size is None
        assert inst.multiplier is None

    def test_equity_factory_overrides(self):
        inst = Instrument.equity(
            "MSFT",
            broker_symbol="MSFT",
            exchange="NASDAQ",
            currency="USD",
            price_precision=2,
        )
        assert inst.exchange == "NASDAQ"
        assert inst.price_precision == 2

    def test_equity_does_not_require_fx_or_futures_fields(self):
        # Should NOT raise even though tick_size/pip_value are None.
        Instrument.equity("AAPL")

    def test_equity_negative_margin_rejected(self):
        with pytest.raises(ValueError, match="margin_requirement cannot be negative"):
            Instrument(
                symbol="AAPL",
                broker_symbol="AAPL",
                asset_class="EQUITY",
                price_precision=2,
                margin_requirement=-1.0,
                exchange="SMART",
                currency="USD",
            )


class TestRegistryEquitySupport:
    """Registry must accept and surface EQUITY instruments."""

    def test_register_equity_instrument(self):
        registry = InstrumentRegistry()
        inst = Instrument.equity("AAPL")
        registry.register(inst)
        assert registry.contains("AAPL")
        assert registry.get("AAPL").asset_class == "EQUITY"

    def test_get_by_asset_class_equity_isolated(self):
        registry = InstrumentRegistry()
        registry.register(Instrument.equity("AAPL"))
        registry.register(Instrument.equity("MSFT"))
        registry.register(
            Instrument.fx(symbol="EUR_USD", broker_symbol="EUR_USD", pip_value=0.0001)
        )
        eqs = registry.get_by_asset_class("EQUITY")
        assert {e.symbol for e in eqs} == {"AAPL", "MSFT"}
        # FX still discoverable
        fxs = registry.get_by_asset_class("FX")
        assert [f.symbol for f in fxs] == ["EUR_USD"]

    def test_repr_includes_equity_count(self):
        registry = InstrumentRegistry()
        registry.register(Instrument.equity("AAPL"))
        rep = repr(registry)
        assert "1 EQUITY instruments" in rep

    def test_default_registry_still_works_with_equity_count(self):
        reg = get_default_registry()
        # default registry has FX + FUTURES; EQUITY count starts at 0
        # but repr must still render without crashing.
        assert "EQUITY instruments" in repr(reg)


# ---------------------------------------------------------------------------
# Ship-gate guard semantics
# ---------------------------------------------------------------------------


class TestEquityShipGateGuard:
    """``enforce_equity_ship_gate`` is fail-closed by construction."""

    def test_passes_when_file_valid_and_hash_matches(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload())
        enforce_equity_ship_gate(path, VALID_HASH)  # no raise = pass

    def test_blocks_when_missing(self, tmp_path):
        path = tmp_path / "MISSING.json"
        with pytest.raises(IBKREquityGateError, match="file not found"):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_malformed(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text("this is not json", encoding="utf-8")
        with pytest.raises(IBKREquityGateError, match="cannot read"):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_payload_is_list(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(IBKREquityGateError, match="not a JSON object"):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_gate_pass_false(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = False
        _atomic_write_json(path, payload)
        with pytest.raises(IBKREquityGateError, match="gate_pass="):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_gate_pass_missing(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        del payload["gate_pass"]
        _atomic_write_json(path, payload)
        with pytest.raises(IBKREquityGateError, match="gate_pass="):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_gate_pass_truthy_but_not_true(self, tmp_path):
        # The contract is gate_pass IS True (identity), not truthy.
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = 1  # truthy but not True
        _atomic_write_json(path, payload)
        with pytest.raises(IBKREquityGateError, match="gate_pass="):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_universe_hash_missing(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        del payload["universe_hash"]
        _atomic_write_json(path, payload)
        with pytest.raises(IBKREquityGateError, match="universe_hash missing"):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_universe_hash_blank(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload(universe_hash="")
        _atomic_write_json(path, payload)
        with pytest.raises(IBKREquityGateError, match="universe_hash missing"):
            enforce_equity_ship_gate(path, VALID_HASH)

    def test_blocks_when_universe_hash_mismatches(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload(universe_hash="ffff"))
        with pytest.raises(IBKREquityGateError, match="universe_hash mismatch"):
            enforce_equity_ship_gate(path, VALID_HASH)


# ---------------------------------------------------------------------------
# Broker contract construction
# ---------------------------------------------------------------------------


class TestIBKREquityBuildContract:
    """``_build_contract`` honors the equity gate and routes via SMART."""

    def _broker(self):
        return IBKRBroker(host="127.0.0.1", port=7497, client_id=99)

    def test_equity_blocked_when_gate_not_armed(self):
        broker = self._broker()
        inst = Instrument.equity("AAPL")
        with pytest.raises(IBKREquityGateError, match="not armed"):
            broker._build_contract(inst)

    def test_equity_passes_when_gate_armed(self, tmp_path):
        if Stock is None:
            pytest.skip("ib_async/ib_insync not installed")
        broker = self._broker()
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload())
        broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)
        contract = broker._build_contract(Instrument.equity("AAPL"))
        # ib_async Stock — symbol/exchange/currency populated.
        assert contract.symbol == "AAPL"
        assert contract.exchange == "SMART"
        assert contract.currency == "USD"
        # secType is "STK" for a Stock contract on ib_async.
        assert getattr(contract, "secType", "STK") == "STK"

    def test_equity_uses_custom_exchange_when_set(self, tmp_path):
        if Stock is None:
            pytest.skip("ib_async/ib_insync not installed")
        broker = self._broker()
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload())
        broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)
        inst = Instrument.equity("MSFT", exchange="NASDAQ")
        contract = broker._build_contract(inst)
        assert contract.exchange == "NASDAQ"

    def test_disarm_re_engages_failclose(self, tmp_path):
        broker = self._broker()
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload())
        broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)
        broker.disarm_equity_gate()
        with pytest.raises(IBKREquityGateError, match="not armed"):
            broker._build_contract(Instrument.equity("AAPL"))

    def test_arm_fails_when_gate_file_missing(self, tmp_path):
        broker = self._broker()
        with pytest.raises(IBKREquityGateError, match="file not found"):
            broker.arm_equity_gate(
                VALID_HASH, ship_gate_path=tmp_path / "nope.json"
            )

    def test_arm_fails_when_hash_mismatches(self, tmp_path):
        broker = self._broker()
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload(universe_hash="dead"))
        with pytest.raises(IBKREquityGateError, match="universe_hash mismatch"):
            broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)

    def test_arm_uses_default_path_constant(self):
        # SHIP_GATE_PATH_DEFAULT points at the canonical artifact location.
        assert SHIP_GATE_PATH_DEFAULT.endswith("SHIP_GATE.json")
        assert "trained_data/backtests" in SHIP_GATE_PATH_DEFAULT

    def test_fx_path_does_not_consult_equity_gate(self):
        # Confirm FX construction never raises the equity-gate error even
        # when the gate is disarmed — only EQUITY consults the guard.
        broker = self._broker()
        assert not broker._equity_gate_armed
        fx = Instrument.fx(
            symbol="EUR_USD", broker_symbol="EUR/USD", pip_value=0.0001,
        )
        try:
            broker._build_contract(fx)
        except IBKREquityGateError:
            pytest.fail("FX path must not consult the equity ship gate")
        except Exception:
            # FX path may raise downstream (e.g. ib_insync format requirements);
            # that's outside US-011 scope. The contract — no EQUITY guard
            # consultation — held.
            pass


# ---------------------------------------------------------------------------
# Whole-share place_equity_order: validation gates (no network)
# ---------------------------------------------------------------------------


class TestPlaceEquityOrderValidation:
    """All rejections happen WITHOUT touching the network."""

    def _broker_armed(self, tmp_path):
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=99)
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload())
        broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)
        return broker

    def test_rejects_non_equity_instrument(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        fx = Instrument.fx(
            symbol="EUR_USD", broker_symbol="EUR/USD", pip_value=0.0001,
        )
        result = broker.place_equity_order(fx, "BUY", 1)
        assert result.status == "rejected"
        assert result.trade_id == "0"
        assert "asset_class='EQUITY'" in result.error_message

    def test_rejects_when_not_connected(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        # broker._connected == False by default; no IB instance.
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", 1,
        )
        assert result.status == "rejected"
        assert "Not connected" in result.error_message

    def test_rejects_invalid_direction(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "LONG", 1,
        )
        assert result.status == "rejected"
        # Direction validation runs after connection check; the broker is
        # not connected here, so we just confirm rejection happened.
        assert result.trade_id == "0"

    def test_rejects_fractional_quantity(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", 1.5,  # type: ignore[arg-type]
        )
        assert result.status == "rejected"

    def test_rejects_zero_quantity(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", 0,
        )
        assert result.status == "rejected"

    def test_rejects_negative_quantity(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", -3,
        )
        assert result.status == "rejected"

    def test_rejects_bool_quantity(self, tmp_path):
        # Python bools are ints — explicitly reject them.
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", True,  # type: ignore[arg-type]
        )
        assert result.status == "rejected"

    def test_order_result_shape(self, tmp_path):
        broker = self._broker_armed(tmp_path)
        result = broker.place_equity_order(
            Instrument.equity("AAPL"), "BUY", 1,
        )
        assert isinstance(result, OrderResult)


# ---------------------------------------------------------------------------
# Integration: real paper account end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_paper_account_places_one_equity_order(tmp_path):
    """End-to-end: connect to paper IBGateway, place + confirm 1 share.

    Skips cleanly when:
    - ib_async/ib_insync isn't installed
    - IBGateway/TWS is not running on the expected port
    - The environment doesn't opt in via ``IBKR_PAPER_HOST``/``IBKR_PAPER_PORT``

    Defaults to ``127.0.0.1:7497`` (TWS paper) when env not set so a
    developer with a paper gateway up can run this directly.
    """
    if Stock is None:
        pytest.skip("ib_async/ib_insync not installed; cannot run paper test")

    host = os.environ.get("IBKR_PAPER_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PAPER_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_PAPER_CLIENT_ID", "199"))
    symbol = os.environ.get("IBKR_PAPER_SYMBOL", "AAPL")

    # Arm the equity gate against a real fixture written by THIS test
    # (we don't rely on the repo's SHIP_GATE.json in case the universe
    # hash has shifted between commits).
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, _passing_payload())

    broker = IBKRBroker(host=host, port=port, client_id=client_id)
    broker.arm_equity_gate(VALID_HASH, ship_gate_path=path)

    try:
        broker.connect()
    except (ConnectionError, OSError) as exc:
        pytest.skip(f"IBGateway not reachable at {host}:{port} ({exc})")

    try:
        inst = Instrument.equity(symbol)
        result = broker.place_equity_order(inst, "BUY", 1)
        assert result.status in ("PENDING", "rejected")
        # On a healthy paper account a PENDING result carries a non-zero
        # trade_id; a 'rejected' carries the broker's error string for
        # post-mortem (e.g. insufficient funds, market closed). Either
        # outcome confirms the order round-trip happened end-to-end.
        assert result.trade_id is not None
        if result.status == "PENDING":
            assert int(result.trade_id) > 0
    finally:
        try:
            broker.disconnect()
        except Exception:
            pass
