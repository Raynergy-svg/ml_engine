"""Tests for trading event schemas and factory functions."""

import dataclasses

from src.scanner.automation.trading_events import (
    PolicyBlockedPayload,
    TradeClosedPayload,
    TradeOpenedPayload,
    TradingEvent,
    TradingEventType,
    create_trade_closed_event,
    validate_event,
)


class TestTradingEventType:
    def test_has_8_members(self):
        assert len(TradingEventType) == 8

    def test_member_values(self):
        expected = {
            "trade_opened", "trade_closed", "trade_modified",
            "policy_blocked", "retrain_requested",
            "degraded_mode_entered", "degraded_mode_exited", "queue_failure",
        }
        assert {m.value for m in TradingEventType} == expected


class TestTradingEvent:
    def test_creation_with_all_fields(self):
        evt = TradingEvent(
            event_id="evt-1",
            event_type=TradingEventType.TRADE_CLOSED,
            timestamp="2026-04-03T00:00:00+00:00",
            source="test",
            session_id="sess-1",
            correlation_id="corr-1",
            payload={"key": "value"},
            payload_version=2,
        )
        assert evt.event_id == "evt-1"
        assert evt.event_type == TradingEventType.TRADE_CLOSED
        assert evt.timestamp == "2026-04-03T00:00:00+00:00"
        assert evt.source == "test"
        assert evt.session_id == "sess-1"
        assert evt.correlation_id == "corr-1"
        assert evt.payload == {"key": "value"}
        assert evt.payload_version == 2

    def test_default_payload_version(self):
        evt = TradingEvent(
            event_id="e", event_type=TradingEventType.QUEUE_FAILURE,
            timestamp="t", source="s", session_id="si",
            correlation_id="c", payload={},
        )
        assert evt.payload_version == 1


class TestTradeClosedPayload:
    def test_has_at_least_18_fields(self):
        fields = dataclasses.fields(TradeClosedPayload)
        assert len(fields) >= 18, f"Expected >=18 fields, got {len(fields)}"

    def test_optional_fields_default_none(self):
        payload = TradeClosedPayload(
            trade_id="t1", pair="EUR_USD", direction="BUY",
            entry_price=1.1, exit_price=1.2, realized_pl=100.0,
            pnl_pips=10.0, trade_won=True, exit_reason="tp",
            duration_minutes=30.0, confidence=0.8, regime="trending",
            agent_reasons={}, model="tcn", analysis_context={},
            close_time="2026-04-03T00:00:00+00:00", sl_pips=5.0, tp_pips=10.0,
        )
        assert payload.weighted_vote_score is None
        assert payload.uncertainty_score is None
        assert payload.model_disagreement is None


class TestTradeOpenedPayload:
    def test_has_10_fields(self):
        fields = dataclasses.fields(TradeOpenedPayload)
        assert len(fields) == 10


class TestPolicyBlockedPayload:
    def test_has_5_fields(self):
        fields = dataclasses.fields(PolicyBlockedPayload)
        assert len(fields) == 5


class TestCreateTradeClosedEvent:
    def test_generates_unique_event_ids(self):
        outcome = {"pair": "EUR_USD", "direction": "BUY"}
        evt1 = create_trade_closed_event(outcome, "sess-1")
        evt2 = create_trade_closed_event(outcome, "sess-1")
        assert evt1.event_id != evt2.event_id

    def test_maps_outcome_data_keys(self):
        outcome = {
            "trade_id": "t-99",
            "pair": "GBP_USD",
            "direction": "SELL",
            "entry_price": 1.25,
            "exit_price": 1.24,
            "realized_pl": 50.0,
            "pnl_pips": 10.0,
            "trade_won": True,
            "exit_reason": "tp_hit",
            "duration_minutes": 45.0,
            "confidence": 0.75,
            "regime": "ranging",
            "agent_reasons": {"trend": 0.8},
            "model": "ridge",
            "analysis_context": {"mtf": True},
            "close_time": "2026-04-03T12:00:00+00:00",
            "sl_pips": 8.0,
            "tp_pips": 16.0,
            "weighted_vote_score": 0.7,
            "uncertainty_score": 0.3,
            "model_disagreement": 0.15,
        }
        evt = create_trade_closed_event(outcome, "sess-2")
        assert evt.event_type == TradingEventType.TRADE_CLOSED
        assert evt.session_id == "sess-2"
        p = evt.payload
        assert p["trade_id"] == "t-99"
        assert p["pair"] == "GBP_USD"
        assert p["direction"] == "SELL"
        assert p["entry_price"] == 1.25
        assert p["realized_pl"] == 50.0
        assert p["weighted_vote_score"] == 0.7
        assert p["uncertainty_score"] == 0.3
        assert p["model_disagreement"] == 0.15


class TestValidateEvent:
    def test_rejects_empty_event_id(self):
        evt = TradingEvent(
            event_id="", event_type=TradingEventType.TRADE_CLOSED,
            timestamp="t", source="s", session_id="si",
            correlation_id="c", payload={},
        )
        assert validate_event(evt) is False

    def test_rejects_empty_event_type_string(self):
        """validate_event checks isinstance(event_type, TradingEventType)."""
        evt = TradingEvent(
            event_id="e", event_type="",  # type: ignore[arg-type]
            timestamp="t", source="s", session_id="si",
            correlation_id="c", payload={},
        )
        assert validate_event(evt) is False

    def test_accepts_valid_event(self):
        evt = TradingEvent(
            event_id="e", event_type=TradingEventType.TRADE_OPENED,
            timestamp="t", source="s", session_id="si",
            correlation_id="c", payload={},
        )
        assert validate_event(evt) is True
