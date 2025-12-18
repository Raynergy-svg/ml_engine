import unified_talk


def _dummy_fx_df():
    import pandas as pd

    base = 1.1000
    rows = []
    for i in range(20):
        c = base + (i * 0.00001)
        rows.append(
            {
                "open": c,
                "high": c + 0.00005,
                "low": c - 0.00005,
                "close": c,
                "volume": 100.0,
                "bid_close": c,
                "ask_close": c + 0.00002,  # 0.2 pips on non-JPY pairs
            }
        )
    return pd.DataFrame(rows)


class _StubOandaClient:
    def __init__(self):
        self.orders = []
        self.closes = []

    def create_market_order(self, *, instrument: str, units: int, **kwargs):
        self.orders.append((instrument, int(units)))
        # mimic OANDA response shape
        return {"orderCreateTransaction": {"id": "T-1"}}

    def close_position(self, *, instrument: str, **kwargs):
        self.closes.append(str(instrument))
        return {"closePositionTransaction": {"id": "C-1"}}


def _ctx():
    # We avoid loading checkpoints by constructing a minimal TalkContext.
    return unified_talk.TalkContext(
        config_path="config.yaml",
        checkpoint_path="/dev/null",
        feature_columns=["open", "high", "low", "close", "volume"],
        sequence_length=60,
        target_shift=1,
        engine=None,  # type: ignore[arg-type]
        reasoning=None,  # type: ignore[arg-type]
        oanda_client=_StubOandaClient(),
        oanda_instrument="EUR_USD",
        oanda_execute=False,
        assistant_name="Buddy",
        verbose=False,
    )


def _prime_fx_context(ctx):
    # Ensure Tier-1 session gating is deterministic for tests.
    # 14:00 UTC on Dec 16 == 09:00 America/New_York.
    from datetime import datetime, timezone

    ctx.fx_now = datetime(2025, 12, 16, 14, 0, tzinfo=timezone.utc)
    ctx.active_df = _dummy_fx_df()
    ctx.active_source = "oanda:EUR_USD"
    return ctx


def test_trade_auto_dry_run_uses_last_prediction(capsys):
    ctx = _prime_fx_context(_ctx())
    # Provide state_probs + risk so confidence gating can evaluate.
    ctx.last_result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "buy" in out.lower()
    assert ctx.oanda_client.orders == []


def test_trade_auto_execute_places_order(capsys):
    ctx = _prime_fx_context(_ctx())
    ctx.oanda_execute = True
    ctx.last_result = {
        "prediction": 1.0000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "auto-trade executed" in out.lower()
    expected_units = -unified_talk._default_trade_units(ctx, "EUR_USD")
    assert ctx.oanda_client.orders == [("EUR_USD", expected_units)]


def test_manual_trade_buy_dry_run(capsys):
    ctx = _prime_fx_context(_ctx())

    # Manual trades also require a last prediction (for confidence checks).
    ctx.last_result = {
        "prediction": 1.2000,
        "last_close": 1.1000,
        "state_probs": [0.05, 0.05, 0.90],
        "risk": 0.10,
    }

    assert unified_talk._handle_talk_command(ctx, "trade buy 2000 EUR_USD", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "market order" in out.lower()
    assert ctx.oanda_client.orders == []


def test_manual_trade_close_dry_run(capsys):
    ctx = _prime_fx_context(_ctx())

    assert unified_talk._handle_talk_command(ctx, "trade close EUR_USD", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "close" in out.lower()
    assert ctx.oanda_client.closes == []
