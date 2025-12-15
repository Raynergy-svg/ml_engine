import unified_talk


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


def test_trade_auto_dry_run_uses_last_prediction(capsys):
    ctx = _ctx()
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
    ctx = _ctx()
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
    assert ctx.oanda_client.orders == [("EUR_USD", -100000)]


def test_manual_trade_buy_dry_run(capsys):
    ctx = _ctx()

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
    ctx = _ctx()

    assert unified_talk._handle_talk_command(ctx, "trade close EUR_USD", period="5d", interval="1h") is True
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "close" in out.lower()
    assert ctx.oanda_client.closes == []
