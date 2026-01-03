import sys
import types


def test_launch_buddy_repl_calls_unified_talk(monkeypatch):
    called = {}

    def fake_run_unified_talk(*a, **k):
        called['args'] = a
        called['kwargs'] = k

    class FakeOandaSettings:
        def __init__(self, instrument='USD_JPY', granularity='M5', candles=300, execute=True):
            self.instrument = instrument
            self.granularity = granularity
            self.candles = candles
            self.execute = execute

    fake_mod = types.SimpleNamespace(run_unified_talk=fake_run_unified_talk, OandaSettings=FakeOandaSettings)
    monkeypatch.setitem(sys.modules, 'unified_talk', fake_mod)

    import importlib
    import main
    importlib.reload(main)

    main.launch_buddy_repl_from_wizard(
        'config_tuned.yaml',
        checkpoint_path='ckpt',
        instrument='EUR_USD',
        granularity='M15',
        candles=500,
        execute=False,
        all_features=True,
        verbose=True,
        assistant_name='Buddy',
    )

    assert 'kwargs' in called
    assert called['kwargs'].get('checkpoint_path') == 'ckpt'
    assert called['kwargs'].get('oanda') is True
    assert hasattr(called['kwargs']['oanda_settings'], 'instrument')
    assert called['kwargs'].get('assistant_name') == 'Buddy'
# — Raynergy-svg —
