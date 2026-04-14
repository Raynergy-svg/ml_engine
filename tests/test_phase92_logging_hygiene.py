from __future__ import annotations

def test_us001_module_dispatcher_init():
    from pathlib import Path
    src = (Path(__file__).parent.parent).joinpath(*['src', 'scanner', 'automation', 'orchestrator.py']).read_text()
    assert 'self._module_dispatcher = ModuleDispatcher' in src

def test_us001_close_calls_module_dispatcher():
    from src.scanner.automation.orchestrator import Orchestrator
    o = Orchestrator.__new__(Orchestrator)
    called = []
    class Fake:
        def save_state(self): called.append(1)
    for a in ['_state', '_config_adjuster', '_observation_consumer', '_drift_remediator', '_online_rl', '_gate_health', '_agent_health']:
        setattr(o, a, None)
    o._module_dispatcher = Fake()
    o.close()
    assert called, 'close() did not call save_state on _module_dispatcher'

def test_us002_atexit_in_continuous():
    from pathlib import Path
    src = (Path(__file__).parent.parent).joinpath(*['src', 'scanner', 'automation', 'continuous.py']).read_text()
    assert 'import atexit' in src
    assert 'atexit.register(self._save_peak_nav)' in src
    assert 'def _save_peak_nav' in src

def test_us003_warnings_escalated():
    from pathlib import Path
    src = (Path(__file__).parent.parent).joinpath(*['src', 'scanner', 'automation', 'orchestrator.py']).read_text()
    for w in ['Config tuner skipped', 'Online RL update skipped', 'Promotion check skipped']:
        matches = [l for l in src.splitlines() if w in l]
        assert matches
        for l in matches:
            assert 'logger.warning' in l or 'logger.info' in l

def test_us004_no_fstring_warning_loggers():
    from pathlib import Path
    src = (Path(__file__).parent.parent).joinpath(*['src', 'scanner', 'automation', 'orchestrator.py']).read_text()
    bad = [l for l in src.splitlines() if 'logger.warning(f' in l or 'logger.error(f' in l]
    assert not bad, 'f-string warning/error loggers remain: ' + str(len(bad))
