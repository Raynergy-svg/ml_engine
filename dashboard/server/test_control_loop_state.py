import subprocess
from types import SimpleNamespace

from dashboard.server import control


def test_loop_pids_match_exact_script_arg_not_probe_text(monkeypatch):
    ps_out = "\n".join([
        "101 /opt/python scripts/run_tier7_loop.py",
        "202 /bin/zsh -c ps -axo pid,command | rg 'run_tier7_loop.py'",
        "303 rg run_tier7_loop.py",
        "404 /opt/python -c \"print('scripts/run_tier7_loop.py')\"",
    ])

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=ps_out)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert control._loop_pids("tier7") == [101]


def test_loop_pids_match_macos_framework_python(monkeypatch):
    ps_out = (
        "22421 /Applications/Xcode.app/Contents/Developer/Library/Frameworks/"
        "Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python "
        "/Users/buddy/Documents/ml_engine/scripts/run_tier7_loop.py\n"
    )

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=ps_out)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert control._loop_pids("tier7") == [22421]
