"""Import/runtime separation is not a substitute for OS credential isolation."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]


def test_portfolio_evaluates_without_training_broker_or_external_side_effects():
    code = r"""
import json, runpy, sys
sys.path.insert(0, sys.argv[1])
sys.dont_write_bytecode = True
blocked = []
def audit(event, args):
    if event in {'socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system', 'os.posix_spawn'}:
        blocked.append(event)
        raise RuntimeError('forbidden external side effect')
sys.addaudithook(audit)
from src.axiom2.portfolio.authority import evaluate_portfolio
fixture = runpy.run_path(sys.argv[1] + '/tests/axiom2/portfolio/test_portfolio_authority.py')
_, request, policy = fixture['case']()
result = evaluate_portfolio(request, policy)
for prefix in ('src.axiom2.research', 'src.brokers', 'src.scanner', 'src.training',
               'src.axiom_operator', 'lightgbm', 'pandas', 'numpy', 'wandb'):
    assert not any(m == prefix or m.startswith(prefix + '.') for m in sys.modules), prefix
assert not blocked
assert result.status == 'READY_SHADOW'
assert result.execution_enabled is False and result.capital_authorized is False
print(json.dumps({'status': result.status, 'blocked_side_effects': blocked,
                  'execution_enabled': result.execution_enabled}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-W", "error", "-c", code, str(ROOT)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "status": "READY_SHADOW", "blocked_side_effects": [], "execution_enabled": False,
    }


def test_research_artifact_cannot_import_portfolio_or_execution(tmp_path):
    bundle = tmp_path / "research.zip"
    result = subprocess.run(
        [sys.executable, "-B", "-W", "error", str(ROOT / "scripts/axiom2_verify_isolation.py"),
         "--bundle", str(bundle)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS" and receipt["execution_enabled"] is False
    with zipfile.ZipFile(bundle) as archive:
        assert not any(name.startswith(("src/axiom2/portfolio/", "src/axiom2/execution/"))
                       for name in archive.namelist())
    code = r"""
import importlib.util, sys
sys.path.insert(0, sys.argv[1])
assert importlib.util.find_spec('src.axiom2.portfolio') is None
assert importlib.util.find_spec('src.axiom2.execution') is None
print('NO_PORTFOLIO_OR_EXECUTION_AUTHORITY')
"""
    probe = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(bundle)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "NO_PORTFOLIO_OR_EXECUTION_AUTHORITY"
