"""Real-git tests for scripts/ci/generate_policy_input.py (CI tooling, no-mock).

Builds an actual temp git repo per test and diffs real commits — the same
git plumbing the script uses against this repo, just pointed at a throwaway
one via --repo-root / build_input(repo_root=...).
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_policy_input",
    Path(__file__).resolve().parents[1] / "scripts" / "ci" / "generate_policy_input.py",
)
generate_policy_input = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_policy_input)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "fixture_repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "config", "user.email", "ci@example.com")
    _git(repo_dir, "config", "user.name", "CI Test")
    (repo_dir / "README.md").write_text("base\n")
    _git(repo_dir, "add", "README.md")
    _git(repo_dir, "commit", "-q", "-m", "base")
    return repo_dir


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_eval_report_requires_timestamp_pattern(repo: Path) -> None:
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    backtests_dir = repo / "trained_data" / "backtests"
    backtests_dir.mkdir(parents=True)
    (backtests_dir / "notes.md").write_text("not a timestamped eval report\n")
    (backtests_dir / "equity_harvester_singlestock_20260702T050603Z.json").write_text("{}\n")
    head_sha = _commit_all(repo, "touch backtests dir")

    doc = generate_policy_input.build_input(
        base_sha, head_sha, "1", [], repo_root=str(repo)
    )

    assert doc["gate_evidence"]["eval_report_files_changed"] == [
        "trained_data/backtests/equity_harvester_singlestock_20260702T050603Z.json"
    ]
    assert "trained_data/backtests/notes.md" not in doc["gate_evidence"]["eval_report_files_changed"]


def test_ship_gate_file_not_double_counted_as_eval_report(repo: Path) -> None:
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    backtests_dir = repo / "trained_data" / "backtests"
    backtests_dir.mkdir(parents=True)
    (backtests_dir / "SHIP_GATE.json").write_text("{}\n")
    head_sha = _commit_all(repo, "add ship gate")

    doc = generate_policy_input.build_input(
        base_sha, head_sha, "1", [], repo_root=str(repo)
    )

    assert doc["gate_evidence"]["ship_gate_files_changed"] == ["trained_data/backtests/SHIP_GATE.json"]
    assert doc["gate_evidence"]["eval_report_files_changed"] == []


def test_live_arm_checklist_file_detected_and_template_excluded(repo: Path) -> None:
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    evidence_dir = repo / ".ci" / "policy" / "evidence" / "live-arm"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "TEMPLATE.md").write_text("template\n")
    head_sha = _commit_all(repo, "add template only")

    doc = generate_policy_input.build_input(base_sha, head_sha, "1", [], repo_root=str(repo))
    assert doc["review_evidence"]["checklist_file_changed"] is False

    (evidence_dir / "2026-07-07-review.md").write_text("real review\n")
    head_sha_2 = _commit_all(repo, "add real checklist")

    doc_2 = generate_policy_input.build_input(base_sha, head_sha_2, "1", [], repo_root=str(repo))
    assert doc_2["review_evidence"]["checklist_file_changed"] is True


def test_risky_diff_pattern_detected_in_added_lines_only(repo: Path) -> None:
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    config_file = repo / "config.py"
    config_file.write_text('oanda_environment = "practice"\n')
    head_sha = _commit_all(repo, "add config with practice pin")

    doc = generate_policy_input.build_input(base_sha, head_sha, "1", [], repo_root=str(repo))
    assert doc["diff_signals"]["risky_live_patterns_found"] == []

    config_file.write_text('oanda_environment = "live"\n')
    head_sha_2 = _commit_all(repo, "flip to live")

    doc_2 = generate_policy_input.build_input(base_sha, head_sha_2, "1", [], repo_root=str(repo))
    assert len(doc_2["diff_signals"]["risky_live_patterns_found"]) == 1
