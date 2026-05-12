"""Tests for scripts/lock_skills.py — skill content-hash lockfile."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lock_skills.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lock_skills", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    """A throwaway .claude/skills/ tree with two skills."""
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "alpha").mkdir()
    (skills_dir / "alpha" / "SKILL.md").write_text("alpha v1\n")
    (skills_dir / "beta").mkdir()
    (skills_dir / "beta" / "SKILL.md").write_text("beta v1\n")
    (skills_dir / "beta" / "extra.md").write_text("supporting\n")

    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(mod, "LOCK_PATH", tmp_path / ".claude" / "skills-lock.json")
    return mod, skills_dir


def test_build_lockfile_lists_all_skills(skills_root):
    mod, _ = skills_root
    lock = mod.build_lockfile()
    assert lock["version"] == 1
    assert set(lock["skills"].keys()) == {"alpha", "beta"}
    for meta in lock["skills"].values():
        assert meta["sourceType"] == "local"
        assert len(meta["computedHash"]) == 64  # sha256 hex


def test_hash_changes_when_skill_content_changes(skills_root):
    mod, skills_dir = skills_root
    before = mod.compute_skill_hash(skills_dir / "alpha")
    (skills_dir / "alpha" / "SKILL.md").write_text("alpha v2 — edited\n")
    after = mod.compute_skill_hash(skills_dir / "alpha")
    assert before != after


def test_hash_stable_across_calls(skills_root):
    mod, skills_dir = skills_root
    h1 = mod.compute_skill_hash(skills_dir / "beta")
    h2 = mod.compute_skill_hash(skills_dir / "beta")
    assert h1 == h2


def test_check_passes_after_generate(skills_root):
    mod, _ = skills_root
    lock = mod.build_lockfile()
    mod.LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    locked = json.loads(mod.LOCK_PATH.read_text())
    assert locked == lock


def test_check_detects_drift(skills_root):
    mod, skills_dir = skills_root
    lock = mod.build_lockfile()
    mod.LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (skills_dir / "alpha" / "SKILL.md").write_text("alpha v2\n")
    fresh = mod.build_lockfile()
    assert fresh != lock
    assert fresh["skills"]["alpha"]["computedHash"] != lock["skills"]["alpha"]["computedHash"]
