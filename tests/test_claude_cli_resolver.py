"""Tests for src.utils.claude_cli — the claude-CLI-under-minimal-PATH resolver.

Root cause under test: launchd daemons (e.g. com.axiom.api) run with a hardcoded
minimal PATH that excludes ~/.local/bin, so shutil.which("claude") returns None
even though the binary is installed. See src/axiom_operator/runner.py and
src/scanner/automation/claude_subprocess.py for the call sites this backs.
"""
import os
import stat

from src.utils.claude_cli import resolve_claude_cli, augmented_path_env


def _make_executable(path):
    path.write_text("#!/bin/sh\necho fake-claude\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_resolves_via_current_path(tmp_path):
    bin_dir = tmp_path / "on_path"
    bin_dir.mkdir()
    claude_bin = bin_dir / "claude"
    _make_executable(claude_bin)

    found = resolve_claude_cli(env={"PATH": str(bin_dir)}, home=tmp_path / "home")

    assert found == str(claude_bin)


def test_resolves_via_local_bin_fallback_when_daemon_path_is_minimal(tmp_path):
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    claude_bin = local_bin / "claude"
    _make_executable(claude_bin)

    # Minimal daemon-style PATH (mirrors com.axiom.api.plist) — no ~/.local/bin.
    minimal_path = "/usr/bin:/bin:/usr/sbin:/sbin"

    found = resolve_claude_cli(env={"PATH": minimal_path}, home=home)

    assert found == str(claude_bin)


def test_honest_unavailable_when_binary_truly_absent(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    found = resolve_claude_cli(env={"PATH": "/usr/bin:/bin"}, home=home)

    assert found is None


def test_explicit_env_override_takes_priority(tmp_path):
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    _make_executable(local_bin / "claude")

    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override_bin = override_dir / "claude"
    _make_executable(override_bin)

    found = resolve_claude_cli(
        env={"PATH": "/usr/bin:/bin", "CLAUDE_CLI_PATH": str(override_bin)},
        home=home,
    )

    assert found == str(override_bin)


def test_override_env_ignored_if_not_executable_file(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    _make_executable(local_bin / "claude")

    bogus = tmp_path / "does_not_exist"

    found = resolve_claude_cli(
        env={"PATH": "/usr/bin:/bin", "CLAUDE_CLI_PATH": str(bogus)},
        home=home,
    )

    # Falls through to the ~/.local/bin fallback rather than trusting a bad override.
    assert found == str(local_bin / "claude")


def test_augmented_path_env_includes_local_bin(tmp_path):
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)

    result = augmented_path_env(env={"PATH": "/usr/bin:/bin"}, home=home)

    assert str(local_bin) in result.split(os.pathsep)
    assert "/usr/bin" in result.split(os.pathsep)
