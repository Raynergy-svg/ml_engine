"""Tests for the launchd user-agent installer (US-601)."""
from __future__ import annotations

import plistlib
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_launchd_service.sh"


@pytest.fixture(scope="module")
def installed_plist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the installer with HOME pointed at a temp dir; return the plist path."""
    fake_home = tmp_path_factory.mktemp("home")
    env = {"HOME": str(fake_home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    subprocess.run(
        ["bash", str(INSTALL_SCRIPT)],
        check=True,
        env=env,
        capture_output=True,
    )
    plist = fake_home / "Library" / "LaunchAgents" / "com.buddy.trader.plist"
    assert plist.exists(), f"plist not written to {plist}"
    return plist


def test_plist_is_valid_xml(installed_plist: Path) -> None:
    """plistlib parses without raising — guarantees well-formed XML+plist DTD."""
    with installed_plist.open("rb") as f:
        data = plistlib.load(f)
    assert isinstance(data, dict)


def test_required_keys_present(installed_plist: Path) -> None:
    with installed_plist.open("rb") as f:
        data = plistlib.load(f)
    for key in ("Label", "ProgramArguments", "KeepAlive", "StandardOutPath",
                "StandardErrorPath", "ExitTimeOut", "ThrottleInterval"):
        assert key in data, f"missing required plist key: {key}"
    assert data["Label"] == "com.buddy.trader"
    assert data["KeepAlive"] is True
    assert data["ExitTimeOut"] == 30
    assert data["ThrottleInterval"] == 60
    assert isinstance(data["ProgramArguments"], list)
    assert any("buddy" in arg for arg in data["ProgramArguments"])
    assert "new-session" in data["ProgramArguments"]
    assert "-A" in data["ProgramArguments"]


def test_paths_are_absolute(installed_plist: Path) -> None:
    with installed_plist.open("rb") as f:
        data = plistlib.load(f)
    path_keys = ("StandardOutPath", "StandardErrorPath", "WorkingDirectory")
    for key in path_keys:
        value = data[key]
        assert value.startswith("/"), f"{key} must be absolute, got: {value}"
    for arg in data["ProgramArguments"]:
        if "/" in arg:
            assert arg.startswith("/"), f"ProgramArgument path must be absolute: {arg}"
