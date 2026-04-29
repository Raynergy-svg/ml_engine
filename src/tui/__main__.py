"""Entry point for: python -m src.tui"""
# Idempotent env bootstrap — safe even when launched via ./buddy
# (which already sourced scripts/init.sh). Ensures direct
# `python -m src.tui` invocations see the same environment.
from src.bootstrap.env import ensure_runtime_env

ensure_runtime_env()

from src.tui.app import run  # noqa: E402

run()
