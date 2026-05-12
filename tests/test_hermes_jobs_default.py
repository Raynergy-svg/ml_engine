"""Hermes watchdog: default_jobs() includes the 30-min watchdog and daily brief."""
from __future__ import annotations

from src.scanner.automation.scheduled_jobs import default_jobs
from src.scanner.automation.brain_caps import caps


def test_default_jobs_includes_hermes_watchdog():
    jobs = {j.job_id: j for j in default_jobs()}
    assert "hermes_watchdog" in jobs
    j = jobs["hermes_watchdog"]
    assert j.schedule == "every_30_minutes"
    assert j.enabled is True
    assert "hermes_watchdog" in j.command


def test_default_jobs_includes_hermes_daily_brief():
    jobs = {j.job_id: j for j in default_jobs()}
    assert "hermes_daily_brief" in jobs
    j = jobs["hermes_daily_brief"]
    assert j.schedule == "daily_07:00"
    assert j.enabled is True
    assert "hermes_daily_brief" in j.command


def test_default_jobs_preserves_homework_weekly():
    """Don't drop the existing default."""
    jobs = {j.job_id: j for j in default_jobs()}
    assert "homework_weekly" in jobs


def test_brain_caps_includes_hermes_watchdog():
    c = caps()
    assert "hermes_watchdog.md" in c
    hard_cap, warn_ratio = c["hermes_watchdog.md"]
    assert hard_cap == 8000
    assert warn_ratio == 1.15
