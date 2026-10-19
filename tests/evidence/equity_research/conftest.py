"""Task 5 fixtures with uniquely qualified worker helpers."""

from datetime import timedelta

import pytest

from ._support import NOW, COMMIT, build_context, proposal


@pytest.fixture
def context(tmp_path):
    return build_context(tmp_path / "evidence")


@pytest.fixture
def frozen(context):
    context.registry.register_experiment(proposal())
    context.operator.register_holdout("sealed-1", "b" * 64, NOW - timedelta(days=60), NOW - timedelta(days=30))
    return context.registry.freeze_candidate("exp-1", "c" * 64, COMMIT)
