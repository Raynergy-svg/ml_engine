from cli import training


def test_pair_config_overrides_direction_threshold(monkeypatch):
    printed = []
    monkeypatch.setattr(training.console, "print", lambda *args, **kwargs: printed.append((args, kwargs)))

    cfg = {
        "direction_threshold": 0.005,
        "direction_lookahead": 24,
        "training": {"epochs": 200},
    }

    merged = training._apply_pair_config_overrides(
        cfg=cfg,
        config_path="config/config_intel_optimized.yaml",
        training_instrument="USD_CAD",
    )

    assert merged["direction_threshold"] == 0.003
    assert merged["direction_lookahead"] == 24
    assert merged["training"]["epochs"] == 200
    assert printed


def test_pair_config_overrides_skip_generic_instrument(monkeypatch):
    monkeypatch.setattr(training.console, "print", lambda *args, **kwargs: None)

    cfg = {
        "direction_threshold": 0.005,
        "direction_lookahead": 24,
    }

    merged = training._apply_pair_config_overrides(
        cfg=cfg,
        config_path="config/config_intel_optimized.yaml",
        training_instrument="GENERIC",
    )

    assert merged == cfg
