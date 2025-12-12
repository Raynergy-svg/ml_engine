import pytest


def _minimal_config():
    return {
        "device": "cpu",
        "mixed_precision": False,
        "batch_size": 2,
        "model": {
            "type": "lstm",
            "input_size": 7,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        },
    }


def test_integrated_predict_once_smoke():
    import main

    result = main.integrated_predict_once(_minimal_config())

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "reasoning" in result

    pred = result["prediction"]
    assert getattr(pred, "shape", None) is not None
    assert pred.shape[0] == 2


def test_compute_slm_indicators_smoke():
    pd = pytest.importorskip("pandas")

    import main

    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    indicators = main.compute_slm_indicators(df)

    assert "rsi_14" in indicators
    assert len(indicators["rsi_14"]) == len(df)
