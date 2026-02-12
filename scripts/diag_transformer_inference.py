import numpy as np

from src.core.modular_inference import ModularEnsembleInference
from src.data.feature_engineering import FeatureEngineering
from src.utils.fx_paper import candles_to_ohlcv_df
from src.utils.oanda_practice import OandaPracticeClient
from src.utils.utils import load_config


def main() -> None:
    cfg = load_config("config/config_intel_optimized.yaml")
    fe = FeatureEngineering(cfg.get("feature_engineering", {}))

    ensemble = ModularEnsembleInference(
        instrument="EUR_USD",
        enable_market_intelligence=False,
        enable_drift_detection=False,
    )
    ensemble.load_models(instrument="EUR_USD")

    model = ensemble.tcn
    print("Transformer loaded:", model is not None)
    if model is None:
        return

    feature_names = list(getattr(model, "feature_names", []) or [])
    print("feature_names len:", len(feature_names))
    print("feature_names first10:", feature_names[:10])

    selected_indices = getattr(model, "selected_indices", None)
    print("selected_indices:", selected_indices is not None)
    if selected_indices is not None:
        print("selected_indices len:", len(selected_indices))

    print("model expects n_features:", model.model.input_shape[-1])

    client = OandaPracticeClient.from_env()
    resp = client.get_candles("EUR_USD", granularity="H1", count=200, price="MBA")
    df = candles_to_ohlcv_df(resp)
    df = fe.create_features(df, include_all=True)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    X = ensemble._extract_tcn_features(df)
    print("X shape:", X.shape)

    flat = X.reshape(-1, X.shape[-1])
    print(
        "flat stats min/mean/max/std:",
        float(np.min(flat)),
        float(np.mean(flat)),
        float(np.max(flat)),
        float(np.std(flat)),
    )
    print("fraction |x|<1e-8:", float(np.mean(np.abs(flat) < 1e-8)))

    pred = model.predict(X)
    print(
        "pred:",
        {
            "direction": pred.get("direction"),
            "probability": pred.get("probability"),
            "threshold": pred.get("threshold"),
            "ema_used": pred.get("ema_used"),
        },
    )


if __name__ == "__main__":
    main()
