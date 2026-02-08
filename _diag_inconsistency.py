"""Diagnose scanner vs inference inconsistency for GBP_USD."""
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from multi_pair_inference import MultiPairInference  # noqa: E402
from src.core.modular_inference import ModularEnsembleInference  # noqa: E402
from src.core.modular_data_loaders import compute_normalized_features  # noqa: E402
from src.utils.oanda_practice import OandaPracticeClient  # noqa: E402
from src.utils.fx_paper import candles_to_ohlcv_df  # noqa: E402

print('=== Fetching GBP_USD data ===')
client = OandaPracticeClient.from_env()
resp = client.get_candles('GBP_USD', granularity='H1', count=300, price='MBA')
df_raw = candles_to_ohlcv_df(resp)
print(f'Raw data: {len(df_raw)} rows')

# -----------------------------------------------------------
# PATH 1: MODULAR INFERENCE (what buddy -i GBP_USD uses)
# -----------------------------------------------------------
print('\n=== MODULAR INFERENCE PATH (buddy -i) ===')
mei = ModularEnsembleInference(instrument='GBP_USD')
mei.load_models()
tcn = mei.tcn

mod_dir = None
mod_prob = None
mod_threshold = 0.5

if tcn:
    print(f'Type: {type(tcn).__name__}')
    fn = tcn.feature_names
    print(f'Features: {len(fn) if fn else 0}, seq_len={tcn.seq_len}')
    print(f'Scaler: {tcn.scaler is not None}')
    cal = getattr(tcn, 'output_calibration', None)
    print(f'Calibration: {cal}')

    df_feat = compute_normalized_features(df_raw.copy())
    features = mei._extract_tcn_features(df_feat)
    print(f'Feature shape: {features.shape}')
    tcn_pred = tcn.predict(features)
    mod_prob = tcn_pred['probability']
    mod_threshold = tcn_pred.get('threshold', 0.5)
    mod_dir = 'LONG' if tcn_pred['direction'] == 1 else 'SHORT'
    print(f'Result: dir={tcn_pred["direction"]} ({mod_dir}), prob={mod_prob:.4f}, threshold={mod_threshold:.4f}')
else:
    print('NO TCN MODEL LOADED')

# -----------------------------------------------------------
# PATH 2: MULTI-PAIR INFERENCE (what scanner uses)
# -----------------------------------------------------------
print('\n=== MULTI-PAIR INFERENCE PATH (scanner) ===')
mpi = MultiPairInference()
mpi.load()
pm = mpi._pair_models.get('GBP_USD')

scan_dir = None
scan_prob = None
scan_raw_prob = None

if pm:
    print(f'Features: {len(pm.feature_names) if pm.feature_names else 0}, seq_len={pm.seq_len}')
    print(f'Scaler: {pm.scaler is not None}')
    print(f'Accuracy: {pm.accuracy:.3f}')

    direction, confidence = mpi.predict(df_raw.copy(), 'GBP_USD')
    scan_dir = direction
    scan_prob = confidence
    print(f'Result: dir={direction}, conf={confidence:.4f}')

    # Replicate raw model prediction
    df_feat2 = compute_normalized_features(df_raw.copy())
    fn2 = pm.feature_names or []
    sl = pm.seq_len or 60
    X_list = []
    missing = []
    for f in fn2:
        if f in df_feat2.columns:
            X_list.append(df_feat2[f].iloc[-sl:].values)
        else:
            missing.append(f)
            X_list.append(np.zeros(sl))

    print(f'Missing features in scanner path: {len(missing)}')
    if missing:
        print(f'  First 10: {missing[:10]}')

    X = np.column_stack(X_list)
    X = np.nan_to_num(X)
    if pm.scaler:
        X = pm.scaler.transform(X)
    raw_pred = pm.transformer.predict(X.reshape(1, sl, -1).astype(np.float32), verbose=0)
    scan_raw_prob = float(raw_pred[0, 0])
    print(f'Raw model output: {scan_raw_prob:.4f} -> {"LONG" if scan_raw_prob > 0.5 else "SHORT"}')
    print('Scanner direction logic: prob > 0.5 => direction, confidence = |prob-0.5|*2')
else:
    print('NO PAIR MODEL LOADED')

# -----------------------------------------------------------
# CHECK: Are they loading the same .keras file?
# -----------------------------------------------------------
import pathlib  # noqa: E402
mei_path = mei._get_model_path('transformer_direction', '.keras')
mpi_path = pathlib.Path('trained_data/models/GBP_USD/transformer_direction.keras')
print('\n=== MODEL FILES ===')
print(f'Modular inference: {mei_path} (exists={mei_path.exists()})')
print(f'Multi-pair:        {mpi_path} (exists={mpi_path.exists()})')
print(f'Same file: {mei_path.resolve() == mpi_path.resolve()}')

# -----------------------------------------------------------
# CHECK: Feature differences between paths
# -----------------------------------------------------------
if tcn and pm:
    fn1 = set(tcn.feature_names) if tcn.feature_names else set()
    fn2 = set(pm.feature_names) if pm.feature_names else set()
    print('\n=== FEATURE COMPARISON ===')
    print(f'Modular inference features: {len(fn1)}')
    print(f'Multi-pair features:        {len(fn2)}')
    only_mod = fn1 - fn2
    only_scan = fn2 - fn1
    if only_mod:
        print(f'Only in modular: {sorted(only_mod)[:10]}')
    if only_scan:
        print(f'Only in scanner: {sorted(only_scan)[:10]}')
    if fn1 == fn2:
        print('Feature names MATCH')

# -----------------------------------------------------------
# ROOT CAUSE ANALYSIS
# -----------------------------------------------------------
print(f'\n{"="*60}')
print('ROOT CAUSE ANALYSIS')
print(f'{"="*60}')

if mod_prob is not None and scan_raw_prob is not None:
    print(f'Modular inf: raw_prob={mod_prob:.4f}, threshold={mod_threshold:.4f} -> {mod_dir}')
    print(f'Scanner:     raw_prob={scan_raw_prob:.4f}, threshold=0.5000 -> {"LONG" if scan_raw_prob > 0.5 else "SHORT"}')
    print()

    prob_diff = abs(mod_prob - scan_raw_prob)
    if prob_diff > 0.01:
        print(f'[BUG 1] DIFFERENT RAW PROBABILITIES (diff={prob_diff:.4f})')
        print('  The two paths compute different features or apply scaling differently.')
        print('  Even though they load the same .keras file, the input differs.')

    if mod_threshold != 0.5:
        print('[BUG 2] THRESHOLD MISMATCH')
        print(f'  Modular inference uses calibrated threshold: {mod_threshold:.4f}')
        print('  Scanner always uses hardcoded threshold: 0.5')
        print('  This means the scanner ignores output calibration entirely.')

    if mod_dir != ("LONG" if scan_raw_prob > 0.5 else "SHORT"):
        print(f'\n  >>> CONFIRMED: Scanner says {"SHORT" if scan_raw_prob < 0.5 else "LONG"} while inference says {mod_dir}')
        print('  >>> This explains the GBP/USD SHORT vs LONG inconsistency')
