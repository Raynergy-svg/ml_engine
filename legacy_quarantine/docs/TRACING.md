# Tracing (OpenTelemetry)

This repo supports optional distributed tracing for training runs using OpenTelemetry + the VS Code AI Toolkit trace viewer.

## Setup (macOS / conda `tf-metal`)

Tracing deps are included in `requirements_tf_metal.txt`.

```bash
/Users/mirelacertan/miniforge3/bin/conda run -n tf-metal python -m pip install -r requirements_tf_metal.txt
```

## Start the AI Toolkit collector

In VS Code run the command:
- `AI Toolkit: Tracing: Open`

(Programmatically this is `ai-mlstudio.tracing.open`.)

The collector listens on:
- `http://localhost:4318` (OTLP/HTTP)

## Enable tracing

Set `ML_ENGINE_TRACING=1`.

```bash
cd /Users/mirelacertan/Documents/ml_engine
/Users/mirelacertan/miniforge3/bin/conda run -n tf-metal env \
  ML_ENGINE_TRACING=1 \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  PYTHONUNBUFFERED=1 \
  python -u train_visual.py --framework tensorflow --model tcn --multi-task \
  --samples 512 --features 32 --epochs 1 --batch-size 16 --patience 1 --run-eagerly
```

## What you’ll see

`train_visual.py` emits spans for:
- `train_visual.train_tensorflow`
- `tensorflow.gpu_probe`
- `engine.init`
- `engine.build_model`
- `engine.train`
- `engine.evaluate`

If the collector is not running, training still works (tracing becomes a no-op).
