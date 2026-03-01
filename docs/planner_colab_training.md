# Planner Colab Training

The planner fine-tune path can be exported into a self-contained Colab bundle. The bundle keeps the same PEFT LoRA adapter layout as the local trainer, so a Colab-trained adapter can be copied back into `trained_data/planner/checkpoints/`.

The current planner dataset generator now produces 1,146 rows and now mixes two supervision modes:

- planner JSON / tool-routing examples
- direct Buddy reply examples for grounded conversational responses

It also includes the newer chat-planner tool surface:

- `use_market_source`
- `run_prediction`
- `summarize_last_prediction`
- `run_runtime_command`
- `run_oanda_command`
- `run_trade_command`

If Colab is only letting you open notebook files from its launcher, upload [planner_colab_train.ipynb](/Users/davidcertan/Desktop/ml_engine/notebooks/planner_colab_train.ipynb) first, open it, and use its upload cells from inside the notebook.

## Local export

From the repo root:

```bash
python scripts/train_planner_model.py \
  --prepare-only \
  --write-colab-bundle trained_data/planner/planner_colab_bundle
```

This writes:

- `trained_data/planner/planner_colab_bundle/`
- `trained_data/planner/planner_colab_bundle.zip`

This repo now includes the generated bundle at that path.

The bundle contains:

- `train_planner_model.py`
- `planner_training_colab.yaml`
- `sft_train.jsonl`
- `sft_valid.jsonl`
- `README.md`

## Colab run

The notebook at [planner_colab_train.ipynb](/Users/davidcertan/Desktop/ml_engine/notebooks/planner_colab_train.ipynb) is the canonical Colab workflow now. It supports:

- Google Drive-backed run directories
- automatic resume from the latest checkpoint
- named A100 run profiles
- base-model overrides, including 3B runs
- custom train/valid dataset path overrides
- persistent `train.log` output

Use a GPU runtime. The default training config now targets `Qwen/Qwen2.5-3B-Instruct`. L4 or A100 is preferred for useful 3B runs.

Install dependencies:

```bash
pip install -q transformers accelerate peft bitsandbytes sentencepiece safetensors pyyaml
```

Upload `trained_data/planner/planner_colab_bundle.zip` to Colab:

1. Open Colab with a GPU runtime.
2. In the left sidebar, open `Files`.
3. Click `Upload to session storage`.
4. Upload `planner_colab_bundle.zip`.
5. In a code cell, run:

```bash
!unzip -q planner_colab_bundle.zip -d planner_colab_bundle
%cd planner_colab_bundle
```

Then run:

```bash
python train_planner_model.py \
  --config planner_training_colab.yaml \
  --skip-prepare \
  --output-dir planner_adapter \
  --colab-preset t4 \
  --max-steps 100
```

For an A100 run, use:

```bash
python train_planner_model.py \
  --config planner_training_colab.yaml \
  --skip-prepare \
  --output-dir planner_adapter \
  --colab-preset a100 \
  --max-steps 200
```

What the presets do:

- `t4`: enables QLoRA 4-bit loading, NF4 + double quant, packed sequences, `adamw_bnb_8bit`, auto batch-size search, pinned memory, worker preloading, and fp16.
- `a100`: enables packed sequences, larger starting batches, auto batch-size search, TF32, SDPA, pinned memory, worker preloading, and bf16 when supported. The preset now starts at per-device batch size `32` on the 80GB A100 path. Leave `torch.compile` off unless you are doing a longer run and can tolerate a slower first step.

The notebook's main A100 profiles are:

- `a100_fast`: quick validation on 1.5B
- `a100_long`: longer 1.5B run
- `a100_2h`: 1.5B profile sized for roughly a 2-hour Colab window
- `a100_3b_long`: 3B profile with batch size `32`
- `a100_3b_2h`: 3B profile sized for a longer A100 session, batch size `40`

Recommended next run with the current tiny dataset:

- `--max-steps 50` for a quick validation pass
- `--max-steps 100` to `--max-steps 200` for a more useful adapter

On the `NVIDIA A100-SXM4-80GB`, start with:

- batch size `32`
- gradient accumulation `1`
- warmup steps `10`

If that is stable, push batch size to `40` or `48` using the notebook override cell.

For `Qwen/Qwen2.5-3B-Instruct`, use the notebook's `a100_3b_*` profiles rather than the 1.5B settings. They now push the A100 harder with larger batches and disable gradient checkpointing by default to improve throughput. If the `a100_3b_2h` profile OOMs on your session, drop its batch from `40` to `36`.

If Colab gives you a newer GPU with BF16 support, the trainer will select BF16 automatically. On T4 it will fall back to FP16 automatically.

## Bring the adapter back

Download `planner_adapter/` from Colab and copy its contents into:

```text
trained_data/planner/checkpoints/
```

Expected files include:

- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer files

In Colab, the fastest way to download the trained adapter is:

```python
import shutil
shutil.make_archive("planner_adapter", "zip", "planner_adapter")
```

Then use the Files sidebar to download `planner_adapter.zip`, unzip it locally, and place the extracted files in `trained_data/planner/checkpoints/`.

`PlannerRuntime` now auto-attempts to load the adapter from `trained_data/planner/checkpoints/` on startup. You can override that with `BUDDY_PLANNER_ADAPTER_PATH=/custom/path`.
