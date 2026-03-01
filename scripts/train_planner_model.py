#!/usr/bin/env python3
"""Fine-tuning entrypoint for Buddy's planner model.

Supports both local runs and exporting a self-contained Colab bundle that
preserves the same PEFT adapter output format.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _format_example(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    training_mode = str(row.get("training_mode") or "planner_json")
    if training_mode == "buddy_reply":
        prompt = {
            "system": str(row.get("response_system_prompt") or system_prompt),
            "user_message": row["user_message"],
            "chat_history": row.get("chat_history") or [],
            "runtime_context": row.get("runtime_context") or {},
            "grounded_answer": row.get("grounded_answer") or row.get("target_response") or "",
            "tool_outputs": row.get("tool_observations") or [],
        }
        completion = str(row.get("target_reply") or row.get("target_response") or "")
        return {"prompt": json.dumps(prompt, ensure_ascii=True), "completion": completion}

    prompt = {
        "system": system_prompt,
        "user_message": row["user_message"],
        "chat_history": row.get("chat_history") or [],
        "runtime_context": row.get("runtime_context") or {},
        "available_tools": row.get("available_tools") or [],
    }
    completion = {
        "target_plan": row["target_plan"],
        "target_response": row["target_response"],
    }
    return {"prompt": json.dumps(prompt, ensure_ascii=True), "completion": json.dumps(completion, ensure_ascii=True)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_path(path_str: str, *, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidates = [
        (base_dir / path).resolve(),
        path.resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _prepare_sft_files(
    cfg: dict[str, Any],
    *,
    config_path: Path,
    prepared_dir: Path | None = None,
) -> tuple[Path, Path]:
    base_dir = config_path.resolve().parent
    train_path = _resolve_path(str(cfg["dataset"]["train_path"]), base_dir=base_dir)
    valid_path = _resolve_path(str(cfg["dataset"]["valid_path"]), base_dir=base_dir)
    system_prompt = str(cfg["format"]["system_prompt"])
    response_system_prompt = str(cfg["format"].get("response_system_prompt") or system_prompt)

    train_rows = _load_jsonl(train_path)
    valid_rows = _load_jsonl(valid_path)
    for row in train_rows + valid_rows:
        row.setdefault("response_system_prompt", response_system_prompt)

    prepared_dir = prepared_dir or Path("trained_data/planner")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    train_out = prepared_dir / "sft_train.jsonl"
    valid_out = prepared_dir / "sft_valid.jsonl"

    train_out.write_text(
        "\n".join(json.dumps(_format_example(row, system_prompt), ensure_ascii=True) for row in train_rows) + "\n",
        encoding="utf-8",
    )
    valid_out.write_text(
        "\n".join(json.dumps(_format_example(row, system_prompt), ensure_ascii=True) for row in valid_rows) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared SFT data at {train_out} and {valid_out}")
    return train_out, valid_out


def _copy_if_needed(src: Path, dest: Path) -> None:
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _write_colab_readme(
    readme_path: Path,
    *,
    config_name: str,
    train_file_name: str,
    valid_file_name: str,
) -> None:
    readme_path.write_text(
        "\n".join(
            [
                "# Buddy Planner Colab Bundle",
                "",
                "This bundle is self-contained for Colab GPU training.",
                "",
                "## Files",
                f"- `{config_name}`: Colab-ready training config",
                f"- `{train_file_name}` / `{valid_file_name}`: prepared SFT jsonl files",
                "- `train_planner_model.py`: trainer entrypoint copied from this repo",
                "",
                "## Colab Steps",
                "1. Start a GPU runtime in Colab.",
                "2. Upload or mount this bundle directory.",
                "3. Install dependencies:",
                "   `pip install -q transformers accelerate peft bitsandbytes sentencepiece safetensors pyyaml`",
                "4. Run training on T4:",
                f"   `python train_planner_model.py --config {config_name} --skip-prepare --output-dir planner_adapter --colab-preset t4 --max-steps 100`",
                "5. Run training on A100:",
                f"   `python train_planner_model.py --config {config_name} --skip-prepare --output-dir planner_adapter --colab-preset a100 --max-steps 200`",
                "6. Download `planner_adapter/` and copy its contents into `trained_data/planner/checkpoints/` locally.",
                "",
                "Preset notes:",
                "- T4: QLoRA 4-bit, packed sequences, 8-bit optimizer, fp16.",
                "- A100: packed sequences, larger batches, TF32/bf16 when available. Enable torch.compile only for longer runs.",
                "",
                "Recommended first Colab run: 50 to 200 steps with the current tiny dataset.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_colab_bundle(
    cfg: dict[str, Any],
    *,
    bundle_dir: Path,
    prepared_train_path: Path,
    prepared_valid_path: Path,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    train_name = "sft_train.jsonl"
    valid_name = "sft_valid.jsonl"
    config_name = "planner_training_colab.yaml"

    train_dest = bundle_dir / train_name
    valid_dest = bundle_dir / valid_name
    _copy_if_needed(prepared_train_path, train_dest)
    _copy_if_needed(prepared_valid_path, valid_dest)

    colab_cfg = json.loads(json.dumps(cfg))
    colab_cfg.setdefault("dataset", {})
    colab_cfg["dataset"]["train_path"] = train_name
    colab_cfg["dataset"]["valid_path"] = valid_name
    colab_cfg.setdefault("model", {})
    colab_cfg["model"]["output_dir"] = "planner_adapter"

    (bundle_dir / config_name).write_text(yaml.safe_dump(colab_cfg, sort_keys=False), encoding="utf-8")
    shutil.copy2(Path(__file__), bundle_dir / "train_planner_model.py")
    _write_colab_readme(
        bundle_dir / "README.md",
        config_name=config_name,
        train_file_name=train_name,
        valid_file_name=valid_name,
    )

    archive_path = bundle_dir.parent / f"{bundle_dir.name}.zip"
    if archive_path.exists():
        archive_path.unlink()
    shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
    print(f"Wrote Colab bundle to {bundle_dir}")
    print(f"Wrote Colab zip archive to {archive_path}")
    return bundle_dir


def _override_config(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("dataset", {})
    cfg.setdefault("model", {})
    cfg.setdefault("lora", {})
    if args.train_path:
        cfg["dataset"]["train_path"] = args.train_path
    if args.valid_path:
        cfg["dataset"]["valid_path"] = args.valid_path
    if args.base_model:
        cfg["model"]["base_model"] = args.base_model
    if args.output_dir:
        cfg["model"]["output_dir"] = args.output_dir
    if args.max_seq_length > 0:
        cfg["model"]["max_seq_length"] = args.max_seq_length
    if args.learning_rate > 0:
        cfg["model"]["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size > 0:
        cfg["model"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size > 0:
        cfg["model"]["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.gradient_accumulation_steps > 0:
        cfg["model"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.logging_steps > 0:
        cfg["model"]["logging_steps"] = args.logging_steps
    if args.warmup_steps >= 0:
        cfg["model"]["warmup_steps"] = args.warmup_steps
    if args.save_steps > 0:
        cfg["model"]["save_steps"] = args.save_steps
    if args.save_total_limit > 0:
        cfg["model"]["save_total_limit"] = args.save_total_limit
    if args.num_train_epochs > 0:
        cfg["model"]["num_train_epochs"] = args.num_train_epochs
    if args.gradient_checkpointing is not None:
        cfg["model"]["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    if args.lora_r > 0:
        cfg["lora"]["r"] = args.lora_r
    if args.lora_alpha > 0:
        cfg["lora"]["alpha"] = args.lora_alpha
    if args.lora_dropout >= 0:
        cfg["lora"]["dropout"] = args.lora_dropout


def _training_precision(torch: Any, args: argparse.Namespace) -> tuple[Any, bool, bool]:
    if args.bf16:
        return torch.bfloat16, True, False
    if args.fp16:
        return torch.float16, False, True
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        return torch.float16, False, True
    if torch.backends.mps.is_available():
        return torch.float16, False, True
    return None, False, False


def _detect_hardware(torch: Any) -> dict[str, Any]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        name = str(props.name)
        return {
            "device": "cuda",
            "name": name,
            "name_upper": name.upper(),
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "supports_bf16": torch.cuda.is_bf16_supported(),
            "supports_tf32": getattr(torch.cuda, "is_tf32_supported", lambda: False)(),
        }
    if torch.backends.mps.is_available():
        return {"device": "mps", "name": "Apple Metal", "name_upper": "APPLE METAL", "total_memory_gb": None}
    return {"device": "cpu", "name": "CPU", "name_upper": "CPU", "total_memory_gb": None}


def _default_preset_name(args: argparse.Namespace, hardware: dict[str, Any]) -> str:
    if args.colab_preset != "auto":
        return args.colab_preset
    if hardware["device"] != "cuda":
        return "none"
    name = hardware["name_upper"]
    if "A100" in name:
        return "a100"
    if "T4" in name:
        return "t4"
    return "cuda"


def _default_setting(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _training_plan(args: argparse.Namespace, hardware: dict[str, Any]) -> dict[str, Any]:
    preset = _default_preset_name(args, hardware)
    is_cuda = hardware["device"] == "cuda"
    cpu_workers = max(1, min(4, os.cpu_count() or 1))

    defaults: dict[str, Any] = {
        "preset": preset,
        "use_4bit": False,
        "packing": False,
        "group_by_length": True,
        "pad_to_multiple_of": 8 if hardware["device"] in {"cuda", "mps"} else None,
        "attn_implementation": "sdpa" if is_cuda else None,
        "optim": "adamw_torch",
        "dataloader_num_workers": cpu_workers if is_cuda else 0,
        "dataloader_pin_memory": is_cuda,
        "auto_find_batch_size": False,
        "torch_compile": False,
        "torch_compile_backend": "inductor",
        "torch_empty_cache_steps": 0,
        "allow_tf32": False,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "save_strategy": None,
        "evaluation_strategy": None,
    }

    if preset == "t4":
        defaults.update(
            {
                "use_4bit": True,
                "packing": True,
                "group_by_length": False,
                "optim": "adamw_bnb_8bit",
                "auto_find_batch_size": True,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "save_strategy": "no",
                "evaluation_strategy": "no",
            }
        )
    elif preset == "a100":
        defaults.update(
            {
                "packing": True,
                "group_by_length": False,
                "auto_find_batch_size": True,
                "torch_compile": False,
                "allow_tf32": True,
                "per_device_train_batch_size": 32,
                "gradient_accumulation_steps": 1,
                "save_strategy": "no",
                "evaluation_strategy": "no",
            }
        )
    elif preset == "cuda":
        defaults.update(
            {
                "packing": True,
                "group_by_length": False,
                "auto_find_batch_size": True,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "save_strategy": "no",
                "evaluation_strategy": "no",
            }
        )

    plan = {
        "preset": preset,
        "use_4bit": _default_setting(args.use_4bit, defaults["use_4bit"]),
        "packing": _default_setting(args.packing, defaults["packing"]),
        "group_by_length": _default_setting(args.group_by_length, defaults["group_by_length"]),
        "pad_to_multiple_of": args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else defaults["pad_to_multiple_of"],
        "attn_implementation": args.attn_implementation or defaults["attn_implementation"],
        "optim": args.optim or defaults["optim"],
        "dataloader_num_workers": (
            args.dataloader_num_workers if args.dataloader_num_workers >= 0 else defaults["dataloader_num_workers"]
        ),
        "dataloader_pin_memory": _default_setting(args.dataloader_pin_memory, defaults["dataloader_pin_memory"]),
        "auto_find_batch_size": _default_setting(args.auto_find_batch_size, defaults["auto_find_batch_size"]),
        "torch_compile": _default_setting(args.torch_compile, defaults["torch_compile"]),
        "torch_compile_backend": args.torch_compile_backend or defaults["torch_compile_backend"],
        "torch_empty_cache_steps": (
            args.torch_empty_cache_steps if args.torch_empty_cache_steps > 0 else defaults["torch_empty_cache_steps"]
        ),
        "allow_tf32": _default_setting(args.allow_tf32, defaults["allow_tf32"]),
        "per_device_train_batch_size": defaults["per_device_train_batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "save_strategy": args.save_strategy or defaults["save_strategy"],
        "evaluation_strategy": args.evaluation_strategy or defaults["evaluation_strategy"],
    }
    if plan["packing"]:
        plan["group_by_length"] = False
    return plan


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        supported = set(inspect.signature(callable_obj).parameters)
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in supported}


def _resolve_warmup_steps(
    model_cfg: dict[str, Any],
    *,
    max_steps: int,
    train_dataset_len: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    explicit_steps = model_cfg.get("warmup_steps")
    if explicit_steps is not None:
        return max(0, int(explicit_steps))

    warmup_ratio = float(model_cfg.get("warmup_ratio", 0.03))
    if max_steps > 0:
        return max(0, int(math.ceil(max_steps * warmup_ratio)))

    effective_batch = max(1, per_device_train_batch_size * gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(train_dataset_len / effective_batch))
    total_steps = max(1, int(math.ceil(float(model_cfg.get("num_train_epochs", 1)) * steps_per_epoch)))
    return max(0, int(math.ceil(total_steps * warmup_ratio)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or run Buddy planner fine-tuning.")
    parser.add_argument("--config", default="config/planner_training.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="Only write formatted SFT JSONL files.")
    parser.add_argument("--skip-prepare", action="store_true", help="Use prepared SFT JSONL files directly from config dataset paths.")
    parser.add_argument("--prepared-data-dir", default="trained_data/planner", help="Directory for prepared SFT files.")
    parser.add_argument("--write-colab-bundle", help="Export a self-contained Colab bundle to this directory.")
    parser.add_argument("--bundle-only", action="store_true", help="Prepare data and write the Colab bundle without training locally.")
    parser.add_argument("--train-path", help="Override dataset.train_path from config.")
    parser.add_argument("--valid-path", help="Override dataset.valid_path from config.")
    parser.add_argument("--output-dir", help="Override model.output_dir from config.")
    parser.add_argument("--base-model", help="Override model.base_model from config.")
    parser.add_argument("--resume-from-checkpoint", default="", help="Resume Trainer state from a checkpoint directory.")
    parser.add_argument("--bf16", action="store_true", help="Force bfloat16 training precision.")
    parser.add_argument("--fp16", action="store_true", help="Force float16 training precision.")
    parser.add_argument("--colab-preset", choices=["auto", "none", "t4", "a100", "cuda"], default="auto")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=None, help="Enable QLoRA 4-bit base-model loading on CUDA.")
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=None, help="Pack multiple samples into fixed-length token blocks.")
    parser.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=None, help="Bucket variable-length samples together when packing is disabled.")
    parser.add_argument("--auto-find-batch-size", action=argparse.BooleanOptionalAction, default=None, help="Let Trainer shrink the batch size automatically on OOM.")
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=None, help="Pin host memory for CUDA dataloading.")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=None, help="Enable torch.compile for faster steady-state CUDA training.")
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=None, help="Enable TF32 matmuls on supported NVIDIA GPUs.")
    parser.add_argument("--attn-implementation", default="", help="Attention backend override, e.g. sdpa.")
    parser.add_argument("--optim", default="", help="Optimizer name override for TrainingArguments.")
    parser.add_argument("--save-strategy", default="", help="TrainingArguments save_strategy override.")
    parser.add_argument("--evaluation-strategy", default="", help="TrainingArguments evaluation_strategy override.")
    parser.add_argument("--torch-compile-backend", default="", help="torch.compile backend override.")
    parser.add_argument("--pad-to-multiple-of", type=int, default=0, help="Pad batches to a token multiple, typically 8 on fp16/bf16.")
    parser.add_argument("--dataloader-num-workers", type=int, default=-1, help="Dataloader worker count override.")
    parser.add_argument("--torch-empty-cache-steps", type=int, default=0, help="Call torch empty_cache every N steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=0, help="Override model per-device train batch size.")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=0, help="Override model per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0, help="Override gradient accumulation steps.")
    parser.add_argument("--logging-steps", type=int, default=0, help="Override TrainingArguments logging_steps.")
    parser.add_argument("--warmup-steps", type=int, default=-1, help="Override warmup steps. Use 0 to disable warmup.")
    parser.add_argument("--save-steps", type=int, default=0, help="Save checkpoints every N optimizer steps.")
    parser.add_argument("--save-total-limit", type=int, default=0, help="Override how many checkpoints to keep.")
    parser.add_argument("--learning-rate", type=float, default=0.0, help="Override model learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=0, help="Override model max sequence length.")
    parser.add_argument("--num-train-epochs", type=float, default=0.0, help="Override number of train epochs when max-steps is not used.")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None, help="Override model gradient checkpointing.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training examples.")
    parser.add_argument("--max-valid-samples", type=int, default=0, help="Optional cap on validation examples.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on optimizer steps.")
    parser.add_argument("--lora-r", type=int, default=0, help="Override LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=0, help="Override LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=-1.0, help="Override LoRA dropout. Use 0 for none.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    _override_config(cfg, args)

    if args.skip_prepare:
        base_dir = config_path.parent
        train_out = _resolve_path(str(cfg["dataset"]["train_path"]), base_dir=base_dir)
        valid_out = _resolve_path(str(cfg["dataset"]["valid_path"]), base_dir=base_dir)
    else:
        train_out, valid_out = _prepare_sft_files(
            cfg,
            config_path=config_path,
            prepared_dir=Path(args.prepared_data_dir),
        )

    if args.write_colab_bundle:
        _write_colab_bundle(
            cfg,
            bundle_dir=Path(args.write_colab_bundle),
            prepared_train_path=train_out,
            prepared_valid_path=valid_out,
        )

    if args.prepare_only or args.bundle_only:
        return 0

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        print(f"Training dependencies unavailable: {e}")
        print("Install transformers + accelerate, or use --prepare-only.")
        return 1

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
        peft_available = True
    except Exception:
        LoraConfig = None  # type: ignore
        get_peft_model = None  # type: ignore
        peft_available = False

    model_cfg = cfg.get("model") or {}
    lora_cfg = cfg.get("lora") or {}
    model_name = str(model_cfg["base_model"])
    max_seq_length = int(model_cfg.get("max_seq_length", 1024))
    use_lora = bool(lora_cfg.get("enabled", True) and peft_available)
    dtype, bf16, fp16 = _training_precision(torch, args)
    hardware = _detect_hardware(torch)
    plan = _training_plan(args, hardware)

    if hardware["device"] == "cuda" and plan["allow_tf32"] and hardware.get("supports_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_load_kwargs["dtype"] = dtype
    if plan["attn_implementation"]:
        model_load_kwargs["attn_implementation"] = plan["attn_implementation"]

    use_4bit = bool(plan["use_4bit"] and hardware["device"] == "cuda")
    if plan["use_4bit"] and hardware["device"] != "cuda":
        print("Ignoring --use-4bit because CUDA is not available.")
    if use_4bit:
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore
            from transformers import BitsAndBytesConfig
        except Exception as e:
            print(f"4-bit training dependencies unavailable: {e}")
            print("Install bitsandbytes-compatible transformers/peft, or disable --use-4bit.")
            return 1

        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype or torch.float16,
        )
        model_load_kwargs["device_map"] = {"": 0}
        model_load_kwargs["low_cpu_mem_usage"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    except TypeError as e:
        if "attn_implementation" not in str(e):
            raise
        model_load_kwargs.pop("attn_implementation", None)
        print("Retrying model load without attn_implementation because this transformers build does not support it.")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        )

    if use_lora:
        target_modules = list(lora_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        lora = LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
        print(f"LoRA enabled on target modules: {target_modules}")
    else:
        print("LoRA disabled; full-model fine-tune will be attempted.")

    if bool(model_cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        # Required for LoRA training with checkpointing so gradients flow from embeddings.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if use_lora and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    print(
        "Training hardware: "
        f"{hardware['name']} preset={plan['preset']} dtype="
        f"{'bf16' if bf16 else 'fp16' if fp16 else 'fp32'} "
        f"use_4bit={use_4bit} packing={plan['packing']} "
        f"attn={plan['attn_implementation'] or 'default'} optim={plan['optim']}"
    )

    class PlannerSFTDataset(Dataset):
        def __init__(self, path: Path, limit: int = 0, *, packing: bool = False):
            rows = _load_jsonl(path)
            if limit > 0:
                rows = rows[:limit]
            self.samples = []
            if packing:
                eos = tokenizer.eos_token or ""
                token_stream: list[int] = []
                for row in rows:
                    text = row["prompt"] + "\n" + row["completion"] + eos
                    encoded = tokenizer(text, truncation=False, add_special_tokens=True)
                    token_stream.extend(encoded["input_ids"])
                for start in range(0, len(token_stream), max_seq_length):
                    chunk = token_stream[start : start + max_seq_length]
                    if len(chunk) < max_seq_length:
                        continue
                    self.samples.append(
                        {
                            "input_ids": chunk,
                            "attention_mask": [1] * len(chunk),
                            "labels": list(chunk),
                        }
                    )
                if not self.samples and token_stream:
                    self.samples.append(
                        {
                            "input_ids": token_stream[:max_seq_length],
                            "attention_mask": [1] * min(len(token_stream), max_seq_length),
                            "labels": list(token_stream[:max_seq_length]),
                        }
                    )
            else:
                for row in rows:
                    text = row["prompt"] + "\n" + row["completion"]
                    encoded = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=True)
                    encoded["labels"] = list(encoded["input_ids"])
                    self.samples.append(encoded)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return self.samples[idx]

    train_dataset = PlannerSFTDataset(train_out, limit=args.max_train_samples, packing=bool(plan["packing"]))
    valid_dataset = PlannerSFTDataset(valid_out, limit=args.max_valid_samples, packing=bool(plan["packing"]))
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=plan["pad_to_multiple_of"],
    )

    output_dir = str(model_cfg.get("output_dir", "trained_data/planner/checkpoints"))
    per_device_train_batch_size = int(model_cfg.get("per_device_train_batch_size", 1))
    if plan["per_device_train_batch_size"] is not None:
        per_device_train_batch_size = int(plan["per_device_train_batch_size"])
    gradient_accumulation_steps = int(model_cfg.get("gradient_accumulation_steps", 4))
    if plan["gradient_accumulation_steps"] is not None:
        gradient_accumulation_steps = int(plan["gradient_accumulation_steps"])

    max_steps = int(args.max_steps) if args.max_steps > 0 else -1
    warmup_steps = _resolve_warmup_steps(
        model_cfg,
        max_steps=max_steps,
        train_dataset_len=len(train_dataset),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    training_kwargs = {
        "output_dir": output_dir,
        "learning_rate": float(model_cfg.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": int(model_cfg.get("per_device_eval_batch_size", per_device_train_batch_size)),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": float(model_cfg.get("num_train_epochs", 1)),
        "warmup_steps": warmup_steps,
        "weight_decay": float(model_cfg.get("weight_decay", 0.01)),
        "evaluation_strategy": str(plan["evaluation_strategy"] or "epoch"),
        "save_strategy": str(plan["save_strategy"] or "epoch"),
        "logging_steps": int(model_cfg.get("logging_steps", 10)),
        "save_total_limit": int(model_cfg.get("save_total_limit", 1)),
        "remove_unused_columns": False,
        "bf16": bf16,
        "fp16": fp16,
        "optim": str(plan["optim"]),
        "group_by_length": bool(plan["group_by_length"]),
        "auto_find_batch_size": bool(plan["auto_find_batch_size"]),
        "dataloader_num_workers": int(plan["dataloader_num_workers"]),
        "dataloader_pin_memory": bool(plan["dataloader_pin_memory"]),
        "torch_compile": bool(plan["torch_compile"]),
        "torch_compile_backend": str(plan["torch_compile_backend"]),
        "torch_empty_cache_steps": int(plan["torch_empty_cache_steps"]) if plan["torch_empty_cache_steps"] else None,
        "use_mps_device": torch.backends.mps.is_available(),
        "max_steps": max_steps,
        "report_to": [],
        "save_safetensors": True,
        "logging_first_step": True,
    }
    save_steps = int(model_cfg.get("save_steps", 0))
    if save_steps > 0:
        training_kwargs["save_steps"] = save_steps
        if not args.save_strategy:
            training_kwargs["save_strategy"] = "steps"
    training_args = TrainingArguments(**_filter_supported_kwargs(TrainingArguments, training_kwargs))
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
    }
    trainer = Trainer(**_filter_supported_kwargs(Trainer, trainer_kwargs))
    resume_from_checkpoint = args.resume_from_checkpoint.strip() or None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved planner model to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
