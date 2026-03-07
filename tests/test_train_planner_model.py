from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import zipfile

import yaml


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "train_planner_model.py"
    spec = importlib.util.spec_from_file_location("train_planner_model", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_dataset_builder_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_planner_dataset.py"
    spec = importlib.util.spec_from_file_location("build_planner_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _planner_row() -> dict[str, object]:
    return {
        "id": "planner_000001",
        "split": "train",
        "user_message": "show status",
        "chat_history": [],
        "runtime_context": {"granularity": "H1"},
        "available_tools": [{"name": "get_status", "description": "status", "args_schema": {}}],
        "target_plan": {
            "needs_clarification": False,
            "clarification_question": None,
            "final_answer": None,
            "steps": [{"tool": "get_status", "args": {}}],
        },
        "tool_observations": [],
        "target_response": "Status loaded.",
        "safety_label": "tool_grounded",
        "tags": ["status"],
        "training_mode": "planner_json",
        "grounded_answer": "Status loaded.",
        "target_reply": "I'm loaded and ready with the current status.",
    }


def test_prepare_files_and_write_colab_bundle(tmp_path: Path):
    module = _load_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train_path = raw_dir / "train.jsonl"
    valid_path = raw_dir / "valid.jsonl"
    row = _planner_row()
    train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    valid_path.write_text(json.dumps({**row, "split": "valid", "id": "planner_000002"}) + "\n", encoding="utf-8")

    config_path = tmp_path / "planner_training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "output_dir": "trained_data/planner/checkpoints",
                },
                "dataset": {
                    "train_path": "raw/train.jsonl",
                    "valid_path": "raw/valid.jsonl",
                },
                "lora": {"enabled": True},
                "format": {"system_prompt": "You are Buddy's planner model."},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = module._load_config(config_path)
    prepared_dir = tmp_path / "prepared"
    train_out, valid_out = module._prepare_sft_files(cfg, config_path=config_path, prepared_dir=prepared_dir)

    prepared_row = json.loads(train_out.read_text(encoding="utf-8").strip())
    assert set(prepared_row) == {"prompt", "completion"}
    assert "Buddy's planner model" in prepared_row["prompt"]
    assert valid_out.exists()

    bundle_dir = tmp_path / "planner_colab_bundle"
    module._write_colab_bundle(
        cfg,
        bundle_dir=bundle_dir,
        prepared_train_path=train_out,
        prepared_valid_path=valid_out,
    )

    bundle_cfg = yaml.safe_load((bundle_dir / "planner_training_colab.yaml").read_text(encoding="utf-8"))
    assert bundle_cfg["dataset"]["train_path"] == "sft_train.jsonl"
    assert bundle_cfg["dataset"]["valid_path"] == "sft_valid.jsonl"
    assert bundle_cfg["model"]["output_dir"] == "planner_adapter"
    assert (bundle_dir / "train_planner_model.py").exists()
    assert (bundle_dir / "README.md").exists()

    archive_path = bundle_dir.parent / f"{bundle_dir.name}.zip"
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
    assert "planner_training_colab.yaml" in names
    assert "sft_train.jsonl" in names
    assert "train_planner_model.py" in names


def test_format_example_supports_direct_buddy_reply_mode():
    module = _load_module()
    row = {
        **_planner_row(),
        "training_mode": "buddy_reply",
        "grounded_answer": "Status loaded.",
        "target_reply": "I'm on EUR/USD and ready to walk through the current mode.",
        "tool_observations": [{"tool": "get_status", "args": {}, "result": {"execute_mode": "dry-run"}}],
        "response_system_prompt": "You are Buddy Planner 3B speaking directly to the user.",
    }

    formatted = module._format_example(row, "unused system prompt")

    prompt = json.loads(formatted["prompt"])
    assert prompt["system"] == "You are Buddy Planner 3B speaking directly to the user."
    assert prompt["grounded_answer"] == "Status loaded."
    assert prompt["tool_outputs"][0]["tool"] == "get_status"
    assert formatted["completion"] == "I'm on EUR/USD and ready to walk through the current mode."


def test_dataset_builder_routes_plain_english_prompts_to_scan():
    module = _load_dataset_builder_module()
    rows = module._generate_records()

    by_prompt = {row["user_message"]: row for row in rows if row["training_mode"] == "planner_json"}

    for prompt in ("do the thing", "go ahead", "find best pair to trade tomorrow"):
        row = by_prompt[prompt]
        assert row["target_plan"]["needs_clarification"] is False
        assert row["target_plan"]["steps"][0]["tool"] == "scan_market"

    clarification_row = by_prompt["write me a poem"]
    assert clarification_row["target_plan"]["needs_clarification"] is True


def test_resolve_warmup_steps_from_max_steps():
    module = _load_module()
    steps = module._resolve_warmup_steps(
        {"warmup_ratio": 0.05},
        max_steps=200,
        train_dataset_len=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
    )
    assert steps == 10


def test_override_config_supports_long_run_overrides():
    module = _load_module()
    cfg = {
        "model": {"base_model": "Qwen/Qwen2.5-1.5B-Instruct"},
        "dataset": {"train_path": "train.jsonl", "valid_path": "valid.jsonl"},
        "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.05},
    }
    args = argparse.Namespace(
        train_path="",
        valid_path="",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        output_dir="planner_runs/run_001",
        max_seq_length=2048,
        learning_rate=1.5e-5,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=1,
        warmup_steps=40,
        save_steps=100,
        save_total_limit=4,
        num_train_epochs=0.0,
        gradient_checkpointing=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    module._override_config(cfg, args)

    assert cfg["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg["model"]["output_dir"] == "planner_runs/run_001"
    assert cfg["model"]["max_seq_length"] == 2048
    assert cfg["model"]["learning_rate"] == 1.5e-5
    assert cfg["model"]["per_device_train_batch_size"] == 48
    assert cfg["model"]["gradient_accumulation_steps"] == 2
    assert cfg["model"]["warmup_steps"] == 40
    assert cfg["model"]["save_steps"] == 100
    assert cfg["model"]["save_total_limit"] == 4
    assert cfg["model"]["gradient_checkpointing"] is False
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["alpha"] == 32
    assert cfg["lora"]["dropout"] == 0.1


def test_build_planner_dataset_includes_new_chat_tools_and_more_rows():
    module = _load_dataset_builder_module()
    rows = module._generate_records()

    assert len(rows) >= 1100
    tools = {step["tool"] for row in rows for step in row["target_plan"].get("steps", [])}
    modes = {row.get("training_mode") for row in rows}
    assert {
        "use_market_source",
        "run_prediction",
        "summarize_last_prediction",
        "run_runtime_command",
        "override_guardrail",
        "run_oanda_command",
        "run_trade_command",
    }.issubset(tools)
    assert {"planner_json", "buddy_reply"}.issubset(modes)
    assert any("capabilities" in (row.get("tags") or []) for row in rows)


def test_default_planner_training_config_uses_3b_and_response_prompt():
    config_path = Path(__file__).resolve().parents[1] / "config" / "planner_training.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["model"]["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert "response_system_prompt" in cfg["format"]
