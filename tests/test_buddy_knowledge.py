from __future__ import annotations

import json
from pathlib import Path

from memory_client import MLEngineMemory
from src.utils.buddy_knowledge import answer_buddy_question


def test_answer_buddy_question_reports_grounded_workspace_state(tmp_path: Path):
    memory = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory")
    memory.log_trade({
        "timestamp": "2026-02-28T12:00:00+00:00",
        "trade_id": "T001",
        "instrument": "EUR_USD",
        "direction": "long",
        "pnl": 12.5,
    })
    memory.update_runtime_state("online_learning", {"trades_since_retrain": 7})

    model_dir = tmp_path / "trained_data" / "models" / "EUR_USD"
    model_dir.mkdir(parents=True)
    (model_dir / "modular_ensemble.meta.json").write_text(json.dumps({"pair": "EUR_USD"}))

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config_test.yaml"
    config_path.write_text(
        "fx:\n"
        "  risk:\n"
        "    risk_per_trade_pct: 0.02\n"
        "data:\n"
        "  granularity: H1\n"
    )

    answer = answer_buddy_question(
        "How do you work and what do you remember?",
        config_path="config/config_test.yaml",
        root_dir=tmp_path,
        use_llm=False,
    )

    assert "current workspace" in answer.lower()
    assert "modular ensemble" in answer.lower()
    assert "shared ledger counts" in answer.lower()
    assert "trades since last retrain" in answer.lower()
    assert "Sources:" in answer


def test_answer_buddy_question_explains_veto_path_from_code(tmp_path: Path):
    # The veto-path explanation is derived from REAL engine source, so the
    # test root symlinks the actual code files (never a hardcoded absolute
    # path — the previous /Users/davidcertan/... literal broke when the
    # repo moved machines). Memory stays hermetic: a fresh empty
    # trained_data/.memory is created inside tmp_path, so the
    # "does not persist by default" branch is deterministic regardless of
    # what the LIVE runtime state currently holds.
    repo_root = Path(__file__).resolve().parents[1]
    (tmp_path / "cli").symlink_to(repo_root / "cli")
    (tmp_path / "buddy_intelligent_mode.py").symlink_to(
        repo_root / "buddy_intelligent_mode.py"
    )
    answer = answer_buddy_question(
        "Why did you veto that trade?",
        root_dir=tmp_path,
        use_llm=False,
    )

    answer_lower = answer.lower()
    assert "no-trade paths" in answer_lower
    assert "llm_validation.reason" in answer_lower
    assert "losing_streak" in answer_lower
    assert "does not persist every no-trade veto reason by default" in answer_lower


def test_answer_buddy_question_uses_exact_persisted_no_trade_reason(tmp_path: Path):
    memory = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory")
    memory.update_runtime_state(
        "buddy_runtime",
        {
            "last_no_trade_decision": {
                "decision_type": "intelligent_reject",
                "reason": "event risk",
                "failed_gates": ["Ridge confidence (44/100)"],
            }
        },
    )

    answer = answer_buddy_question(
        "What exact no-trade veto reason was used on the last rejected trade?",
        root_dir=tmp_path,
        use_llm=False,
    )

    answer_lower = answer.lower()
    assert "exact last no-trade decision is persisted in runtime state" in answer_lower
    assert "event risk" in answer_lower
    assert "ridge confidence (44/100)" in answer_lower


def test_answer_buddy_question_reports_selected_provider(tmp_path: Path):
    memory = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory")
    memory.update_runtime_state("buddy_runtime", {"selected_llm_provider": "ollama"})

    answer = answer_buddy_question(
        "What provider did you choose?",
        root_dir=tmp_path,
        use_llm=False,
    )

    answer_lower = answer.lower()
    assert "last selected buddy llm provider: ollama" in answer_lower


def test_answer_buddy_question_uses_exact_persisted_trade_execution_reason(tmp_path: Path):
    memory = MLEngineMemory(storage_path=tmp_path / "trained_data" / ".memory")
    memory.update_runtime_state(
        "buddy_runtime",
        {
            "last_trade_execution_decision": {
                "decision_type": "execute",
                "reason": "All model gates passed and intelligent review approved",
                "instrument": "EUR_USD",
                "direction": "long",
                "intelligent_review_reason": "event risk cleared",
                "adaptive_reason": "Adaptive memory neutral after 4/4 wins",
            }
        },
    )

    answer = answer_buddy_question(
        "Why did you take the last trade?",
        root_dir=tmp_path,
        use_llm=False,
    )

    answer_lower = answer.lower()
    assert "exact last trade execution decision is persisted in runtime state" in answer_lower
    assert "all model gates passed and intelligent review approved" in answer_lower
    assert "last persisted trade decision was long on eur_usd" in answer_lower
