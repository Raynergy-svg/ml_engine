"""Grounded self-knowledge helpers for Buddy's conversational layer.

This module answers "Buddy about Buddy" questions using deterministic
workspace/runtime retrieval first, with optional LLM synthesis on top.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from memory_client import MLEngineMemory
from src.utils import load_config

logger = logging.getLogger(__name__)

_TOPIC_ARCHITECTURE = "architecture"
_TOPIC_MEMORY = "memory"
_TOPIC_MODELS = "models"
_TOPIC_PROVIDER = "provider"
_TOPIC_VETO = "veto"
_TOPIC_EXECUTION = "execution"
_TOPIC_CONFIG = "config"
_TOPIC_GENERAL = "general"

_ARCHITECTURE_SOURCES = (
    "buddy_intelligent_mode.py",
    "cli/commands.py",
    "src/utils/unified_talk.py",
    "memory_client.py",
)
_MODEL_SOURCES = (
    "trained_data/models",
    "cli/commands.py",
    "llm_providers.py",
)
_MEMORY_SOURCES = (
    "memory_client.py",
    "market_intelligence.py",
    "buddy_intelligent_mode.py",
)
_VETO_SOURCES = (
    "cli/commands.py",
    "buddy_intelligent_mode.py",
)
_CONFIG_SOURCES = (
    "config",
    "cli/argparser.py",
    "cli/commands.py",
)


def _repo_root(root_dir: Optional[Path] = None) -> Path:
    return Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]


def _resolve_config_path(root: Path, config_path: Optional[str]) -> Optional[Path]:
    if not config_path:
        return None

    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate if candidate.exists() else None


def _classify_topic(query: str) -> str:
    ql = query.lower()
    mentions_veto = any(
        phrase in ql
        for phrase in (
            "why did you veto",
            "why did buddy veto",
            "why did you reject",
            "why was there no trade",
            "why no trade",
            "why did you block",
            "why did intelligent mode reject",
            "veto reason",
            "rejected trade",
            "no-trade veto",
        )
    )
    mentions_config = any(
        phrase in ql
        for phrase in (
            "risk per trade",
            "granularity",
            "what config",
            "which config",
            "what setting",
            "what threshold",
        )
    )
    mentions_execution = any(
        phrase in ql
        for phrase in (
            "why did you take the last trade",
            "why did you take that trade",
            "why did you execute the last trade",
            "why did buddy take the last trade",
            "why did you enter the last trade",
            "why did you buy",
            "why did you sell",
        )
    )
    mentions_memory = any(phrase in ql for phrase in ("memory", "learn", "adapt", "recent trades", "ledger", "remember"))
    mentions_models = any(phrase in ql for phrase in ("model", "models", "provider", "ollama", "llm", "backend"))
    mentions_arch = any(
        phrase in ql
        for phrase in ("how do you work", "who are you", "what are you", "architecture", "intelligent mode", "how are you wired")
    )

    active_topics = sum((mentions_veto, mentions_execution, mentions_config, mentions_memory, mentions_models, mentions_arch))
    if active_topics >= 2:
        return _TOPIC_GENERAL
    if mentions_veto:
        return _TOPIC_VETO
    if mentions_execution:
        return _TOPIC_EXECUTION
    if mentions_config:
        return _TOPIC_CONFIG
    if mentions_memory:
        return _TOPIC_MEMORY
    if mentions_models:
        return _TOPIC_MODELS if "provider" not in ql and "ollama" not in ql and "llm" not in ql else _TOPIC_PROVIDER
    if mentions_arch:
        return _TOPIC_ARCHITECTURE
    return _TOPIC_GENERAL


def _scan_model_inventory(root: Path) -> dict[str, Any]:
    model_root = root / "trained_data" / "models"
    pair_dirs: list[str] = []
    meta_files: list[str] = []

    if model_root.exists():
        for child in sorted(model_root.iterdir()):
            if child.is_dir() and any(child.glob("*.meta.json")):
                pair_dirs.append(child.name)
            elif child.is_file() and child.suffix == ".json" and child.name.endswith(".meta.json"):
                meta_files.append(child.name)

    return {
        "model_root": str(model_root),
        "pair_count": len(pair_dirs),
        "pairs": pair_dirs[:8],
        "meta_files": meta_files[:8],
    }


def _load_memory_snapshot(root: Path) -> dict[str, Any]:
    memory = MLEngineMemory(storage_path=root / "trained_data" / ".memory")
    if hasattr(memory, "get_storage_stats"):
        stats = memory.get_storage_stats()
    else:
        stats = memory.get_stats()
    recent = memory.get_recent_trades(limit=3)
    online_state = memory.get_runtime_state("online_learning")
    buddy_runtime = memory.get_runtime_state("buddy_runtime")
    latest_trade = recent[0] if recent else None

    return {
        "stats": stats,
        "recent_count": len(recent),
        "latest_trade": latest_trade,
        "online_state": online_state,
        "buddy_runtime": buddy_runtime,
    }


def _load_config_snapshot(root: Path, config_path: Optional[str]) -> dict[str, Any]:
    resolved = _resolve_config_path(root, config_path)
    if resolved is None:
        return {"config_path": None, "risk_per_trade_pct": None, "granularity": None}

    try:
        cfg = load_config(str(resolved))
    except Exception as e:
        logger.debug("Failed to load Buddy config from %s: %s", resolved, e)
        return {"config_path": str(resolved), "risk_per_trade_pct": None, "granularity": None}

    return {
        "config_path": str(resolved),
        "risk_per_trade_pct": cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct"),
        "granularity": cfg.get("data", {}).get("granularity") or cfg.get("buddy", {}).get("granularity"),
    }


def _format_latest_trade(latest_trade: dict[str, Any] | None) -> Optional[str]:
    if not latest_trade:
        return None

    instrument = latest_trade.get("instrument") or "unknown"
    direction = latest_trade.get("direction") or "unknown"
    pnl = latest_trade.get("pnl")
    timestamp = latest_trade.get("timestamp") or "unknown time"
    if pnl is None:
        return f"Latest shared trade: {instrument} {direction} at {timestamp}"
    return f"Latest shared trade: {instrument} {direction} pnl={float(pnl):+.2f} at {timestamp}"


def _first_existing_line(path: Path, needles: tuple[str, ...]) -> Optional[int]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for idx, line in enumerate(lines, start=1):
        if any(needle in line for needle in needles):
            return idx
    return None


def _lookup_veto_facts(root: Path, memory: dict[str, Any]) -> tuple[list[str], list[str]]:
    cli_path = root / "cli" / "commands.py"
    intel_path = root / "buddy_intelligent_mode.py"
    cli_reason_line = _first_existing_line(cli_path, ("llm_validation.reason", "NO TRADE: LLM VALIDATION FAILED"))
    gate_reason_line = _first_existing_line(cli_path, ("Reason: {signal.reason}", "failed_gates.append"))
    memory_veto_line = _first_existing_line(intel_path, ("Adaptive memory veto:", "losing_streak >= 4"))

    facts = [
        "Buddy has two main no-trade paths: gate failure and intelligent-review rejection",
        (
            "If a tradable signal is rejected by intelligent mode, cli/commands.py converts the final decision "
            f"to NO TRADE using llm_validation.reason (around line {cli_reason_line or 'unknown'})"
        ),
        (
            "If model gates fail before intelligent approval, cli/commands.py prints signal.reason and lists the specific "
            f"failed gates (around line {gate_reason_line or 'unknown'})"
        ),
        (
            "Adaptive memory can veto in buddy_intelligent_mode.py when recent similar trades show a harsh losing pattern, "
            f"including losing_streak >= 4 or very weak weighted win rate (around line {memory_veto_line or 'unknown'})"
        ),
    ]

    latest_trade = memory.get("latest_trade") or {}
    metadata = latest_trade.get("metadata") if isinstance(latest_trade, dict) else {}
    runtime_state = memory.get("buddy_runtime") or {}
    last_no_trade = runtime_state.get("last_no_trade_decision") if isinstance(runtime_state, dict) else None
    veto_reason = None
    if isinstance(metadata, dict):
        veto_reason = metadata.get("pre_trade_block_reason") or metadata.get("gate_reason")
    if isinstance(last_no_trade, dict) and last_no_trade.get("reason"):
        facts.append(
            "Exact last no-trade decision is persisted in runtime state: "
            f"{last_no_trade.get('decision_type', 'unknown')} because {last_no_trade.get('reason')}"
        )
        failed_gates = last_no_trade.get("failed_gates")
        if isinstance(failed_gates, list) and failed_gates:
            facts.append("Last persisted failed gates: " + ", ".join(str(item) for item in failed_gates[:4]))
    elif veto_reason:
        facts.append(f"Latest stored trade metadata mentions a pre-trade block reason: {veto_reason}")
    else:
        facts.append(
            "The shared trade ledger stores executed trades and runtime counters, but it does not persist every no-trade veto reason by default"
        )

    return facts, list(_VETO_SOURCES)


def _lookup_config_facts(config: dict[str, Any], *, root: Path) -> tuple[list[str], list[str]]:
    cli_argparser_path = root / "cli" / "argparser.py"
    llm_arg_line = _first_existing_line(cli_argparser_path, ("--llm-provider",))
    explain_arg_line = _first_existing_line(cli_argparser_path, ("--explain",))

    facts: list[str] = []
    if config.get("config_path"):
        facts.append(f"Active config path: {config['config_path']}")
    if config.get("granularity"):
        facts.append(f"Configured default granularity: {config['granularity']}")
    if config.get("risk_per_trade_pct") is not None:
        facts.append(f"Configured FX risk per trade: {float(config['risk_per_trade_pct']):.4f}")
    facts.append(
        f"CLI intelligent-mode controls are defined in cli/argparser.py around lines {llm_arg_line or 'unknown'} and {explain_arg_line or 'unknown'}"
    )
    return facts, list(_CONFIG_SOURCES)


def _lookup_execution_facts(memory: dict[str, Any]) -> tuple[list[str], list[str]]:
    runtime_state = memory.get("buddy_runtime") or {}
    last_execution = runtime_state.get("last_trade_execution_decision") if isinstance(runtime_state, dict) else None

    if isinstance(last_execution, dict) and last_execution.get("reason"):
        facts = [
            "Exact last trade execution decision is persisted in runtime state: "
            f"{last_execution.get('decision_type', 'unknown')} because {last_execution.get('reason')}",
        ]
        instrument = last_execution.get("instrument")
        direction = last_execution.get("direction")
        if instrument and direction:
            facts.append(f"Last persisted trade decision was {direction} on {instrument}")
        review_reason = last_execution.get("intelligent_review_reason")
        if review_reason:
            facts.append(f"Intelligent review note for that trade: {review_reason}")
        adaptive_reason = last_execution.get("adaptive_reason")
        if adaptive_reason:
            facts.append(f"Adaptive memory note for that trade: {adaptive_reason}")
        return facts, ["cli/commands.py", "memory_client.py"]

    return [
        "No Buddy trade execution decision has been persisted yet"
    ], ["cli/commands.py", "memory_client.py"]


def _lookup_provider_facts(memory: dict[str, Any]) -> tuple[list[str], list[str]]:
    facts: list[str] = []

    runtime_state = memory.get("buddy_runtime") or {}
    selected_provider = runtime_state.get("selected_llm_provider") if isinstance(runtime_state, dict) else None
    if selected_provider:
        facts.append(f"Last selected Buddy LLM provider: {selected_provider}")

    try:
        from llm_providers import select_buddy_provider_name

        preferred = select_buddy_provider_name(None)
        if preferred:
            facts.append(f"Current Buddy provider preference resolves to: {preferred}")
        else:
            facts.append("Current Buddy provider preference resolves to: unavailable")
    except Exception:
        pass

    if not facts:
        facts.append("No Buddy LLM provider has been persisted yet")

    return facts, list(_MODEL_SOURCES)


def _collect_facts(topic: str, *, root: Path, config_path: Optional[str]) -> tuple[list[str], list[str]]:
    inventory = _scan_model_inventory(root)
    memory = _load_memory_snapshot(root)
    config = _load_config_snapshot(root, config_path)

    facts: list[str] = []
    sources: list[str] = []

    if topic in {_TOPIC_ARCHITECTURE, _TOPIC_GENERAL}:
        facts.extend([
            (
                "Trading core is a pair-specific modular ensemble under "
                f"{inventory['model_root']} with {inventory['pair_count']} pair model directories discovered"
            ),
            "Reasoning and action are connected through cli/commands.py calling buddy_intelligent_mode.py",
            "Conversation planning and tool execution live in src/utils/unified_talk.py",
            "Shared runtime memory is persisted through trained_data/.memory via memory_client.py",
        ])
        sources.extend(_ARCHITECTURE_SOURCES)

    if topic in {_TOPIC_MEMORY, _TOPIC_GENERAL}:
        stats = memory["stats"]
        facts.append(
            "Shared ledger counts: "
            f"{stats.get('trade_records_count', 0)} trades, "
            f"{stats.get('scan_results_count', 0)} scans, "
            f"{stats.get('training_sessions_count', 0)} training sessions"
        )
        online_state = memory["online_state"] or {}
        if online_state:
            trades_since_retrain = online_state.get("trades_since_retrain")
            if trades_since_retrain is not None:
                facts.append(
                    f"Online learner runtime state shows {int(trades_since_retrain)} trades since last retrain"
                )
        latest_trade_line = _format_latest_trade(memory["latest_trade"])
        if latest_trade_line:
            facts.append(latest_trade_line)
        sources.extend(_MEMORY_SOURCES)

    if topic in {_TOPIC_MODELS, _TOPIC_PROVIDER, _TOPIC_GENERAL}:
        pair_names = ", ".join(inventory["pairs"][:5]) if inventory["pairs"] else "none found"
        facts.append(f"Pair model directories found: {pair_names}")
        if inventory["meta_files"]:
            facts.append("Top-level model metadata files: " + ", ".join(inventory["meta_files"][:4]))
        if config.get("config_path"):
            facts.append(f"Active config path: {config['config_path']}")
        if config.get("risk_per_trade_pct") is not None:
            facts.append(f"Configured FX risk per trade: {float(config['risk_per_trade_pct']):.4f}")
        sources.extend(_MODEL_SOURCES)
        if topic in {_TOPIC_PROVIDER, _TOPIC_GENERAL}:
            provider_facts, provider_sources = _lookup_provider_facts(memory)
            facts.extend(provider_facts)
            sources.extend(provider_sources)

    if topic in {_TOPIC_VETO, _TOPIC_GENERAL}:
        veto_facts, veto_sources = _lookup_veto_facts(root, memory)
        facts.extend(veto_facts)
        sources.extend(veto_sources)

    if topic in {_TOPIC_EXECUTION, _TOPIC_GENERAL}:
        execution_facts, execution_sources = _lookup_execution_facts(memory)
        facts.extend(execution_facts)
        sources.extend(execution_sources)

    if topic in {_TOPIC_CONFIG, _TOPIC_GENERAL}:
        config_facts, config_sources = _lookup_config_facts(config, root=root)
        facts.extend(config_facts)
        sources.extend(config_sources)

    deduped_sources = list(dict.fromkeys(sources))
    return facts, deduped_sources


def _local_llm_grounded_summary(query: str, facts: list[str], sources: list[str]) -> Optional[str]:
    if not facts:
        return None

    try:
        from llm_providers import llm_call, select_buddy_provider_name
    except Exception:
        return None

    provider_name = select_buddy_provider_name(None)
    if not provider_name:
        return None

    prompt = (
        "Answer the user's Buddy-self question using only the grounded facts below.\n"
        "If a detail is not in the facts, say that directly.\n"
        "Keep the answer concise and factual.\n\n"
        f"User question: {query}\n\n"
        "Grounded facts:\n- " + "\n- ".join(facts) + "\n\n"
        "Sources:\n- " + "\n- ".join(sources)
    )
    system_prompt = (
        "You are Buddy's self-knowledge assistant. "
        "Never invent architecture, runtime state, model counts, or memory details. "
        "Only summarize the provided facts."
    )

    try:
        return llm_call(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=provider_name,
            temperature=0.1,
            max_tokens=280,
        )
    except Exception as e:
        logger.debug("Grounded Buddy LLM summary failed: %s", e)
        return None


def _deterministic_summary(query: str, facts: list[str], sources: list[str]) -> str:
    head = "Based on my current workspace:"
    if re.search(r"\bwho are you\b|\bwhat are you\b", query.lower()):
        head = "Based on my current workspace, I am:"

    lines = [head]
    lines.extend(f"- {fact}" for fact in facts[:6])
    if sources:
        lines.append("Sources: " + ", ".join(sources[:4]))
    return "\n".join(lines)


def answer_buddy_question(
    query: str,
    *,
    config_path: Optional[str] = None,
    root_dir: Optional[Path] = None,
    use_llm: bool = True,
) -> str:
    """Answer Buddy self-questions from grounded workspace/runtime data."""
    root = _repo_root(root_dir)
    topic = _classify_topic(query)
    facts, sources = _collect_facts(topic, root=root, config_path=config_path)

    if use_llm:
        grounded = _local_llm_grounded_summary(query, facts, sources)
        if grounded:
            return grounded.strip()

    return _deterministic_summary(query, facts, sources)
