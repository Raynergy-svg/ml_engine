"""Interactive chat for unified multi-head training diagnostics.

This is intentionally minimal: it reads the most recent saved unified training
metrics (JSON) and uses the reasoning engine to answer questions like:
- which head is falling behind?
- show latest head health
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from reasoning_enhanced import ReasoningEngine
from utils import load_config


def _default_metrics_path(cfg: Dict[str, Any]) -> Path:
    return Path(
        cfg.get("METRICS_DIR")
        or cfg.get("paths", {}).get("metrics_dir")
        or cfg.get("paths", {}).get("log_dir")
        or "trained_data/logs"
    ) / "unified_training_metrics.json"


def _find_newest_metrics_file(root: Path) -> Optional[Path]:
    try:
        candidates = list(root.rglob("unified_training_metrics.json"))
    except Exception:
        return None
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_newest_training_log(root: Path) -> Optional[Path]:
    """Locate a log file that likely contains unified training epoch lines."""
    names = ("unified_multitask_training.log", "cli.log")
    candidates: list[Path] = []
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            candidates.append(p)

    # Fall back to searching under root (bounded to these filenames).
    try:
        for name in names:
            candidates.extend([p for p in root.rglob(name) if p.is_file()])
    except Exception:
        pass

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


_EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+val\s+"
    r"price=(?P<price>[0-9.]+)\s+"
    r"trend=(?P<trend>[0-9.]+)\s+"
    r"risk=(?P<risk>[0-9.]+)\s+"
    r"state=(?P<state>[0-9.]+)\s+"
    r"acc=(?P<acc>[0-9.]+)"
)


def _reconstruct_metrics_from_log(log_path: Path) -> Optional[Dict[str, Any]]:
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None

    history: list[Dict[str, Any]] = []
    by_epoch: dict[int, Dict[str, Any]] = {}
    for line in text.splitlines():
        m = _EPOCH_RE.search(line)
        if not m:
            continue
        epoch = int(m.group("epoch"))
        by_epoch[epoch] = {
            "epoch": epoch,
            "val": {
                "price_loss": float(m.group("price")),
                "trend_loss": float(m.group("trend")),
                "risk_loss": float(m.group("risk")),
                "state_loss": float(m.group("state")),
                "state_acc": float(m.group("acc")),
            },
        }

    if not by_epoch:
        return None

    for epoch in sorted(by_epoch):
        history.append(by_epoch[epoch])

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv": None,
        "model_path": None,
        "history": history,
        "source": str(log_path),
    }


def _common_metrics_roots() -> list[Path]:
    return [Path.cwd(), Path("trained_data"), Path("output")]


def _locate_metrics_file(default_path: Path, roots: list[Path]) -> Path:
    if default_path.exists():
        return default_path
    for root in roots:
        found = _find_newest_metrics_file(root)
        if found is not None:
            return found
    return default_path


def _reconstruct_metrics_to_default(cfg: Dict[str, Any], roots: list[Path]) -> Optional[Path]:
    for root in roots:
        log_path = _find_newest_training_log(root)
        if log_path is None:
            continue

        reconstructed = _reconstruct_metrics_from_log(log_path)
        if reconstructed is None:
            continue

        out_path = _default_metrics_path(cfg)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(reconstructed, indent=2, sort_keys=True))
        except Exception:
            continue
        return out_path

    return None


def _load_metrics(path: Path, *, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    roots = _common_metrics_roots()
    cfg = cfg or {}

    path = _locate_metrics_file(path, roots)
    if not path.exists():
        reconstructed_path = _reconstruct_metrics_to_default(cfg, roots)
        if reconstructed_path is not None:
            path = reconstructed_path

    if not path.exists():
        raise FileNotFoundError(
            "Metrics file not found. Run unified training once to generate it:\n"
            "  python main.py train-unified --config ./config.yaml --csv <file.csv>\n"
            "Then run:\n"
            "  python main.py chat-unified --config ./config.yaml"
        )

    payload = json.loads(path.read_text())
    # Attach the resolved path so callers can print it.
    if isinstance(payload, dict) and "metrics_path" not in payload:
        payload["metrics_path"] = str(path)
    return payload


def _extract_latest(history: Any) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    items: list[Dict[str, Any]] = [x for x in (history or []) if isinstance(x, dict)]
    if not items:
        return {}, []
    latest = items[-1].get("val", {}) if isinstance(items[-1].get("val", {}), dict) else {}
    return latest, items


_EXIT_WORDS = {"exit", "quit", "q"}


def _read_chat_question() -> Optional[str]:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not q:
        return ""
    if q.lower() in _EXIT_WORDS:
        return None
    return q


def _reload_metrics_for_chat(resolved: Path, cfg: Dict[str, Any]) -> Optional[tuple[Dict[str, Any], list[Dict[str, Any]]]]:
    try:
        payload = _load_metrics(resolved, cfg=cfg)
        return _extract_latest(payload.get("history"))
    except Exception as e:
        print(f"Assistant: I couldn't read metrics ({e}).")
        return None


def _wants_numbers(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ("numbers", "metrics", "loss", "acc", "show"))


def _format_latest_val(latest_val: Dict[str, Any]) -> str:
    compact = ", ".join(
        f"{k}={latest_val[k]:.4f}"
        for k in ("price_loss", "trend_loss", "risk_loss", "state_loss")
        if k in latest_val
    )
    if "state_acc" in latest_val:
        compact += f", state_acc={float(latest_val['state_acc']):.3f}"
    return compact


def _print_insights(summary: Dict[str, Any]) -> None:
    for line in summary.get("insights", []):
        print(f"Assistant: {line}")


def run_unified_chat(config_path: str, *, metrics_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    reasoning = ReasoningEngine(cfg.get("reasoning", {}))

    path = Path(metrics_path) if metrics_path else _default_metrics_path(cfg)
    payload = _load_metrics(path, cfg=cfg)
    resolved = Path(payload.get("metrics_path") or path)

    print(f"Unified chat ready. Metrics: {resolved}")
    print("Ask about head health (type 'exit' to quit).")

    while True:
        q = _read_chat_question()
        if q is None:
            print("Bye.")
            return
        if not q:
            continue

        reloaded = _reload_metrics_for_chat(resolved, cfg)
        if reloaded is None:
            continue
        latest_val, history = reloaded

        summary = reasoning.summarize_head_health(latest_val, history=history)

        if _wants_numbers(q) and latest_val:
            print(f"Assistant: latest val -> {_format_latest_val(latest_val)}")

        _print_insights(summary)
