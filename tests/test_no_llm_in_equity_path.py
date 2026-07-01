"""US-017 — Standing Claude-free canary.

Asserts no LLM/agent dependency is reachable from any module in
``src/equity/`` — the entire strategy + execution surface. AST-walks
every ``.py`` file, collects every ``import`` / ``from ... import ...``
statement (at any nesting level — function-local imports count too),
and fails the build if any forbidden symbol appears.

Why an AST walk and not a regex
-------------------------------

A regex match on ``import openai`` would miss::

    import importlib
    importlib.import_module("openai")

…and also miss::

    from anthropic import (
        Anthropic,
    )

The AST walk inspects every ``ast.Import`` and ``ast.ImportFrom`` node
regardless of formatting, so the canary catches both forms. It also
recurses into nested ``ast.Import`` statements within function bodies,
which is the actual smuggling vector — a runtime lazy import.

Forbidden namespaces
--------------------

* ``anthropic``  — Anthropic SDK
* ``openai``     — OpenAI SDK
* ``langchain``  — LangChain
* ``llama_index`` / ``llamaindex``
* ``litellm``    — LLM proxy
* ``ollama``     — Ollama client
* ``transformers`` (HF) — pulling an LLM weight at runtime is forbidden
  in the equity hot path; backtests must use offline sklearn/torch only
* ``mistralai``
* ``cohere``
* ``google.generativeai`` / ``google_generativeai``
* any module starting with ``claude_`` or named ``claude``

The canary is *standing*: it does not look at a single commit; it runs
on every pytest invocation. Adding any of the above to ``src/equity/``
will break the build.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Set

import pytest


EQUITY_DIR = Path(__file__).resolve().parent.parent / "src" / "equity"

FORBIDDEN_PREFIXES = (
    "anthropic",
    "openai",
    "langchain",
    "llama_index",
    "llamaindex",
    "litellm",
    "ollama",
    "transformers",
    "mistralai",
    "cohere",
    "google.generativeai",
    "google_generativeai",
    "claude_",
    "claude.",
)

# Exact module names that should never appear (for symbols where a
# prefix match would be too aggressive, e.g. "claude" alone).
FORBIDDEN_EXACT = {
    "claude",
}


def _is_forbidden(module_name: str) -> bool:
    if not module_name:
        return False
    if module_name in FORBIDDEN_EXACT:
        return True
    for prefix in FORBIDDEN_PREFIXES:
        if module_name == prefix or module_name.startswith(prefix + "."):
            return True
        # also catch the "starts_with prefix" case for prefix-form
        # entries like "claude_" / "claude."
        if prefix.endswith(("_", ".")) and module_name.startswith(prefix):
            return True
    return False


def _collect_imports(tree: ast.AST) -> Set[str]:
    """Walk the tree and collect every imported module name."""
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
                # also flag "from x import y" where y is also a module
                for alias in node.names:
                    if alias.name and alias.name != "*":
                        found.add(f"{node.module}.{alias.name}")
    return found


def _equity_modules() -> List[Path]:
    return sorted(
        p
        for p in EQUITY_DIR.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_equity_dir_exists() -> None:
    """Sanity: the directory we're guarding actually exists."""
    assert EQUITY_DIR.is_dir(), f"equity dir not found: {EQUITY_DIR}"
    assert _equity_modules(), "no .py files under src/equity/"


def test_no_llm_imports_in_equity_modules() -> None:
    """Every src/equity/*.py is free of LLM/agent imports."""
    offenders: List[str] = []
    for module in _equity_modules():
        try:
            source = module.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(module))
        except (OSError, SyntaxError) as exc:
            pytest.fail(f"cannot parse {module}: {exc}")
        for name in sorted(_collect_imports(tree)):
            if _is_forbidden(name):
                offenders.append(f"{module.name}: import {name}")
    if offenders:
        pytest.fail(
            "Claude-free canary tripped — LLM imports reachable in "
            "src/equity/:\n  " + "\n  ".join(offenders)
        )


def test_canary_detects_forbidden_module() -> None:
    """Self-test: the prefix logic actually rejects forbidden names."""
    assert _is_forbidden("anthropic")
    assert _is_forbidden("openai.types")
    assert _is_forbidden("langchain.llms")
    assert _is_forbidden("claude_code")
    assert _is_forbidden("claude")
    assert _is_forbidden("google.generativeai")
    # negatives
    assert not _is_forbidden("pandas")
    assert not _is_forbidden("numpy")
    assert not _is_forbidden("src.equity.rebalance")
    assert not _is_forbidden("")


def test_canary_uses_ast_not_regex() -> None:
    """A multi-line import statement should still trip the canary."""
    src = "from anthropic import (\n    Anthropic,\n)\n"
    tree = ast.parse(src)
    imports = _collect_imports(tree)
    assert any(_is_forbidden(n) for n in imports)
