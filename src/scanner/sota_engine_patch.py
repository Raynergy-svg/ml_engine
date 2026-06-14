"""
Minimal patch layer for Scanner engine SOTA integration.

Instead of editing the 6,000-line engine.py directly, this module
provides a `patch_scanner_for_sota(scanner)` function that monkey-patches
the Scanner instance at runtime.  Call it immediately after Scanner.__init__
when either use_sota_inference or use_neural_agents is True.

This keeps the patch isolated, reviewable, and reversible.
"""

from __future__ import annotations

import logging
from typing import Any

from src.scanner.config import ScannerConfig
from src.scanner.sota_integration import resolve_inference_engine, resolve_agent_team

logger = logging.getLogger(__name__)


def patch_scanner_for_sota(scanner: Any) -> None:
    """Swap inference engine and/or agent team on a Scanner instance.

    Safe to call even when SOTA flags are False (no-op).
    """
    if not isinstance(getattr(scanner, "config", None), ScannerConfig):
        logger.debug("patch_scanner_for_sota: scanner has no config, skipping")
        return

    cfg: ScannerConfig = scanner.config

    # --- Goal 1: SOTA Inference Engine ---
    if getattr(cfg, "use_sota_inference", False):
        # MED-4 FIX: clear stale state before patching
        scanner._ensemble_health = {}
        scanner._ensemble_loaded = False
        scanner._ensemble_type = None
        try:
            new_engine = resolve_inference_engine(cfg, scanner)
            if new_engine is not None:
                scanner._modular_ensemble = new_engine
                scanner._ensemble_loaded = True
                scanner._ensemble_type = "SOTAInference"
                logger.info("Scanner patched: using SOTAInference")
        except Exception as e:
            logger.warning(f"Failed to patch SOTA inference: {e}")

    # --- Goal 2: Neural Agent Team ---
    if getattr(cfg, "use_neural_agents", False):
        try:
            new_team = resolve_agent_team(cfg)
            if new_team is not None:
                scanner._agent_team = new_team
                logger.info("Scanner patched: using NeuralAgentTeam")
        except Exception as e:
            logger.warning(f"Failed to patch neural agents: {e}")
