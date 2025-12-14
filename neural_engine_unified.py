"""Unified neural engine.

Goal: replace separate ML/MT/MR engines + integrator composition with a single
shared-backbone network that can output multiple heads.

This module is intentionally lightweight so it can be used from existing
entrypoints without forcing a full training pipeline refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def safe_torch_load(checkpoint_path: str, *, map_location: str):
    """Load project checkpoints safely on PyTorch>=2.6.

    PyTorch changed `torch.load()` default `weights_only=True`, which uses a
    restricted unpickler. Our checkpoints may include sklearn scalers inside
    metadata; we allowlist the known scaler classes used by this project.
    """

    import inspect

    def _torch_load(*, weights_only: Optional[bool]):
        kwargs: Dict[str, Any] = {"map_location": map_location}
        try:
            if weights_only is not None and "weights_only" in inspect.signature(torch.load).parameters:
                kwargs["weights_only"] = bool(weights_only)
        except Exception:
            pass
        return torch.load(checkpoint_path, **kwargs)

    try:
        repo_root = Path(__file__).resolve().parent
        ckpt_resolved = Path(checkpoint_path).resolve()
        in_repo = repo_root == ckpt_resolved or repo_root in ckpt_resolved.parents
    except Exception:
        in_repo = False

    # For trusted, repo-local checkpoints (trained by this project), load with
    # full pickle support to avoid PyTorch 2.6+ `weights_only=True` restrictions.
    if in_repo:
        return _torch_load(weights_only=False)

    try:
        from torch.serialization import safe_globals  # type: ignore
    except Exception:
        # Older torch versions without safe globals.
        return _torch_load(weights_only=False)

    allow: list[object] = []
    try:
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler  # type: ignore

        allow.extend([RobustScaler, StandardScaler, MinMaxScaler])
    except Exception:
        allow = []

    try:
        if allow:
            with safe_globals(allow):
                return _torch_load(weights_only=True)
        return _torch_load(weights_only=True)
    except Exception as e:
        # If restricted loading still fails, fall back to full pickle loading
        # only for checkpoints that live inside this repo directory.
        # NOTE: `weights_only=False` can execute arbitrary code during unpickling.
        # Use only for trusted checkpoints.
        if in_repo:
            return _torch_load(weights_only=False)

        raise e


class UnifiedMarketNet(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        state_classes: int = 3,
    ) -> None:
        super().__init__()

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        rep_size = hidden_size * (2 if bidirectional else 1)

        self.price_head = nn.Linear(rep_size, 1)
        self.trend_head = nn.Linear(rep_size, 1)
        self.risk_head = nn.Sequential(nn.Linear(rep_size, 1), nn.Sigmoid())
        self.state_head = nn.Linear(rep_size, int(state_classes))

    def forward(self, x: torch.Tensor, *, return_all: bool = False):
        out, _ = self._lstm(x)
        rep = out[:, -1, :]

        price = self.price_head(rep)
        if not return_all:
            return price

        trend = self.trend_head(rep)
        risk = self.risk_head(rep)
        state_logits = self.state_head(rep)
        return {
            "price": price,
            "trend": trend,
            "risk": risk,
            "state_logits": state_logits,
        }


@dataclass
class UnifiedPrediction:
    prediction: np.ndarray
    uncertainty: float
    trend: Optional[np.ndarray] = None
    risk: Optional[np.ndarray] = None
    state_probs: Optional[np.ndarray] = None


class UnifiedNeuralEngine:
    """Thin wrapper that exposes a `predict()` API compatible with the integrator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        model_cfg = self.config.get("model", {})
        input_size = int(model_cfg.get("input_size", 32))
        hidden_size = int(model_cfg.get("hidden_size", 64))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.1))
        bidirectional = bool(model_cfg.get("bidirectional", False))
        state_classes = int(model_cfg.get("state_classes", 3))

        self.model = UnifiedMarketNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            state_classes=state_classes,
        ).to(self.device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, *, device: Optional[str] = None) -> "UnifiedNeuralEngine":
        ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
        model_params = dict(ckpt.get("model_params") or {})
        cfg = dict(ckpt.get("config") or {})
        
        # Infer model params from state_dict if not in checkpoint
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or {}
        if state_dict and not model_params.get("input_size"):
            # Infer input_size from LSTM weight shape
            lstm_key = "_lstm.weight_ih_l0"
            if lstm_key in state_dict:
                model_params["input_size"] = int(state_dict[lstm_key].shape[1])
            # Infer hidden_size
            if lstm_key in state_dict:
                model_params["hidden_size"] = int(state_dict[lstm_key].shape[0] // 4)
            # Infer state_classes from state_head
            state_head_key = "state_head.weight"
            if state_head_key in state_dict:
                model_params["state_classes"] = int(state_dict[state_head_key].shape[0])
        
        if device is not None:
            cfg["device"] = device
        cfg = {"device": cfg.get("device", device), "model": {**(cfg.get("model") or {}), **model_params}}
        eng = cls(cfg)
        eng.load_checkpoint(checkpoint_path)
        return eng

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        ckpt = safe_torch_load(checkpoint_path, map_location=str(self.device))
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if state is None:
            raise ValueError("Checkpoint missing model_state_dict")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        return ckpt

    @staticmethod
    def _coerce_tensor(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def _align_and_concat(seq_a: torch.Tensor, seq_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align by truncating to the smallest sequence length."""
        if seq_a.dim() != 3 or seq_b.dim() != 3:
            raise ValueError("Expected (batch, seq, features) tensors")
        min_len = min(int(seq_a.shape[1]), int(seq_b.shape[1]))
        return seq_a[:, :min_len, :], seq_b[:, :min_len, :]

    def _combine_features(self, features: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            x = features
            if x.dim() == 2:
                x = x.unsqueeze(1)
            return x

        if not isinstance(features, dict):
            raise TypeError("features must be a Tensor or dict")

        # Common callers pass these keys.
        parts = []
        for key in ("features", "ml_features", "mt_features", "mr_features"):
            val = features.get(key)
            if val is None:
                continue
            parts.append(self._coerce_tensor(val))

        if not parts:
            raise ValueError("No usable features found")

        # Normalize each part to (batch, seq, feat)
        norm_parts = []
        for p in parts:
            if p.dim() == 1:
                p = p.view(p.shape[0], 1, 1)
            elif p.dim() == 2:
                p = p.unsqueeze(1)
            elif p.dim() > 3:
                p = p.reshape(p.shape[0], p.shape[1], -1)
            norm_parts.append(p)

        x = norm_parts[0]
        for nxt in norm_parts[1:]:
            x, nxt = self._align_and_concat(x, nxt)
            x = torch.cat([x, nxt], dim=-1)

        return x

    def predict(self, features: Union[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        x = self._combine_features(features).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x, return_all=True)

        price = out["price"].detach().cpu().numpy()
        trend = out["trend"].detach().cpu().numpy()
        risk = out["risk"].detach().cpu().numpy()
        state_probs = torch.softmax(out["state_logits"], dim=-1).detach().cpu().numpy()

        # Use mean risk as a simple uncertainty proxy.
        uncertainty = float(np.clip(np.mean(risk), 0.0, 1.0))

        return {
            "prediction": price,
            "uncertainty": uncertainty,
            "trend": trend,
            "risk": risk,
            "state_probs": state_probs,
        }
