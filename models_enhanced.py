"""
Enhanced model implementations with state-of-the-art architectures
for time series prediction.
Includes advanced models with better performance and numerical
stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union, List


class StockPredictor(nn.Module):
    """Enhanced LSTM-based predictor with residual connections and layer normalization."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
    ):
        super(StockPredictor, self).__init__()

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)

        self.dropout = nn.Dropout(dropout)

        # Improved architecture with residual connections
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Skip connection
        self.skip = nn.Linear(lstm_output_size, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)

        lstm_out = self.dropout(lstm_out)

        # Skip connection
        skip_out = self.skip(lstm_out)

        # Main path
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out + skip_out


class AttentiveLSTM(nn.Module):
    """LSTM with multi-head self-attention for enhanced sequence modeling."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_flash_attention: bool = True,
    ):
        super(AttentiveLSTM, self).__init__()

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.use_flash_attention = use_flash_attention and hasattr(
            F, "scaled_dot_product_attention"
        )

        if self.use_flash_attention:
            self.num_heads = num_heads
            self.head_dim = lstm_output_size // num_heads
            self.scaling = float(self.head_dim) ** -0.5

            self.q_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.k_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.v_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.out_proj = nn.Linear(lstm_output_size, lstm_output_size)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.layer_norm1 = nn.LayerNorm(lstm_output_size)
        self.layer_norm2 = nn.LayerNorm(lstm_output_size)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size * 4, lstm_output_size),
            nn.Dropout(dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        lstm_out, _ = self.lstm(x)

        if lstm_out.dim() == 2:
            lstm_out = lstm_out.unsqueeze(1)

        if self.use_flash_attention:
            residual = lstm_out
            lstm_out = self.layer_norm1(lstm_out)

            q = self.q_proj(lstm_out)
            k = self.k_proj(lstm_out)
            v = self.v_proj(lstm_out)

            batch_size, seq_len, embed_dim = q.shape
            q = q.view(
                batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
            ).transpose(1, 2)
            k = k.view(
                batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
            ).transpose(1, 2)
            v = v.view(
                batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
            ).transpose(1, 2)

            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, embed_dim)
            )
            attn_output = self.out_proj(attn_output)
            attn_output = self.dropout(attn_output)

            lstm_out = residual + attn_output
        else:
            residual = lstm_out
            lstm_out = self.layer_norm1(lstm_out)
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = residual + self.dropout(attn_output)

        residual = lstm_out
        lstm_out = self.layer_norm2(lstm_out)
        lstm_out = residual + self.ffn(lstm_out)

        lstm_out = lstm_out[:, -1, :]

        return self.fc(lstm_out)


class GRUPredictor(nn.Module):
    """GRU-based predictor with improved architecture."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super(GRUPredictor, self).__init__()

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(gru_output_size)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.layer_norm(gru_out)
        gru_out = self.dropout(gru_out)

        return self.fc(gru_out)


class TransformerPredictor(nn.Module):
    """Transformer-based predictor for time series."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_flash_attention: bool = True,
        positional_encoding: str = "learned",
    ):
        super(TransformerPredictor, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding_type = positional_encoding

        if positional_encoding == "learned":
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_projection(x)

        if self.positional_encoding_type == "learned":
            seq_len = x.size(1)
            x = x + self.pos_encoding[:, :seq_len, :]

        x = self.transformer_encoder(x)
        x = x[:, -1, :]

        return self.fc(x)


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network for time series prediction."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super(TCNPredictor, self).__init__()

        layers = []
        num_channels = [hidden_size] * num_layers

        for i in range(num_layers):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = num_channels[i]

            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dilation=dilation_size,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x[:, :, -1]

        return self.fc(x)


class EnsemblePredictor(nn.Module):
    """Ensemble of multiple models for robust predictions."""

    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "average",
    ):
        super(EnsemblePredictor, self).__init__()

        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method

        if ensemble_method == "weighted":
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)

        if self.ensemble_method == "average":
            return predictions.mean(dim=0)
        elif self.ensemble_method == "weighted":
            weights = F.softmax(self.weights, dim=0)
            return (predictions * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            return predictions.mean(dim=0)
