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


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))
    Better than ReLU for deep networks with smoother gradients."""
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)
    Also known as SiLU, performs better than ReLU in many cases."""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


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
        use_batch_norm: bool = False,
        activation: str = "relu",
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
        self.use_batch_norm = use_batch_norm
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)

        self.dropout = nn.Dropout(dropout)

        # Improved architecture with residual connections and batch norm
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32) if use_batch_norm else nn.Identity()
        
        self.fc4 = nn.Linear(32, 1)

        # Skip connection
        self.skip = nn.Linear(lstm_output_size, 1)
        
        # Activation function selection
        if activation == "mish":
            self.activation = Mish()
        elif activation == "swish" or activation == "silu":
            self.activation = Swish()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using improved initialization strategies."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    # Orthogonal initialization for LSTM recurrent weights (weight_hh)
                    # Xavier for input weights (weight_ih)
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                elif "bn" in name or "layer_norm" in name:
                    # Initialize batch norm and layer norm weights to 1
                    nn.init.ones_(param)
                else:
                    # He initialization for ReLU-like activations, Xavier for others
                    if isinstance(self.activation, (nn.ReLU, Mish, Swish)):
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections and improved activations."""
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)

        lstm_out = self.dropout(lstm_out)

        # Skip connection for better gradient flow
        skip_out = self.skip(lstm_out)

        # Main path with improved activations and optional batch norm
        out = self.fc1(lstm_out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.fc4(out)

        # Residual connection improves gradient flow and model performance
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

            # Use single projection for efficiency (fused QKV)
            self.qkv_proj = nn.Linear(lstm_output_size, lstm_output_size * 3)
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

            # Optimized: Fused QKV projection for better memory efficiency
            batch_size, seq_len, embed_dim = lstm_out.shape
            qkv = self.qkv_proj(lstm_out)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Use scaled_dot_product_attention for better performance (flash attention)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            
            # Reshape back
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
    """GRU-based predictor with improved architecture and modern enhancements."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_batch_norm: bool = False,
        activation: str = "relu",
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
        self.use_batch_norm = use_batch_norm
        self.dropout = nn.Dropout(dropout)

        # Enhanced architecture with configurable activation and optional batch norm
        self.fc1 = nn.Linear(gru_output_size, 128)
        self.bn1 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        
        self.fc3 = nn.Linear(64, 1)
        
        # Skip connection for better gradient flow
        self.skip = nn.Linear(gru_output_size, 1)
        
        # Activation function selection
        if activation == "mish":
            self.activation = Mish()
        elif activation == "swish" or activation == "silu":
            self.activation = Swish()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved strategies."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    # Orthogonal for GRU recurrent weights, Xavier for input weights
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                elif "bn" in name or "layer_norm" in name:
                    nn.init.ones_(param)
                else:
                    # Kaiming for ReLU-like, Xavier for others
                    if isinstance(self.activation, (nn.ReLU, Mish, Swish)):
                        nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.layer_norm(gru_out)
        gru_out = self.dropout(gru_out)

        # Skip connection
        skip_out = self.skip(gru_out)
        
        # Main path with improved activations
        out = self.fc1(gru_out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # Residual connection
        return out + skip_out


class TransformerPredictor(nn.Module):
    """Transformer-based predictor for time series with enhanced features."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_flash_attention: bool = True,
        positional_encoding: str = "learned",
        activation: str = "gelu",
    ):
        super(TransformerPredictor, self).__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.positional_encoding_type = positional_encoding
        self.hidden_size = hidden_size

        if positional_encoding == "learned":
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_size))
        elif positional_encoding == "sinusoidal":
            # Register sinusoidal encoding as buffer (not trainable)
            self.register_buffer('pos_encoding', self._get_sinusoidal_encoding(1000, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Improved output layers with configurable activation
        if activation == "mish":
            act_fn = Mish()
        elif activation == "swish":
            act_fn = Swish()
        else:
            act_fn = nn.GELU()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()
    
    def _get_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings with improved numerical stability."""
        position = torch.arange(max_len).unsqueeze(1).float()
        # More numerically stable formulation
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe

    def _init_weights(self):
        """Initialize weights with improved strategies."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Use Xavier uniform for transformer weights
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with positional encoding."""
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        if self.positional_encoding_type in ["learned", "sinusoidal"]:
            x = x + self.pos_encoding[:, :seq_len, :]

        x = self.transformer_encoder(x)
        x = x[:, -1, :]

        return self.fc(x)


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network for time series prediction with enhanced features."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        use_residual: bool = True,
    ):
        super(TCNPredictor, self).__init__()

        self.use_residual = use_residual
        
        # Activation function selection
        if activation == "mish":
            self.activation = Mish()
        elif activation == "swish" or activation == "silu":
            self.activation = Swish()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Build TCN layers with residual connections
        self.tcn_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size

            # Main convolutional block
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dilation=dilation_size,
                ),
                nn.BatchNorm1d(out_channels),
            )
            self.tcn_layers.append(conv_block)
            
            # Residual connection (1x1 conv if dimensions don't match)
            if use_residual and in_channels != out_channels:
                self.residual_layers.append(nn.Conv1d(in_channels, out_channels, 1))
            else:
                self.residual_layers.append(nn.Identity())
        
        self.dropout = nn.Dropout(dropout)

        # Enhanced output layers - create fresh activation instances
        if activation == "mish":
            act_fn1, act_fn2 = Mish(), Mish()
        elif activation == "swish" or activation == "silu":
            act_fn1, act_fn2 = Swish(), Swish()
        elif activation == "gelu":
            act_fn1, act_fn2 = nn.GELU(), nn.GELU()
        else:
            act_fn1, act_fn2 = nn.ReLU(), nn.ReLU()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            act_fn1,
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            act_fn2,
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with improved strategies."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming for conv layers (best for ReLU-like activations)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        
        for i, (conv_block, residual_layer) in enumerate(zip(self.tcn_layers, self.residual_layers)):
            residual = x
            
            # Apply convolution
            out = conv_block(x)
            
            # Crop to proper length (causal padding)
            if out.size(-1) != residual.size(-1):
                out = out[:, :, :residual.size(-1)]
            
            # Apply activation
            out = self.activation(out)
            out = self.dropout(out)
            
            # Add residual connection
            if self.use_residual:
                residual = residual_layer(residual)
                x = out + residual
            else:
                x = out
        
        # Take last timestep
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
