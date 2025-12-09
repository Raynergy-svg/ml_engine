"""
Optimized model implementations with enhanced CPU and GPU performance.
Includes improved attention mechanisms, memory efficiency, and numerical stability.

Key improvements:
- Enhanced weight initialization for faster convergence
- Better numerical stability with layer normalization
- Improved attention mechanisms with flash attention support
- Residual connections for better gradient flow
- Robust error handling and validation
"""

import logging
import math
from typing import Tuple, Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logger
logger = logging.getLogger(__name__)


class StockPredictor(nn.Module):
    """LSTM-based predictor with residual connections and optimized performance.

    Accepts an input sequence (e.g., technical indicators) and predicts
    the next closing price.
    """

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
        
        # Validate inputs
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        logger.info(f"Initializing StockPredictor: input_size={input_size}, "
                   f"hidden_size={hidden_size}, num_layers={num_layers}, "
                   f"dropout={dropout}, bidirectional={bidirectional}")
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # LSTM with optimized settings
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Optional layer normalization for better training stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Skip connection from LSTM output to final prediction
        self.skip = nn.Linear(lstm_output_size, 1)
        
        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights benefit from uniform initialization
                    nn.init.xavier_uniform_(param)
                else:
                    # FC layers use normal initialization
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and optimized computation.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)
        
        # Skip connection (direct path from LSTM to output)
        skip_out = self.skip(lstm_out)
        
        # Main path through fully connected layers
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        # Combine skip connection and main path
        return out + skip_out
        
    def to_device(self, device=None):
        """
        Move model to specified device (CPU/GPU) with proper error handling.
        
        Args:
            device: Target device or None to use CUDA if available
        
        Returns:
            Self for method chaining
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            return self.to(device)
        except Exception as e:
            print(f"Failed to move model to {device}: {e}")
            print("Falling back to CPU")
            return self.to("cpu")


class AttentiveLSTM(nn.Module):
    """LSTM with optimized multi-head self-attention for enhanced sequence modeling."""

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
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # LSTM with optimized settings
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Determine if we can use flash attention
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Multi-head attention mechanism
        if self.use_flash_attention:
            # For PyTorch 2.0+ with flash attention
            self.num_heads = num_heads
            self.head_dim = lstm_output_size // num_heads
            self.scaling = float(self.head_dim) ** -0.5
            
            # Projections for query, key, value
            self.q_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.k_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.v_proj = nn.Linear(lstm_output_size, lstm_output_size)
            self.out_proj = nn.Linear(lstm_output_size, lstm_output_size)
        else:
            # Traditional multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size, 
                num_heads=num_heads, 
                dropout=dropout,
                batch_first=True
            )
        
        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(lstm_output_size)
        self.layer_norm2 = nn.LayerNorm(lstm_output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size * 4, lstm_output_size),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights benefit from uniform initialization
                    nn.init.xavier_uniform_(param)
                else:
                    # FC layers use normal initialization
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _reshape_for_multihead(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized attention computation.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Ensure lstm_out is 3D before attention
        if lstm_out.dim() == 2:
            lstm_out = lstm_out.unsqueeze(1)
        
        # Apply attention mechanism
        if self.use_flash_attention:
            # Flash attention implementation (PyTorch 2.0+)
            residual = lstm_out
            
            # Apply layer normalization
            lstm_out = self.layer_norm1(lstm_out)
            
            # Project to queries, keys, values
            q = self.q_proj(lstm_out)
            k = self.k_proj(lstm_out)
            v = self.v_proj(lstm_out)
            
            # Reshape for multi-head attention
            batch_size, seq_len, embed_dim = q.shape
            q = q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads).transpose(1, 2)
            
            # Apply scaled dot-product attention with flash attention
            attn_output = F.scaled_dot_product_attention(q, k, v)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            
            # Output projection
            attn_output = self.out_proj(attn_output)
            
            # Apply dropout
            attn_output = self.dropout(attn_output)
            
            # Add residual connection
            lstm_out = residual + attn_output
        else:
            # Traditional multi-head attention
            residual = lstm_out
            lstm_out = self.layer_norm1(lstm_out)
            attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = residual + self.dropout(attn_output)
        
        # Feed-forward network with residual connection
        residual = lstm_out
        lstm_out = self.layer_norm2(lstm_out)
        lstm_out = residual + self.ffn(lstm_out)
        
        # Get the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Final prediction
        return self.fc(lstm_out)
        
    def to_device(self, device=None):
        """
        Move model to specified device (CPU/GPU) with proper error handling.
        
        Args:
            device: Target device or None to use CUDA if available
        
        Returns:
            Self for method chaining
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            return self.to(device)
        except Exception as e:
            print(f"Failed to move model to {device}: {e}")
            print("Falling back to CPU")
            return self.to("cpu")


class GRUPredictor(nn.Module):
    """Optimized GRU-based predictor with enhanced performance.

    Accepts an input sequence and predicts the next closing price using a
    GRU network with improved architecture.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
    ):
        super(GRUPredictor, self).__init__()
        
        # Calculate output size based on bidirectional setting
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # GRU with optimized settings
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Optional layer normalization for better training stability
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gru_output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(gru_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Skip connection from GRU output to final prediction
        self.skip = nn.Linear(gru_output_size, 1)
        
        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    # GRU weights benefit from uniform initialization
                    nn.init.xavier_uniform_(param)
                else:
                    # FC layers use normal initialization
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and optimized computation.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        # Process through GRU
        gru_out, _ = self.gru(x)
        
        # Get the output from the last time step
        gru_out = gru_out[:, -1, :]
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)
        
        # Apply dropout for regularization
        gru_out = self.dropout(gru_out)
        
        # Skip connection (direct path from GRU to output)
        skip_out = self.skip(gru_out)
        
        # Main path through fully connected layers
        out = F.relu(self.fc1(gru_out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        # Combine skip connection and main path
        return out + skip_out
        
    def to_device(self, device=None):
        """
        Move model to specified device (CPU/GPU) with proper error handling.
        
        Args:
            device: Target device or None to use CUDA if available
        
        Returns:
            Self for method chaining
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            return self.to(device)
        except Exception as e:
            print(f"Failed to move model to {device}: {e}")
            print("Falling back to CPU")
            return self.to("cpu")


class TransformerPredictor(nn.Module):
    """Optimized Transformer-based predictor for sequential data."""
    
    # Class constants
    POSITIONAL_ENCODING_BASE = 10000.0  # Base for sinusoidal positional encoding

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        dropout: float,
        num_heads: int = 8,
        use_flash_attention: bool = True,
        activation: str = "gelu",
        positional_encoding: str = "learned",
    ):
        super().__init__()
        
        # Determine if we can use flash attention
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.positional_encoding = positional_encoding
        if positional_encoding == "learned":
            self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, hidden_size))  # Max sequence length of 1000
            nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        elif positional_encoding == "sinusoidal":
            # Sinusoidal positional encoding will be computed in forward pass
            pass
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def _get_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_embedding.out_features, 2, dtype=torch.float32, device=device) * 
                            (-math.log(self.POSITIONAL_ENCODING_BASE) / self.input_embedding.out_features))
        
        pe = torch.zeros(seq_len, self.input_embedding.out_features, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        if self.positional_encoding == "learned":
            x = x + self.pos_encoder[:, :seq_len, :]
        elif self.positional_encoding == "sinusoidal":
            pe = self._get_positional_encoding(seq_len, x.device)
            x = x + pe
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TCNPredictor(nn.Module):
    """Temporal Convolutional Network for time series prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        kernel_size: int = 3
    ):
        super().__init__()
        
        # Validate inputs
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        
        logger.info(f"Initializing TCNPredictor: input_size={input_size}, "
                   f"hidden_size={hidden_size}, num_layers={num_layers}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            self.tcn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_size, kernel_size, 
                             padding=(kernel_size - 1) * dilation // 2, dilation=dilation),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def _apply_residual_connection(self, layer_output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply residual connection if dimensions match."""
        if residual.shape[1] == self.hidden_size:
            return layer_output + residual
        return layer_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        # Transpose for Conv1d: [batch, features, sequence]
        x = x.transpose(1, 2)
        
        # Apply TCN layers with residual connections
        for layer in self.tcn_layers:
            layer_output = layer(x)
            x = self._apply_residual_connection(layer_output, x)
        
        # Take the last time step
        x = x[:, :, -1]
        
        # Output prediction
        return self.fc(x)


class EnsemblePredictor(nn.Module):
    """Ensemble of multiple models with attention-based weighting."""
    
    SUPPORTED_METHODS = {"mean", "weighted", "attention"}
    
    def __init__(
        self,
        models: List[nn.Module],
        input_size: int,
        hidden_size: int,
        dropout: float = 0.2,
        ensemble_method: str = "attention"
    ):
        super().__init__()
        
        # Validate inputs
        if not models:
            raise ValueError("models list cannot be empty")
        if ensemble_method not in self.SUPPORTED_METHODS:
            raise ValueError(f"ensemble_method must be one of {self.SUPPORTED_METHODS}, got '{ensemble_method}'")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        logger.info(f"Initializing EnsemblePredictor with {len(models)} models, method={ensemble_method}")
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        if ensemble_method == "attention":
            # Attention mechanism for weighting models
            self.attention = nn.Sequential(
                nn.Linear(self.num_models, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.num_models),
                nn.Softmax(dim=1)
            )
        elif ensemble_method == "weighted":
            # Learnable weights
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]
            
        Returns:
            Predicted values of shape [batch_size, 1]
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions: [batch_size, num_models]
        predictions = torch.cat(predictions, dim=1)
        
        if self.ensemble_method == "mean":
            # Simple averaging
            return predictions.mean(dim=1, keepdim=True)
        elif self.ensemble_method == "weighted":
            # Weighted average
            weights = F.softmax(self.weights, dim=0)
            return (predictions * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        elif self.ensemble_method == "attention":
            # Attention-based weighting
            attention_weights = self.attention(predictions)
            return (predictions * attention_weights).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


class GRUPredictor(nn.Module):
    """GRU-based predictor similar to StockPredictor but using GRU cells."""
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc1 = nn.Linear(gru_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU."""
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        
        out = F.relu(self.fc1(gru_out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
