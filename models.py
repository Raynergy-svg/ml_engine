"""
Optimized model implementations with enhanced CPU and GPU performance.
Includes improved attention mechanisms, memory efficiency, and numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union, List


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
