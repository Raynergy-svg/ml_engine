from typing import Tuple, Dict, Union
from enum import Enum
import torch
import torch.nn as nn
import xarray as xr  # Added for multi-dimensional labeled arrays (Xarray)


class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SELU = "selu"
    TANH = "tanh"


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        explanation_dim: int = 3,
        activation: Union[str, ActivationType] = ActivationType.RELU,
    ):
        """
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Hidden dimension of the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability applied in LSTM and heads.
            explanation_dim (int): Dimensionality of the
                explanation vector (e.g., factors like sentiment, news,
                technical).
            activation (Union[str, ActivationType]): Activation function
                to use in the heads.
        """
        super(MultiTaskModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.attention_layer = nn.Identity()  # added placeholder
        self.fc_price = nn.Linear(hidden_size, 1)
        self.fc_explanation = nn.Linear(hidden_size, explanation_dim)
        self.fc_attn = nn.Linear(hidden_size, hidden_size)
        # Added risk head for risk prediction.
        self.fc_risk = nn.Linear(hidden_size, 1)
        # Added missing dropout and layer normalization layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "selu":
            self.activation = nn.SELU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: [batch, seq_len, input_size]
        lstm_out, _ = self.lstm(x)
        # Process attention using the identity layer is optional;
        # use fc_attn output as attention vector.
        out = lstm_out[:, -1, :]
        context_vector = self.fc_attn(out)
        context_vector = self.layer_norm(context_vector)
        context_vector = self.dropout(context_vector)
        # Use the correctly defined heads.
        price = self.fc_price(context_vector)
        explanation = self.fc_explanation(context_vector)
        # Optionally compute risk if needed:
        _ = self.fc_risk(context_vector)
        # Return the processed attention vector.
        return price, explanation, context_vector

    def _validate_inputs(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

    def _get_activation(self, activation: Union[str, ActivationType]) -> nn.Module:
        if isinstance(activation, str):
            activation = ActivationType(activation.lower())

        activations = {
            ActivationType.RELU: nn.ReLU(),
            ActivationType.GELU: nn.GELU(),
            ActivationType.SELU: nn.SELU(),
            ActivationType.TANH: nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())

    def _build_head(
        self, in_features: int, hidden_dim: int, out_features: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            self.activation,
            self.dropout,
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features),
        )

    def get_interpretation(self, explanation_vector: torch.Tensor) -> Dict[str, float]:
        """Convert explanation vector to human-readable format."""
        if len(explanation_vector.shape) == 2:
            explanation_vector = explanation_vector.mean(dim=0)

        return {
            label: float(value)
            for label, value in zip(self.explanation_labels, explanation_vector)
        }

    def checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "explanation_labels": self.explanation_labels,
                "config": {
                    "input_size": self.lstm.input_size,
                    "hidden_size": self.lstm.hidden_size,
                    "num_layers": self.lstm.num_layers,
                    "dropout": self.dropout.p,
                },
            },
            path,
        )
        print(f"MultiTaskModel checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str) -> "MultiTaskModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.explanation_labels = checkpoint["explanation_labels"]
        return model

    def process_with_xarray(self, data) -> xr.DataArray:
        """
        Convert input data (e.g. a NumPy/Modin array) into an Xarray DataArray.
        """
        # data: assumed to be an array-like input
        return xr.DataArray(data)


# Example usage:
if __name__ == "__main__":
    # Example input: batch_size=16, sequence_length=50, input_size=11
    sample_input = torch.randn(16, 50, 11)
    model = MultiTaskModel()
    price, explanation, attn_weights = model(sample_input)
    print("Price output shape:", price.shape)
    print("Explanation output shape:", explanation.shape)
    print("Attention weights shape:", attn_weights.shape)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
