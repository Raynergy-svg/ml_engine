from typing import Tuple, Dict, Union
import torch
import torch.nn as nn
from enum import Enum
import vaex  # Added for optimized, memory-efficient computations (Vaex)


class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SELU = "selu"
    TANH = "tanh"


class MREngine(nn.Module):
    """
    MREngine: Multi-branch regression model for price prediction, explanation, and risk estimation.

    This model uses a bidirectional LSTM to encode sequential market data, an attention mechanism
    (optionally multi-head) to weigh timesteps, and separate fully-connected heads for predicting
    price (regression), explanation factors, and risk/volatility.
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        explanation_dim: int = 3,
        risk_dim: int = 1,
        activation: Union[str, ActivationType] = ActivationType.RELU,
        num_attention_heads: int = 1,  # New parameter for multi-head attention
    ):
        """
        Initialize the MREngine.

        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Hidden dimension of the LSTM layers.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability for LSTM, heads and attention.
            explanation_dim (int): Dimensionality of the explanation vector.
            risk_dim (int): Dimensionality of the risk output.
            activation (Union[str, ActivationType]): Activation function for head layers.
            num_attention_heads (int): Number of attention heads; 1 for standard attention.
        """
        super().__init__()
        self._validate_inputs(
            input_size, hidden_size, num_layers, dropout, num_attention_heads
        )
        self.activation = self._get_activation(activation)
        self.num_attention_heads = num_attention_heads

        # LSTM backbone to process sequential data
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )

        # Linear layers for the different outputs:
        self.fc_price = nn.Linear(hidden_size, 1)
        self.fc_explanation = nn.Linear(hidden_size, explanation_dim)
        self.fc_risk = nn.Linear(hidden_size, risk_dim)
        self.fc_attn = nn.Linear(hidden_size, hidden_size)

        # Initialize explanation_labels and risk_labels
        self.explanation_labels = [
            f"Explanation_{i}" for i in range(explanation_dim)
        ]  # Initialize explanation_labels
        self.risk_labels = [f"Risk_{i}" for i in range(risk_dim)]

        # Initialize weights for all submodules.
        self.apply(self._init_weights)

    def _validate_inputs(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_attention_heads: int,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1")

    def _get_activation(self, activation: Union[str, ActivationType]) -> nn.Module:
        if isinstance(activation, str):
            try:
                activation = ActivationType(activation.lower())
            except ValueError as e:
                raise ValueError(f"Unsupported activation: {activation}") from e
        activations = {
            ActivationType.RELU: nn.ReLU(),
            ActivationType.GELU: nn.GELU(),
            ActivationType.SELU: nn.SELU(),
            ActivationType.TANH: nn.Tanh(),
        }
        return activations.get(activation, nn.ReLU())

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for linear layers using Xavier Uniform initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            price (Tensor): Predicted price output of shape (batch_size, 1).
            explanation (Tensor): Explanation vector of shape (batch_size, explanation_dim).
            risk (Tensor): Risk/volatility output of shape (batch_size, risk_dim).
            attn_weights (Tensor): Attention weights over the sequence.
        """
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, sequence, features), got {len(x.shape)}D"
            )

        # Process the sequence with the LSTM
        lstm_out, _ = self.lstm(x)
        # Select the output at the last timestep
        out = lstm_out[:, -1, :]
        price = self.fc_price(out)
        explanation = self.fc_explanation(out)
        risk = self.fc_risk(out)
        # Apply activation to the attention layer output
        attn = self.activation(self.fc_attn(out))
        return price, explanation, risk, attn

    def get_interpretation(self, explanation_vector: torch.Tensor) -> Dict[str, float]:
        """
        Convert explanation vector to a human-readable dictionary.

        Args:
            explanation_vector (Tensor): Tensor of shape (batch_size, explanation_dim) or (explanation_dim,).

        Returns:
            Dict[str, float]: Mapping from explanation label to value.
        """
        # If multiple samples, average across batch dimension.
        if len(explanation_vector.shape) == 2:
            explanation_vector = explanation_vector.mean(dim=0)
        return {
            label: float(value)
            for label, value in zip(self.explanation_labels, explanation_vector)
        }

    def checkpoint(self, path: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            path (str): File path for saving the checkpoint.
        """
        checkpoint_data = {
            "state_dict": self.state_dict(),
            "explanation_labels": self.explanation_labels,
            "config": {
                "input_size": self.lstm.input_size,
                "hidden_size": self.lstm.hidden_size,
                "num_layers": self.lstm.num_layers,
                "dropout": self.dropout.p,
                "num_attention_heads": self.num_attention_heads,
            },
        }
        torch.save(checkpoint_data, path)
        print(f"MREngine checkpoint saved to {path}")

    @classmethod
    def from_checkpoint(cls, path: str) -> "MREngine":
        """
        Load the model from a checkpoint file.

        Args:
            path (str): File path of the checkpoint.

        Returns:
            MREngine: An instance of MREngine loaded from the checkpoint.
        """
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_attention_heads=config.get("num_attention_heads", 1),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.explanation_labels = checkpoint["explanation_labels"]
        return model

    def summary(self) -> None:
        """
        Print a summary of the model: layers, output shapes, and total trainable parameters.
        """
        print("MREngine Model Summary:")
        print("-----------------------")
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def load_data_with_vaex(self, file_path: str) -> vaex.dataframe.DataFrame:
        """
        Load large CSV data into a Vaex DataFrame.
        """
        try:
            df = vaex.from_csv(file_path, convert=True)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset with Vaex: {e}")


# Example usage:
if __name__ == "__main__":
    # Example input: batch_size=16, sequence_length=50, input_size=11
    sample_input = torch.randn(16, 50, 11)
    # Instantiate with multi-head attention enabled (e.g., 4 heads)
    model = MREngine(num_attention_heads=4, activation="gelu")
    price, explanation, risk, attn_weights = model(sample_input)

    print("Price output shape:", price.shape)  # Expected: (16, 1)
    print(
        "Explanation output shape:", explanation.shape
    )  # Expected: (16, explanation_dim)
    print("Risk output shape:", risk.shape)  # Expected: (16, 1)
    print(
        "Attention weights shape:", attn_weights.shape
    )  # Expected: (16, 50, num_attention_heads) if multi-head
    model.summary()

    # Demonstrate interpretation of explanation vector.
    interpretation = model.get_interpretation(explanation)
    print("Interpretation of explanation vector:", interpretation)
