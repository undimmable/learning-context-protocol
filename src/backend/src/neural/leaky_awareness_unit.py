import torch
import torch.nn as nn


class LeakyAwarenessUnit(nn.Module):
    """A custom neural network layer using LeakyReLU activations.

    This is a simple feed-forward network with two linear layers and LeakyReLU activations.
    It is not equivalent to a transformer block used in GPT models, which would include
    self-attention mechanisms, layer normalization, and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, alpha=0.01):
        """Initialize the LeakyAwarenessUnit.

        Args:
            input_dim: Dimension of the input features.
            hidden_dim: Dimension of the hidden layer.
            alpha: Negative slope for LeakyReLU activation.
        """
        super(LeakyAwarenessUnit, self).__init__()
        self.alpha = alpha
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.alpha)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x * attention_mask  # Apply mask before transformation

        # Step 1: Encode awareness impulse
        hidden = self.leaky_relu(self.input_to_hidden(x))

        if attention_mask is not None:
            hidden = hidden * attention_mask  # Re-apply mask during processing

        # Step 2: Decode response with retained dual polarity
        out = self.leaky_relu(self.hidden_to_output(hidden))

        if attention_mask is not None:
            out = out * attention_mask  # Final gating

        return out


class CogCore(nn.Module):
    """A stack of LeakyAwarenessUnit layers.

    This is a simplified alternative to the transformer blocks used in GPT models.
    Unlike transformer blocks, this does not include self-attention mechanisms,
    which are central to GPT's ability to model long-range dependencies.
    """
    def __init__(self, input_dim, hidden_dim, depth=3, alpha=0.01):
        """Initialize the CogCore.

        Args:
            input_dim: Dimension of the input features.
            hidden_dim: Dimension of the hidden layers in LeakyAwarenessUnits.
            depth: Number of LeakyAwarenessUnit layers.
            alpha: Negative slope for LeakyReLU activation.
        """
        super(CogCore, self).__init__()
        self.layers = nn.ModuleList([
            LeakyAwarenessUnit(input_dim, hidden_dim, alpha) for _ in range(depth)
        ])

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


# Example usage
if __name__ == "__main__":
    core = CogCore(input_dim=1024, hidden_dim=2048, depth=5, alpha=0.05)
    dummy_input = torch.randn(1, 1024)
    attention_mask = torch.ones_like(dummy_input)  # All attention open
    output = core(dummy_input, attention_mask)
    print("Conscious vector output:", output)
