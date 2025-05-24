import torch
import torch.nn as nn

from src.neural.leaky_awareness_unit import CogCore
from src.neural.attention import AttentionCogCore


class GPTLikeModel(nn.Module):
    """A language model inspired by GPT architecture.

    This model can use either LeakyAwarenessUnits or transformer-style attention mechanisms.
    When use_attention=True, it includes self-attention mechanisms similar to GPT models.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, max_seq_len, 
                 use_attention=False, num_heads=8, dropout=0.1, alpha=0.01):
        """Initialize the GPTLikeModel.

        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the embeddings.
            hidden_dim: Dimension of the hidden layers.
            num_layers: Number of layers (LeakyAwarenessUnit or attention blocks).
            max_seq_len: Maximum sequence length for positional embeddings.
            use_attention: Whether to use attention mechanisms (True) or LeakyAwarenessUnits (False).
            num_heads: Number of attention heads (only used if use_attention=True).
            dropout: Dropout probability (only used if use_attention=True).
            alpha: Alpha parameter for LeakyReLU (only used if use_attention=False).
        """
        super(GPTLikeModel, self).__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Use either attention-based core or LeakyAwarenessUnit-based core
        if use_attention:
            self.cog_core = AttentionCogCore(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                ff_dim=hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout
            )
        else:
            self.cog_core = CogCore(
                input_dim=embed_dim, 
                hidden_dim=hidden_dim, 
                depth=num_layers, 
                alpha=alpha
            )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, causal=True):
        """Forward pass through the model.

        Args:
            input_ids: Tensor of token IDs.
            attention_mask: Optional mask to avoid attending to padding tokens.
            causal: Whether to use causal attention (each position can only attend to previous positions).
                Only used if use_attention=True in the model initialization.

        Returns:
            logits: Tensor of logits for next token prediction.
        """
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Token and positional embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + position_embeds

        # Process through CogCore
        # If using AttentionCogCore, pass the causal parameter
        if hasattr(self.cog_core, 'layers') and hasattr(self.cog_core.layers[0], 'attention'):
            hidden_states = self.cog_core(hidden_states, attention_mask, causal=causal)
        else:
            hidden_states = self.cog_core(hidden_states, attention_mask)

        # Language modeling head
        logits = self.lm_head(hidden_states)
        return logits


# Example usage
if __name__ == "__main__":
    vocab_size = 30522  # Example vocab size (e.g., BERT tokenizer)
    embed_dim = 128
    hidden_dim = 256
    depth = 6
    max_seq_len = 512
    alpha = 0.01

    model = GPTLikeModel(vocab_size, embed_dim, hidden_dim, num_layers=depth, max_seq_len=max_seq_len, alpha=alpha)
    dummy_input_ids = torch.randint(0, vocab_size, (1, 50))  # Batch size 1, sequence length 50
    attention_mask = torch.ones_like(dummy_input_ids)  # No masking
    output_logits = model(dummy_input_ids, attention_mask)
    print("Output logits shape:", output_logits.shape)  # Should be (1, 50, vocab_size)
