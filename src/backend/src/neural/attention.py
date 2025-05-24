"""Attention mechanisms for neural network models.

This module provides implementations of various attention mechanisms
that can be used in transformer-based models like GPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism as used in transformer models.

    This implementation follows the "Attention is All You Need" paper
    by Vaswani et al. (2017).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """Initialize the multi-head attention module.

        Args:
            embed_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attention_mask=None, causal=False):
        """Forward pass through the multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim).
            key: Key tensor of shape (batch_size, seq_len, embed_dim).
            value: Value tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional mask to avoid attending to certain positions.
                Shape: (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len).
            causal: Whether to use causal attention (each position can only attend to previous positions).

        Returns:
            output: Attention output of shape (batch_size, seq_len, embed_dim).
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape for multi-head attention
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        query = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to shape: (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention scores
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask from shape (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
            if attention_mask.dim() == 2:
                # Expand the provided mask to the right shape
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

                # Create a mask where 0 -> -inf (don't attend) and 1 -> 0 (attend)
                padding_mask = (1 - attention_mask) * float('-inf')

                # Apply padding mask
                attention_scores = attention_scores + padding_mask
            else:
                # If the mask is already in the right shape, just apply it
                attention_scores = attention_scores.masked_fill(
                    attention_mask == 0, float('-inf')
                )

        # Apply causal mask if requested
        if causal:
            # Create a causal mask that allows each position to attend to all previous positions
            # Shape: (1, 1, seq_len, seq_len)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device) * float('-inf'), 
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # Apply causal mask
            attention_scores = attention_scores + causal_mask

        # Apply softmax to get attention weights
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, value)

        # Transpose and reshape to original dimensions
        # Shape: (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Apply output projection
        output = self.output_proj(context)

        return output, attention_weights


class AttentionBlock(nn.Module):
    """Attention block combining multi-head attention with feed-forward network.

    This block includes self-attention, layer normalization, and a feed-forward
    network with residual connections, similar to the transformer block in GPT.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """Initialize the attention block.

        Args:
            embed_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            ff_dim: Dimension of the feed-forward network.
            dropout: Dropout probability.
        """
        super(AttentionBlock, self).__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, causal=False):
        """Forward pass through the attention block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional mask to avoid attending to certain positions.
            causal: Whether to use causal attention (each position can only attend to previous positions).

        Returns:
            output: Processed tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Self-attention with residual connection and layer normalization
        residual = x
        x = self.layer_norm1(x)
        attention_output, _ = self.attention(x, x, x, attention_mask, causal=causal)
        x = residual + self.dropout(attention_output)

        # Feed-forward network with residual connection and layer normalization
        residual = x
        x = self.layer_norm2(x)
        ff_output = self.ff_network(x)
        x = residual + ff_output

        return x


class AttentionCogCore(nn.Module):
    """A stack of attention blocks for use in GPTLikeModel.

    This is a replacement for the CogCore class that uses attention blocks
    instead of LeakyAwarenessUnits.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        """Initialize the attention-based CogCore.

        Args:
            embed_dim: Dimension of the embeddings.
            num_heads: Number of attention heads.
            ff_dim: Dimension of the feed-forward network.
            num_layers: Number of attention blocks.
            dropout: Dropout probability.
        """
        super(AttentionCogCore, self).__init__()

        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None, causal=False):
        """Forward pass through the attention-based CogCore.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional mask to avoid attending to certain positions.
            causal: Whether to use causal attention (each position can only attend to previous positions).

        Returns:
            output: Processed tensor of shape (batch_size, seq_len, embed_dim).
        """
        for layer in self.layers:
            x = layer(x, attention_mask, causal=causal)
        return x


# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_layers = 2

    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    # Test MultiHeadAttention
    mha = MultiHeadAttention(embed_dim, num_heads)
    mha_output, attention_weights = mha(x, x, x, attention_mask)
    print("MultiHeadAttention output shape:", mha_output.shape)
    print("Attention weights shape:", attention_weights.shape)

    # Test AttentionBlock
    attn_block = AttentionBlock(embed_dim, num_heads, ff_dim)
    attn_block_output = attn_block(x, attention_mask)
    print("AttentionBlock output shape:", attn_block_output.shape)

    # Test AttentionCogCore
    attn_cogcore = AttentionCogCore(embed_dim, num_heads, ff_dim, num_layers)
    attn_cogcore_output = attn_cogcore(x, attention_mask)
    print("AttentionCogCore output shape:", attn_cogcore_output.shape)
