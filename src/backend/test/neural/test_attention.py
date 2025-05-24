"""Tests for the attention mechanisms in the neural module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn

from src.neural.attention import MultiHeadAttention, AttentionBlock, AttentionCogCore
from src.neural.gpt import GPTLikeModel


class TestMultiHeadAttention(unittest.TestCase):
    """Tests for the MultiHeadAttention class."""

    def test_initialization(self):
        """Test that the MultiHeadAttention initializes correctly."""
        embed_dim = 64
        num_heads = 4

        mha = MultiHeadAttention(embed_dim, num_heads)

        # Check that the parameters are set correctly
        self.assertEqual(mha.embed_dim, embed_dim)
        self.assertEqual(mha.num_heads, num_heads)
        self.assertEqual(mha.head_dim, embed_dim // num_heads)

        # Check that the projections are initialized
        self.assertIsInstance(mha.query_proj, nn.Linear)
        self.assertIsInstance(mha.key_proj, nn.Linear)
        self.assertIsInstance(mha.value_proj, nn.Linear)
        self.assertIsInstance(mha.output_proj, nn.Linear)

        # Check that the dimensions are correct
        self.assertEqual(mha.query_proj.in_features, embed_dim)
        self.assertEqual(mha.query_proj.out_features, embed_dim)
        self.assertEqual(mha.key_proj.in_features, embed_dim)
        self.assertEqual(mha.key_proj.out_features, embed_dim)
        self.assertEqual(mha.value_proj.in_features, embed_dim)
        self.assertEqual(mha.value_proj.out_features, embed_dim)
        self.assertEqual(mha.output_proj.in_features, embed_dim)
        self.assertEqual(mha.output_proj.out_features, embed_dim)

    def test_forward(self):
        """Test the forward pass of MultiHeadAttention."""
        batch_size = 2
        seq_len = 10
        embed_dim = 64
        num_heads = 4

        # Create random input
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)

        # Initialize MultiHeadAttention
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Forward pass without mask and with causal=False for testing
        output, attention_weights = mha(query, key, value, attention_mask=None, causal=False)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

        # Check attention weights shape
        self.assertEqual(attention_weights.shape, (batch_size, num_heads, seq_len, seq_len))

        # Check that attention weights are valid probabilities
        self.assertTrue((attention_weights >= 0).all())
        self.assertTrue((attention_weights <= 1).all())


class TestAttentionBlock(unittest.TestCase):
    """Tests for the AttentionBlock class."""

    def test_initialization(self):
        """Test that the AttentionBlock initializes correctly."""
        embed_dim = 64
        num_heads = 4
        ff_dim = 128

        block = AttentionBlock(embed_dim, num_heads, ff_dim)

        # Check that the components are initialized
        self.assertIsInstance(block.attention, MultiHeadAttention)
        self.assertIsInstance(block.layer_norm1, nn.LayerNorm)
        self.assertIsInstance(block.layer_norm2, nn.LayerNorm)
        self.assertIsInstance(block.ff_network, nn.Sequential)

        # Check that the dimensions are correct
        self.assertEqual(block.layer_norm1.normalized_shape[0], embed_dim)
        self.assertEqual(block.layer_norm2.normalized_shape[0], embed_dim)

        # Check feed-forward network
        self.assertEqual(len(block.ff_network), 4)  # Linear, GELU, Linear, Dropout
        self.assertEqual(block.ff_network[0].in_features, embed_dim)
        self.assertEqual(block.ff_network[0].out_features, ff_dim)
        self.assertEqual(block.ff_network[2].in_features, ff_dim)
        self.assertEqual(block.ff_network[2].out_features, embed_dim)

    def test_forward(self):
        """Test the forward pass of AttentionBlock."""
        batch_size = 2
        seq_len = 10
        embed_dim = 64
        num_heads = 4
        ff_dim = 128

        # Create random input
        x = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        # Initialize AttentionBlock
        block = AttentionBlock(embed_dim, num_heads, ff_dim)

        # Forward pass (with causal=False for testing)
        output = block(x, attention_mask, causal=False)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestAttentionCogCore(unittest.TestCase):
    """Tests for the AttentionCogCore class."""

    def test_initialization(self):
        """Test that the AttentionCogCore initializes correctly."""
        embed_dim = 64
        num_heads = 4
        ff_dim = 128
        num_layers = 2

        core = AttentionCogCore(embed_dim, num_heads, ff_dim, num_layers)

        # Check that the layers are initialized
        self.assertIsInstance(core.layers, nn.ModuleList)
        self.assertEqual(len(core.layers), num_layers)

        # Check that each layer is an AttentionBlock
        for layer in core.layers:
            self.assertIsInstance(layer, AttentionBlock)

    def test_forward(self):
        """Test the forward pass of AttentionCogCore."""
        batch_size = 2
        seq_len = 10
        embed_dim = 64
        num_heads = 4
        ff_dim = 128
        num_layers = 2

        # Create random input
        x = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, seq_len)

        # Initialize AttentionCogCore
        core = AttentionCogCore(embed_dim, num_heads, ff_dim, num_layers)

        # Forward pass (with causal=False for testing)
        output = core(x, attention_mask, causal=False)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))


class TestGPTLikeModelWithAttention(unittest.TestCase):
    """Tests for the GPTLikeModel with attention."""

    def test_initialization_with_attention(self):
        """Test that the GPTLikeModel initializes correctly with attention."""
        vocab_size = 1000
        embed_dim = 64
        hidden_dim = 128
        num_layers = 2
        max_seq_len = 50

        model = GPTLikeModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            use_attention=True,
            num_heads=4,
            dropout=0.1
        )

        # Check that the components are initialized
        self.assertIsInstance(model.token_embedding, nn.Embedding)
        self.assertIsInstance(model.position_embedding, nn.Embedding)
        self.assertIsInstance(model.cog_core, AttentionCogCore)
        self.assertIsInstance(model.lm_head, nn.Linear)

        # Check that the dimensions are correct
        self.assertEqual(model.token_embedding.num_embeddings, vocab_size)
        self.assertEqual(model.token_embedding.embedding_dim, embed_dim)
        self.assertEqual(model.position_embedding.num_embeddings, max_seq_len)
        self.assertEqual(model.position_embedding.embedding_dim, embed_dim)
        self.assertEqual(model.lm_head.in_features, embed_dim)
        self.assertEqual(model.lm_head.out_features, vocab_size)

    def test_forward_with_attention(self):
        """Test the forward pass of GPTLikeModel with attention."""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        embed_dim = 64
        hidden_dim = 128
        num_layers = 2
        max_seq_len = 50

        # Create random input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Initialize GPTLikeModel with attention
        model = GPTLikeModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            use_attention=True,
            num_heads=4,
            dropout=0.1
        )

        # Forward pass (with causal=False for testing)
        logits = model(input_ids, attention_mask, causal=False)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))


if __name__ == '__main__':
    unittest.main()
