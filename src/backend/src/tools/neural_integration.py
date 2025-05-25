"""Integration between MCP Server and GPTLikeModel.

This module provides functions for integrating the MCP Server with the GPTLikeModel.
It extends the MCP Server to use the GPTLikeModel for generating responses to MCP endpoints.

Usage:
    from src.tools.neural_integration import GPTLikeModelProvider

    # Initialize the GPTLikeModel Provider
    llm_provider = GPTLikeModelProvider()

    # Use it in the MCP Server
    _llm_provider = llm_provider
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union

try:
    from transformers import AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

from src.neural.gpt import GPTLikeModel
from src.neural.leaky_awareness_unit import CogCore

# ---------------------------------------------------------------------------
# GPTLikeModel Provider
# ---------------------------------------------------------------------------

# Default model parameters (can be overridden with environment variables)
VOCAB_SIZE = int(os.environ.get("GPT_VOCAB_SIZE", "30522"))  # Example vocab size (e.g., BERT tokenizer)
EMBED_DIM = int(os.environ.get("GPT_EMBED_DIM", "128"))
HIDDEN_DIM = int(os.environ.get("GPT_HIDDEN_DIM", "256"))
DEPTH = int(os.environ.get("GPT_DEPTH", "6"))
MAX_SEQ_LEN = int(os.environ.get("GPT_MAX_SEQ_LEN", "512"))
ALPHA = float(os.environ.get("GPT_ALPHA", "0.01"))

# Attention-related parameters
USE_ATTENTION = os.environ.get("GPT_USE_ATTENTION", "true").lower() == "true"
NUM_HEADS = int(os.environ.get("GPT_NUM_HEADS", "8"))
DROPOUT = float(os.environ.get("GPT_DROPOUT", "0.1"))

# Path to saved model (if available)
MODEL_PATH = os.environ.get("GPT_MODEL_PATH", None)

class GPTLikeModelProvider:
    """Provider for interacting with the GPTLikeModel."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = DEPTH,
        max_seq_len: int = MAX_SEQ_LEN,
        alpha: float = ALPHA,
        use_attention: bool = USE_ATTENTION,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        model_path: Optional[str] = MODEL_PATH,
        tokenizer_name: str = "bert-base-uncased",
    ):
        """Initialize the GPTLikeModel provider.

        Args:
            vocab_size: The size of the vocabulary.
            embed_dim: The dimension of the embeddings.
            hidden_dim: The dimension of the hidden layers.
            num_layers: The number of layers.
            max_seq_len: The maximum sequence length.
            alpha: The alpha parameter for the LeakyAwarenessUnit.
            use_attention: Whether to use attention mechanisms.
            num_heads: Number of attention heads (only used if use_attention=True).
            dropout: Dropout probability (only used if use_attention=True).
            model_path: Optional path to a saved model.
            tokenizer_name: The name of the HuggingFace tokenizer to use.
        """
        # Use CUDA if available, otherwise MPS if available, otherwise CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize the model
        self.model = GPTLikeModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            use_attention=use_attention,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )

        # Load saved model if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Using initialized model instead")

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Initialize tokenizer
        if HAVE_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                print(f"Using HuggingFace tokenizer: {tokenizer_name}")
                # Ensure the vocab size matches
                if self.tokenizer.vocab_size != vocab_size:
                    print(f"Warning: Tokenizer vocab size ({self.tokenizer.vocab_size}) "
                          f"doesn't match model vocab size ({vocab_size})")
            except Exception as e:
                print(f"Error initializing HuggingFace tokenizer: {e}")
                print("Falling back to SimpleTokenizer")
                self.tokenizer = SimpleTokenizer(vocab_size)
        else:
            print("Transformers library not available, using SimpleTokenizer")
            self.tokenizer = SimpleTokenizer(vocab_size)

        # Store context for conversation history
        self.conversation_history = []
        self.max_history_length = int(os.environ.get("GPT_MAX_HISTORY", "5"))

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, 
              use_history: bool = True, top_k: int = 50, top_p: float = 0.9) -> str:
        """Generate text using the GPTLikeModel with advanced sampling options.

        Args:
            prompt: The prompt to generate from.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness. Lower values make responses more deterministic.
            use_history: Whether to include conversation history for context.
            top_k: Number of highest probability tokens to consider for sampling.
            top_p: Cumulative probability threshold for nucleus sampling.

        Returns:
            The generated text.
        """
        # Include conversation history if requested
        if use_history and self.conversation_history:
            context = "\n".join(self.conversation_history[-self.max_history_length:])
            full_prompt = f"{context}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Tokenize the prompt
        if HAVE_TRANSFORMERS and hasattr(self.tokenizer, 'encode') and callable(getattr(self.tokenizer, 'encode')):
            try:
                # Use HuggingFace tokenizer if available
                encoded = self.tokenizer.encode(full_prompt, return_tensors="pt")
                if isinstance(encoded, torch.Tensor):
                    input_ids = encoded.tolist()
                elif hasattr(encoded, 'input_ids'):
                    input_ids = encoded.input_ids[0].tolist()
                else:
                    input_ids = encoded[0].tolist()
            except (AttributeError, IndexError):
                # Fallback if the tokenizer doesn't behave as expected
                input_ids = self.tokenizer.encode(full_prompt)
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else [input_ids]
        else:
            # Fallback to SimpleTokenizer
            input_ids = self.tokenizer.encode(full_prompt)

        input_tensor = torch.tensor([input_ids], device=self.device)
        attention_mask = torch.ones_like(input_tensor)

        # Generate tokens
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions
                logits = self.model(input_tensor, attention_mask)

                # Get the next token prediction (from the last position)
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Append the predicted token
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.tensor([[1]], device=self.device)], dim=1)

                # Stop if we predict the end of sequence token
                if hasattr(self.tokenizer, 'eos_token_id') and next_token == self.tokenizer.eos_token_id:
                    break

        # Decode the generated tokens
        generated_ids = input_tensor[0, len(input_ids):].tolist()

        if HAVE_TRANSFORMERS and hasattr(self.tokenizer, 'decode') and callable(getattr(self.tokenizer, 'decode')):
            # Use HuggingFace tokenizer if available
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            # Fallback to SimpleTokenizer
            generated_text = self.tokenizer.decode(generated_ids)

        # Update conversation history
        if use_history:
            self.conversation_history.append(f"User: {prompt}")
            self.conversation_history.append(f"Assistant: {generated_text}")
            # Trim history if it gets too long
            if len(self.conversation_history) > self.max_history_length * 2:
                self.conversation_history = self.conversation_history[-self.max_history_length * 2:]

        return generated_text

    def answer_shell_command(self, command: str) -> str:
        """Generate a response for a shell command.

        Args:
            command: The shell command to simulate.

        Returns:
            The simulated output of the shell command.
        """
        prompt = (
            "You are an advanced shell interpreter running on a Unix-like system.\n"
            "Simulate running the following command and provide *only* the raw\n"
            "stdout that a user would see (no additional commentary).  If the\n"
            "command would normally produce no output, return an empty string.\n\n"
            f"$ {command}"
        )
        return self.generate(prompt, max_new_tokens=128, use_history=False)

    def enhance_memory_query(self, query: str, entries: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhance memory query results using semantic understanding.

        This method takes the query and a list of potential entries, then uses the
        GPTLikeModel to re-rank them based on semantic relevance rather than just
        string similarity.

        Args:
            query: The search query.
            entries: List of memory entries to rank.
            top_k: Number of top results to return.

        Returns:
            List of entries ranked by semantic relevance.
        """
        if not entries:
            return []

        # Create embeddings for the query and entries
        query_embedding = self._get_text_embedding(query)

        # Calculate semantic similarity scores
        scored_entries = []
        for entry in entries:
            entry_text = entry.get("text", "")
            entry_embedding = self._get_text_embedding(entry_text)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry_embedding)
            scored_entries.append((similarity, entry))

        # Sort by similarity score (descending)
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        # Return top_k entries
        return [entry for _, entry in scored_entries[:top_k]]

    def summarize_file_content(self, content: str, max_length: int = 500) -> str:
        """Summarize file content using the GPTLikeModel.

        Args:
            content: The file content to summarize.
            max_length: Maximum length of the summary.

        Returns:
            A summary of the file content.
        """
        # Truncate content if it's too long to fit in the model's context
        if len(content) > 10000:  # Arbitrary limit to avoid tokenizer issues
            content = content[:10000] + "..."

        prompt = (
            "You are an expert at summarizing code and text files. "
            "Provide a concise summary of the following file content, "
            "focusing on its main purpose and key components:\n\n"
            f"{content}\n\n"
            "Summary:"
        )

        return self.generate(
            prompt, 
            max_new_tokens=min(max_length // 4, 128),  # Rough estimate of tokens
            temperature=0.3,  # Lower temperature for more focused summary
            use_history=False  # Don't include in conversation history
        )

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """Get embedding for a text using the model's token embeddings.

        Args:
            text: The text to embed.

        Returns:
            Tensor containing the text embedding.
        """
        # Tokenize the text
        if HAVE_TRANSFORMERS and hasattr(self.tokenizer, 'encode') and callable(getattr(self.tokenizer, 'encode')):
            try:
                # Use HuggingFace tokenizer if available
                encoded = self.tokenizer.encode(text, return_tensors="pt")
                if isinstance(encoded, torch.Tensor):
                    input_ids = encoded.tolist()
                elif hasattr(encoded, 'input_ids'):
                    input_ids = encoded.input_ids[0].tolist()
                else:
                    input_ids = encoded[0].tolist()
            except (AttributeError, IndexError):
                # Fallback if the tokenizer doesn't behave as expected
                input_ids = self.tokenizer.encode(text)
                if not isinstance(input_ids, list):
                    input_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else [input_ids]
        else:
            # Fallback to SimpleTokenizer
            input_ids = self.tokenizer.encode(text)

        input_tensor = torch.tensor([input_ids], device=self.device)

        # Get embeddings from the model's token embedding layer
        with torch.no_grad():
            # Use only token embeddings without position embeddings
            embeddings = self.model.token_embedding(input_tensor)

            # Average the embeddings across the sequence dimension
            mean_embedding = torch.mean(embeddings, dim=1)

        return mean_embedding.squeeze()

    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Calculate cosine similarity between two tensors.

        Args:
            tensor1: First tensor.
            tensor2: Second tensor.

        Returns:
            Cosine similarity score.
        """
        if tensor1.dim() == 0 or tensor2.dim() == 0:
            return 0.0

        # Normalize the tensors
        tensor1 = F.normalize(tensor1, p=2, dim=0)
        tensor2 = F.normalize(tensor2, p=2, dim=0)

        # Calculate cosine similarity
        return torch.dot(tensor1, tensor2).item()


# ---------------------------------------------------------------------------
# Simple Tokenizer (for demonstration purposes)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """A simple character-level tokenizer for demonstration purposes."""

    def __init__(self, vocab_size: int):
        """Initialize the tokenizer.

        Args:
            vocab_size: The size of the vocabulary.
        """
        self.vocab_size = vocab_size
        self.eos_token_id = 0  # End of sequence token

    def encode(self, text: str, return_tensors=None) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The text to encode.
            return_tensors: If specified, returns tensors of the specified format.
                            Currently only supports "pt" for PyTorch tensors.

        Returns:
            A list of token IDs or a tensor if return_tensors is specified.
        """
        # This is a very simple character-level encoding
        # In a real implementation, you would use a proper tokenizer
        token_ids = [ord(c) % (self.vocab_size - 1) + 1 for c in text]

        # If return_tensors is specified, convert to tensor
        if return_tensors == "pt":
            return torch.tensor([token_ids])

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens=False) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: The token IDs to decode.
            skip_special_tokens: Whether to skip special tokens like EOS.

        Returns:
            The decoded text.
        """
        # This is a very simple character-level decoding
        # In a real implementation, you would use a proper tokenizer
        return ''.join([chr(t) for t in token_ids if (t > 0 and (not skip_special_tokens or t != self.eos_token_id))])
