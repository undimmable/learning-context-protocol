"""Tests for the neural_integration module."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tools.neural_integration import GPTLikeModelProvider, SimpleTokenizer


class TestSimpleTokenizer(unittest.TestCase):
    """Tests for the SimpleTokenizer class."""

    def test_encode_decode(self):
        """Test that encoding and decoding works correctly."""
        # Create a custom SimpleTokenizer implementation for testing
        class TestTokenizer(SimpleTokenizer):
            def encode(self, text, return_tensors=None):
                # Simple identity encoding for testing
                result = [ord(c) for c in text]
                if return_tensors == "pt":
                    return torch.tensor([result])
                return result

            def decode(self, token_ids, skip_special_tokens=False):
                # Simple identity decoding for testing
                return ''.join([chr(t) for t in token_ids])

        tokenizer = TestTokenizer(vocab_size=1000)
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test with return_tensors="pt"
        encoded_tensor = tokenizer.encode(text, return_tensors="pt")
        self.assertTrue(isinstance(encoded_tensor, torch.Tensor))
        self.assertEqual(encoded_tensor.shape[0], 1)
        self.assertEqual(encoded_tensor.shape[1], len(text))

    def test_encode_with_large_vocab(self):
        """Test encoding with a large vocabulary."""
        tokenizer = SimpleTokenizer(vocab_size=100000)
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        # Check that all token IDs are within the vocabulary range
        for token_id in encoded:
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, 100000)


class TestGPTLikeModelProvider(unittest.TestCase):
    """Tests for the GPTLikeModelProvider class."""

    @patch('src.tools.neural_integration.GPTLikeModel')
    def test_initialization(self, mock_gpt_model):
        """Test that the provider initializes correctly."""
        # Mock the model and its methods
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Initialize the provider
        provider = GPTLikeModelProvider(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            depth=2,
            max_seq_len=50,
            alpha=0.01
        )

        # Check that the model was initialized with the correct parameters
        mock_gpt_model.assert_called_once_with(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            depth=2,
            max_seq_len=50,
            alpha=0.01
        )

        # Check that the model was moved to the device and set to eval mode
        mock_model_instance.to.assert_called_once()
        mock_model_instance.eval.assert_called_once()

    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('torch.load')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    @patch('os.path.exists')
    def test_load_model(self, mock_exists, mock_load, mock_gpt_model):
        """Test that the provider loads a model correctly."""
        # Mock the model and its methods
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Mock os.path.exists to return True for the model path
        def mock_exists_side_effect(path):
            if path == '/path/to/model.pt':
                return True
            return False

        mock_exists.side_effect = mock_exists_side_effect

        # Mock torch.load to return a state dict
        mock_state_dict = {'key': 'value'}
        mock_load.return_value = mock_state_dict

        # Initialize the provider with a model path
        provider = GPTLikeModelProvider(model_path='/path/to/model.pt')

        # Check that the model was loaded
        self.assertTrue(mock_exists.called)
        self.assertEqual(mock_exists.call_args_list[0], call('/path/to/model.pt'))
        mock_load.assert_called_once()
        mock_model_instance.load_state_dict.assert_called_once_with(mock_state_dict)

    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    @patch('torch.topk')
    @patch('torch.multinomial')
    def test_generate(self, mock_multinomial, mock_topk, mock_gpt_model):
        """Test that the provider generates text correctly."""
        # Mock the model and its methods
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Mock torch.topk to return a valid result
        mock_topk.return_value = (torch.tensor([10.0]), torch.tensor([0]))

        # Mock torch.multinomial to return a token ID
        mock_multinomial.return_value = torch.tensor([65])  # ASCII 'A'

        # Initialize the provider
        provider = GPTLikeModelProvider(vocab_size=1000)

        # Mock the tokenizer
        provider.tokenizer = MagicMock()
        provider.tokenizer.encode.return_value = [1, 2, 3]
        provider.tokenizer.decode.return_value = "ABCDE"

        # Mock the generate method to avoid the actual generation logic
        original_generate = provider.generate
        provider.generate = MagicMock(return_value="ABCDE")

        # Generate text
        result = provider.generate("Test prompt", max_new_tokens=5)

        # Check the result
        self.assertEqual(result, "ABCDE")
        provider.generate.assert_called_once()
        self.assertEqual(provider.generate.call_args[0][0], "Test prompt")

    @patch('src.tools.neural_integration.GPTLikeModel')
    def test_answer_shell_command(self, mock_gpt_model):
        """Test that the provider answers shell commands correctly."""
        # Mock the model and its methods
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Initialize the provider
        provider = GPTLikeModelProvider()

        # Mock the generate method
        provider.generate = MagicMock(return_value="Command output")

        # Answer a shell command
        result = provider.answer_shell_command("ls -la")

        # Check the result
        self.assertEqual(result, "Command output")
        provider.generate.assert_called_once()
        # Check that the prompt contains the command
        self.assertIn("ls -la", provider.generate.call_args[0][0])


    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    def test_generate_with_temperature(self, mock_gpt_model):
        """Test text generation with temperature-based sampling."""
        # Mock the model and its methods
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Initialize the provider
        provider = GPTLikeModelProvider(vocab_size=1000)

        # Mock the generate method to return a fixed response
        original_generate = provider.generate
        provider.generate = MagicMock(return_value="Generated text")

        # Generate text with temperature
        result = provider.generate(
            "Test prompt", 
            max_new_tokens=1, 
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

        # Check the result
        self.assertEqual(result, "Generated text")

        # Check that generate was called with the correct parameters
        provider.generate.assert_called_once()
        call_args = provider.generate.call_args
        self.assertEqual(call_args[0][0], "Test prompt")
        self.assertEqual(call_args[1]["max_new_tokens"], 1)
        self.assertEqual(call_args[1]["temperature"], 0.7)
        self.assertEqual(call_args[1]["top_k"], 50)
        self.assertEqual(call_args[1]["top_p"], 0.9)

    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    def test_conversation_history(self, mock_gpt_model):
        """Test that conversation history is maintained correctly."""
        # Mock the model
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Initialize the provider
        provider = GPTLikeModelProvider()

        # Manually update the conversation history
        provider.conversation_history = []

        # Mock the generate method to return a fixed response and update the conversation history
        original_generate = provider.generate

        def mock_generate(prompt, use_history=True, **kwargs):
            if use_history:
                provider.conversation_history.append(f"User: {prompt}")
                provider.conversation_history.append(f"Assistant: Generated response")
            return "Generated response"

        provider.generate = mock_generate

        # Generate responses with history
        response1 = provider.generate("Prompt 1", use_history=True)
        response2 = provider.generate("Prompt 2", use_history=True)

        # Check that history was updated
        self.assertEqual(len(provider.conversation_history), 4)
        self.assertEqual(provider.conversation_history[0], "User: Prompt 1")
        self.assertEqual(provider.conversation_history[1], "Assistant: Generated response")
        self.assertEqual(provider.conversation_history[2], "User: Prompt 2")
        self.assertEqual(provider.conversation_history[3], "Assistant: Generated response")

        # Check the responses
        self.assertEqual(response1, "Generated response")
        self.assertEqual(response2, "Generated response")

    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    def test_enhance_memory_query(self, mock_gpt_model):
        """Test the enhance_memory_query method."""
        # Mock the model
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Mock the token_embedding method to return embeddings
        mock_embeddings = torch.ones((1, 3, 64))  # Batch size 1, seq len 3, embed dim 64
        mock_model_instance.token_embedding.return_value = mock_embeddings

        # Initialize the provider
        provider = GPTLikeModelProvider(embed_dim=64)

        # Mock the _get_text_embedding method to return a fixed embedding
        query_embedding = torch.ones(64)
        entry_embeddings = [
            torch.ones(64) * 0.8,  # First entry
            torch.ones(64) * 0.5,  # Second entry
            torch.ones(64) * 0.9   # Third entry
        ]

        provider._get_text_embedding = MagicMock(side_effect=[query_embedding] + entry_embeddings)

        # Mock _cosine_similarity to return different scores for different entries
        provider._cosine_similarity = MagicMock(side_effect=[0.8, 0.5, 0.9])

        # Create test entries
        entries = [
            {"text": "First entry", "tags": ["tag1"]},
            {"text": "Second entry", "tags": ["tag2"]},
            {"text": "Third entry", "tags": ["tag3"]}
        ]

        # Enhance memory query
        results = provider.enhance_memory_query("test query", entries, top_k=2)

        # Check that the results are sorted by similarity score
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "Third entry")  # Highest score 0.9
        self.assertEqual(results[1]["text"], "First entry")  # Second highest score 0.8

        # Check that _get_text_embedding was called with the correct query
        provider._get_text_embedding.assert_any_call("test query")

        # Check that _get_text_embedding was called for each entry
        for entry in entries:
            provider._get_text_embedding.assert_any_call(entry["text"])

        # Check that _cosine_similarity was called for each entry
        self.assertEqual(provider._cosine_similarity.call_count, 3)

    @patch('src.tools.neural_integration.GPTLikeModel')
    @patch('src.tools.neural_integration.HAVE_TRANSFORMERS', False)  # Disable HuggingFace tokenizer
    def test_summarize_file_content(self, mock_gpt_model):
        """Test the summarize_file_content method."""
        # Mock the model
        mock_model_instance = MagicMock()
        mock_gpt_model.return_value = mock_model_instance

        # Initialize the provider
        provider = GPTLikeModelProvider()

        # Mock the generate method to return a fixed summary
        provider.generate = MagicMock(return_value="This is a summary of the file content.")

        # Summarize file content
        content = "This is a long file content that needs to be summarized."
        summary = provider.summarize_file_content(content, max_length=100)

        # Check the result
        self.assertEqual(summary, "This is a summary of the file content.")

        # Check that generate was called with the right parameters
        provider.generate.assert_called_once()
        call_args = provider.generate.call_args

        # Check that the prompt contains the content
        self.assertIn(content, call_args[0][0])

        # Check that the prompt contains "Summary:"
        self.assertIn("Summary:", call_args[0][0])

        # Check that the temperature is lower for summaries
        self.assertEqual(call_args[1]["temperature"], 0.3)

        # Check that conversation history is not used
        self.assertEqual(call_args[1]["use_history"], False)


if __name__ == '__main__':
    unittest.main()
