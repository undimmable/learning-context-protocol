import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Union
from torch.utils.data import DataLoader, Dataset

from neural.gpt import GPTLikeModel
from neural.leaky_awareness_unit import LeakyAwarenessUnit

try:
    from transformers import AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


# Example dataset
class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()  # For language modeling, labels are the same as input
        return input_ids, labels


class MCPDataset(Dataset):
    """Dataset for training on MCP server data."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 512):
        """Initialize the MCP dataset.

        Args:
            data_path: Path to the JSON file containing MCP data.
            tokenizer: Tokenizer to use for encoding text.
            max_seq_len: Maximum sequence length.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from a JSON file.

        Args:
            data_path: Path to the JSON file.

        Returns:
            List of data entries.
        """
        if not os.path.exists(data_path):
            print(f"Warning: Data file {data_path} not found. Using empty dataset.")
            return []

        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading data from {data_path}: {e}")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Format the entry as a prompt-response pair
        if isinstance(entry, dict):
            prompt = entry.get('prompt', '')
            response = entry.get('response', '')
            text = f"{prompt}\n{response}"
        else:
            text = str(entry)

        # Tokenize the text
        if HAVE_TRANSFORMERS and hasattr(self.tokenizer, 'encode') and callable(getattr(self.tokenizer, 'encode')):
            # Use HuggingFace tokenizer
            encodings = self.tokenizer(text, max_length=self.max_seq_len, truncation=True, padding="max_length", return_tensors="pt")
            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()
        else:
            # Use simple tokenizer
            if hasattr(self.tokenizer, 'encode') and callable(getattr(self.tokenizer, 'encode')):
                input_ids = torch.tensor(self.tokenizer.encode(text)[:self.max_seq_len])
                attention_mask = torch.ones_like(input_ids)
            else:
                # Fallback to character encoding
                input_ids = torch.tensor([ord(c) % 30000 for c in text[:self.max_seq_len]])
                attention_mask = torch.ones_like(input_ids)

        # For causal language modeling, labels are the same as input_ids
        labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# Training loop
def train_model(model, dataset, epochs, batch_size, learning_rate, device, save_path=None):
    """Train the model on the given dataset.

    Args:
        model: The model to train.
        dataset: The dataset to train on.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        device: Device to train on.
        save_path: Path to save the trained model.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Handle both dictionary and tuple batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch['labels'].to(device)
            else:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = None

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Reshape logits and labels for loss computation
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = criterion(logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save the best model
        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    return model


def fine_tune_on_mcp_data(
    model_path: Optional[str] = None,
    data_path: str = "data/mcp_training_data.json",
    output_path: str = "models/gpt_model_finetuned.pt",
    vocab_size: int = 30522,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    depth: int = 6,
    max_seq_len: int = 512,
    use_attention: bool = True,
    num_heads: int = 8,
    dropout: float = 0.1,
    alpha: float = 0.01,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    tokenizer_name: str = "bert-base-uncased"
):
    """Fine-tune the GPTLikeModel on MCP-specific data.

    Args:
        model_path: Path to a pre-trained model to fine-tune.
        data_path: Path to the training data.
        output_path: Path to save the fine-tuned model.
        vocab_size: Size of the vocabulary.
        embed_dim: Dimension of the embeddings.
        hidden_dim: Dimension of the hidden layers.
        depth: Number of layers.
        max_seq_len: Maximum sequence length.
        use_attention: Whether to use attention mechanisms.
        num_heads: Number of attention heads (only used if use_attention=True).
        dropout: Dropout probability (only used if use_attention=True).
        alpha: Alpha parameter for LeakyReLU (only used if use_attention=False).
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        epochs: Number of training epochs.
        tokenizer_name: Name of the HuggingFace tokenizer to use.
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer
    if HAVE_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print(f"Using HuggingFace tokenizer: {tokenizer_name}")
            # Update vocab_size to match tokenizer
            vocab_size = tokenizer.vocab_size
        except Exception as e:
            print(f"Error initializing HuggingFace tokenizer: {e}")
            print("Using simple tokenizer")
            from src.tools.neural_integration import SimpleTokenizer
            tokenizer = SimpleTokenizer(vocab_size)
    else:
        print("Transformers library not available, using simple tokenizer")
        from src.tools.neural_integration import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size)

    # Initialize or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = GPTLikeModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=depth,
            max_seq_len=max_seq_len,
            use_attention=use_attention,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Initializing new model")
        model = GPTLikeModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=depth,
            max_seq_len=max_seq_len,
            use_attention=use_attention,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )

    # Initialize dataset
    dataset = MCPDataset(data_path, tokenizer, max_seq_len)
    print(f"Dataset size: {len(dataset)}")

    # Train the model
    train_model(model, dataset, epochs, batch_size, learning_rate, device, output_path)

    print(f"Fine-tuning complete. Model saved to {output_path}")
    return model


# Example usage
if __name__ == "__main__":
    # Example 1: Basic training with dummy data (without attention)
    vocab_size = 30522
    embed_dim = 128
    hidden_dim = 256
    num_layers = 6
    max_seq_len = 50
    alpha = 0.01
    batch_size = 32
    learning_rate = 5e-4
    epochs = 5
    num_samples = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model without attention
    model = GPTLikeModel(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        max_seq_len=max_seq_len, 
        use_attention=False,
        alpha=alpha
    )
    dataset = DummyDataset(vocab_size, max_seq_len, num_samples)

    train_model(model, dataset, epochs, batch_size, learning_rate, device)

    # Example 2: Training with attention
    # Create model with attention
    model_with_attention = GPTLikeModel(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        max_seq_len=max_seq_len, 
        use_attention=True,
        num_heads=8,
        dropout=0.1
    )

    train_model(model_with_attention, dataset, epochs, batch_size, learning_rate, device)

    # Example 3: Fine-tuning on MCP data with attention
    # Uncomment to run
    # fine_tune_on_mcp_data(
    #     model_path=None,  # Start with a fresh model
    #     data_path="data/mcp_training_data.json",
    #     output_path="models/gpt_model_finetuned.pt",
    #     use_attention=True,
    #     num_heads=8,
    #     dropout=0.1
    # )
