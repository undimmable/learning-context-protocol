# Vector Database Provider

This directory contains the implementation of the Vector Database Provider, which is responsible for storing and retrieving vectors from a Weaviate vector database.

## Overview

The Vector Database Provider is a FastAPI server that exposes endpoints for storing and retrieving vectors. It uses Weaviate as the vector database and sentence-transformers for generating embeddings.

## Components

- `vector_db_provider.py`: The FastAPI server that exposes endpoints for storing and retrieving vectors.
- `embeddings.py`: Functions for converting text to embeddings using sentence-transformers.
- `mcp_vector_integration.py`: Integration between the MCP Server and the Vector Database Provider.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Weaviate vector database:

```bash
docker-compose up -d
```

3. Start the Vector Database Provider:

```bash
cd src/backend
python -m uvicorn src.tools.vector_db_provider:app --host 127.0.0.1 --port 4321
```

## Usage

### Vector Database Provider

The Vector Database Provider exposes the following endpoints:

- `POST /vectors/upsert`: Store a text snippet with its embedding.
- `POST /vectors/query`: Retrieve up to `top_k` snippets most similar to a query string.
- `GET /vectors/health`: Check if the vector database is healthy.

Example usage:

```python
import requests

# Store a text snippet
response = requests.post(
    "http://localhost:4321/vectors/upsert",
    json={
        "text": "This is a test vector",
        "tags": ["test", "vector"],
        "timestamp": "2023-05-01T12:00:00Z"
    }
)
print(response.json())

# Query for similar vectors
response = requests.post(
    "http://localhost:4321/vectors/query",
    json={
        "query": "test vector",
        "top_k": 5,
        "tags": ["test"]
    }
)
print(response.json())
```

### MCP Server Integration

#### Vector Database Integration

To use the Vector Database Provider with the MCP Server, set the `USE_VECTOR_DB` environment variable to `true`:

```bash
export USE_VECTOR_DB=true
python -m uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

This will make the MCP Server use the Vector Database Provider for storing and retrieving embeddings.

#### GPTLikeModel Integration

To use the custom GPTLikeModel for LLM endpoints instead of the default FLAN-T5 model, set the `USE_GPT_MODEL` environment variable to `true`:

```bash
export USE_GPT_MODEL=true
python -m uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

You can also specify model parameters using environment variables:

```bash
export GPT_VOCAB_SIZE=30522
export GPT_EMBED_DIM=128
export GPT_HIDDEN_DIM=256
export GPT_DEPTH=6
export GPT_MAX_SEQ_LEN=512
export GPT_ALPHA=0.01
export GPT_MODEL_PATH=/path/to/saved/model.pt
export GPT_MAX_HISTORY=5  # Number of conversation turns to remember

# Attention mechanism parameters
export GPT_USE_ATTENTION=true  # Enable transformer-style attention (default: true)
export GPT_NUM_HEADS=8  # Number of attention heads
export GPT_DROPOUT=0.1  # Dropout probability for attention layers
```

If a saved model is available at the specified path, it will be loaded. Otherwise, a new model will be initialized.

##### Enhanced Features

The GPTLikeModel integration includes several advanced features:

1. **Transformer-Style Attention**: Implements multi-head self-attention mechanisms similar to those used in GPT models, enabling the model to better capture long-range dependencies and relationships between tokens in the input sequence.

2. **Better Tokenization**: Uses HuggingFace tokenizers when available, with fallback to a simple tokenizer.

3. **Advanced Text Generation**: Supports temperature-based sampling, top-k and top-p (nucleus) filtering for more diverse and higher quality text generation.

4. **Conversation History**: Maintains context across multiple interactions for more coherent responses.

5. **Semantic Search**: Enhances the `/memory/query` endpoint with semantic understanding instead of just string similarity.

6. **File Summarization**: Adds summarization capabilities to the `/read_file` endpoint when the `summarize` parameter is set to `true`.

7. **Fine-Tuning Support**: Includes a comprehensive fine-tuning system for training on MCP-specific data.

##### Attention Mechanism

The GPTLikeModel now supports transformer-style attention mechanisms, which significantly improve its ability to model complex relationships in text:

1. **Multi-Head Self-Attention**: Allows the model to attend to different parts of the input sequence simultaneously, capturing various types of relationships between tokens.

2. **Scaled Dot-Product Attention**: Efficiently computes attention weights using the dot product of queries and keys, scaled to prevent vanishing gradients.

3. **Layer Normalization**: Stabilizes training and improves convergence by normalizing activations within each layer.

4. **Residual Connections**: Helps with gradient flow during training by providing direct paths for gradients to flow through the network.

5. **Position-Aware Processing**: The attention mechanism is inherently position-aware, allowing it to better understand the sequential nature of text.

To enable attention mechanisms, set the `GPT_USE_ATTENTION` environment variable to `true` (enabled by default). You can configure the number of attention heads and dropout rate using the `GPT_NUM_HEADS` and `GPT_DROPOUT` environment variables.

##### Fine-Tuning the Model

To fine-tune the GPTLikeModel on MCP-specific data:

1. Prepare your training data in JSON format:

```json
[
  {
    "prompt": "User query or command",
    "response": "Desired model response"
  },
  ...
]
```

2. Run the fine-tuning script:

```python
from src.neural.training import fine_tune_on_mcp_data

fine_tune_on_mcp_data(
    model_path=None,  # Start with a fresh model, or provide path to existing model
    data_path="path/to/your/data.json",
    output_path="path/to/save/finetuned/model.pt"
)
```

3. Use the fine-tuned model with the MCP Server:

```bash
export GPT_MODEL_PATH=path/to/save/finetuned/model.pt
export USE_GPT_MODEL=true
python -m uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

You can use both the Vector Database Provider and the GPTLikeModel together:

```bash
export USE_VECTOR_DB=true
export USE_GPT_MODEL=true
python -m uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

## Testing

To run the tests for the Vector Database Provider:

```bash
cd src/backend
python -m pytest test/tools/test_vector_db_provider.py -v
python -m pytest test/tools/test_mcp_vector_integration.py -v
```

## Implementation Details

### Embeddings

The Vector Database Provider uses sentence-transformers to generate embeddings for text. By default, it uses the `all-MiniLM-L6-v2` model, which is a good balance between quality and performance.

You can change the model by setting the `DEFAULT_EMBEDDING_MODEL` environment variable:

```bash
export DEFAULT_EMBEDDING_MODEL=all-mpnet-base-v2
```

### Fallback Mechanism

If the Vector Database Provider is not available, the MCP Server will fall back to using the DB Memory Store (PostgreSQL or SQLite) for storing and retrieving embeddings.

Similarly, if the embeddings module is not available, the Vector Database Provider will fall back to using Weaviate's built-in vectorizer.
