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
pip install -r requirements-vector-db.txt
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

To use the Vector Database Provider with the MCP Server, set the `USE_VECTOR_DB` environment variable to `true`:

```bash
export USE_VECTOR_DB=true
python -m uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

This will make the MCP Server use the Vector Database Provider for storing and retrieving embeddings.

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