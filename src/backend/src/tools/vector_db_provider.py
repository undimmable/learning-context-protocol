"""Vector Database Provider for the AGI system.

This module provides a FastAPI server that exposes endpoints for storing and
retrieving vectors from a Weaviate vector database.

Run with:

    uvicorn vector_db_provider:app --host 127.0.0.1 --port 4321

Environment variable `VECTOR_DB_URL` should then be set to
```
export VECTOR_DB_URL=http://127.0.0.1:4321
```

The server exposes the following endpoints:

1. POST /vectors/upsert - Store a text snippet with its embedding.
2. POST /vectors/query - Retrieve up to `top_k` snippets most similar to a query string.
3. GET /vectors/health - Check if the vector database is healthy.
"""

from __future__ import annotations

import os
import uuid
from typing import Dict, List, Optional, Any

import weaviate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import the embeddings module
try:
    from src.tools.embeddings import get_embedding
    HAVE_EMBEDDINGS = True
except ImportError:
    print("Warning: embeddings module not found. Vector search will use Weaviate's built-in vectorizer.")
    HAVE_EMBEDDINGS = False

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Vector Database Provider", version="0.1.0")

# ---------------------------------------------------------------------------
# Weaviate client
# ---------------------------------------------------------------------------

# Default Weaviate URL (can be overridden with environment variable)
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:4321")

# Initialize the Weaviate client
try:
    client = weaviate.Client(WEAVIATE_URL)
    # Create the schema if it doesn't exist
    if not client.schema.contains({"classes": [{"class": "TextSnippet"}]}):
        schema = {
            "classes": [
                {
                    "class": "TextSnippet",
                    "description": "A text snippet with its embedding",
                    "vectorizer": "text2vec-transformers",
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "The text content",
                        },
                        {
                            "name": "tags",
                            "dataType": ["string[]"],
                            "description": "Optional tags for the text",
                        },
                        {
                            "name": "timestamp",
                            "dataType": ["string"],
                            "description": "When the snippet was created",
                        },
                    ],
                }
            ]
        }
        client.schema.create(schema)
except Exception as e:
    print(f"Error connecting to Weaviate: {e}")
    # We'll continue and let the health endpoint report the error

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class VectorUpsertArgs(BaseModel):
    text: str = Field(..., description="Text to store")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags")
    timestamp: Optional[str] = Field(default=None, description="Timestamp")

class VectorQueryArgs(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/vectors/upsert")
def vector_upsert(req: VectorUpsertArgs):
    """Store a text snippet with its embedding."""
    try:
        # Generate a UUID for the object
        object_uuid = str(uuid.uuid4())

        # Prepare the data object
        data_object = {
            "text": req.text,
            "tags": req.tags or [],
            "timestamp": req.timestamp or "",
        }

        # If we have the embeddings module, we can generate the embedding ourselves
        # Otherwise, Weaviate will use its built-in vectorizer
        if HAVE_EMBEDDINGS:
            try:
                # Generate the embedding
                embedding = get_embedding(req.text)

                # Store the object in Weaviate with the embedding
                client.data_object.create(
                    data_object=data_object,
                    class_name="TextSnippet",
                    uuid=object_uuid,
                    vector=embedding
                )
                return {"status": "ok", "uuid": object_uuid, "embedding_source": "local"}
            except Exception as e:
                print(f"Error generating embedding: {e}")
                print("Falling back to Weaviate's built-in vectorizer")
                # Fall back to Weaviate's built-in vectorizer
                client.data_object.create(
                    data_object=data_object,
                    class_name="TextSnippet",
                    uuid=object_uuid
                )
                return {"status": "ok", "uuid": object_uuid, "embedding_source": "weaviate"}
        else:
            # Store the object in Weaviate using its built-in vectorizer
            client.data_object.create(
                data_object=data_object,
                class_name="TextSnippet",
                uuid=object_uuid
            )
            return {"status": "ok", "uuid": object_uuid, "embedding_source": "weaviate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing vector: {str(e)}")

@app.post("/vectors/query")
def vector_query(req: VectorQueryArgs):
    """Retrieve up to `top_k` snippets most similar to a query string."""
    try:
        # Prepare the query
        query = client.query.get("TextSnippet", ["text", "tags", "timestamp"])

        # If we have the embeddings module, we can generate the embedding ourselves
        # and use it for the vector search
        if HAVE_EMBEDDINGS:
            try:
                # Generate the embedding for the query
                embedding = get_embedding(req.query)

                # Add the vector search using the embedding
                query = query.with_near_vector({"vector": embedding})

                # Add tag filter if provided
                if req.tags:
                    filter_query = {
                        "operator": "ContainsAny",
                        "path": ["tags"],
                        "valueString": req.tags
                    }
                    query = query.with_where(filter_query)

                # Set the limit
                query = query.with_limit(req.top_k)

                # Execute the query
                result = query.do()

                # Extract the results
                if "data" in result and "Get" in result["data"] and "TextSnippet" in result["data"]["Get"]:
                    snippets = result["data"]["Get"]["TextSnippet"]
                    return {"results": snippets, "embedding_source": "local"}

                return {"results": [], "embedding_source": "local"}
            except Exception as e:
                print(f"Error using local embedding for query: {e}")
                print("Falling back to Weaviate's built-in vectorizer")
                # Fall back to Weaviate's built-in vectorizer

        # Use Weaviate's built-in vectorizer
        # Add the vector search
        query = query.with_near_text({"concepts": [req.query]})

        # Add tag filter if provided
        if req.tags:
            filter_query = {
                "operator": "ContainsAny",
                "path": ["tags"],
                "valueString": req.tags
            }
            query = query.with_where(filter_query)

        # Set the limit
        query = query.with_limit(req.top_k)

        # Execute the query
        result = query.do()

        # Extract the results
        if "data" in result and "Get" in result["data"] and "TextSnippet" in result["data"]["Get"]:
            snippets = result["data"]["Get"]["TextSnippet"]
            return {"results": snippets, "embedding_source": "weaviate"}

        return {"results": [], "embedding_source": "weaviate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying vectors: {str(e)}")

@app.get("/vectors/health")
def health_check():
    """Check if the vector database is healthy."""
    try:
        # Check if we can connect to Weaviate
        is_ready = client.is_ready()
        if is_ready:
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy", "details": {"message": "Weaviate is not ready"}}
    except Exception as e:
        return {"status": "unhealthy", "details": {"message": str(e)}}
