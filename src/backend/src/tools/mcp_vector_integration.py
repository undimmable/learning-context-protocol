"""Integration between MCP Server and Vector Database Provider.

This module provides functions for integrating the MCP Server with the Vector
Database Provider. It extends the MCP Server to use the Vector Database Provider
for storing and retrieving embeddings.

Usage:
    from src.tools.mcp_vector_integration import VectorDBMemoryStore

    # Initialize the Vector DB Memory Store
    memory_store = VectorDBMemoryStore()

    # Use it in the MCP Server
    _memory_store = memory_store
"""

from __future__ import annotations

import datetime as dt
import json
import os
import requests
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# Vector Database Provider client
# ---------------------------------------------------------------------------

# Default Vector DB URL (can be overridden with environment variable)
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://localhost:4321")

class VectorDBClient:
    """Client for interacting with the Vector Database Provider."""

    def __init__(self, base_url: str = VECTOR_DB_URL):
        """Initialize the Vector DB client.

        Args:
            base_url: The base URL of the Vector Database Provider.
        """
        self.base_url = base_url

    def health(self) -> Dict[str, Any]:
        """Check if the Vector Database Provider is healthy.

        Returns:
            A dictionary with the health status.
        """
        try:
            response = requests.get(f"{self.base_url}/vectors/health")
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "details": {"message": str(e)}}

    def upsert(self, text: str, tags: Optional[List[str]] = None, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Store a text snippet with its embedding.

        Args:
            text: The text to store.
            tags: Optional tags for the text.
            timestamp: Optional timestamp for the text.

        Returns:
            A dictionary with the status and UUID of the stored text.
        """
        data = {
            "text": text,
            "tags": tags or [],
            "timestamp": timestamp or dt.datetime.now(dt.UTC).isoformat() + "Z",
        }
        response = requests.post(f"{self.base_url}/vectors/upsert", json=data)
        return response.json()

    def query(self, query: str, top_k: int = 5, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieve up to `top_k` snippets most similar to a query string.

        Args:
            query: The search query.
            top_k: The number of results to return.
            tags: Optional tags to filter by.

        Returns:
            A dictionary with the search results.
        """
        data = {
            "query": query,
            "top_k": top_k,
            "tags": tags or [],
        }
        response = requests.post(f"{self.base_url}/vectors/query", json=data)
        return response.json()

# ---------------------------------------------------------------------------
# Vector DB Memory Store
# ---------------------------------------------------------------------------

class VectorDBMemoryStore:
    """Memory store that uses the Vector Database Provider for storage."""

    def __init__(self, vector_db_url: str = VECTOR_DB_URL):
        """Initialize the Vector DB Memory Store.

        Args:
            vector_db_url: The URL of the Vector Database Provider.
        """
        self.client = VectorDBClient(vector_db_url)
        
        # Check if the Vector Database Provider is healthy
        health = self.client.health()
        if health["status"] != "healthy":
            print(f"Warning: Vector Database Provider is not healthy: {health}")
            print("Falling back to in-memory storage")
            self._entries = []
        else:
            self._entries = None  # We'll use the Vector Database Provider

    def append(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the memory store.

        Args:
            entry: The entry to append.
        """
        if self._entries is not None:
            # Fallback to in-memory storage
            self._entries.append(entry)
        else:
            # Use the Vector Database Provider
            self.client.upsert(
                text=entry["text"],
                tags=entry.get("tags", []),
                timestamp=entry.get("timestamp", None),
            )

    def all(self) -> List[Dict[str, Any]]:
        """Get all entries from the memory store.

        Returns:
            A list of all entries.
        """
        if self._entries is not None:
            # Fallback to in-memory storage
            return self._entries
        else:
            # Use the Vector Database Provider
            # Since there's no "get all" endpoint, we'll use a query that should match everything
            result = self.client.query("", top_k=1000)
            return result.get("results", [])

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the memory store for entries similar to the query.

        Args:
            query: The search query.
            top_k: The number of results to return.

        Returns:
            A list of entries similar to the query.
        """
        if self._entries is not None:
            # Fallback to in-memory storage
            # This is a simplified version of the query logic in the MCP Server
            import difflib
            scored = []
            q = query.lower()
            for ent in self._entries:
                score = difflib.SequenceMatcher(None, q, ent["text"].lower()).ratio()
                scored.append((score, ent))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:top_k]]
        else:
            # Use the Vector Database Provider
            result = self.client.query(query, top_k=top_k)
            return result.get("results", [])