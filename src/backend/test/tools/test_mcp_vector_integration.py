"""Tests for the integration between MCP Server and Vector Database Provider.

This module contains tests for the integration between the MCP Server and the
Vector Database Provider. It verifies that the MCP Server can use the Vector
Database Provider for storing and retrieving embeddings.

To run these tests, you need to have the Vector Database Provider running:

    docker-compose up -d
    cd src/backend
    python -m uvicorn src.tools.vector_db_provider:app --host 127.0.0.1 --port 4321

Then run the tests:

    cd src/backend
    python -m pytest test/tools/test_mcp_vector_integration.py -v
"""

import os
import time
import unittest
from typing import Dict, List, Any

import requests

# Import the VectorDBMemoryStore
from src.tools.mcp_vector_integration import VectorDBMemoryStore, VectorDBClient

# URL of the Vector Database Provider
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://localhost:4321")

class TestMCPVectorIntegration(unittest.TestCase):
    """Tests for the integration between MCP Server and Vector Database Provider."""

    def setUp(self):
        """Set up the test case."""
        # Wait for the Vector Database Provider to be ready
        max_retries = 5
        retry_delay = 2
        for i in range(max_retries):
            try:
                client = VectorDBClient(VECTOR_DB_URL)
                health = client.health()
                if health["status"] == "healthy":
                    break
                print(f"Vector Database Provider not ready yet, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Error connecting to Vector Database Provider: {e}")
                if i == max_retries - 1:
                    self.skipTest("Vector Database Provider not available")
                time.sleep(retry_delay)
        
        # Initialize the VectorDBMemoryStore
        self.memory_store = VectorDBMemoryStore(VECTOR_DB_URL)

    def test_append_and_all(self):
        """Test appending entries and retrieving all entries."""
        # Append an entry
        entry = {
            "text": "This is a test entry",
            "tags": ["test", "entry"],
            "timestamp": "2023-05-01T12:00:00Z",
        }
        self.memory_store.append(entry)
        
        # Wait for the entry to be indexed
        time.sleep(2)
        
        # Retrieve all entries
        entries = self.memory_store.all()
        
        # Check if the entry is in the list
        found = False
        for e in entries:
            if e["text"] == entry["text"]:
                found = True
                self.assertEqual(e["tags"], entry["tags"])
                break
        self.assertTrue(found, "The appended entry was not found in the list of all entries")

    def test_query(self):
        """Test querying entries."""
        # Append entries
        entries = [
            {
                "text": "The quick brown fox jumps over the lazy dog",
                "tags": ["test", "fox"],
                "timestamp": "2023-05-01T12:00:00Z",
            },
            {
                "text": "The lazy dog sleeps all day",
                "tags": ["test", "dog"],
                "timestamp": "2023-05-01T12:01:00Z",
            },
            {
                "text": "The quick brown fox is very clever",
                "tags": ["test", "fox"],
                "timestamp": "2023-05-01T12:02:00Z",
            },
        ]
        for entry in entries:
            self.memory_store.append(entry)
        
        # Wait for the entries to be indexed
        time.sleep(2)
        
        # Query for entries about foxes
        results = self.memory_store.query("fox", top_k=2)
        
        # Check if the results contain entries about foxes
        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertIn("fox", result["text"].lower())

    def test_fallback_to_in_memory(self):
        """Test fallback to in-memory storage when Vector DB is not available."""
        # Create a VectorDBMemoryStore with an invalid URL
        memory_store = VectorDBMemoryStore("http://invalid-url:9999")
        
        # Check if it's using in-memory storage
        self.assertIsNotNone(memory_store._entries)
        
        # Append an entry
        entry = {
            "text": "This is a test entry for in-memory storage",
            "tags": ["test", "in-memory"],
            "timestamp": "2023-05-01T12:00:00Z",
        }
        memory_store.append(entry)
        
        # Retrieve all entries
        entries = memory_store.all()
        
        # Check if the entry is in the list
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["text"], entry["text"])
        
        # Query for entries
        results = memory_store.query("in-memory", top_k=1)
        
        # Check if the results contain the entry
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], entry["text"])

if __name__ == "__main__":
    unittest.main()