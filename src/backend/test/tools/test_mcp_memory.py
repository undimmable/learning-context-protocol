import unittest
import requests
import json
import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the MCP server module
from src.tools.mcp_server import app, _memory_store

# Use FastAPI's TestClient for testing
from fastapi.testclient import TestClient

class TestMCPMemory(unittest.TestCase):
    def setUp(self):
        # Create a test client
        self.client = TestClient(app)
        
        # Backup the original memory.jsonl file if it exists
        self.memory_file = _memory_store.file_path
        self.backup_file = self.memory_file.with_suffix('.jsonl.bak')
        if self.memory_file.exists():
            self.memory_file.rename(self.backup_file)
        
        # Create a fresh memory store for testing
        _memory_store._entries = []
        
    def tearDown(self):
        # Remove the test memory file
        if self.memory_file.exists():
            self.memory_file.unlink()
        
        # Restore the original memory file if it was backed up
        if self.backup_file.exists():
            self.backup_file.rename(self.memory_file)
    
    def test_memory_upsert(self):
        """Test that we can add entries to the memory store."""
        # Add a test entry
        response = self.client.post(
            "/memory/upsert",
            json={"text": "Test memory entry", "tags": ["test", "memory"]}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        
        # Verify the entry was added to the memory store
        self.assertEqual(len(_memory_store._entries), 1)
        self.assertEqual(_memory_store._entries[0]["text"], "Test memory entry")
        self.assertEqual(_memory_store._entries[0]["tags"], ["test", "memory"])
        
        # Verify the entry was written to the file
        self.assertTrue(self.memory_file.exists())
        with open(self.memory_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["text"], "Test memory entry")
            self.assertEqual(entry["tags"], ["test", "memory"])
    
    def test_memory_query(self):
        """Test that we can query entries from the memory store."""
        # Add some test entries
        entries = [
            {"text": "First test entry", "tags": ["test", "first"]},
            {"text": "Second test entry", "tags": ["test", "second"]},
            {"text": "Third entry with different content", "tags": ["different"]}
        ]
        
        for entry in entries:
            self.client.post("/memory/upsert", json=entry)
        
        # Query for entries containing "test"
        response = self.client.post(
            "/memory/query",
            json={"query": "test", "top_k": 2}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 2)
        
        # The first two entries should be returned (they have higher similarity to "test")
        texts = [result["text"] for result in results]
        self.assertIn("First test entry", texts)
        self.assertIn("Second test entry", texts)
        
        # Query for entries containing "different"
        response = self.client.post(
            "/memory/query",
            json={"query": "different", "top_k": 1}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Third entry with different content")
    
    def test_memory_persistence(self):
        """Test that memory entries persist across server restarts."""
        # Add a test entry
        self.client.post(
            "/memory/upsert",
            json={"text": "Persistent entry", "tags": ["persistent"]}
        )
        
        # Simulate server restart by creating a new memory store instance
        new_memory_store = _memory_store.__class__(_memory_store.file_path)
        
        # Verify the entry was loaded from the file
        self.assertEqual(len(new_memory_store._entries), 1)
        self.assertEqual(new_memory_store._entries[0]["text"], "Persistent entry")
        self.assertEqual(new_memory_store._entries[0]["tags"], ["persistent"])

if __name__ == "__main__":
    unittest.main()