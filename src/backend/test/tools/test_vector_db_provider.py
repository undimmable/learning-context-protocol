"""Tests for the Vector Database Provider.

This module contains tests for the Vector Database Provider, which is responsible
for storing and retrieving vectors from a Weaviate vector database.

To run these tests, you need to have the Vector Database Provider running:

    docker-compose up -d
    cd src/backend
    python -m uvicorn src.tools.vector_db_provider:app --host 127.0.0.1 --port 4321

Then run the tests:

    cd src/backend
    python -m pytest test/tools/test_vector_db_provider.py -v
"""

import os
import time
import unittest
from typing import Dict, List, Any

import requests

# URL of the Vector Database Provider
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://localhost:4321")

class TestVectorDBProvider(unittest.TestCase):
    """Tests for the Vector Database Provider."""

    def setUp(self):
        """Set up the test case."""
        # Wait for the Vector Database Provider to be ready
        max_retries = 5
        retry_delay = 2
        for i in range(max_retries):
            try:
                response = requests.get(f"{VECTOR_DB_URL}/vectors/health")
                if response.status_code == 200 and response.json()["status"] == "healthy":
                    break
                print(f"Vector Database Provider not ready yet, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Error connecting to Vector Database Provider: {e}")
                if i == max_retries - 1:
                    self.skipTest("Vector Database Provider not available")
                time.sleep(retry_delay)

    def test_health_check(self):
        """Test the health check endpoint."""
        response = requests.get(f"{VECTOR_DB_URL}/vectors/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_upsert_and_query(self):
        """Test upserting and querying vectors."""
        # Upsert a vector
        text = "This is a test vector"
        tags = ["test", "vector"]
        timestamp = "2023-05-01T12:00:00Z"
        
        response = requests.post(
            f"{VECTOR_DB_URL}/vectors/upsert",
            json={"text": text, "tags": tags, "timestamp": timestamp}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("uuid", data)
        
        # Wait for the vector to be indexed
        time.sleep(2)
        
        # Query the vector
        response = requests.post(
            f"{VECTOR_DB_URL}/vectors/query",
            json={"query": "test vector", "top_k": 5}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertGreaterEqual(len(data["results"]), 1)
        
        # Check if the result contains the text we upserted
        found = False
        for result in data["results"]:
            if result["text"] == text:
                found = True
                self.assertEqual(result["tags"], tags)
                self.assertEqual(result["timestamp"], timestamp)
                break
        self.assertTrue(found, "The upserted vector was not found in the query results")

    def test_query_with_tags(self):
        """Test querying vectors with tag filtering."""
        # Upsert vectors with different tags
        texts = [
            "This is a test vector with tag1",
            "This is a test vector with tag2",
            "This is a test vector with both tag1 and tag2"
        ]
        tags_list = [
            ["test", "tag1"],
            ["test", "tag2"],
            ["test", "tag1", "tag2"]
        ]
        
        for text, tags in zip(texts, tags_list):
            response = requests.post(
                f"{VECTOR_DB_URL}/vectors/upsert",
                json={"text": text, "tags": tags}
            )
            self.assertEqual(response.status_code, 200)
        
        # Wait for the vectors to be indexed
        time.sleep(2)
        
        # Query vectors with tag1
        response = requests.post(
            f"{VECTOR_DB_URL}/vectors/query",
            json={"query": "test vector", "top_k": 10, "tags": ["tag1"]}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        
        # Check if the results contain only vectors with tag1
        for result in data["results"]:
            self.assertIn("tag1", result["tags"])

if __name__ == "__main__":
    unittest.main()