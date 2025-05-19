import unittest

from fastapi.testclient import TestClient

# The MCP server is our system-under-test.
from src.tools.mcp_server import app, _memory_store


class TestMCPMemoryDB(unittest.TestCase):
    """Validate that the MCP memory endpoints persist data in the database."""

    def setUp(self):
        # FastAPI test client – spins up the app in-process.
        self.client = TestClient(app)

        # Ensure a clean slate for every test case.
        _memory_store.clear()

    def test_memory_upsert(self):
        """The /memory/upsert endpoint writes rows to the DB."""

        payload = {"text": "Remember me", "tags": ["unit", "memory"]}
        res = self.client.post("/memory/upsert", json=payload)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

        # The store should now contain exactly one row with matching content.
        rows = _memory_store.all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "Remember me")
        self.assertEqual(rows[0]["tags"], ["unit", "memory"])

    def test_memory_query(self):
        """The /memory/query endpoint uses fuzzy matching and respects top_k."""

        entries = [
            {"text": "First unit entry", "tags": ["unit"]},
            {"text": "Second unit entry", "tags": ["unit"]},
            {"text": "Different content", "tags": ["other"]},
        ]

        for entry in entries:
            self.client.post("/memory/upsert", json=entry)

        # Query for "unit" – expect the two unit entries back.
        res = self.client.post("/memory/query", json={"query": "unit", "top_k": 2})
        self.assertEqual(res.status_code, 200)
        results = res.json()["results"]
        self.assertEqual(len(results), 2)
        texts = {r["text"] for r in results}
        self.assertSetEqual(texts, {"First unit entry", "Second unit entry"})

        # Query for the uncommon word only present in the third entry.
        res = self.client.post("/memory/query", json={"query": "different", "top_k": 1})
        self.assertEqual(res.status_code, 200)
        results = res.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Different content")

    def test_memory_persistence(self):
        """Rows must survive across *new* store instances (fresh sessions)."""

        self.client.post("/memory/upsert", json={"text": "Persistent", "tags": []})

        # Simulate a new store instance (e.g. after server restart).
        from src.db.memory_store import MemoryStore

        new_store = MemoryStore()
        rows = new_store.all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "Persistent")
