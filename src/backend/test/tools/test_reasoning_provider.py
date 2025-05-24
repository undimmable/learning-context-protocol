"""Tests for the Reasoning Provider."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import requests
from fastapi.testclient import TestClient

from src.tools.reasoning_provider import app
from src.tools.reasoning_integration import ReasoningClient, ReasoningProvider


class TestReasoningProvider(unittest.TestCase):
    """Tests for the Reasoning Provider."""

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/reasoning/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_plan_endpoint(self):
        """Test the plan endpoint."""
        request_data = {
            "task": "Implement a new feature",
            "context": "This is a web application",
            "constraints": ["Must be completed in 2 days", "Must use existing libraries"],
            "max_steps": 5
        }
        response = self.client.post("/reasoning/plan", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("plan", data)
        self.assertIn("reasoning", data)
        self.assertIn("estimated_time", data)
        self.assertIsInstance(data["plan"], list)
        self.assertLessEqual(len(data["plan"]), 5)  # Should respect max_steps

    def test_analyze_endpoint(self):
        """Test the analyze endpoint."""
        request_data = {
            "content": "def add(a, b):\n    return a + b",
            "content_type": "code",
            "focus": ["readability", "performance"]
        }
        response = self.client.post("/reasoning/analyze", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("analysis", data)
        self.assertIn("issues", data)
        self.assertIn("suggestions", data)
        self.assertIsInstance(data["issues"], list)
        self.assertIsInstance(data["suggestions"], list)

    def test_evaluate_endpoint(self):
        """Test the evaluate endpoint."""
        request_data = {
            "solution": "def add(a, b):\n    return a + b",
            "requirements": "Implement a function that adds two numbers",
            "criteria": ["correctness", "readability"]
        }
        response = self.client.post("/reasoning/evaluate", json=request_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIn("evaluation", data)
        self.assertIn("strengths", data)
        self.assertIn("weaknesses", data)
        self.assertIn("recommendations", data)
        self.assertIsInstance(data["score"], float)
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)


class TestReasoningClient(unittest.TestCase):
    """Tests for the Reasoning Client."""

    @patch('requests.get')
    def test_health(self, mock_get):
        """Test the health method."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        # Create a client and call health
        client = ReasoningClient()
        result = client.health()

        # Check the result
        self.assertEqual(result["status"], "healthy")
        mock_get.assert_called_once_with("http://localhost:3210/reasoning/health")

    @patch('requests.post')
    def test_plan(self, mock_post):
        """Test the plan method."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "plan": ["Step 1", "Step 2"],
            "reasoning": "This is the reasoning",
            "estimated_time": "2 hours"
        }
        mock_post.return_value = mock_response

        # Create a client and call plan
        client = ReasoningClient()
        result = client.plan("Implement a feature")

        # Check the result
        self.assertEqual(result["plan"], ["Step 1", "Step 2"])
        self.assertEqual(result["reasoning"], "This is the reasoning")
        self.assertEqual(result["estimated_time"], "2 hours")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_analyze(self, mock_post):
        """Test the analyze method."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "analysis": "This is the analysis",
            "issues": [{"description": "Issue 1"}],
            "suggestions": [{"description": "Suggestion 1"}]
        }
        mock_post.return_value = mock_response

        # Create a client and call analyze
        client = ReasoningClient()
        result = client.analyze("def add(a, b):\n    return a + b", "code")

        # Check the result
        self.assertEqual(result["analysis"], "This is the analysis")
        self.assertEqual(result["issues"], [{"description": "Issue 1"}])
        self.assertEqual(result["suggestions"], [{"description": "Suggestion 1"}])
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_evaluate(self, mock_post):
        """Test the evaluate method."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "score": 0.8,
            "evaluation": "This is the evaluation",
            "strengths": ["Strength 1"],
            "weaknesses": ["Weakness 1"],
            "recommendations": ["Recommendation 1"]
        }
        mock_post.return_value = mock_response

        # Create a client and call evaluate
        client = ReasoningClient()
        result = client.evaluate("def add(a, b):\n    return a + b", "Implement a function that adds two numbers")

        # Check the result
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["evaluation"], "This is the evaluation")
        self.assertEqual(result["strengths"], ["Strength 1"])
        self.assertEqual(result["weaknesses"], ["Weakness 1"])
        self.assertEqual(result["recommendations"], ["Recommendation 1"])
        mock_post.assert_called_once()

    @patch('requests.post', side_effect=Exception("Connection error"))
    def test_plan_fallback(self, mock_post):
        """Test the plan method with fallback."""
        # Create a client and call plan
        client = ReasoningClient()
        result = client.plan("Implement a feature")

        # Check the result
        self.assertIn("plan", result)
        self.assertIn("reasoning", result)
        self.assertIn("estimated_time", result)
        self.assertIsInstance(result["plan"], list)
        mock_post.assert_called_once()


class TestReasoningProvider(unittest.TestCase):
    """Tests for the Reasoning Provider."""

    @patch('src.tools.reasoning_integration.ReasoningClient')
    def test_initialization(self, mock_client_class):
        """Test initialization of the Reasoning Provider."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "healthy"}
        mock_client_class.return_value = mock_client

        # Create a provider
        provider = ReasoningProvider()

        # Check that the client was initialized
        mock_client_class.assert_called_once()
        mock_client.health.assert_called_once()
        self.assertTrue(provider.is_healthy())

    @patch('src.tools.reasoning_integration.ReasoningClient')
    def test_generate_plan(self, mock_client_class):
        """Test the generate_plan method."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "healthy"}
        mock_client.plan.return_value = {
            "plan": ["Step 1", "Step 2"],
            "reasoning": "This is the reasoning",
            "estimated_time": "2 hours"
        }
        mock_client_class.return_value = mock_client

        # Create a provider and call generate_plan
        provider = ReasoningProvider()
        result = provider.generate_plan("Implement a feature")

        # Check the result
        self.assertEqual(result["plan"], ["Step 1", "Step 2"])
        self.assertEqual(result["reasoning"], "This is the reasoning")
        self.assertEqual(result["estimated_time"], "2 hours")
        mock_client.plan.assert_called_once_with("Implement a feature", None, None, 10)

    @patch('src.tools.reasoning_integration.ReasoningClient')
    def test_analyze_content(self, mock_client_class):
        """Test the analyze_content method."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "healthy"}
        mock_client.analyze.return_value = {
            "analysis": "This is the analysis",
            "issues": [{"description": "Issue 1"}],
            "suggestions": [{"description": "Suggestion 1"}]
        }
        mock_client_class.return_value = mock_client

        # Create a provider and call analyze_content
        provider = ReasoningProvider()
        result = provider.analyze_content("def add(a, b):\n    return a + b", "code")

        # Check the result
        self.assertEqual(result["analysis"], "This is the analysis")
        self.assertEqual(result["issues"], [{"description": "Issue 1"}])
        self.assertEqual(result["suggestions"], [{"description": "Suggestion 1"}])
        mock_client.analyze.assert_called_once_with("def add(a, b):\n    return a + b", "code", None)

    @patch('src.tools.reasoning_integration.ReasoningClient')
    def test_evaluate_solution(self, mock_client_class):
        """Test the evaluate_solution method."""
        # Mock the client
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "healthy"}
        mock_client.evaluate.return_value = {
            "score": 0.8,
            "evaluation": "This is the evaluation",
            "strengths": ["Strength 1"],
            "weaknesses": ["Weakness 1"],
            "recommendations": ["Recommendation 1"]
        }
        mock_client_class.return_value = mock_client

        # Create a provider and call evaluate_solution
        provider = ReasoningProvider()
        result = provider.evaluate_solution("def add(a, b):\n    return a + b", "Implement a function that adds two numbers")

        # Check the result
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["evaluation"], "This is the evaluation")
        self.assertEqual(result["strengths"], ["Strength 1"])
        self.assertEqual(result["weaknesses"], ["Weakness 1"])
        self.assertEqual(result["recommendations"], ["Recommendation 1"])
        mock_client.evaluate.assert_called_once_with("def add(a, b):\n    return a + b", "Implement a function that adds two numbers", None)


if __name__ == '__main__':
    unittest.main()