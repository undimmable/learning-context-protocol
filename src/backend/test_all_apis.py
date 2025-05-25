"""Test script to verify all APIs work as expected.

This script tests the MCP server and its integrations with the Vector DB Provider,
Reasoning Provider, and Neural Integration.

To run this script:
1. Start the Vector DB Provider: uvicorn src.tools.vector_db_provider:app --host 127.0.0.1 --port 4321
2. Start the Reasoning Provider: uvicorn src.tools.reasoning_provider:app --host 127.0.0.1 --port 3210
3. Start the MCP server: uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
4. Run this script: python test_all_apis.py
"""

import os
import requests
import time
import json
import tempfile
from typing import Dict, Any, List, Optional

# URLs for the different services
MCP_URL = os.environ.get("CODEX_MCP_URL", "http://127.0.0.1:7821")
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://127.0.0.1:4321")
REASONING_URL = os.environ.get("REASONING_PROVIDER_URL", "http://127.0.0.1:3210")

def test_mcp_health():
    """Test if the MCP server is running."""
    try:
        # Try to access the llm/generate endpoint with a simple request
        # This is just to check if the server is running, not to test the endpoint
        data = {
            "prompt": "test",
            "max_new_tokens": 1
        }
        response = requests.post(f"{MCP_URL}/llm/generate", json=data)
        is_running = response.status_code != 404  # Any response other than 404 means the server is running
        print(f"MCP server health check: {'OK' if is_running else 'Failed'}")
        return is_running
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        return False

def test_vector_db_health():
    """Test if the Vector DB Provider is running and healthy."""
    try:
        response = requests.get(f"{VECTOR_DB_URL}/vectors/health")
        result = response.json()
        is_healthy = result.get("status") == "healthy"
        print(f"Vector DB health check: {'OK' if is_healthy else 'Failed'}")
        return is_healthy
    except Exception as e:
        print(f"Error connecting to Vector DB Provider: {e}")
        return False

def test_memory_upsert():
    """Test the memory/upsert endpoint."""
    try:
        data = {
            "text": "This is a test entry for the memory store",
            "tags": ["test", "memory"]
        }
        response = requests.post(f"{MCP_URL}/memory/upsert", json=data)
        result = response.json()
        is_success = result.get("status") == "ok"
        print(f"Memory upsert test: {'OK' if is_success else 'Failed'}")
        return is_success
    except Exception as e:
        print(f"Error testing memory/upsert: {e}")
        return False

def test_memory_query():
    """Test the memory/query endpoint."""
    try:
        # First, insert some test data
        test_memory_upsert()

        # Wait a moment for the data to be indexed
        time.sleep(1)

        # Query for the test data
        data = {
            "query": "test entry",
            "top_k": 5
        }
        response = requests.post(f"{MCP_URL}/memory/query", json=data)
        result = response.json()
        has_results = len(result.get("results", [])) > 0
        print(f"Memory query test: {'OK' if has_results else 'Failed'}")
        return has_results
    except Exception as e:
        print(f"Error testing memory/query: {e}")
        return False

def test_llm_generate():
    """Test the llm/generate endpoint."""
    try:
        data = {
            "prompt": "What is the capital of France?",
            "max_new_tokens": 50
        }
        response = requests.post(f"{MCP_URL}/llm/generate", json=data)
        result = response.json()
        has_text = "generated_text" in result and result["generated_text"]
        print(f"LLM generate test: {'OK' if has_text else 'Failed'}")
        if has_text:
            print(f"Generated text: {result['generated_text']}")
        return has_text
    except Exception as e:
        print(f"Error testing llm/generate: {e}")
        return False

def test_shell():
    """Test the shell endpoint."""
    try:
        data = {
            "command": ["ls", "-la"],
            "workdir": None,
            "timeout": 10000
        }
        response = requests.post(f"{MCP_URL}/shell", json=data)
        result = response.json()
        has_stdout = "stdout" in result
        print(f"Shell test: {'OK' if has_stdout else 'Failed'}")
        if has_stdout:
            print(f"Shell output: {result['stdout'][:100]}...")
        return has_stdout
    except Exception as e:
        print(f"Error testing shell: {e}")
        return False

def test_read_file():
    """Test the read_file endpoint."""
    try:
        data = {
            "path": "requirements.txt",
            "max_bytes": 1000,
            "summarize": True
        }
        response = requests.post(f"{MCP_URL}/read_file", json=data)
        result = response.json()
        has_content = "content" in result and result["content"]
        print(f"Read file test: {'OK' if has_content else 'Failed'}")
        if has_content:
            print(f"File content (first 100 chars): {result['content'][:100]}...")
            if "summary" in result:
                print(f"File summary: {result['summary']}")
        return has_content
    except Exception as e:
        print(f"Error testing read_file: {e}")
        return False

def test_apply_patch():
    """Test the apply_patch endpoint."""
    try:
        # Create a temporary file to patch
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("old line\n")
            temp_path = temp_file.name

        # Create a patch
        patch_content = f"--- {temp_path}\n+++ {temp_path}\n@@ -1,1 +1,1 @@\n-old line\n+new line\n"

        # Apply the patch
        data = {
            "patch": patch_content
        }
        response = requests.post(f"{MCP_URL}/apply_patch", json=data)
        result = response.json()
        is_success = result.get("status") == "ok"

        # Verify the patch was applied
        if is_success:
            with open(temp_path, 'r') as f:
                content = f.read()
            is_success = "new line" in content

        print(f"Apply patch test: {'OK' if is_success else 'Failed'}")

        # Clean up
        os.remove(temp_path)

        return is_success
    except Exception as e:
        print(f"Error testing apply_patch: {e}")
        return False

def test_fine_tuning_status():
    """Test the fine-tuning/status endpoint."""
    try:
        response = requests.get(f"{MCP_URL}/fine-tuning/status")
        result = response.json()
        has_status = "enabled" in result
        print(f"Fine-tuning status test: {'OK' if has_status else 'Failed'}")
        if has_status:
            print(f"Fine-tuning enabled: {result['enabled']}")
            if result['enabled'] and 'status' in result:
                print(f"Fine-tuning status: {result['status']}")
        return has_status
    except Exception as e:
        print(f"Error testing fine-tuning/status: {e}")
        return False

def test_fine_tuning_trigger():
    """Test the fine-tuning/trigger endpoint."""
    try:
        # First check if fine-tuning is enabled
        status_response = requests.get(f"{MCP_URL}/fine-tuning/status")
        status_result = status_response.json()

        if not status_result.get('enabled', False):
            print("Fine-tuning trigger test: Skipped (fine-tuning not enabled)")
            return True  # Skip test if fine-tuning is not enabled

        # Try to trigger fine-tuning
        response = requests.post(f"{MCP_URL}/fine-tuning/trigger")
        result = response.json()
        is_success = "status" in result
        print(f"Fine-tuning trigger test: {'OK' if is_success else 'Failed'}")
        if is_success:
            print(f"Trigger response: {result['status']}")
        return is_success
    except Exception as e:
        print(f"Error testing fine-tuning/trigger: {e}")
        return False

def test_reasoning_health():
    """Test if the Reasoning Provider is running."""
    try:
        response = requests.get(f"{REASONING_URL}/reasoning/health")
        result = response.json()
        is_healthy = result.get("status") == "healthy"
        print(f"Reasoning Provider health check: {'OK' if is_healthy else 'Failed'}")
        return is_healthy
    except Exception as e:
        print(f"Error connecting to Reasoning Provider: {e}")
        return False

def test_reasoning_plan():
    """Test the reasoning/plan endpoint."""
    try:
        data = {
            "task": "Create a simple web application that displays a list of items",
            "context": "The application should use HTML, CSS, and JavaScript",
            "constraints": ["Must be responsive", "Should work in modern browsers"],
            "max_steps": 5
        }
        response = requests.post(f"{REASONING_URL}/reasoning/plan", json=data)
        result = response.json()
        has_plan = "plan" in result and isinstance(result["plan"], list) and len(result["plan"]) > 0
        print(f"Reasoning plan test: {'OK' if has_plan else 'Failed'}")
        if has_plan:
            print(f"Plan first step: {result['plan'][0]}")
            print(f"Reasoning: {result['reasoning'][:100]}...")
        return has_plan
    except Exception as e:
        print(f"Error testing reasoning/plan: {e}")
        return False

def test_reasoning_analyze():
    """Test the reasoning/analyze endpoint."""
    try:
        data = {
            "content": "function add(a, b) { return a + b; }",
            "content_type": "code",
            "focus": ["readability", "best practices"]
        }
        response = requests.post(f"{REASONING_URL}/reasoning/analyze", json=data)
        result = response.json()
        has_analysis = "analysis" in result and result["analysis"]
        print(f"Reasoning analyze test: {'OK' if has_analysis else 'Failed'}")
        if has_analysis:
            print(f"Analysis: {result['analysis'][:100]}...")
        return has_analysis
    except Exception as e:
        print(f"Error testing reasoning/analyze: {e}")
        return False

def test_reasoning_evaluate():
    """Test the reasoning/evaluate endpoint."""
    try:
        data = {
            "solution": "function add(a, b) { return a + b; }",
            "requirements": "Create a function that adds two numbers and returns the result",
            "criteria": ["correctness", "simplicity"]
        }
        response = requests.post(f"{REASONING_URL}/reasoning/evaluate", json=data)
        result = response.json()
        has_evaluation = "evaluation" in result and result["evaluation"]
        print(f"Reasoning evaluate test: {'OK' if has_evaluation else 'Failed'}")
        if has_evaluation:
            print(f"Score: {result['score']}")
            print(f"Evaluation: {result['evaluation'][:100]}...")
        return has_evaluation
    except Exception as e:
        print(f"Error testing reasoning/evaluate: {e}")
        return False

def test_neural_integration():
    """Test the Neural Integration through the LLM generate endpoint."""
    try:
        # Check if GPT model is enabled
        use_gpt_model = os.environ.get("USE_GPT_MODEL", "false").lower() == "true"

        # Test with a prompt that would produce different results with different models
        data = {
            "prompt": "Explain how neural networks work in one sentence",
            "max_new_tokens": 50
        }
        response = requests.post(f"{MCP_URL}/llm/generate", json=data)
        result = response.json()

        has_text = "generated_text" in result and result["generated_text"]
        print(f"Neural Integration test: {'OK' if has_text else 'Failed'}")

        if has_text:
            print(f"Generated text: {result['generated_text']}")
            print(f"Using GPT model: {use_gpt_model}")

            # If we're using the GPT model, we can check for specific characteristics
            # of the output, but this is optional and might not be reliable
            if use_gpt_model:
                # GPT models often produce more fluent and comprehensive responses
                # This is just a heuristic check
                is_gpt_like = len(result['generated_text']) > 20
                print(f"Output appears to be from GPT model: {is_gpt_like}")

        return has_text
    except Exception as e:
        print(f"Error testing Neural Integration: {e}")
        return False

def run_all_tests():
    """Run all API tests."""
    print("Starting API tests...")

    # Test MCP server health
    mcp_healthy = test_mcp_health()
    if not mcp_healthy:
        print("MCP server is not running or not healthy. Aborting tests.")
        return False

    # Test Vector DB health
    vector_db_healthy = test_vector_db_health()
    if not vector_db_healthy:
        print("Vector DB is not running or not healthy. Some tests may fail.")

    # Test Reasoning Provider health
    reasoning_healthy = test_reasoning_health()
    if not reasoning_healthy:
        print("Reasoning Provider is not running or not healthy. Some tests may fail.")

    # Test memory endpoints
    memory_upsert_ok = test_memory_upsert()
    memory_query_ok = test_memory_query()

    # Test LLM endpoint
    llm_generate_ok = test_llm_generate()

    # Test shell endpoint
    shell_ok = test_shell()

    # Test read_file endpoint
    read_file_ok = test_read_file()

    # Test apply_patch endpoint
    apply_patch_ok = test_apply_patch()

    # Test fine-tuning endpoints
    fine_tuning_status_ok = test_fine_tuning_status()
    fine_tuning_trigger_ok = test_fine_tuning_trigger()

    # Test neural integration
    neural_integration_ok = test_neural_integration()

    # Test reasoning endpoints (only if the reasoning provider is healthy)
    reasoning_plan_ok = False
    reasoning_analyze_ok = False
    reasoning_evaluate_ok = False

    if reasoning_healthy:
        reasoning_plan_ok = test_reasoning_plan()
        reasoning_analyze_ok = test_reasoning_analyze()
        reasoning_evaluate_ok = test_reasoning_evaluate()

    # Print summary
    print("\nTest Summary:")
    print(f"MCP server health: {'OK' if mcp_healthy else 'Failed'}")
    print(f"Vector DB health: {'OK' if vector_db_healthy else 'Failed'}")
    print(f"Reasoning Provider health: {'OK' if reasoning_healthy else 'Failed'}")
    print(f"Memory upsert: {'OK' if memory_upsert_ok else 'Failed'}")
    print(f"Memory query: {'OK' if memory_query_ok else 'Failed'}")
    print(f"LLM generate: {'OK' if llm_generate_ok else 'Failed'}")
    print(f"Shell: {'OK' if shell_ok else 'Failed'}")
    print(f"Read file: {'OK' if read_file_ok else 'Failed'}")
    print(f"Apply patch: {'OK' if apply_patch_ok else 'Failed'}")
    print(f"Fine-tuning status: {'OK' if fine_tuning_status_ok else 'Failed'}")
    print(f"Fine-tuning trigger: {'OK' if fine_tuning_trigger_ok else 'Failed'}")
    print(f"Neural integration: {'OK' if neural_integration_ok else 'Failed'}")

    if reasoning_healthy:
        print(f"Reasoning plan: {'OK' if reasoning_plan_ok else 'Failed'}")
        print(f"Reasoning analyze: {'OK' if reasoning_analyze_ok else 'Failed'}")
        print(f"Reasoning evaluate: {'OK' if reasoning_evaluate_ok else 'Failed'}")
    else:
        print("Reasoning endpoints: Skipped (Reasoning Provider not healthy)")

    # Overall result
    all_ok = (
        mcp_healthy and
        memory_upsert_ok and
        memory_query_ok and
        llm_generate_ok and
        shell_ok and
        read_file_ok and
        apply_patch_ok and
        fine_tuning_status_ok and
        fine_tuning_trigger_ok and
        neural_integration_ok
    )

    # Include reasoning tests in overall result only if the reasoning provider is healthy
    if reasoning_healthy:
        all_ok = all_ok and reasoning_plan_ok and reasoning_analyze_ok and reasoning_evaluate_ok

    print(f"\nOverall result: {'All tests passed!' if all_ok else 'Some tests failed.'}")
    return all_ok

if __name__ == "__main__":
    run_all_tests()
