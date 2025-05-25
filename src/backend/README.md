# LainGPT API Testing and Integration

This repository contains a set of APIs for the LainGPT system, including:

1. MCP Server - A server that provides memory, LLM, and file manipulation capabilities
2. Vector DB Provider - A server that provides vector database capabilities
3. Neural Integration - Integration with neural models for text generation
4. Reasoning Provider - Integration with reasoning capabilities for planning and analysis

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install with `pip install -r requirements.txt`):
  - transformers
  - flask
  - numpy
  - scikit-learn
  - torch
  - python-dotenv
  - sqlalchemy
  - uvicorn
  - alembic
  - psycopg2-binary
  - fastapi
  - weaviate-client
  - pydantic
  - sentence-transformers
  - requests

## Running the Tests

We've provided a script that starts all the necessary services and runs the tests:

```bash
chmod +x run_services.sh
./run_services.sh
```

This script:
1. Starts the Vector DB Provider with a mock Weaviate client
2. Starts the Reasoning Provider
3. Starts the MCP server
4. Runs the API tests
5. Stops the services when done

## Manual Testing

If you want to test the APIs manually, you can:

1. Start the Vector DB Provider:
```bash
export USE_MOCK_WEAVIATE=true
uvicorn src.tools.vector_db_provider:app --host 127.0.0.1 --port 4321
```

2. Start the Reasoning Provider:
```bash
export REASONING_PROVIDER_URL=http://127.0.0.1:3210
uvicorn src.tools.reasoning_provider:app --host 127.0.0.1 --port 3210
```

3. Start the MCP server:
```bash
export CODEX_MCP_URL=http://127.0.0.1:7821
export VECTOR_DB_URL=http://127.0.0.1:4321
export REASONING_PROVIDER_URL=http://127.0.0.1:3210
uvicorn src.tools.mcp_server:app --host 127.0.0.1 --port 7821
```

4. Use the APIs:

### Memory API

Store a text snippet:
```bash
curl -X POST http://127.0.0.1:7821/memory/upsert \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test entry", "tags": ["test", "example"]}'
```

Query for similar snippets:
```bash
curl -X POST http://127.0.0.1:7821/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test entry", "top_k": 5}'
```

### LLM API

Generate text:
```bash
curl -X POST http://127.0.0.1:7821/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_new_tokens": 50}'
```

### Shell API

Run a shell command:
```bash
curl -X POST http://127.0.0.1:7821/shell \
  -H "Content-Type: application/json" \
  -d '{"command": ["ls", "-la"], "workdir": null, "timeout": 10000}'
```

### File API

Read a file:
```bash
curl -X POST http://127.0.0.1:7821/read_file \
  -H "Content-Type: application/json" \
  -d '{"path": "requirements.txt", "max_bytes": 1000, "summarize": true}'
```

Apply a patch:
```bash
curl -X POST http://127.0.0.1:7821/apply_patch \
  -H "Content-Type: application/json" \
  -d '{"patch": "--- file.txt\n+++ file.txt\n@@ -1,1 +1,1 @@\n-old line\n+new line"}'
```

### Fine-tuning API

Get fine-tuning status:
```bash
curl -X GET http://127.0.0.1:7821/fine-tuning/status
```

Trigger fine-tuning:
```bash
curl -X POST http://127.0.0.1:7821/fine-tuning/trigger
```

### Reasoning API

Generate a plan:
```bash
curl -X POST http://127.0.0.1:3210/reasoning/plan \
  -H "Content-Type: application/json" \
  -d '{"task": "Create a simple web application", "context": "Using HTML, CSS, and JavaScript", "constraints": ["Must be responsive"], "max_steps": 5}'
```

Analyze code:
```bash
curl -X POST http://127.0.0.1:3210/reasoning/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "function add(a, b) { return a + b; }", "content_type": "code", "focus": ["readability", "best practices"]}'
```

Evaluate a solution:
```bash
curl -X POST http://127.0.0.1:3210/reasoning/evaluate \
  -H "Content-Type: application/json" \
  -d '{"solution": "function add(a, b) { return a + b; }", "requirements": "Create a function that adds two numbers", "criteria": ["correctness", "simplicity"]}'
```

## Configuration

The system can be configured using environment variables:

- `CODEX_MCP_URL` - URL of the MCP server (default: http://127.0.0.1:7821)
- `VECTOR_DB_URL` - URL of the Vector DB Provider (default: http://127.0.0.1:4321)
- `REASONING_PROVIDER_URL` - URL of the Reasoning Provider (default: http://127.0.0.1:3210)
- `USE_VECTOR_DB` - Whether to use the Vector DB Provider (default: false)
- `USE_GPT_MODEL` - Whether to use the GPTLikeModel (default: false)
- `USE_MOCK_WEAVIATE` - Whether to use a mock Weaviate client (default: false)
- `USE_PERIODIC_FINE_TUNING` - Whether to enable periodic fine-tuning (default: false)

## Project Structure

- `src/tools/mcp_server.py` - The MCP server implementation
- `src/tools/vector_db_provider.py` - The Vector DB Provider implementation
- `src/tools/neural_integration.py` - Integration with neural models
- `src/tools/reasoning_integration.py` - Integration with reasoning capabilities
- `src/tools/mcp_vector_integration.py` - Integration between MCP and Vector DB
- `src/neural/` - Neural model implementations
- `test/tools/` - Tests for the tools
- `test_all_apis.py` - Script to test all APIs
- `run_services.sh` - Script to start all services and run tests
