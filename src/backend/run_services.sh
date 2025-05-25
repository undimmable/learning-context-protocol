#!/bin/bash

# This script starts all the necessary services for the MCP server
# and runs the API tests.

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed."
    exit 1
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "Error: uvicorn is not installed."
    echo "Please install it with: pip install uvicorn"
    exit 1
fi

# Set environment variables if not already set
export VECTOR_DB_URL="${VECTOR_DB_URL:-http://localhost:4321}"
export CODEX_MCP_URL="${CODEX_MCP_URL:-http://localhost:7821}"
export REASONING_PROVIDER_URL="${REASONING_PROVIDER_URL:-http://localhost:3210}"
export USE_MOCK_WEAVIATE="${USE_MOCK_WEAVIATE:-true}"

# In Docker environment, we don't need to kill processes
# as each service runs in its own container
if [ "$DOCKER_ENV" != "true" ]; then
    # Function to check if a port is in use
    is_port_in_use() {
        if command -v lsof &> /dev/null; then
            lsof -i :$1 &> /dev/null
            return $?
        elif command -v netstat &> /dev/null; then
            netstat -tuln | grep -q ":$1 "
            return $?
        else
            echo "Warning: Cannot check if port $1 is in use. Proceeding anyway."
            return 1
        fi
    }

    # Kill processes using specific ports
    kill_process_on_port() {
        if command -v lsof &> /dev/null; then
            pid=$(lsof -t -i:$1)
            if [ ! -z "$pid" ]; then
                echo "Killing process $pid on port $1"
                kill -9 $pid
            fi
        elif command -v netstat &> /dev/null && command -v grep &> /dev/null && command -v awk &> /dev/null; then
            pid=$(netstat -tuln | grep ":$1 " | awk '{print $7}' | cut -d'/' -f1)
            if [ ! -z "$pid" ]; then
                echo "Killing process $pid on port $1"
                kill -9 $pid
            fi
        else
            echo "Warning: Cannot kill process on port $1. Please ensure the port is free."
        fi
    }

    # Kill any existing processes on the ports we need
    kill_process_on_port 4321
    kill_process_on_port 7821
    kill_process_on_port 3210
fi

# Start the Vector DB Provider in the background
echo "Starting Vector DB Provider..."
uvicorn src.tools.vector_db_provider:app --host 0.0.0.0 --port 4321 &
VECTOR_DB_PID=$!

# Wait for the Vector DB Provider to start
echo "Waiting for Vector DB Provider to start..."
sleep 5

# Start the Reasoning Provider in the background
echo "Starting Reasoning Provider..."
uvicorn src.tools.reasoning_provider:app --host 0.0.0.0 --port 3210 &
REASONING_PID=$!

# Wait for the Reasoning Provider to start
echo "Waiting for Reasoning Provider to start..."
sleep 5

# Start the MCP server in the background
echo "Starting MCP server..."
uvicorn src.tools.mcp_server:app --host 0.0.0.0 --port 7821 &
MCP_PID=$!

# Wait for the MCP server to start
echo "Waiting for MCP server to start..."
sleep 5

# If we're not running tests, keep the services running
if [ "$RUN_TESTS" != "true" ]; then
    echo "Services started. Press Ctrl+C to stop."
    # Wait for all background processes to finish
    wait
else
    # Run the API tests
    echo "Running API tests..."
    python test_all_apis.py

    # Capture the exit code
    EXIT_CODE=$?

    # Kill the background processes
    echo "Stopping services..."
    kill $VECTOR_DB_PID
    kill $REASONING_PID
    kill $MCP_PID

    echo "Done."
    exit $EXIT_CODE
fi
