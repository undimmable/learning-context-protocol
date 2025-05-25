#!/bin/bash

# This script runs GitHub Actions locally using act
# https://github.com/nektos/act

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo "Error: act is not installed."
    echo "Please install act first:"
    echo "  - On macOS: brew install act"
    echo "  - On Linux: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
    echo "  - Or visit https://github.com/nektos/act for other installation methods"
    exit 1
fi

# Navigate to the repository root (assuming this script is in the repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run act with default settings
# This will use the workflows defined in .github/workflows
echo "Running GitHub Actions locally using act..."
act "$@"

echo ""
echo "For more options, run: act --help"
echo "Common options:"
echo "  - Run a specific workflow: act -W .github/workflows/python-tests.yaml"
echo "  - Run a specific job: act -j test"
echo "  - Run with a specific event: act pull_request"
echo "  - List all actions: act -l"