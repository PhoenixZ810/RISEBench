#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Replace with your own keys
export API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxx"
export TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m rise-agent --config "${SCRIPT_DIR}/agent.cfg" "$@"
