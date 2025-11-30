#!/bin/bash

# format.sh - Runs pre-commit on edited files after file edits
#
# This hook is triggered by Cursor's afterFileEdit event. It:
# 1. Extracts the file path from the edit event
# 2. Runs pre-commit on that specific file
# 3. Logs the result to /tmp/agent-logs.jsonl with status and output
#
# Exit codes from pre-commit:
# - 0: All hooks passed
# - 1: Some hooks failed or modified files
# - >1: Error occurred
#
# The hook always exits 0 to avoid blocking the editor.

set -euo pipefail

# Read JSON input from stdin
input=$(cat)

# Parse the file path from the JSON input
file_path=$(echo "$input" | jq -r '.file_path // empty')

# Create timestamp for logging
timestamp=$(date '+%Y-%m-%d %H:%M:%S')

# Initialize log entry
log_entry=$(jq -n \
    --arg timestamp "$timestamp" \
    --arg hook "format.sh" \
    --arg event "afterFileEdit" \
    --arg file "$file_path" \
    '{timestamp: $timestamp, hook: $hook, event: $event, file: $file}')

# Check if file_path is empty or doesn't exist
if [[ -z "$file_path" ]]; then
    log_entry=$(echo "$log_entry" | \
        jq '. + {status: "skipped", reason: "no file path provided"}')
    echo "$log_entry" >> /tmp/agent-logs.jsonl
    exit 0
fi

if [[ ! -f "$file_path" ]]; then
    log_entry=$(echo "$log_entry" | \
        jq '. + {status: "skipped", reason: "file does not exist"}')
    echo "$log_entry" >> /tmp/agent-logs.jsonl
    exit 0
fi

# Check if pre-commit is available
if ! command -v pre-commit &> /dev/null; then
    log_entry=$(echo "$log_entry" | \
        jq '. + {status: "skipped", reason: "pre-commit not installed"}')
    echo "$log_entry" >> /tmp/agent-logs.jsonl
    exit 0
fi

# Check if pre-commit config exists
if [[ ! -f ".pre-commit-config.yaml" ]]; then
    log_entry=$(echo "$log_entry" | \
        jq '. + {status: "skipped", reason: "no config found"}')
    echo "$log_entry" >> /tmp/agent-logs.jsonl
    exit 0
fi

# Run pre-commit on the edited file
# Capture both stdout and stderr, and the exit code
output=$(pre-commit run --files "$file_path" 2>&1) || exit_code=$?
exit_code=${exit_code:-0}

# Determine status based on exit code
if [[ $exit_code -eq 0 ]]; then
    status="success"
elif [[ $exit_code -eq 1 ]]; then
    status="modified"
else
    status="failed"
fi

# Add results to log entry
log_entry=$(echo "$log_entry" | jq \
    --arg status "$status" \
    --argjson exit_code "$exit_code" \
    --arg output "$output" \
    '. + {status: $status, exit_code: $exit_code, output: $output}')

# Write to log file
echo "$log_entry" >> /tmp/agent-logs.jsonl

# Always allow the operation to continue
exit 0
