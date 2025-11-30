#!/bin/bash

# test-format-hook.sh - Test the format.sh hook

set -euo pipefail

echo "Testing format.sh hook..."
echo

# Clear previous logs
true > /tmp/agent-logs.jsonl

# Test 1: Valid file edit
echo "Test 1: Valid file edit (README.md)"
cat test/events/afterFileEdit/example.json | .cursor/hooks/format.sh
echo "✓ Completed"
echo

# Test 2: Non-existent file
echo "Test 2: Non-existent file"
cat << 'EOF' | .cursor/hooks/format.sh
{
    "conversation_id": "test",
    "generation_id": "test",
    "model": "claude-4.5-sonnet",
    "file_path": "/nonexistent/file.txt",
    "edits": [],
    "hook_event_name": "afterFileEdit",
    "cursor_version": "2.1.39",
    "workspace_roots": ["/Users/gat/personal/cursorrules"],
    "user_email": "test@example.com"
}
EOF
echo "✓ Completed"
echo

# Test 3: Empty file path
echo "Test 3: Empty file path"
cat << 'EOF' | .cursor/hooks/format.sh
{
    "conversation_id": "test",
    "generation_id": "test",
    "model": "claude-4.5-sonnet",
    "edits": [],
    "hook_event_name": "afterFileEdit",
    "cursor_version": "2.1.39",
    "workspace_roots": ["/Users/gat/personal/cursorrules"],
    "user_email": "test@example.com"
}
EOF
echo "✓ Completed"
echo

# Display logs
echo "═══════════════════════════════════════════════════════════"
echo "Log entries:"
echo "═══════════════════════════════════════════════════════════"
cat /tmp/agent-logs.jsonl | jq -c '.'
echo

echo "✓ All tests completed successfully"

