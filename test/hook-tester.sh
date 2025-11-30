#!/bin/bash

# hook-tester.sh - Test cursor hooks locally
#
# Usage:
#   ./test/hook-tester.sh --event=<event-json-file> --hook=<hook-name>
#   ./test/hook-tester.sh --event=<event-json-file> --command=<specific-command>
#
# Examples:
#   ./test/hook-tester.sh --event=test/events/beforeShellExecution/ls.json --hook=beforeShellExecution
#   ./test/hook-tester.sh --event=test/events/beforeShellExecution/ls.json --command=".cursor/hooks/block-git.sh"

set -euo pipefail

# Parse arguments
EVENT_FILE=""
HOOK_NAME=""
SPECIFIC_COMMAND=""

for arg in "$@"; do
    case $arg in
        --event=*)
            EVENT_FILE="${arg#*=}"
            shift
            ;;
        --hook=*)
            HOOK_NAME="${arg#*=}"
            shift
            ;;
        --command=*)
            SPECIFIC_COMMAND="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 --event=<file> [--hook=<name>|--command=<cmd>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EVENT_FILE" ]]; then
    echo "Error: --event argument is required"
    echo "Usage: $0 --event=<file> [--hook=<name>|--command=<cmd>]"
    exit 1
fi

if [[ ! -f "$EVENT_FILE" ]]; then
    echo "Error: Event file not found: $EVENT_FILE"
    exit 1
fi

# Read event data
EVENT_DATA=$(cat "$EVENT_FILE")

# Get hooks configuration
HOOKS_FILE=".cursor/hooks.json"
if [[ ! -f "$HOOKS_FILE" ]]; then
    echo "Error: Hooks configuration not found: $HOOKS_FILE"
    exit 1
fi

# Function to run a single hook command
run_hook_command() {
    local command="$1"
    local event_data="$2"
    
    echo "═══════════════════════════════════════════════════════════"
    echo "Running: $command"
    echo "───────────────────────────────────────────────────────────"
    
    # Run the command with event data as stdin
    result=$(echo "$event_data" | bash -c "$command" 2>&1) || {
        echo "Command failed with exit code: $?"
        echo "Output: $result"
        return 1
    }
    
    # Display output if any
    if [[ -n "$result" ]]; then
        echo "Output:"
        echo "$result"
    else
        echo "(no output)"
    fi
    
    echo "═══════════════════════════════════════════════════════════"
    echo
}

# If specific command provided, run only that
if [[ -n "$SPECIFIC_COMMAND" ]]; then
    run_hook_command "$SPECIFIC_COMMAND" "$EVENT_DATA"
    exit 0
fi

# If hook name provided, run all commands for that hook
if [[ -n "$HOOK_NAME" ]]; then
    # Get number of commands in the hook
    num_commands=$(jq -r ".hooks.${HOOK_NAME} | length" "$HOOKS_FILE")
    
    if [[ "$num_commands" == "null" ]] || [[ "$num_commands" == "0" ]]; then
        echo "Error: No hooks found for: $HOOK_NAME"
        exit 1
    fi
    
    echo "Testing hook: $HOOK_NAME (${num_commands} commands)"
    echo
    
    # Run each command in sequence
    for ((i=0; i<num_commands; i++)); do
        command=$(jq -r ".hooks.${HOOK_NAME}[$i].command" "$HOOKS_FILE")
        
        if [[ "$command" == "null" ]]; then
            echo "Warning: Skipping empty command at index $i"
            continue
        fi
        
        run_hook_command "$command" "$EVENT_DATA"
    done
    
    echo "✓ All hooks completed"
    exit 0
fi

# If neither provided, show usage
echo "Error: Either --hook or --command must be specified"
echo "Usage: $0 --event=<file> [--hook=<name>|--command=<cmd>]"
echo
echo "Available hooks:"
jq -r '.hooks | keys[]' "$HOOKS_FILE" | sed 's/^/  - /'
exit 1