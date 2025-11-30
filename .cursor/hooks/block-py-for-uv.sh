#!/bin/bash

# block-py-for-uv.sh - Blocks python calls, opting to use uv.

echo "Hook execution started" >> /tmp/hooks.log

input=$(cat)
echo "Received input: $input" >> /tmp/hooks.log

command=$(echo "$input" | jq -r '.command // empty')
echo "Parsed command: '$command'" >> /tmp/hooks.log

if [[ "$command" =~ python[[:space:]] ]] || [[ "$command" == "python" ]]; then
    echo "Python command detected - blocking: '$command'" >> /tmp/hooks.log
    cat <<'__DENY__'
{
  "continue": true,
  "permission": "deny",
  "userMessage": "Python command blocked. Use UV instead.",
  "agentMessage": "The git command '$command' has been blocked by a project hook. Instead of using raw python commands, use 'uv run <script>'."
}
__DENY__
elif [[ "$command" =~ uv[[:space:]]run[[:space:]] ]] || [[ "$command" == "uv" ]]; then
    echo "UV command detected - asking for permission: '$command'" >> /tmp/hooks.log
    cat <<'__ASK__'
{
  "continue": true,
  "permission": "ask",
  "userMessage": "UV CLI command requires permission: $command",
  "agentMessage": "The command '$command' uses UV. Please review and approve this command if you want to proceed."
}
__ASK__
else
    echo "Non-python/non-uv command detected - allowing: '$command'" >> /tmp/hooks.log
    cat <<'__ALLOW__'
{
  "continue": true,
  "permission": "allow"
}
__ALLOW__
fi