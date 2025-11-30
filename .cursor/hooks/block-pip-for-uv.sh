#!/bin/bash

# block-py-for-uv.sh - Blocks python calls, opting to use uv.

echo "Hook execution started" >> /tmp/hooks.log

input=$(cat)
echo "Received input: $input" >> /tmp/hooks.log

command=$(echo "$input" | jq -r '.command // empty')
echo "Parsed command: '$command'" >> /tmp/hooks.log

if [[ "$command" =~ pip[[:space:]] ]] || [[ "$command" == "pip" ]]; then
    echo "Pip command detected - blocking: '$command'" >> /tmp/hooks.log
    cat <<'__DENY__'
{
  "continue": true,
  "permission": "deny",
  "userMessage": "Pip command blocked. Use UV instead.",
  "agentMessage": "The git command '$command' has been blocked by a project hook. Instead of using raw pip commands e.g. pip install "<package-name>[<extra-name>]<=2.0.0", use 'uv add "<package-name><=2.0.0" --extra <extra-name>' for a permanent install or 'uv pip install ...' for a temporary install..
}
__DENY__
else
    echo "Non-pip command detected - allowing: '$command'" >> /tmp/hooks.log
    cat <<'__ALLOW__'
{
  "continue": true,
  "permission": "allow"
}
__ALLOW__
fi