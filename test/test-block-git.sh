#!/bin/bash

./test/hook-tester.sh --event=test/events/beforeShellExecution/ls.json --hook=beforeShellExecution

# echo '{"command": "ls -la"}' | ./.cursor/hooks/block-git.sh
# echo '{"command": "gh repo view"}' | ./.cursor/hooks/block-git.sh
# echo '{"command": "git status"}' | ./.cursor/hooks/block-git.sh

cat /tmp/hooks.log