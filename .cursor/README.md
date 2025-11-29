# Cursor IDE Integration

The Cursor IDE has many features that greatly improve its ability to both work with your style and imporve performance. T hese features can be read about [online](https://cursor.com/en-US/docs). I will explain how they are being used here. Some of these are cross compatible with other tools like Claude Code.

## `.cursorignore`

Use this to actively disable rules, commands, etc. For example, if you have too many rules you may wish to disable some of them so you're not adding too much context to your model which may result in longer load times, more API usage, and decreased performance. You could obviously delete the file instead, but it may be helpful to keep it around for the future.

## `./rules/`

Context that is applied either (A) automatically, (B) when an opened/referenced file is added with a particular path-pattern, (C) The agent decides to read from the description.

- Root `*.mdc`: files are always applied within the workspace.
- Folders with file extensions `md`, `nix`, etc.: Applied when files with those extensions (or related, e.g. `pyproject.toml` for `py`) are referenced.
- `vcs`: Version control usage and preferences
- `writing`: General writing preferences regardless of medium
- `prog`: General programming preferences
- `domain`: Domain-specific knowledge or preferences (e.g. about game theory)
- `gotem`: Specific preferences and information related to my template `gatlens-opinionated-template`. Ignored by default.
- `devops`: Devops tools and preferences
- `cursor`: Info and preferences about cursor itself (e.g. how to use cursor to write cursor rules)
- `misc`: Self explanatory

_Note: To check whether these are added, check Cursor Settings._

## `./commands/`

A collection of commands accessible using `/<command-name>` in the cursor dialogue. Similar to manually-added cursor rules.


## `./mcp/`

TODO

## `./hooks/`


## Custom

- `artifacts` -- Similar to Claude artifacts, these are temporary files (typically markdown) created and updated by LLMs where the chat interface is not enough. Managed by `rules/use-artifacts.mdc`
- `mem` -- Memory bank for LLMs to update a knowledge base of your repository. Currently not functioning. Managed by `rules/use-memory.mdc`
- `notes` -- Agentic note taking system. This is described in `commands/write/take-notes.md`

## Model Context Protocol (MCP)

GOTem recommends using [ComposeIO](https://mcp.composio.dev/). These make adding tools to your model extremely easy.

https://github.com/hamzafer/cursor-commands