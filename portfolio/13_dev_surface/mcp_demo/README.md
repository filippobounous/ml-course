# MCP demo — calculator server

A minimal Model Context Protocol server exposing three arithmetic tools. Lets
you see the end-to-end pattern: register a tool, spin up the server, wire it
into an MCP client (Claude Code, Claude Desktop, etc.).

## Install & run

```bash
pip install mcp
python portfolio/13_dev_surface/mcp_demo/server.py
```

The server communicates over **stdio** by default — Claude Code and Claude
Desktop spawn it as a subprocess and speak JSON-RPC on its pipes.

## Wire into Claude Code

Edit your Claude Code MCP config (typical path: `~/.config/claude-code/mcp.json`):

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["/abs/path/to/portfolio/13_dev_surface/mcp_demo/server.py"]
    }
  }
}
```

Restart Claude Code. It will auto-discover the `add`, `multiply`, and
`compound_interest` tools.

## Register in Claude Desktop

Claude Desktop has the same config format at `~/Library/Application\ Support/Claude/claude_desktop_config.json` (macOS). Restart the app; the tools appear under the MCP menu.

## Extend

- Add a new `@mcp.tool()`-decorated function — it becomes available to the
  client after restart.
- For non-trivial resources (files, DB rows) use `@mcp.resource(...)`.
- For remote deployment, replace stdio with HTTP/SSE transport (see the
  `mcp` package's HTTP examples).

## What's it for?

The point of MCP is not the calculator — it's the protocol. Once your
development environment speaks MCP, adding a new capability to Claude Code
is just: write a Python function, decorate it, point the config at the
server. No client code changes.

## Tests

`tests/week_13/test_llm_judge_and_mcp.py` verifies the tool list the server
exposes; running the actual MCP server over stdio requires the `mcp` SDK
(not installed by default in CI).
