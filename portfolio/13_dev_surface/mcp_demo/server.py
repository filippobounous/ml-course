"""Minimal MCP server exposing a calculator tool.

Run it directly:

    python portfolio/13_dev_surface/mcp_demo/server.py

Wire into Claude Code by adding to `~/.config/claude-code/mcp.json` (or the
platform-equivalent config path):

    {
      "mcpServers": {
        "calculator": {
          "command": "python",
          "args": ["/abs/path/to/portfolio/13_dev_surface/mcp_demo/server.py"]
        }
      }
    }

After editing, restart Claude Code; it will auto-discover the `add` and
`multiply` tools.

Requires `pip install mcp`.
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover - environment guard
    raise ImportError(
        "The `mcp` package is not installed. Run `pip install mcp` to use the MCP demo."
    ) from e


mcp = FastMCP("calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return a * b


@mcp.tool()
def compound_interest(principal: float, rate: float, years: float) -> float:
    """Return the final balance of `principal` at annual `rate` after `years`.

    Compounded continuously: balance = principal * exp(rate * years).
    """
    import math

    return principal * math.exp(rate * years)


def _list_tools() -> list[tuple[str, str]]:
    """Return (name, docstring) for each registered tool — used by tests."""
    return [
        ("add", add.__doc__ or ""),
        ("multiply", multiply.__doc__ or ""),
        ("compound_interest", compound_interest.__doc__ or ""),
    ]


if __name__ == "__main__":
    # Default transport is stdio, which is what Claude Code and Claude
    # Desktop expect.
    mcp.run()
