"""
AgentMesh · mesh_mcp.py
Exposes mesh as an MCP tool so Aura (and other agents) can call it
for diagnostic tasks without leaving their own context.

Tools exposed:
  mesh_diagnose(goal: str) -> str
    Run mesh in diagnostic mode on a goal. Returns the finding.
  mesh_run(goal: str, tools: str = "") -> str
    Run mesh in normal mode. tools = comma-separated tool names or empty for all.
    Before any file write, returns a proposed action instead of executing;
    call again with an explicit confirmation in the goal to proceed.
  mesh_auto(goal: str, tools: str = "") -> str
    Run mesh in auto mode — executes file writes without confirmation.

Usage (as MCP server):
  python3 mesh_mcp.py

Register with Claude Code:
  claude mcp add mesh -- python3 /mnt/bigdata/aura/projects/agentmesh/mesh_mcp.py

Register with Aura (add to ~/.aura/mcp.json or equivalent):
  {
    "mesh": {
      "command": "python3",
      "args": ["/mnt/bigdata/aura/projects/agentmesh/mesh_mcp.py"]
    }
  }
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# SkillRegistry loads "skills/" relative to cwd; MCP hosts launch us anywhere.
os.chdir(Path(__file__).parent)

from lean_agent import run, READ_ONLY_TOOLS
from core.hermes_client import Provider

# --- MCP server using the simplest possible protocol ---
# FastMCP or mcp library would be cleaner but adds a dep.
# This uses raw JSON-RPC over stdio, same as MCP spec requires.

import json

TOOLS = [
    {
        "name": "mesh_diagnose",
        "description": (
            "Run mesh in diagnostic mode. Finds what is broken in the system, "
            "reads logs and files, reports the cause. Does not modify anything. "
            "Use for: service failures, error investigation, health checks, "
            "log analysis, process inspection."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What to investigate. E.g. 'why is aura-telegram.service failing'"
                }
            },
            "required": ["goal"]
        }
    },
    {
        "name": "mesh_run",
        "description": (
            "Run mesh in normal mode with all tools available. "
            "Use for: web search, file reads, running python snippets, "
            "anything that doesn't need full Aura agent loop. "
            "Will not write files silently: for file writes it returns a "
            "proposed action instead of executing. Call again with the "
            "confirmation in the goal (e.g. 'confirm write to X') to proceed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What you want done."},
                "tools": {"type": "string", "description": "Comma-separated tool names. Empty = all."}
            },
            "required": ["goal"]
        }
    },
    {
        "name": "mesh_auto",
        "description": (
            "Run mesh in auto mode — executes without asking for confirmation. "
            "Use only when you are certain the action is safe. "
            "For file writes or destructive actions, prefer mesh_run which confirms first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What you want done."},
                "tools": {"type": "string", "description": "Comma-separated tool names. Empty = all."}
            },
            "required": ["goal"]
        }
    }
]

def handle_call(name: str, args: dict) -> str:
    provider = Provider(os.environ.get("AGENTMESH_PROVIDER", "deepseek"))
    goal = args.get("goal", "")
    if name == "mesh_diagnose":
        return run(goal, READ_ONLY_TOOLS, provider, None, debug=True)
    elif name in ("mesh_run", "mesh_auto"):
        tools_str = args.get("tools", "")
        allowed = tools_str.split(",") if tools_str else None
        auto = name == "mesh_auto"
        # No interactive stdin over MCP: mesh_run denies writes and returns
        # the proposed action; mesh_auto executes them.
        return run(goal, allowed, provider, None, auto=auto,
                   write_policy="allow" if auto else "deny")
    return f"Unknown tool: {name}"

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        rid = req.get("id")
        method = req.get("method", "")

        if method == "initialize":
            resp = {"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mesh", "version": "1.0.0"}
            }}
        elif method == "tools/list":
            resp = {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
        elif method == "tools/call":
            tool_name = req.get("params", {}).get("name", "")
            tool_args = req.get("params", {}).get("arguments", {})
            try:
                result = handle_call(tool_name, tool_args)
                resp = {"jsonrpc": "2.0", "id": rid, "result": {
                    "content": [{"type": "text", "text": result}]
                }}
            except Exception as e:
                resp = {"jsonrpc": "2.0", "id": rid, "error": {
                    "code": -32000, "message": str(e)
                }}
        elif method == "notifications/initialized":
            continue
        else:
            resp = {"jsonrpc": "2.0", "id": rid, "error": {
                "code": -32601, "message": f"Method not found: {method}"
            }}

        print(json.dumps(resp), flush=True)

if __name__ == "__main__":
    main()
