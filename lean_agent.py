"""
AgentMesh · lean_agent.py
The simple version. One model, one tool loop, no orchestrator.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.hermes_client import HermesClient, Provider
from core.skill_registry import SkillRegistry

SYSTEM_PROMPT = (
    "You are a direct, efficient assistant. Use tools when you need current "
    "information or need to compute/run something. Otherwise just answer. "
    "Don't narrate your plan -- act, then give the final answer plainly."
)

MAX_STEPS = 8

def run(goal: str, allowed_tools, provider, model) -> str:
    client = HermesClient(provider=provider, model=model)
    registry = SkillRegistry("skills")
    messages = [{"role": "user", "content": goal}]
    return run_turn(client, registry, messages, allowed_tools)

def run_turn(client, registry, messages, allowed_tools) -> str:
    tool_names = allowed_tools or registry.list_all()
    tools = registry.get_openai_tools(tool_names)

    for step in range(MAX_STEPS):
        resp = client.chat(messages, system=SYSTEM_PROMPT, tools=tools)

        if not resp.has_tool_call:
            messages.append({"role": "assistant", "content": resp.content or ""})
            return resp.content

        tool_call_ids = [f"call_{step}_{i}" for i in range(len(resp.tool_calls))]
        messages.append({
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": tool_call_ids[i],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for i, call in enumerate(resp.tool_calls)
            ],
        })
        for call_id, call in zip(tool_call_ids, resp.tool_calls):
            try:
                result = registry.execute(call["name"], call["arguments"])
            except Exception as e:
                result = f"ERROR: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(result),
            })

    return "Stopped after MAX_STEPS without a final answer -- goal likely too broad for the lean path."

def repl(allowed_tools, provider, model) -> None:
    client = None
    registry = SkillRegistry("skills")
    messages = []

    print("mesh -- thin terminal. /tools  /clear  /exit (or Ctrl+D)")
    while True:
        try:
            goal = input("mesh> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not goal:
            continue
        if goal in ("/exit", "/quit"):
            break
        if goal == "/tools":
            print(list_tools())
            continue
        if goal == "/clear":
            messages = []
            print("context cleared")
            continue

        if client is None:
            try:
                client = HermesClient(provider=provider, model=model)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        messages.append({"role": "user", "content": goal})
        try:
            answer = run_turn(client, registry, messages, allowed_tools)
        except Exception as e:
            answer = f"ERROR: {e}"
        print(answer)

def list_tools() -> str:
    registry = SkillRegistry("skills")
    lines = ["Available tools:"]
    for name in sorted(registry.list_all()):
        skill = registry.get(name)
        desc = skill.description.strip().splitlines()[0] if skill and skill.description else ""
        lines.append(f"  {name:<16} {desc}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(
        prog="mesh",
        description="Lean AgentMesh: one model, one tool loop. No orchestrator, no ceremony.",
        epilog=(
            "Examples:\n"
            "  mesh \"what's the current price of bitcoin\"\n"
            "  mesh --tools web_search,run_python \"pull the last 5 HN headlines\"\n"
            "  mesh --provider anthropic --model claude-sonnet-4-6 \"summarize this repo\"\n"
            "  mesh --list-tools\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("goal", nargs="?", help="What you want done.")
    ap.add_argument("--tools", help="Comma-separated allowed tool names (default: all registered).")
    ap.add_argument("--list-tools", action="store_true", help="List available tools and exit.")
    ap.add_argument("--provider", default=os.environ.get("AGENTMESH_PROVIDER", "deepseek"),
                     help="deepseek | openrouter | groq | anthropic | gemini | together (default: deepseek)")
    ap.add_argument("--model", default=None, help="Override the provider's default model.")
    args = ap.parse_args()

    if args.list_tools:
        print(list_tools())
        return

    allowed = args.tools.split(",") if args.tools else None

    if not args.goal:
        repl(allowed, Provider(args.provider), args.model)
        return

    answer = run(args.goal, allowed, Provider(args.provider), args.model)
    print(answer)

if __name__ == "__main__":
    main()
