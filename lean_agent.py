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

MAX_STEPS = 8  # hard ceiling so a broken tool loop can't run forever / burn tokens


def run(goal: str, allowed_tools: list[str] | None, provider: Provider, model: str | None) -> str:
    client = HermesClient(provider=provider, model=model)
    registry = SkillRegistry("skills")
    tool_names = allowed_tools or registry.list_all()
    tools = registry.get_openai_tools(tool_names)

    messages = [{"role": "user", "content": goal}]

    for step in range(MAX_STEPS):
        resp = client.chat(messages, system=SYSTEM_PROMPT, tools=tools)

        if not resp.has_tool_call:
            return resp.content

        # DeepSeek (and OpenAI-compatible APIs generally) require every tool
        # call to carry an id, and every tool-role reply to echo it back.
        # hermes_client.py's parsed tool_calls don't include one, so we
        # manufacture stable ids here -- same workaround the old
        # agents/base_agent.py used before it was removed.
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


def main():
    ap = argparse.ArgumentParser(description="Lean AgentMesh: one model, one tool loop.")
    ap.add_argument("goal", help="What you want done.")
    ap.add_argument("--tools", help="Comma-separated allowed tool names (default: all registered).")
    ap.add_argument("--provider", default=os.environ.get("AGENTMESH_PROVIDER", "deepseek"))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    allowed = args.tools.split(",") if args.tools else None
    answer = run(args.goal, allowed, Provider(args.provider), args.model)
    print(answer)


if __name__ == "__main__":
    main()
