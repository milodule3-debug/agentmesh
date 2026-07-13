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
    "Don't narrate your plan -- act, then give the final answer plainly. "
    "IMPORTANT: Before writing or modifying any file, state what you are "
    "about to do and ask for confirmation. Never write files silently."
)

AUTO_PROMPT = (
    "You are a direct, efficient assistant. Use tools when you need current "
    "information or need to compute/run something. Otherwise just answer. "
    "Don't narrate your plan -- act, then give the final answer plainly."
)

DIAGNOSTIC_PROMPT = (
    "You are a diagnostic agent. Your job is to find what is broken, "
    "not to fix it. Read files, check logs, inspect processes, examine "
    "errors. Report findings clearly: what is wrong, where it is, "
    "and what the likely cause is. Suggest a fix in one sentence but "
    "do not implement it — that is someone else's job. "
    "Be direct and concise. No prose padding."
)

READ_ONLY_TOOLS = ["read_file", "list_files", "run_python", "web_search", "http_get"]

# Tools that modify files. Guarded at code level, not just in the prompt.
WRITE_TOOLS = {"write_file"}

MAX_STEPS = 8

# Running totals across all turns in this process (REPL session / MCP server).
SESSION_TOKENS = {"in": 0, "out": 0}

def pick_prompt(debug: bool, auto: bool) -> str:
    if debug:
        return DIAGNOSTIC_PROMPT
    return AUTO_PROMPT if auto else SYSTEM_PROMPT

def pick_write_policy(debug: bool, auto: bool) -> str:
    if debug:
        return "deny"
    return "allow" if auto else "ask"

def run(goal: str, allowed_tools, provider, model, debug=False, auto=False,
        write_policy=None) -> str:
    client = HermesClient(provider=provider, model=model)
    registry = SkillRegistry("skills")
    messages = [{"role": "user", "content": goal}]
    if debug and not allowed_tools:
        allowed_tools = READ_ONLY_TOOLS
    if write_policy is None:
        write_policy = pick_write_policy(debug, auto)
    return run_turn(client, registry, messages, allowed_tools,
                    pick_prompt(debug, auto), write_policy, debug=debug)

def _guard_write(call, write_policy):
    """Returns None to allow execution, or a result string that replaces it."""
    if call["name"] not in WRITE_TOOLS or write_policy == "allow":
        return None
    proposal = f"{call['name']}({json.dumps(call['arguments'])})"
    if write_policy == "ask":
        print(f"About to execute: {proposal}. Confirm? (y/n) ", end="", file=sys.stderr, flush=True)
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes"):
            return None
        return "User denied the write. Do not retry it; report what you would have done."
    # deny (diagnostic mode / MCP mesh_run): never execute, surface the proposal
    return (f"BLOCKED: writes are not allowed in this mode. Proposed action: {proposal}. "
            "Report this proposal to the caller instead of executing it.")

def run_turn(client, registry, messages, allowed_tools, system_prompt=None,
             write_policy="ask", debug=False) -> str:
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    tool_names = allowed_tools or registry.list_all()
    tools = registry.get_openai_tools(tool_names)
    turn_in = turn_out = 0

    def finish(answer: str) -> str:
        print(f"[tokens] turn: in={turn_in:,} out={turn_out:,} | "
              f"session: in={SESSION_TOKENS['in']:,} out={SESSION_TOKENS['out']:,}",
              file=sys.stderr)
        return answer

    for step in range(MAX_STEPS):
        resp = client.chat(messages, system=system_prompt, tools=tools)
        turn_in += resp.tokens_in
        turn_out += resp.tokens_out
        SESSION_TOKENS["in"] += resp.tokens_in
        SESSION_TOKENS["out"] += resp.tokens_out
        print(f"[tokens] in={resp.tokens_in} out={resp.tokens_out}", file=sys.stderr)

        if not resp.has_tool_call:
            messages.append({"role": "assistant", "content": resp.content or ""})
            return finish(resp.content)

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
            result = _guard_write(call, write_policy)
            if result is None:
                try:
                    result = registry.execute(call["name"], call["arguments"])
                except Exception as e:
                    result = f"ERROR: {e}"
            result_str = str(result)
            if debug:
                print(f"[debug] tool {call['name']} result ~{len(result_str) // 4} tokens",
                      file=sys.stderr)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str,
            })

    return finish("Stopped after MAX_STEPS without a final answer -- goal likely too broad for the lean path.")

def repl(allowed_tools, provider, model, debug=False, auto=False) -> None:
    client = None
    registry = SkillRegistry("skills")
    messages = []

    prompt = pick_prompt(debug, auto)
    if debug and not allowed_tools:
        allowed_tools = READ_ONLY_TOOLS

    if debug:
        mode = "diagnostic — read-only"
    elif auto:
        mode = "auto — no write confirmation"
    else:
        mode = "thin terminal"
    print(f"mesh -- {mode}. /tools  /clear  /exit (or Ctrl+D)")
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
            answer = run_turn(client, registry, messages, allowed_tools, prompt,
                              pick_write_policy(debug, auto), debug=debug)
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
    ap.add_argument("--debug", action="store_true",
                    help="Diagnostic mode: read-only tools, find what is broken.")
    ap.add_argument("--auto", action="store_true",
                    help="Auto mode: execute without asking for confirmation on file writes.")
    args = ap.parse_args()

    if args.list_tools:
        print(list_tools())
        return

    allowed = args.tools.split(",") if args.tools else None

    if not args.goal:
        repl(allowed, Provider(args.provider), args.model, debug=args.debug, auto=args.auto)
        return

    answer = run(args.goal, allowed, Provider(args.provider), args.model, debug=args.debug, auto=args.auto)
    print(answer)

if __name__ == "__main__":
    main()
