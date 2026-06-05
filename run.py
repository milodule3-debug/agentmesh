#!/usr/bin/env python3
"""
AgentMesh · run.py
Entry point.

Usage:
    python run.py "Research DeepSeek pricing and write a summary"
    python run.py --stream "Build a Python JSON parser"   # stream responses
    python run.py --status        # show memory stats
    python run.py --test          # run smoke tests
"""

import sys, os, argparse, json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def cmd_run(goal: str, workspace: str, stream: bool = False):
    from orchestrator import Orchestrator
    from core.hermes_client import Provider

    provider_str = os.environ.get("AGENTMESH_PROVIDER", "deepseek").lower()
    provider = {
        "deepseek":   Provider.DEEPSEEK,
        "groq":       Provider.GROQ,
        "openrouter": Provider.OPENROUTER,
        "anthropic":  Provider.ANTHROPIC,
        "gemini":     Provider.GEMINI,
        "together":   Provider.TOGETHER,
    }.get(provider_str, Provider.DEEPSEEK)

    orch = Orchestrator(provider=provider, workspace=workspace, stream=stream)
    result = orch.run(goal)

    print("\n── Final Output ─────────────────────────────────────")
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, indent=2))
    print(f"\nFull output: {result.get('output_file')}")


def cmd_status(workspace: str):
    from core.memory import AgentMemory
    from core.honcho_bridge import get_honcho_bridge

    mem = AgentMemory(f"{workspace}/memory.db")
    stats = mem.stats()
    print("\n── AgentMesh Status ─────────────────────────────────")
    print(f"  Episodes:       {stats['total_episodes']} (success rate: {stats['success_rate']:.0%})")
    print(f"  Lessons:        {stats['total_lessons']}")
    print(f"  Tracked skills: {stats['tracked_skills']}")

    skill_stats = mem.get_skill_stats()
    if skill_stats:
        print("\n  Skill effectiveness:")
        for s in skill_stats[:8]:
            bar = "█" * int(s.success_rate * 10) + "░" * (10 - int(s.success_rate * 10))
            print(f"    {s.skill_name:<18} {bar} {s.success_rate:.0%} ({s.total_calls} calls)")

    honcho = get_honcho_bridge()
    print(f"\n  Honcho: {'connected' if honcho.is_available() else 'not configured (set HONCHO_API_KEY)'}")


def cmd_test():
    print("Running AgentMesh smoke tests...\n")
    os.system("python test_step1.py")
    os.system("python test_step2_memory.py")
    os.system("python test_step2_honcho.py")


def main():
    parser = argparse.ArgumentParser(description="AgentMesh — Multi-Agent Harness")
    parser.add_argument("goal", nargs="?", help="Goal for the agent mesh to accomplish")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument("--status", action="store_true", help="Show memory + skill stats")
    parser.add_argument("--test", action="store_true", help="Run smoke tests")
    parser.add_argument("--stream", action="store_true", help="Stream agent responses in real-time")
    args = parser.parse_args()

    if args.test:
        cmd_test()
    elif args.status:
        cmd_status(args.workspace)
    elif args.goal:
        cmd_run(args.goal, args.workspace, stream=args.stream)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python run.py "Research LLM memory systems and write a summary"')
        print('  python run.py --status')


if __name__ == "__main__":
    main()
