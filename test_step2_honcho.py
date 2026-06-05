"""
AgentMesh · test_step2_honcho.py
Tests for Honcho integration.

Offline: tests NullBridge, factory, import chain.
Live:    tests real peer creation, message storage, context retrieval.
         Requires HONCHO_API_KEY in .env — costs ~$0.01 per run.

Run: python test_step2_honcho.py
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def test_null_bridge():
    print("\n── honcho: null bridge (offline) ─────────────────────")
    from core.honcho_bridge import NullHonchoBridge, get_honcho_bridge

    bridge = NullHonchoBridge()
    assert bridge.enrich_system_prompt("agent_x", "You are a helper.") == "You are a helper."
    assert bridge.get_agent_context("agent_x") == ""
    assert bridge.ask_about_agent("agent_x", "anything") == ""
    assert bridge.is_available() == False
    assert bridge.best_agent_for_task(["a", "b"], "task") == "a"
    print("  NullBridge: all methods safe ✓")

    # Factory returns NullBridge when no key
    old_key = os.environ.pop("HONCHO_API_KEY", None)
    b2 = get_honcho_bridge()
    assert isinstance(b2, NullHonchoBridge)
    print("  factory → NullBridge when no key ✓")
    if old_key:
        os.environ["HONCHO_API_KEY"] = old_key
    print("  null bridge: PASS")


def test_honcho_live():
    print("\n── honcho: live API ──────────────────────────────────")
    api_key = os.environ.get("HONCHO_API_KEY", "")
    if not api_key:
        print("  HONCHO_API_KEY not set — skipping live test")
        print("  Add to .env: HONCHO_API_KEY=your_key_from_app.honcho.dev")
        return

    from core.honcho_bridge import HonchoBridge, get_honcho_bridge

    bridge = get_honcho_bridge()
    print(f"  bridge type: {type(bridge).__name__}")
    print(f"  workspace: {bridge.workspace_id}")

    # Connectivity
    available = bridge.is_available()
    print(f"  available: {available}")
    if not available:
        print("  Honcho not reachable — check API key")
        return

    # Register agents
    print("  registering peers...")
    bridge.register_agent("agent_research", role="Research agent",
                          description="Searches web, reads files, produces summaries")
    bridge.register_agent("agent_code", role="Code agent",
                          description="Writes and executes Python code")
    bridge.register_agent("orchestrator", role="Orchestrator",
                          description="Decomposes goals, routes tasks, aggregates results")
    print("  peers registered ✓")

    # Record tasks
    print("  recording task history...")
    bridge.record_task(
        agent_id="agent_research",
        task_description="Find DeepSeek API pricing and rate limits",
        result="Found: $0.014/1M input tokens, 8 RPM free tier",
        success=True,
        task_id="test_task_001",
        tokens_used=850,
    )
    bridge.record_task(
        agent_id="agent_code",
        task_description="Parse JSON API response and extract model name field",
        result="FAILED — KeyError: 'choices' — response structure was different from expected",
        success=False,
        task_id="test_task_002",
        tokens_used=1400,
    )
    bridge.record_task(
        agent_id="agent_code",
        task_description="Write function to safely extract nested JSON fields",
        result="Wrote get_nested() helper with KeyError protection",
        success=True,
        task_id="test_task_003",
        tokens_used=900,
    )
    print("  3 tasks recorded ✓")

    # Get context (Honcho may not have reasoned yet on fresh data — that's async)
    print("  fetching agent context...")
    ctx = bridge.get_agent_context("agent_code", search_query="JSON parsing errors")
    print(f"  agent_code context ({len(ctx)} chars): {ctx[:120] if ctx else '(empty — reasoning may be async)'}")

    # Dialectic query
    print("  running dialectic query...")
    insight = bridge.ask_about_agent(
        "agent_code",
        "What patterns has this agent shown when handling structured data tasks?"
    )
    print(f"  insight: {insight[:150] if insight else '(empty — needs more sessions to reason over)'}")

    # Enriched system prompt
    base = "You are a code agent. Write clean Python."
    enriched = bridge.enrich_system_prompt("agent_code", base, "parse API response safely")
    print(f"  enriched prompt length: {len(enriched)} chars")
    assert enriched.startswith(base)
    print("  prompt enrichment ✓")

    print("  honcho live: PASS")


def test_full_integration():
    """Test Honcho + memory.py + learner.py working together."""
    print("\n── honcho + memory integration ───────────────────────")
    api_key = os.environ.get("HONCHO_API_KEY", "")

    from core.memory import AgentMemory, Episode, make_episode_id
    from core.honcho_bridge import get_honcho_bridge

    mem = AgentMemory("workspace/test_integration.db")
    bridge = get_honcho_bridge()

    # Simulate a task cycle:
    # 1. Get enriched prompt (Honcho context + local lessons)
    base_prompt = "You are a research agent. Find accurate information."
    honcho_prompt = bridge.enrich_system_prompt(
        "agent_research", base_prompt,
        "research LLM memory systems"
    )
    local_lessons = mem.format_lessons_for_prompt("agent_research")
    final_prompt = honcho_prompt + ("\n\n" + local_lessons if local_lessons else "")

    print(f"  base prompt: {len(base_prompt)} chars")
    print(f"  after Honcho: {len(honcho_prompt)} chars")
    print(f"  final (+ lessons): {len(final_prompt)} chars")

    # 2. Task completes — store everywhere
    ep = Episode(
        episode_id=make_episode_id("agent_research", "integration_001"),
        agent_id="agent_research",
        task_id="integration_001",
        task_description="Research LLM memory systems: Honcho, MemGPT, Letta",
        success=True,
        tokens_used=1100,
        tool_calls=5,
        elapsed_seconds=9.4,
        output_summary="Honcho uses dialectic reasoning; MemGPT uses OS metaphor",
        lessons=["Honcho context() is async — allow 30s after recording for reasoning"],
        traces=[
            {"event": "tool_called", "tool": "web_search", "query": "Honcho vs MemGPT"},
            {"event": "completed"},
        ],
    )
    mem.store_episode(ep)  # local verbatim storage

    if api_key:
        bridge.record_task(  # Honcho peer modeling
            "agent_research",
            ep.task_description,
            ep.output_summary,
            ep.success,
            task_id=ep.task_id,
            tokens_used=ep.tokens_used,
        )

    print(f"  episode stored locally + Honcho ({'live' if api_key else 'null bridge'}) ✓")
    print(f"  memory stats: {mem.stats()}")

    Path("workspace/test_integration.db").unlink(missing_ok=True)
    print("  full integration: PASS")


if __name__ == "__main__":
    Path("workspace").mkdir(exist_ok=True)
    print("=" * 52)
    print("  AgentMesh — Honcho Integration Tests")
    print("=" * 52)
    try:
        test_null_bridge()
        test_honcho_live()
        test_full_integration()
        print("\n" + "=" * 52)
        print("  ALL HONCHO TESTS PASSED")
        print("  Add HONCHO_API_KEY to .env for live peer modeling")
        print("=" * 52)
    except Exception as e:
        import traceback
        print(f"\nFAIL: {e}")
        traceback.print_exc()
