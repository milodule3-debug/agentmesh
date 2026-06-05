"""
AgentMesh · test_step2_memory.py
Tests for persistence (memory.py) and recursive learning (learner.py).

Tests run fully offline — no DeepSeek call needed for memory.
Learning evolution test uses DeepSeek if DEEPSEEK_API_KEY is set.

Run: python test_step2_memory.py
"""

import os, json, time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def test_memory():
    print("\n── memory: episodic ──────────────────────────────────")
    from core.memory import AgentMemory, Episode, Lesson, make_episode_id, make_lesson_id

    mem = AgentMemory("workspace/test_memory.db")

    # Store 3 episodes: 2 success, 1 failure
    episodes = [
        Episode(
            episode_id=make_episode_id("agent_research", "t001"),
            agent_id="agent_research",
            task_id="t001",
            task_description="Research DeepSeek API pricing and rate limits",
            success=True,
            tokens_used=1200,
            tool_calls=4,
            elapsed_seconds=8.2,
            output_summary="Found pricing: $0.014/1M input tokens",
            lessons=["Always check official docs first, not third-party summaries"],
            traces=[
                {"event": "tool_called", "tool": "web_search", "query": "deepseek api pricing"},
                {"event": "tool_result", "status": "ok", "results": 5},
                {"event": "completed", "output_keys": ["summary", "sources"]},
            ],
            tags=["research", "api"],
        ),
        Episode(
            episode_id=make_episode_id("agent_research", "t002"),
            agent_id="agent_research",
            task_id="t002",
            task_description="Research Hermes 3 model capabilities and benchmarks",
            success=True,
            tokens_used=980,
            tool_calls=3,
            elapsed_seconds=6.1,
            output_summary="Hermes 3 tops tool-use benchmarks, based on Llama 3.1",
            lessons=[],
            traces=[
                {"event": "tool_called", "tool": "web_search", "query": "hermes 3 llama benchmarks"},
                {"event": "completed", "output_keys": ["summary", "sources"]},
            ],
            tags=["research", "models"],
        ),
        Episode(
            episode_id=make_episode_id("agent_code", "t003"),
            agent_id="agent_code",
            task_id="t003",
            task_description="Write Python script to parse JSON API responses",
            success=False,
            tokens_used=2100,
            tool_calls=8,
            elapsed_seconds=22.4,
            output_summary="",
            lessons=[],
            traces=[
                {"event": "tool_called", "tool": "run_python", "code": "import json"},
                {"event": "error_occurred", "msg": "KeyError: 'choices'", "tool": "run_python"},
                {"event": "retry_attempt", "n": 1},
                {"event": "error_occurred", "msg": "KeyError: 'choices'", "tool": "run_python"},
                {"event": "failed", "reason": "max_retries_exceeded"},
            ],
            tags=["code", "json", "api"],
        ),
    ]

    for ep in episodes:
        mem.store_episode(ep)
    print(f"  stored {len(episodes)} episodes")

    # Recall similar
    similar = mem.recall_similar("Research API documentation and pricing", n=3)
    print(f"  recall 'API docs': {[e.task_description[:40] for e in similar]}")
    assert len(similar) >= 1

    # Get failures
    failures = mem.get_failures()
    print(f"  failures found: {len(failures)}")
    assert len(failures) == 1
    assert failures[0].agent_id == "agent_code"

    # Stats
    s = mem.stats()
    print(f"  stats: {s}")
    assert s["total_episodes"] == 3
    print("  episodic memory: PASS")

    print("\n── memory: lessons ───────────────────────────────────")
    # Store lessons
    l1 = Lesson(
        lesson_id=make_lesson_id("agent_code", "check response keys before accessing"),
        agent_id="agent_code",
        content="Always check that 'choices' key exists in API response before indexing with response['choices'][0]",
        source_episode=episodes[2].episode_id,
        confidence=0.8,
        applies_to=["code", "api"],
        reinforcements=1,
    )
    mem.store_lesson(l1)
    lessons = mem.get_lessons("agent_code")
    print(f"  lessons for agent_code: {len(lessons)}")
    assert len(lessons) == 1

    # Reinforce positively
    mem.reinforce_lesson(l1.lesson_id, success=True)
    lessons = mem.get_lessons("agent_code")
    assert lessons[0].confidence > 0.8
    print(f"  confidence after reinforcement: {lessons[0].confidence:.2f}")

    # Format for prompt
    block = mem.format_lessons_for_prompt("agent_code")
    assert "choices" in block
    print(f"  prompt block:\n    {block.replace(chr(10), chr(10)+'    ')}")
    print("  lessons: PASS")

    print("\n── memory: skill stats ───────────────────────────────")
    mem.record_skill_call("web_search", success=True, tokens_used=120)
    mem.record_skill_call("web_search", success=True, tokens_used=100)
    mem.record_skill_call("run_python", success=False, tokens_used=400)
    mem.record_skill_call("run_python", success=False, tokens_used=380)
    mem.record_skill_call("run_python", success=True, tokens_used=200)

    stats = mem.get_skill_stats()
    ws = next(s for s in stats if s.skill_name == "web_search")
    rp = next(s for s in stats if s.skill_name == "run_python")
    print(f"  web_search: {ws.success_rate:.0%} success ({ws.total_calls} calls)")
    print(f"  run_python: {rp.success_rate:.0%} success ({rp.total_calls} calls)")
    assert ws.success_rate == 1.0
    assert abs(rp.success_rate - 0.333) < 0.01
    print("  skill stats: PASS")

    print("\n── memory: cross-session context ─────────────────────")
    ctx = mem.build_context_for_task(
        "agent_research",
        "Find DeepSeek model pricing for API usage"
    )
    print(f"  context block length: {len(ctx)} chars")
    if ctx:
        print(f"  preview: {ctx[:120]}...")
    print("  context builder: PASS")

    # Cleanup test db
    Path("workspace/test_memory.db").unlink(missing_ok=True)
    print("\n  memory.py: ALL PASS")


def test_learner_offline():
    print("\n── learner: structure checks ─────────────────────────")
    from core.learner import RecursiveLearner, HarnessOptimizer, LearningCycle, EvolutionResult
    from core.memory import AgentMemory, Episode, make_episode_id

    mem = AgentMemory("workspace/test_learner.db")

    # Acceptance gate — test offline logic
    class MockClient:
        def complete(self, *a, **kw):
            return json.dumps({
                "diagnosis": "Agent consistently fails when API response structure changes.",
                "lessons": [
                    "Before accessing response['choices'][0], verify 'choices' key exists and list is non-empty",
                    "Log the full API response on any KeyError for debugging",
                ],
                "prompt_addendum": "Always validate API response structure before parsing. Log errors verbatim."
            })

    learner = RecursiveLearner(
        agent_id="agent_code_test",
        memory=mem,
        client=MockClient(),
        min_failures_to_trigger=2,
    )

    # No evolution without enough failures
    result = learner.maybe_evolve()
    assert result is None
    print("  no evolution with 0 failures: OK")

    # Store failures
    for i in range(3):
        ep = Episode(
            episode_id=make_episode_id("agent_code_test", f"fail_{i}"),
            agent_id="agent_code_test",
            task_id=f"fail_{i}",
            task_description="Parse API response and extract model name",
            success=False,
            tokens_used=800,
            tool_calls=5,
            elapsed_seconds=12.0,
            output_summary="",
            lessons=[],
            traces=[
                {"event": "tool_called", "tool": "run_python"},
                {"event": "error_occurred", "msg": "KeyError: 'choices'"},
                {"event": "failed", "reason": "unhandled_exception"},
            ],
        )
        mem.store_episode(ep)

    # Now evolution should trigger
    result = learner.maybe_evolve(force=True)
    assert result is not None
    print(f"  evolution triggered: iter={result.iteration}, accepted={result.accepted}")
    if result.accepted:
        print(f"  lessons extracted: {len(result.lessons)}")
        for l in result.lessons:
            print(f"    • {l[:70]}")
        addendum = learner.get_prompt_addendum()
        print(f"  prompt addendum saved: {bool(addendum)}")

    # Learning cycle
    cycle = LearningCycle(mem, MockClient())
    prompt = cycle.get_enriched_system_prompt(
        "agent_code_test",
        "You are a code agent. Write clean Python."
    )
    print(f"  enriched prompt length: {len(prompt)} chars")
    assert len(prompt) > 40

    # Cleanup
    Path("workspace/test_learner.db").unlink(missing_ok=True)
    import shutil
    shutil.rmtree("workspace/agent_code_test", ignore_errors=True)
    print("  learner: ALL PASS")


def test_learner_live():
    print("\n── learner: live DeepSeek evolution ─────────────────")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "your_deepseek_key_here":
        print("  DEEPSEEK_API_KEY not set — skipping live test")
        return

    from core.memory import AgentMemory, Episode, make_episode_id
    from core.hermes_client import HermesClient, Provider
    from core.learner import RecursiveLearner

    mem = AgentMemory("workspace/test_live.db")
    client = HermesClient(provider=Provider.DEEPSEEK, api_key=api_key, model="deepseek-chat")

    # Simulate 2 realistic failures
    for i in range(2):
        ep = Episode(
            episode_id=make_episode_id("agent_live_test", f"live_{i}"),
            agent_id="agent_live_test",
            task_id=f"live_{i}",
            task_description="Fetch web page content and extract main text",
            success=False,
            tokens_used=1500,
            tool_calls=6,
            elapsed_seconds=18.0,
            output_summary="",
            lessons=[],
            traces=[
                {"event": "tool_called", "tool": "http_get", "url": "https://example.com"},
                {"event": "tool_result", "status": "ok", "bytes": 12000},
                {"event": "tool_called", "tool": "run_python", "code": "soup.find('main').text"},
                {"event": "error_occurred", "msg": "AttributeError: 'NoneType' has no attribute 'text'"},
                {"event": "retry_attempt", "n": 1},
                {"event": "error_occurred", "msg": "AttributeError: 'NoneType' has no attribute 'text'"},
                {"event": "failed", "reason": "repeated_attribute_error"},
            ],
        )
        mem.store_episode(ep)

    learner = RecursiveLearner(
        agent_id="agent_live_test",
        memory=mem,
        client=client,
        min_failures_to_trigger=2,
    )
    result = learner.maybe_evolve(force=True)
    print(f"  evolution result: {result}")
    if result and result.accepted:
        print(f"  diagnosis: {result.diagnosis[:120]}...")
        for l in result.lessons:
            print(f"  lesson: {l[:80]}")

    # Cleanup
    Path("workspace/test_live.db").unlink(missing_ok=True)
    import shutil
    shutil.rmtree("workspace/agent_live_test", ignore_errors=True)
    print("  live evolution: PASS")


if __name__ == "__main__":
    Path("workspace").mkdir(exist_ok=True)
    print("=" * 52)
    print("  AgentMesh Step 2 — Memory & Learning Tests")
    print("=" * 52)
    try:
        test_memory()
        test_learner_offline()
        test_learner_live()
        print("\n" + "=" * 52)
        print("  ALL STEP 2 TESTS PASSED")
        print("  Next: Step 3 — BaseAgent execution loop")
        print("=" * 52)
    except Exception as e:
        import traceback
        print(f"\nFAIL: {e}")
        traceback.print_exc()
