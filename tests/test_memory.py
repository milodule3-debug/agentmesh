import json
from core.memory import AgentMemory, Episode, Lesson, SkillStat, make_episode_id, make_lesson_id


def _make_episode(agent_id="agent_research", task_id="t1", success=True, **kwargs):
    defaults = dict(
        episode_id=make_episode_id(agent_id, task_id),
        agent_id=agent_id, task_id=task_id, task_description=f"task {task_id}",
        success=success, tokens_used=100, tool_calls=2, elapsed_seconds=1.5,
        output_summary="ok", lessons=[], traces=[], tags=[],
        provider="deepseek", model="deepseek-chat", latency_ms=500.0,
        workflow_nodes=["ai"], input_text="hello", rating=None,
    )
    defaults.update(kwargs)
    return Episode(**defaults)


def test_store_and_recall(memory):
    ep = _make_episode()
    memory.store_episode(ep)
    assert memory.episode_count() == 1
    similar = memory.recall_similar("task t1", agent_id="agent_research")
    assert len(similar) == 1


def test_get_failures(memory):
    memory.store_episode(_make_episode(success=True))
    memory.store_episode(_make_episode(agent_id="a2", task_id="t2", success=False))
    failures = memory.get_failures()
    assert len(failures) == 1
    assert not failures[0].success


def test_get_recent(memory):
    for i in range(5):
        memory.store_episode(_make_episode(task_id=f"t{i}"))
    recent = memory.get_recent(limit=3)
    assert len(recent) == 3


def test_episode_count_filter(memory):
    memory.store_episode(_make_episode(agent_id="a1"))
    memory.store_episode(_make_episode(agent_id="a2"))
    assert memory.episode_count() == 2
    assert memory.episode_count("a1") == 1


def test_store_and_get_lessons(memory):
    lesson = Lesson(lesson_id="l1", agent_id="agent_research",
                    content="Always verify sources", source_episode="ep1",
                    confidence=0.8, applies_to=[], reinforcements=1)
    memory.store_lesson(lesson)
    lessons = memory.get_lessons("agent_research")
    assert len(lessons) == 1
    assert lessons[0].content == "Always verify sources"


def test_reinforce_lesson(memory):
    lesson = Lesson(lesson_id="l1", agent_id="a", content="test",
                    source_episode="", confidence=0.5)
    memory.store_lesson(lesson)
    memory.reinforce_lesson("l1", success=True)
    lessons = memory.get_lessons("a", min_confidence=0.0)
    assert lessons[0].confidence > 0.5


def test_reinforce_lesson_prune(memory):
    lesson = Lesson(lesson_id="l1", agent_id="a", content="test",
                    source_episode="", confidence=0.25)
    memory.store_lesson(lesson)
    memory.reinforce_lesson("l1", success=False)  # -0.15 -> 0.1 -> pruned
    lessons = memory.get_lessons("a", min_confidence=0.0)
    assert len(lessons) == 0


def test_format_lessons_for_prompt(memory):
    assert memory.format_lessons_for_prompt("a") == ""
    memory.store_lesson(Lesson(lesson_id="l1", agent_id="a", content="rule 1", source_episode=""))
    result = memory.format_lessons_for_prompt("a")
    assert "rule 1" in result


def test_record_skill_call(memory):
    memory.record_skill_call("web_search", success=True, tokens_used=50)
    memory.record_skill_call("web_search", success=False, tokens_used=30)
    stats = memory.get_skill_stats()
    assert len(stats) == 1
    assert stats[0].total_calls == 2
    assert stats[0].successes == 1


def test_get_weak_skills(memory):
    for _ in range(6):
        memory.record_skill_call("bad_tool", success=False)
    weak = memory.get_weak_skills(threshold=0.5)
    assert len(weak) == 1
    assert weak[0].skill_name == "bad_tool"


def test_build_context_for_task(memory):
    memory.store_episode(_make_episode(task_description="research AI trends"))
    memory.store_lesson(Lesson(lesson_id="l1", agent_id="agent_research",
                               content="cite sources", source_episode=""))
    ctx = memory.build_context_for_task("agent_research", "research AI")
    assert "cite sources" in ctx


def test_stats(memory):
    memory.store_episode(_make_episode())
    s = memory.stats()
    assert s["total_episodes"] == 1
    assert s["success_rate"] == 1.0


def test_provider_stats(memory):
    memory.store_episode(_make_episode(provider="deepseek", model="deepseek-chat", latency_ms=500))
    memory.store_episode(_make_episode(provider="groq", model="llama", latency_ms=200, task_id="t2"))
    ps = memory.get_provider_stats()
    assert len(ps) == 2


def test_workflow_stats(memory):
    memory.store_episode(_make_episode(workflow_nodes=["trigger", "ai", "output"]))
    ws = memory.get_workflow_stats()
    assert len(ws) == 1
    assert "trigger" in ws[0]["pattern"]


def test_rate_episode(memory):
    ep = _make_episode()
    memory.store_episode(ep)
    assert memory.rate_episode(ep.episode_id, 4)
    eps = memory.get_recent()
    assert eps[0].rating == 4


def test_make_ids_deterministic(monkeypatch):
    import time
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    assert make_episode_id("a", "t") == make_episode_id("a", "t")
    assert make_lesson_id("a", "content") == make_lesson_id("a", "content")
