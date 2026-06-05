import json
from core.contracts import ExecutionContract, ContractResult
from core.memory import AgentMemory, Episode
from core.state_manager import FileBackedState
from core.skill_registry import SkillRegistry
from core.learner import LearningCycle
from core.hermes_client import HermesResponse
from agents.base_agent import BaseAgent
from tests.conftest import MockClient


class TestAgent(BaseAgent):
    agent_id = "agent_test"
    system_prompt = "You are a test agent."


def _make_contract(tmp_workspace, max_tool_calls=3):
    return ExecutionContract(
        task_id="t1", agent_id="agent_test", description="test task",
        max_tokens=5000, max_tool_calls=max_tool_calls,
        allowed_tools=["read_file"],
        completion_conditions=["no_error_flag"],
        output_path=f"{tmp_workspace}/agent_test/result.json",
        required_output_keys=[],
    )


def test_agent_run_no_tools(tmp_workspace, memory):
    """Agent that returns final answer immediately (no tool calls)."""
    mock = MockClient(response='{"result": "done"}')
    client = mock

    # Mock call_with_tools to return a response with no tool calls
    def fake_call_with_tools(messages, tools, system="", max_tokens=None):
        return HermesResponse(
            content='{"result": "done"}', tool_calls=[],
            tokens_in=10, tokens_out=20, model="test",
            provider="test", elapsed_ms=100, raw={},
        )
    client.call_with_tools = fake_call_with_tools

    registry = SkillRegistry(f"{tmp_workspace}/skills")
    cycle = LearningCycle(memory, client)
    honcho = type("H", (), {
        "register_agent": lambda *a, **kw: None,
        "start_session": lambda *a, **kw: None,
        "end_session": lambda *a, **kw: None,
        "record_task": lambda *a, **kw: None,
        "record_tool_use": lambda *a, **kw: None,
        "enrich_system_prompt": lambda self, aid, bp, td="": bp,
    })()

    agent = TestAgent(client=client, registry=registry, memory=memory,
                      learning_cycle=cycle, honcho=honcho, workspace=tmp_workspace)
    contract = _make_contract(tmp_workspace)
    result = agent.run(contract)
    assert result.success
    assert result.tokens_used > 0


def test_agent_run_budget_exhaustion(tmp_workspace, memory):
    """Agent that hits token budget ceiling."""
    call_count = [0]
    def fake_call_with_tools(messages, tools, system="", max_tokens=None):
        call_count[0] += 1
        if call_count[0] > 1:
            # Second call: simulate tool call that gets budget exceeded
            return HermesResponse(
                content="", tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}],
                tokens_in=10, tokens_out=20, model="test",
                provider="test", elapsed_ms=100, raw={},
            )
        return HermesResponse(
            content="", tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}],
            tokens_in=10, tokens_out=20, model="test",
            provider="test", elapsed_ms=100, raw={},
        )

    client = MockClient()
    client.call_with_tools = fake_call_with_tools

    registry = SkillRegistry(f"{tmp_workspace}/skills")
    cycle = LearningCycle(memory, client)
    honcho = type("H", (), {
        "register_agent": lambda *a, **kw: None,
        "start_session": lambda *a, **kw: None,
        "end_session": lambda *a, **kw: None,
        "record_task": lambda *a, **kw: None,
        "record_tool_use": lambda *a, **kw: None,
        "enrich_system_prompt": lambda self, aid, bp, td="": bp,
    })()

    agent = TestAgent(client=client, registry=registry, memory=memory,
                      learning_cycle=cycle, honcho=honcho, workspace=tmp_workspace)
    contract = _make_contract(tmp_workspace, max_tool_calls=1)
    result = agent.run(contract)
    # Should complete (either success or failure, but not crash)
    assert result is not None


def test_agent_stores_episode(tmp_workspace, memory):
    mock = MockClient()
    def fake_call_with_tools(messages, tools, system="", max_tokens=None):
        return HermesResponse(
            content='{"done": true}', tool_calls=[],
            tokens_in=10, tokens_out=20, model="test",
            provider="test", elapsed_ms=100, raw={},
        )
    mock.call_with_tools = fake_call_with_tools

    registry = SkillRegistry(f"{tmp_workspace}/skills")
    cycle = LearningCycle(memory, mock)
    honcho = type("H", (), {
        "register_agent": lambda *a, **kw: None,
        "start_session": lambda *a, **kw: None,
        "end_session": lambda *a, **kw: None,
        "record_task": lambda *a, **kw: None,
        "record_tool_use": lambda *a, **kw: None,
        "enrich_system_prompt": lambda self, aid, bp, td="": bp,
    })()

    agent = TestAgent(client=mock, registry=registry, memory=memory,
                      learning_cycle=cycle, honcho=honcho, workspace=tmp_workspace)
    contract = _make_contract(tmp_workspace)
    agent.run(contract)
    assert memory.episode_count() == 1
