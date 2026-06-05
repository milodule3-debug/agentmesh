import json
from orchestrator import Orchestrator, TaskPlan
from core.contracts import ExecutionContract, research_contract, writer_contract


def test_task_plan():
    c1 = research_contract("t1", "query 1")
    c2 = writer_contract("t2", "query 2")
    plan = TaskPlan("test goal", [c1, c2], [["t1"], ["t2"]])
    assert not plan.all_done()
    from core.contracts import ContractResult
    plan.results["t1"] = ContractResult("t1", "agent_research", True, {})
    assert not plan.all_done()
    plan.results["t2"] = ContractResult("t2", "agent_writer", True, {})
    assert plan.all_done()


def test_parse_json_basic():
    orch = Orchestrator.__new__(Orchestrator)
    result = orch._parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_with_markdown():
    orch = Orchestrator.__new__(Orchestrator)
    result = orch._parse_json('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_parse_json_with_think_tags():
    orch = Orchestrator.__new__(Orchestrator)
    text = '<think>reasoning</think>\n{"key": "value"}'
    result = orch._parse_json(text)
    assert result == {"key": "value"}


def test_parse_json_malformed():
    orch = Orchestrator.__new__(Orchestrator)
    result = orch._parse_json("not json at all")
    assert result == {}


def test_parse_json_partial():
    orch = Orchestrator.__new__(Orchestrator)
    result = orch._parse_json('some text {"key": "value"} more text')
    assert result == {"key": "value"}


def test_decompose_fallback(tmp_workspace, mock_client):
    """When the LLM returns bad JSON, decompose falls back to research+write."""
    mock_client._response = "not valid json"
    orch = Orchestrator.__new__(Orchestrator)
    orch.orch_client = mock_client
    orch._agents = {"agent_research": None, "agent_writer": None}
    orch.orch_state = type("S", (), {"register_task": lambda *a: None})()
    orch.workspace = tmp_workspace
    orch.honcho = type("H", (), {
        "get_agent_context": lambda *a: "",
    })()

    from orchestrator import Orchestrator as RealOrch
    plan = RealOrch._decompose(orch, "test goal", "")
    assert len(plan.tasks) == 2  # fallback: research + write
