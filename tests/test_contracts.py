from core.contracts import (
    ExecutionContract, ContractResult, ConditionChecker,
    research_contract, code_contract, writer_contract, data_contract, summary_contract,
)
from pathlib import Path


def test_contract_creation():
    c = ExecutionContract(task_id="t1", agent_id="agent_research", description="test")
    assert c.task_id == "t1"
    assert c.max_tokens == 4000
    assert c.completion_conditions == []


def test_contract_to_prompt_block():
    c = research_contract("t001", "test query")
    block = c.to_prompt_block()
    assert "t001" in block
    assert "agent_research" in block
    assert "MAX_TOKENS" in block


def test_contract_save_load(tmp_workspace):
    c = research_contract("t001", "test", tmp_workspace)
    c.save(tmp_workspace)
    loaded = ExecutionContract.load("agent_research", tmp_workspace)
    assert loaded.task_id == "t001"
    assert loaded.agent_id == "agent_research"


def test_contract_result_save(tmp_workspace):
    r = ContractResult(task_id="t1", agent_id="agent_research", success=True, output={"key": "val"})
    r.save(tmp_workspace)
    assert Path(f"{tmp_workspace}/agent_research/result.json").exists()


def test_condition_checker():
    checker = ConditionChecker()
    c = research_contract("t1", "test")
    data = {"summary": "test summary", "sources": ["http://example.com"]}
    passed, failed = checker.check_all(c, data)
    assert passed


def test_condition_checker_no_error():
    checker = ConditionChecker()
    c = ExecutionContract(task_id="t1", agent_id="a", description="d",
                          completion_conditions=["no_error_flag"])
    assert checker.check_all(c, {})[0]
    assert not checker.check_all(c, {"error": "fail"})[0]


def test_condition_checker_required_keys():
    checker = ConditionChecker()
    c = ExecutionContract(task_id="t1", agent_id="a", description="d",
                          completion_conditions=["required_keys_present"],
                          required_output_keys=["summary", "sources"])
    assert checker.check_all(c, {"summary": "s", "sources": []})[0]
    assert not checker.check_all(c, {"summary": "s"})[0]


def test_condition_checker_key_exists():
    checker = ConditionChecker()
    c = ExecutionContract(task_id="t1", agent_id="a", description="d",
                          completion_conditions=["key_exists:result"])
    assert checker.check_all(c, {"result": 42})[0]
    assert not checker.check_all(c, {"other": 42})[0]


def test_condition_checker_unknown():
    checker = ConditionChecker()
    c = ExecutionContract(task_id="t1", agent_id="a", description="d",
                          completion_conditions=["nonexistent_condition"])
    assert checker.check_all(c, {})[0]  # unknown conditions pass


def test_contract_factories():
    r = research_contract("t1", "query")
    assert r.agent_id == "agent_research"
    co = code_contract("t2", "spec")
    assert co.agent_id == "agent_code"
    w = writer_contract("t3", "topic")
    assert w.agent_id == "agent_writer"
    d = data_contract("t4", "spec")
    assert d.agent_id == "agent_data"
    s = summary_contract("t5", "topic")
    assert s.agent_id == "agent_summary"
