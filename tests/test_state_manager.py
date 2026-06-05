from core.state_manager import FileBackedState, OrchestratorState, AgentStatus
import json


def test_initial_state(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    s = state.get()
    assert s["agent_id"] == "agent_test"
    assert s["status"] == "idle"


def test_set_running(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_running("task_1")
    assert state.is_running
    assert state.get()["task_id"] == "task_1"


def test_advance_step(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_running("t1")
    state.advance_step("step 1")
    state.advance_step("step 2")
    assert state.get()["step"] == 2


def test_token_and_tool_tracking(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.inc_tokens(100)
    state.inc_tokens(50)
    state.inc_tool_calls()
    assert state.get()["tokens_used"] == 150
    assert state.get()["tool_calls"] == 1


def test_context(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_context("key1", "value1")
    assert state.get_context("key1") == "value1"
    assert state.get_context("missing", "default") == "default"


def test_checkpoint(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.checkpoint({"step": 5, "data": "test"})
    loaded = state.load_checkpoint()
    assert loaded["step"] == 5
    assert loaded["data"] == "test"
    state.clear_checkpoint()
    assert state.load_checkpoint() is None


def test_result(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_running("t1")
    state.write_result({"output": "done"})
    assert state.is_done
    r = state.read_result()
    assert r["output"] == "done"


def test_trace(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.trace("event_1", {"key": "val"})
    state.trace("event_2")
    traces = state.get_traces()
    assert len(traces) == 2
    assert traces[0]["event"] == "event_1"


def test_get_failed_traces(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.trace("tool_executed", {"success": True})
    state.trace("error_occurred", {"msg": "fail"})
    failed = state.get_failed_traces()
    assert len(failed) == 1


def test_reset(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_running("t1")
    state.inc_tokens(100)
    state.trace("event")
    state.reset()
    assert state.status == AgentStatus.IDLE
    assert state.get()["tokens_used"] == 0
    assert len(state.get_traces()) == 1  # traces preserved


def test_orchestrator_state(tmp_workspace):
    orch = OrchestratorState(tmp_workspace)
    orch.set_goal("test goal")
    orch.register_task("t1", "agent_research")
    orch.register_task("t2", "agent_code")
    orch.mark_task_done("t1")
    s = orch.get()
    assert s["tasks"]["t1"]["status"] == "done"
    assert s["tasks"]["t2"]["status"] == "pending"
    summary = orch.summary()
    assert "1/2" in summary


def test_atomic_write(tmp_workspace):
    state = FileBackedState("agent_test", tmp_workspace)
    state.set_running("t1")
    # Verify no .tmp file left behind
    from pathlib import Path
    tmp_file = Path(tmp_workspace) / "agent_test" / "state.tmp"
    assert not tmp_file.exists()
