"""
AgentMesh · test_step1.py
Smoke test for Step 1 core modules.
Run: python test_step1.py
No Ollama needed — tests contracts, state, and registry only.
"""

import json
import tempfile
import os
from pathlib import Path

def test_contracts():
    print("\n── contracts ─────────────────────────────────────────")
    from core.contracts import ExecutionContract, ContractResult, ConditionChecker, research_contract

    # Build a contract
    c = research_contract("task_001", "Hermes multi-agent harness design")
    print(f"  contract: {c.task_id} → {c.agent_id}")
    print(f"  budget:   {c.max_tokens} tokens / {c.max_tool_calls} tool calls")
    print(f"  gates:    {c.completion_conditions}")

    # Save / load
    c.save("workspace")
    c2 = ExecutionContract.load("agent_research", "workspace")
    assert c2.task_id == c.task_id
    print(f"  save/load: OK")

    # Prompt block
    block = c.to_prompt_block()
    assert "task_001" in block
    print(f"  prompt block: OK ({len(block)} chars)")

    # Condition checker
    checker = ConditionChecker()
    result_data = {"summary": "Some summary text", "sources": ["url1"], "error": None}
    passed, failed = checker.check_all(c, result_data)
    print(f"  conditions: passed={passed}, failed={failed}")
    print("  contracts: PASS")

def test_state():
    print("\n── state_manager ─────────────────────────────────────")
    from core.state_manager import FileBackedState, OrchestratorState, AgentStatus

    state = FileBackedState("agent_test", "workspace")
    print(f"  status: {state.status}")

    state.set_running("task_001", total_steps=5)
    state.advance_step("doing research")
    state.inc_tokens(150)
    state.inc_tool_calls()
    state.set_context("last_query", "hermes agents")

    s = state.get()
    assert s["step"] == 1
    assert s["tokens_used"] == 150
    assert s["tool_calls"] == 1
    assert state.get_context("last_query") == "hermes agents"
    print(f"  step={s['step']}, tokens={s['tokens_used']}, tools={s['tool_calls']}")

    # Checkpoint
    state.checkpoint({"progress": "halfway", "found": ["item1", "item2"]})
    cp = state.load_checkpoint()
    assert cp["progress"] == "halfway"
    print(f"  checkpoint: OK ({cp['found']})")

    # Result
    state.write_result({"summary": "done", "sources": ["a", "b"]})
    r = state.read_result()
    assert r["summary"] == "done"
    assert state.is_done
    print(f"  result: OK, status={state.status}")

    # Trace log
    state.trace("tool_called", {"tool": "web_search", "query": "hermes agents"})
    state.trace("error_occurred", {"msg": "timeout"})
    traces = state.get_traces()
    failed = state.get_failed_traces()
    print(f"  traces: {len(traces)} total, {len(failed)} failures")

    # Orchestrator state
    orch = OrchestratorState("workspace")
    orch.set_goal("Build a research report on AgentMesh")
    orch.register_task("task_001", "agent_research")
    orch.register_task("task_002", "agent_writer")
    orch.mark_task_done("task_001")
    print(f"  orchestrator: {orch.summary()}")
    print("  state_manager: PASS")

def test_skill_registry():
    print("\n── skill_registry ────────────────────────────────────")
    from core.skill_registry import SkillRegistry

    reg = SkillRegistry("skills")
    all_skills = reg.list_all()
    print(f"  registered: {all_skills}")

    # Get tools for an agent with limited permissions
    allowed = ["read_file", "web_search"]
    tools = reg.get_for_agent(allowed)
    assert len(tools) == 2
    print(f"  filtered for agent: {[t.name for t in tools]}")

    # OpenAI format
    oa = reg.get_openai_tools(allowed)
    assert oa[0]["type"] == "function"
    print(f"  openai format: OK ({oa[0]['function']['name']})")

    # Execute built-in: write then read
    reg.execute("write_file", {"path": "workspace/_test.txt", "content": "hello agentmesh"})
    result = reg.execute("read_file", {"path": "workspace/_test.txt"})
    assert result["content"] == "hello agentmesh"
    print(f"  write+read: OK")
    Path("workspace/_test.txt").unlink(missing_ok=True)

    # List files
    result = reg.execute("list_files", {"directory": "workspace", "pattern": "*.json"})
    print(f"  list_files: {result['count']} json files found")
    print("  skill_registry: PASS")

def test_hermes_client_offline():
    print("\n── hermes_client ────────────────────────────────────")
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from core.hermes_client import HermesClient, ClientPool, Provider

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    if api_key and api_key != "your_deepseek_key_here":
        client = HermesClient(provider=Provider.DEEPSEEK, api_key=api_key)
        print(f"  provider: DeepSeek / {client.model}")
        try:
            resp = client.complete("Reply with exactly: AGENTMESH_OK", max_tokens=20)
            print(f"  live call: {resp.strip()}")
        except Exception as e:
            print(f"  live call failed: {e}")
    else:
        print("  DEEPSEEK_API_KEY not in .env — skipping live call")

    # Offline: pool wiring check
    if api_key and api_key != "your_deepseek_key_here":
        pool = ClientPool(default_provider=Provider.DEEPSEEK, default_api_key=api_key)
        c1 = pool.get("agent_research")
        c2 = pool.get("agent_orch", model="deepseek-reasoner")
        assert c1.model == "deepseek-chat"
        assert c2.model == "deepseek-reasoner"
        print(f"  pool: research={c1.model}, orch={c2.model}")

    print("  hermes_client: PASS")

if __name__ == "__main__":
    from pathlib import Path
    Path("workspace").mkdir(exist_ok=True)
    Path("skills").mkdir(exist_ok=True)
    print("=" * 52)
    print("  AgentMesh Step 1 — Core Module Tests")
    print("=" * 52)
    try:
        test_contracts()
        test_state()
        test_skill_registry()
        test_hermes_client_offline()
        print("\n" + "=" * 52)
        print("  ALL STEP 1 TESTS PASSED")
        print("  Next: Step 2 — BaseAgent + sub-agents")
        print("=" * 52)
    except Exception as e:
        import traceback
        print(f"\nFAIL: {e}")
        traceback.print_exc()
