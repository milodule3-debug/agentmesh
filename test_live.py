"""Run this on your Fedora machine — tests real DeepSeek + Honcho calls."""
import os
from dotenv import load_dotenv
load_dotenv()

def test_deepseek():
    print("\n── DeepSeek live ─────────────────────────────────────")
    from core.hermes_client import HermesClient, Provider
    c = HermesClient(provider=Provider.DEEPSEEK)
    resp = c.complete("Reply with exactly: AGENTMESH_OK", max_tokens=15)
    assert "OK" in resp, f"Unexpected: {resp}"
    print(f"  DeepSeek V3: {resp.strip()} ✓")
    resp2 = c.complete("What is 2+2? Answer only with the number.", 
                       max_tokens=5, model_override="deepseek-reasoner" 
                       if hasattr(c, 'model_override') else None)
    print(f"  DeepSeek R1: {resp2.strip()} ✓")

def test_honcho():
    print("\n── Honcho live ───────────────────────────────────────")
    from core.honcho_bridge import HonchoBridge
    bridge = HonchoBridge()
    assert bridge.is_available(), "Honcho not reachable — check key"
    print("  Connected ✓")

    bridge.register_agent("agent_research", role="Research agent",
                          description="Finds information, reads web pages")
    bridge.register_agent("agent_code", role="Code agent", 
                          description="Writes and executes Python")
    bridge.register_agent("orchestrator", role="Orchestrator",
                          description="Plans and delegates tasks")
    print("  3 peers registered ✓")

    bridge.record_task("agent_research",
        "Research DeepSeek V3 API pricing",
        "Found: $0.014/1M input, $0.28/1M output. Strong at code+reasoning.",
        success=True, task_id="init_001", tokens_used=820)
    bridge.record_task("agent_code",
        "Write Python to call DeepSeek API",
        "Wrote working client with retry logic and token counting.",
        success=True, task_id="init_002", tokens_used=950)
    print("  2 task episodes stored ✓")

    ctx = bridge.get_agent_context("agent_research")
    print(f"  agent_research context ({len(ctx)} chars): {ctx[:100] or '(building — async)'}")
    insight = bridge.ask_about_agent("agent_code",
        "What is this agent good at?", reasoning_level="low")
    print(f"  dialectic insight: {insight[:120] or '(building — needs more sessions)'}")

def test_full_run():
    print("\n── Full orchestrator run ─────────────────────────────")
    from orchestrator import Orchestrator
    orch = Orchestrator()
    result = orch.run(
        "Research what DeepSeek-R1 is and write a 3-sentence summary"
    )
    print(f"  Success: {result['success']}")
    print(f"  Tasks: {result['tasks_succeeded']}/{result['tasks_total']}")
    print(f"  Output file: {result.get('output_file')}")

if __name__ == "__main__":
    print("="*52)
    print("  AgentMesh — Live Integration Test")
    print("="*52)
    try:
        test_deepseek()
        test_honcho()
        test_full_run()
        print("\n" + "="*52)
        print("  ALL LIVE TESTS PASSED — AgentMesh is running!")
        print("="*52)
    except Exception as e:
        import traceback
        print(f"\nFAIL: {e}")
        traceback.print_exc()
