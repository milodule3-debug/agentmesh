"""
AgentMesh · test_providers.py
Test all 6 providers with your actual API keys.
Run on your Fedora machine: python test_providers.py

Sandbox note: external API calls blocked in Claude sandbox.
This script is designed to run on YOUR machine.
"""
import os, time
from dotenv import load_dotenv
load_dotenv()

from core.hermes_client import HermesClient, Provider

PROMPT = "Reply with exactly 3 words: AGENTMESH IS RUNNING"

PROVIDERS = [
    (Provider.DEEPSEEK,   "deepseek-chat",                                    "DEEPSEEK_API_KEY"),
    (Provider.GROQ,       "llama-3.1-8b-instant",                             "GROQ_API_KEY"),
    (Provider.OPENROUTER, "nousresearch/hermes-3-llama-3.1-70b",              "OPENROUTER_API_KEY"),
    (Provider.GEMINI,     "gemini-2.0-flash",                                 "GEMINI_API_KEY"),
    (Provider.TOGETHER,   "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",    "TOGETHER_API_KEY"),
    (Provider.ANTHROPIC,  "claude-haiku-4-5-20251001",                        "ANTHROPIC_API_KEY"),  # optional
]

ROLES = {
    Provider.DEEPSEEK:   "Orchestrator planner (R1) + sub-agents (V3)",
    Provider.GROQ:       "Free fallback — fastest response",
    Provider.OPENROUTER: "Hermes 3 — best tool-calling",
    Provider.GEMINI:     "Multimodal + long context tasks",
    Provider.TOGETHER:   "Open source models — Llama 405B",
    Provider.ANTHROPIC:  "Optional Claude for orchestrator",
}

def test_provider(provider, model, env_key):
    key = os.environ.get(env_key, "")
    if not key:
        return None, "key not set"
    try:
        client = HermesClient(provider=provider, api_key=key, model=model,
                              default_max_tokens=30)
        t0 = time.time()
        resp = client.complete(PROMPT, max_tokens=20)
        elapsed = time.time() - t0
        return resp.strip(), f"{elapsed:.1f}s"
    except Exception as e:
        return None, str(e)[:60]

if __name__ == "__main__":
    print("=" * 60)
    print("  AgentMesh — Provider Live Test")
    print("=" * 60)

    results = {}
    for provider, model, env_key in PROVIDERS:
        name = provider.value.upper()
        print(f"\n  Testing {name} ({model[:40]})...")
        resp, info = test_provider(provider, model, env_key)
        ok = resp is not None and len(resp) > 3
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {resp or info}")
        if info and ok:
            print(f"    └ {info}  |  role: {ROLES[provider]}")
        results[name] = ok

    print("\n" + "=" * 60)
    passed = sum(results.values())
    print(f"  {passed}/{len(results)} providers working")
    working = [k for k,v in results.items() if v]
    print(f"  Active: {', '.join(working)}")
    print()
    print("  Recommended AgentMesh config:")
    if results.get("DEEPSEEK"):
        print("    Orchestrator: deepseek-reasoner (R1)")
        print("    Sub-agents:   deepseek-chat (V3)")
    if results.get("GROQ"):
        print("    Free fallback: groq llama-3.1-8b-instant")
    if results.get("OPENROUTER"):
        print("    Tool-heavy tasks: openrouter hermes-3-llama-3.1-70b")
    if results.get("GEMINI"):
        print("    Long context: gemini-2.0-flash")
    print("=" * 60)
