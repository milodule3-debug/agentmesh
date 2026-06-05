"""
AgentMesh · orchestrator.py
Orchestrator — top PC. Decomposes goals → dispatches contracts →
               aggregates results → runs HarnessOptimizer loop.
"""

from __future__ import annotations
import json, time, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.contracts import ExecutionContract, ContractResult, research_contract, code_contract, writer_contract, data_contract, summary_contract
from core.state_manager import OrchestratorState
from core.skill_registry import SkillRegistry
from core.hermes_client import HermesClient, Provider
from core.memory import AgentMemory
from core.learner import LearningCycle, HarnessOptimizer
from core.utils import parse_json_from_llm
from core.honcho_bridge import get_honcho_bridge
from agents.base_agent import BaseAgent


# ── Concrete sub-agents ───────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    agent_id = "agent_research"
    system_prompt = (
        "You are a research agent. Your job is to find accurate, current information. "
        "Use web_search to find sources. Use http_get to fetch page content. "
        "Always cite sources. Return a JSON with 'summary' and 'sources' keys."
    )

class CodeAgent(BaseAgent):
    agent_id = "agent_code"
    system_prompt = (
        "You are a code agent. Write clean, working Python. "
        "Use run_python to test your code before returning it. "
        "Always handle exceptions. Return JSON with 'code' and 'explanation' keys."
    )

class WriterAgent(BaseAgent):
    agent_id = "agent_writer"
    system_prompt = (
        "You are a content writer. Write clear, structured text. "
        "Read any provided research files first. "
        "Return JSON with 'title' and 'content' keys."
    )

class FileAgent(BaseAgent):
    agent_id = "agent_file"
    system_prompt = (
        "You are a file management agent. Read, write, and organise files. "
        "List directories to understand structure before acting. "
        "Return JSON with 'files_modified' and 'summary' keys."
    )

class DataAnalysisAgent(BaseAgent):
    agent_id = "agent_data"
    system_prompt = (
        "You are a data analysis agent. You process datasets, compute statistics, "
        "and identify patterns. Use run_python for computations. Use read_file to load data. "
        "Return JSON with 'analysis', 'statistics', and 'insights' keys."
    )

class SummaryAgent(BaseAgent):
    agent_id = "agent_summary"
    system_prompt = (
        "You are a summarization agent. Condense long content into clear, "
        "structured summaries. Read source files first. Preserve key facts and numbers. "
        "Return JSON with 'summary', 'key_points', and 'word_count' keys."
    )


# ── Task plan ─────────────────────────────────────────────────────────────────

class TaskPlan:
    def __init__(self, goal: str, tasks: list[ExecutionContract], parallel: list[list[str]] = None):
        self.goal = goal
        self.tasks = {t.task_id: t for t in tasks}
        # parallel groups: tasks that can run concurrently
        # e.g. [["t1","t2"], ["t3"]] means t1+t2 parallel, then t3
        self.parallel = parallel or [[t.task_id] for t in tasks]
        self.results: dict[str, ContractResult] = {}

    def all_done(self) -> bool:
        return all(tid in self.results for tid in self.tasks)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Top PC agent. Uses DeepSeek-R1 (reasoner) for planning.
    Sub-agents use DeepSeek-chat (cheaper) for execution.
    """

    WORKSPACE = "workspace"

    def __init__(
        self,
        orchestrator_api_key: str = None,
        agent_api_key: str = None,
        provider: Provider = Provider.DEEPSEEK,
        workspace: str = "workspace",
        stream: bool = False,
    ):
        self.workspace = workspace
        self.stream = stream
        Path(workspace).mkdir(exist_ok=True)

        orch_key = orchestrator_api_key or os.environ.get("DEEPSEEK_API_KEY")
        agent_key = agent_api_key or os.environ.get("DEEPSEEK_API_KEY")

        # Orchestrator planning: use deepseek-chat (V3) — R1 has <think> tags
        # that break JSON parsing. V3 is cheaper and reliable for structured output.
        self.orch_client = HermesClient(
            provider=provider, api_key=orch_key,
            model="deepseek-chat" if provider == Provider.DEEPSEEK else None,
        )
        self.agent_client = HermesClient(
            provider=provider, api_key=agent_key,
            model="deepseek-chat" if provider == Provider.DEEPSEEK else None,
        )

        self.registry = SkillRegistry("skills")
        self.memory = AgentMemory(f"{workspace}/memory.db")
        self.honcho = get_honcho_bridge(routing_client=self.orch_client)
        self.cycle = LearningCycle(self.memory, self.agent_client)
        self.harness = HarnessOptimizer(self.memory, self.orch_client)
        self.orch_state = OrchestratorState(workspace)

        # Build sub-agent roster
        self._agents = self._build_agents()
        self.honcho.register_agent("orchestrator", role="Orchestrator",
                                   description="Decomposes goals, routes tasks")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, goal: str) -> dict:
        """
        Main entry point. Give it a goal, get back aggregated results.

        Example:
            orch = Orchestrator()
            result = orch.run("Research AgentMesh harness patterns and write a summary doc")
        """
        print(f"\n[Orchestrator] Goal: {goal}")
        print("=" * 60)

        self.orch_state.set_goal(goal)

        # 1. Get harness strategy (from previous evolution cycles)
        strategy = self.harness.get_current_strategy()

        # 2. Decompose goal into task plan
        plan = self._decompose(goal, strategy)
        print(f"[Orchestrator] Plan: {len(plan.tasks)} tasks across {len(plan.parallel)} groups")

        # 3. Execute groups (parallel within group, sequential between groups)
        for group_idx, task_group in enumerate(plan.parallel):
            print(f"\n[Orchestrator] Group {group_idx+1}: {task_group}")
            group_contracts = [plan.tasks[tid] for tid in task_group if tid in plan.tasks]

            if len(group_contracts) == 1:
                result = self._run_single(group_contracts[0])
                plan.results[group_contracts[0].task_id] = result
            else:
                results = self._run_parallel(group_contracts)
                plan.results.update(results)

            self.orch_state.mark_task_done(task_group[0])

        # 4. Aggregate final output
        final = self._aggregate(goal, plan)

        # 5. Run harness optimizer (if enough failure data)
        self.harness.optimize(goal)

        print(f"\n[Orchestrator] Done — {sum(1 for r in plan.results.values() if r.success)}/{len(plan.results)} tasks succeeded")
        return final

    # ── Planning ──────────────────────────────────────────────────────────────

    def _decompose(self, goal: str, strategy: str = "") -> TaskPlan:
        """Ask DeepSeek-R1 to break the goal into subtasks."""

        # Get Honcho's read on each agent's strengths
        agent_profiles = ""
        for aid in self._agents:
            ctx = self.honcho.get_agent_context(aid)
            if ctx:
                agent_profiles += f"\n[{aid}]: {ctx[:150]}"

        prompt = f"""Decompose this goal into 2-4 concrete subtasks for a multi-agent system.

GOAL: {goal}

AVAILABLE AGENTS:
- agent_research: web search, document reading, fact-finding
- agent_code:     write/run Python, data processing, file manipulation
- agent_writer:   structured writing, summaries, reports
- agent_file:     file organisation, reading/writing workspace files
- agent_data:     data analysis, statistics, dataset processing
- agent_summary:  condensing long content, key point extraction

{("AGENT PROFILES FROM HONCHO:" + agent_profiles) if agent_profiles else ""}
{("CURRENT HARNESS STRATEGY:" + strategy[:400]) if strategy else ""}

Rules:
- Each task must be assignable to exactly ONE agent
- Tasks that don't depend on each other can be parallel
- Keep tasks narrow and specific (research finding = agent_research)
- Max 4 tasks total
- Research tasks: max_tokens 2000, max_tool_calls 6, use ONLY web_search (not http_get)
- Writer tasks: max_tokens 1500, max_tool_calls 2, use ONLY read_file
- Code tasks: max_tokens 2500, max_tool_calls 8, use run_python + write_file

Respond with JSON only:
{{
  "tasks": [
    {{
      "task_id": "t001",
      "agent_id": "agent_research",
      "description": "specific task description",
      "allowed_tools": ["web_search", "http_get"],
      "required_output_keys": ["summary", "sources"],
      "max_tokens": 2500
    }}
  ],
  "parallel_groups": [["t001","t002"],["t003"]]
}}"""

        try:
            resp = self.orch_client.complete(prompt, max_tokens=1000)
            print(f"[Orchestrator] Plan response ({len(resp)} chars): {resp[:120].strip()}...")
            data = self._parse_json(resp)
            if not data.get("tasks"):
                raise ValueError("No tasks in decompose response")
            print(f"[Orchestrator] Parsed tasks: {len(data.get('tasks', []))}")
            tasks = []
            for t in data.get("tasks", []):
                contract = ExecutionContract(
                    task_id=t["task_id"],
                    agent_id=t["agent_id"],
                    description=t["description"],
                    max_tokens=t.get("max_tokens", 2500),
                    max_tool_calls=t.get("max_tool_calls", 8),
                    allowed_tools=t.get("allowed_tools", ["web_search", "read_file"]),
                    required_output_keys=t.get("required_output_keys", []),
                    completion_conditions=["no_error_flag"],
                    output_path=f"{self.workspace}/{t['agent_id']}/result_{t['task_id']}.json",
                )
                self.orch_state.register_task(t["task_id"], t["agent_id"])
                tasks.append(contract)

            parallel = data.get("parallel_groups", [[t.task_id] for t in tasks])
            return TaskPlan(goal, tasks, parallel)

        except Exception as e:
            print(f"[Orchestrator] Decompose fallback ({type(e).__name__}): {e}")
            # Fallback: research + write
            c1 = research_contract("t001", goal, self.workspace)
            c2 = writer_contract("t002", goal, self.workspace)
            return TaskPlan(goal, [c1, c2], [["t001"], ["t002"]])

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_single(self, contract: ExecutionContract) -> ContractResult:
        agent = self._agents.get(contract.agent_id)
        if not agent:
            return ContractResult(contract.task_id, contract.agent_id, False, {},
                                  error=f"Unknown agent: {contract.agent_id}")
        return agent.run(contract)

    def _run_parallel(self, contracts: list[ExecutionContract]) -> dict[str, ContractResult]:
        results = {}
        with ThreadPoolExecutor(max_workers=len(contracts)) as pool:
            futures = {pool.submit(self._run_single, c): c.task_id for c in contracts}
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    results[tid] = future.result()
                except Exception as e:
                    contract = next(c for c in contracts if c.task_id == tid)
                    results[tid] = ContractResult(tid, contract.agent_id, False, {}, error=str(e))
        return results

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, goal: str, plan: TaskPlan) -> dict:
        outputs = {}
        for tid, result in plan.results.items():
            outputs[tid] = {
                "agent": plan.tasks[tid].agent_id if tid in plan.tasks else "?",
                "success": result.success,
                "output": result.output,
                "tokens": result.tokens_used,
            }

        # Save aggregate to workspace
        agg_path = Path(self.workspace) / "_final_output.json"
        agg_data = {"goal": goal, "tasks": outputs, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        agg_path.write_text(json.dumps(agg_data, indent=2))

        successes = sum(1 for r in plan.results.values() if r.success)
        return {
            "goal": goal,
            "success": successes == len(plan.results),
            "tasks_succeeded": successes,
            "tasks_total": len(plan.results),
            "outputs": outputs,
            "output_file": str(agg_path),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_agents(self) -> dict[str, BaseAgent]:
        shared = dict(
            client=self.agent_client,
            registry=self.registry,
            memory=self.memory,
            learning_cycle=self.cycle,
            honcho=self.honcho,
            workspace=self.workspace,
            stream=self.stream,
        )
        return {
            "agent_research": ResearchAgent(**shared),
            "agent_code":     CodeAgent(**shared),
            "agent_writer":   WriterAgent(**shared),
            "agent_file":     FileAgent(**shared),
            "agent_data":     DataAnalysisAgent(**shared),
            "agent_summary":  SummaryAgent(**shared),
        }

    def _parse_json(self, text: str) -> dict:
        return parse_json_from_llm(text)
