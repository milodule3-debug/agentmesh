"""
AgentMesh · agents/base_agent.py
BaseAgent — the execution loop every sub-agent inherits.

Wires together: Contract → Honcho context → DeepSeek tool loop →
                FileBackedState → ConditionChecker → Memory → LearningCycle
"""

from __future__ import annotations
import json, time, traceback
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.contracts import ExecutionContract, ContractResult, ConditionChecker
from core.state_manager import FileBackedState, AgentStatus
from core.skill_registry import SkillRegistry, SkillError
from core.hermes_client import HermesClient, Provider
from core.memory import AgentMemory, Episode, make_episode_id
from core.learner import LearningCycle
from core.honcho_bridge import get_honcho_bridge, NullHonchoBridge


class BaseAgent:
    """
    Inherit this for every sub-agent in the mesh.
    Override system_prompt and optionally post_process().
    """

    # Override in subclass
    system_prompt: str = "You are a helpful AI agent."
    agent_id: str = "base_agent"

    def __init__(
        self,
        client: HermesClient,
        registry: SkillRegistry,
        memory: AgentMemory,
        learning_cycle: LearningCycle,
        honcho=None,
        workspace: str = "workspace",
        stream: bool = False,
    ):
        self.client = client
        self.registry = registry
        self.memory = memory
        self.cycle = learning_cycle
        self.honcho = honcho or get_honcho_bridge()
        self.workspace = workspace
        self.state = FileBackedState(self.agent_id, workspace)
        self.checker = ConditionChecker()
        self.stream = stream

        # Register with Honcho on startup
        self.honcho.register_agent(self.agent_id, role=self.__class__.__name__)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, contract: ExecutionContract) -> ContractResult:
        """
        Execute a task defined by an ExecutionContract.
        Returns ContractResult with success/failure + output.
        """
        t0 = time.time()
        contract.save(self.workspace)
        self.state.reset()
        self.state.set_running(contract.task_id)

        # Check for crash recovery
        checkpoint = self.state.load_checkpoint()
        messages = checkpoint.get("messages", []) if checkpoint else []
        step = checkpoint.get("step", 0) if checkpoint else 0
        if checkpoint:
            print(f"[{self.agent_id}] Resuming from checkpoint at step {step}")

        # Build enriched system prompt
        prompt = self._build_system_prompt(contract)
        tools = self.registry.get_openai_tools(contract.allowed_tools)

        self.state.trace("task_started", {
            "task_id": contract.task_id,
            "description": contract.description,
            "budget": {"tokens": contract.max_tokens, "tools": contract.max_tool_calls},
        })

        # Start Honcho session
        self.honcho.start_session(contract.task_id, [self.agent_id])

        # Initial user message
        if not messages:
            messages = [{"role": "user", "content": self._task_message(contract)}]

        output = {}
        error = None
        tool_calls_made = 0
        tokens_used = 0
        ABSOLUTE_TOKEN_CEILING = 50_000  # never burn more than this per task

        try:
            # ── Tool loop ──────────────────────────────────────────────────
            for iteration in range(contract.max_tool_calls + 1):
                if tokens_used >= ABSOLUTE_TOKEN_CEILING:
                    self.state.trace("absolute_ceiling_hit", {"tokens": tokens_used})
                    output = self._parse_output(
                        '{"error": "token ceiling reached", "summary": "Task aborted — token budget exceeded"}',
                        contract
                    )
                    break
                self.state.advance_step(f"iteration {iteration}")

                # Hard stop: if already over 90% budget, force final answer
                remaining = contract.max_tokens - tokens_used
                if remaining < contract.max_tokens * 0.1:
                    messages.append({"role": "user",
                        "content": "BUDGET EXHAUSTED. Stop all tool calls. "
                                   "Write your final JSON answer now using only what you already know."})
                    resp = self.client.chat(messages=messages, system=prompt,
                                           max_tokens=800)
                    output = self._parse_output(resp.content, contract)
                    break

                resp = self.client.call_with_tools(
                    messages=messages,
                    tools=tools,
                    system=prompt,
                    max_tokens=min(remaining, 1200),
                )
                tokens_used += resp.total_tokens
                self.state.inc_tokens(resp.total_tokens)
                self.state.trace("llm_response", {
                    "tokens": resp.total_tokens,
                    "has_tool_call": resp.has_tool_call,
                    "content_preview": resp.content[:80] if resp.content else "",
                })

                # No tool call — model is done
                if not resp.has_tool_call:
                    content = resp.content
                    if self.stream:
                        # Re-stream the final answer for real-time display
                        content = self._stream_final(messages, prompt)
                    messages.append({"role": "assistant", "content": content})
                    output = self._parse_output(content, contract)
                    break

                # Execute tool calls
                # Build tool_call list with stable IDs
                tc_list = resp.tool_calls
                tool_call_msgs = [
                    {"id": f"call_{iteration}_{i}", "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["arguments"])}}
                    for i, tc in enumerate(tc_list)
                ]
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": tool_call_msgs,
                })

                # Execute each tool call — MUST respond to every call_id
                # DeepSeek 400s if any tool_call_id has no matching tool response
                for i, tc in enumerate(tc_list):
                    call_id = f"call_{iteration}_{i}"
                    name = tc["name"]
                    args = tc["arguments"]

                    if tool_calls_made >= contract.max_tool_calls:
                        # Budget exceeded — send empty response to satisfy DeepSeek
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps({"error": "tool_budget_exceeded"}),
                        })
                        self.state.trace("budget_exceeded", {"tool_calls": tool_calls_made})
                        continue

                    # Permission check
                    if name not in contract.allowed_tools:
                        result = {"error": f"Tool '{name}' not permitted by contract"}
                        success = False
                    else:
                        try:
                            result = self.registry.execute(name, args)
                            success = "error" not in result
                        except SkillError as e:
                            result = {"error": str(e)}
                            success = False

                    tool_calls_made += 1
                    self.state.inc_tool_calls()
                    self.memory.record_skill_call(name, success, resp.tokens_out)
                    self.state.trace("tool_executed", {
                        "tool": name, "success": success,
                        "result_preview": str(result)[:100],
                    })
                    self.honcho.record_tool_use(
                        self.agent_id, contract.task_id, name, args, str(result), success
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result),
                    })

                # Checkpoint after each tool round
                self.state.checkpoint({"messages": messages[-10:], "step": iteration})

                # Token budget check
                if tokens_used >= contract.max_tokens * 0.9:
                    self.state.trace("token_budget_near", {"used": tokens_used})
                    # Ask model to wrap up
                    messages.append({"role": "user",
                                     "content": "Token budget almost exhausted. Provide your final answer now in the required JSON format."})

        except Exception as e:
            error = str(e)
            self.state.trace("error_occurred", {"msg": error, "trace": traceback.format_exc()[-300:]})

        # ── Completion gate ────────────────────────────────────────────────
        all_passed, failed_conditions = self.checker.check_all(contract, output)
        success = all_passed and error is None

        elapsed = time.time() - t0
        result = ContractResult(
            task_id=contract.task_id,
            agent_id=self.agent_id,
            success=success,
            output=output,
            tokens_used=tokens_used,
            tool_calls_made=tool_calls_made,
            elapsed_seconds=round(elapsed, 2),
            error=error,
            failed_conditions=failed_conditions,
        )
        result.save(self.workspace)
        self.state.set_status(
            AgentStatus.DONE if success else AgentStatus.FAILED,
            "completed" if success else f"failed: {failed_conditions}"
        )

        # ── Store episode + trigger learning ───────────────────────────────
        episode = Episode(
            episode_id=make_episode_id(self.agent_id, contract.task_id),
            agent_id=self.agent_id,
            task_id=contract.task_id,
            task_description=contract.description,
            success=success,
            tokens_used=tokens_used,
            tool_calls=tool_calls_made,
            elapsed_seconds=elapsed,
            output_summary=str(output)[:300],
            lessons=[],
            traces=self.state.get_traces(),
        )
        evo = self.cycle.after_task(self.agent_id, episode)

        # Store to Honcho
        self.honcho.record_task(
            self.agent_id, contract.description,
            str(output)[:500], success,
            contract.task_id, tokens_used,
        )
        self.honcho.end_session(contract.task_id)

        status = "✓" if success else "✗"
        print(f"[{self.agent_id}] {status} {contract.task_id} "
              f"| {tokens_used}tok | {tool_calls_made} tools | {elapsed:.1f}s"
              + (f" | evolved iter {evo.iteration}" if evo and evo.accepted else ""))

        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self, contract: ExecutionContract) -> str:
        parts = [self.system_prompt]

        # Honcho peer model (who this agent is, learned patterns)
        honcho_ctx = self.honcho.enrich_system_prompt(
            self.agent_id, "", contract.description
        )
        if honcho_ctx:
            parts.append(honcho_ctx)

        # Local lessons (acceptance-gated, from learner.py)
        lessons = self.memory.format_lessons_for_prompt(self.agent_id)
        if lessons:
            parts.append(lessons)

        # Evolved prompt addendum
        addendum = self.cycle.get_learner(self.agent_id).get_prompt_addendum()
        if addendum:
            parts.append(f"EVOLVED BEHAVIOUR:\n{addendum}")

        # Contract block (budget, permissions, required output)
        parts.append(f"CONTRACT:\n{contract.to_prompt_block()}")

        return "\n\n".join(parts)

    def _task_message(self, contract: ExecutionContract) -> str:
        return (
            f"Complete this task: {contract.description}\n\n"
            f"Required output keys: {', '.join(contract.required_output_keys)}\n"
            f"Output file: {contract.output_file()}\n\n"
            f"Respond with a JSON object containing the required keys when done."
        )

    def _stream_final(self, messages: list[dict], system: str) -> str:
        """Stream the final answer token-by-token to stdout."""
        print(f"\n[{self.agent_id}] Streaming response:\n")
        accumulated = []
        for token in self.client.stream(messages, system=system):
            print(token, end="", flush=True)
            accumulated.append(token)
        print()  # newline after stream
        return "".join(accumulated)

    def _parse_output(self, content: str, contract: ExecutionContract) -> dict:
        """Try to extract JSON output from model response."""
        if not content:
            return {}
        text = content.strip()
        # Strip markdown fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if part.startswith("json"):
                    text = part[4:].strip()
                    break
                elif "{" in part:
                    text = part.strip()
                    break
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                # Also write to output_path
                out = contract.output_file()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(data, indent=2))
                return data
        except Exception:
            pass
        return {"content": content, "raw": True}
