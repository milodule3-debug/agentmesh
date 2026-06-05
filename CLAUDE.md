# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentMesh

A multi-agent orchestration framework in Python. An orchestrator decomposes goals into subtasks, dispatches them to specialized AI agents (research, code, writer, file), and recursively learns from outcomes. Two entry points: CLI (`run.py`) and FastAPI backend (`server.py`) with a visual workflow builder.

## Commands

```bash
# Run the CLI harness
python run.py "Research X and write a summary"
python run.py --status          # show memory + skill stats
python run.py --test            # run all smoke tests

# Run the FastAPI server (Hermes backend + workflow UI)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Run individual test files
python test_step1.py
python test_step2_memory.py
python test_step2_honcho.py
python test_providers.py
python test_live.py

# Setup (Fedora)
bash setup_fedora.sh
```

## Architecture

```
run.py / server.py          ← Entry points
orchestrator.py             ← Goal decomposition → task dispatch → aggregation
agents/base_agent.py        ← Execution loop: contract → tool loop → memory → learning
core/
  contracts.py              ← ExecutionContract (typed, bounded task defs with budget/permissions)
  hermes_client.py          ← Multi-provider LLM client (OpenRouter, Groq, Anthropic, DeepSeek, Gemini, Together)
  memory.py                 ← 4-layer SQLite memory (episodic, semantic/lessons, procedural/skill stats, working)
  learner.py                ← RecursiveLearner (per-agent) + HarnessOptimizer (orchestrator-level)
  honcho_bridge.py          ← Honcho peer modeling + dialectic memory (optional, NullBridge fallback)
  skill_registry.py         ← Tool registry with built-in skills (web_search, http_get, read_file, write_file, run_python, list_files)
  state_manager.py          ← File-backed agent state with checkpointing and crash recovery
skills/                     ← JSON skill definitions (auto-discovered)
workspace/                  ← Runtime output: per-agent dirs, memory.db, orchestrator state, evolution log
```

## Key Design Decisions

- **Tight token budgets** per agent contract — research shows same quality at 13x less cost
- **Hard boolean completion gates** instead of verifier agents (verifier agents hurt performance by -8.4)
- **Raw traces never summarized** — Stanford finding: summaries drop accuracy 50% → 34.9%
- **Recursive self-evolution** — agents diagnose own failures, extract lessons, evolve system prompts. Acceptance gate rejects generic lessons
- **File-backed state** — agent working memory on disk, survives truncation/restart/delegation

## Agent Execution Flow

1. Orchestrator calls LLM to decompose goal into 2-4 `ExecutionContract` tasks with parallel groups
2. Tasks dispatched to agents (parallel within group, sequential between groups)
3. Each agent runs a tool loop: LLM decides tool calls → registry executes → results fed back
4. After task: episode stored in SQLite, `RecursiveLearner` may evolve agent's system prompt
5. After full cycle: `HarnessOptimizer` rewrites orchestrator's decomposition strategy

## Provider Configuration

Set in `.env` (copy from `.env.example`). The `AGENTMESH_PROVIDER` env var selects the sub-agent provider. Orchestrator always uses `deepseek-chat` for planning. Supported: `deepseek`, `groq`, `openrouter`, `anthropic`, `gemini`, `together`.

## Server (server.py) — Separate System

`server.py` is a standalone FastAPI app ("Hermes Backend") with its own `RecursiveLearner` that stores episodes in `hermes_memory.json` (not SQLite). It powers the visual workflow builder UI at `/ui` and has 16 provider configs. The CLI harness (`run.py` + `orchestrator.py`) uses the `core/` modules instead.

## Adding Skills

Drop a `.json` file in `skills/` with `{name, description, parameters, required, tags, safe}`. Auto-registered on startup. Attach a Python handler via `registry.register_handler(name, fn)`.
