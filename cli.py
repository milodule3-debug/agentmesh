#!/usr/bin/env python3
"""
AgentMesh · cli.py
Click-based CLI for the Hermes backend server.

Usage:
    python cli.py chat --provider deepseek --model deepseek-chat "What is Python?"
    python cli.py workflow run workflow.json
    python cli.py learn
    python cli.py learn recommend
    python cli.py providers
    python cli.py serve
"""

import asyncio
import json
import sys
from pathlib import Path

import click

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))


def _get_server_modules():
    """Import server modules lazily to avoid circular imports."""
    from server import (
        hermes_complete, PROVIDERS, KEYS, load_keys,
        memory, learner,
    )
    return hermes_complete, PROVIDERS, KEYS, load_keys, memory, learner


@click.group()
def cli():
    """AgentMesh CLI — multi-agent orchestration from the terminal."""
    pass


# ── chat ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("message")
@click.option("--provider", "-p", default="deepseek", help="LLM provider name")
@click.option("--model", "-m", default=None, help="Model ID (defaults to provider default)")
@click.option("--system", "-s", default="", help="System prompt")
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", default=1024, type=int)
@click.option("--no-learn", is_flag=True, help="Skip recording to learning memory")
def chat(message, provider, model, system, temperature, max_tokens, no_learn):
    """Send a message to an LLM provider and print the response."""
    _, PROVIDERS, KEYS, load_keys, memory, learner = _get_server_modules()

    # Reload keys from .env
    KEYS.update(load_keys())

    if provider not in PROVIDERS:
        click.echo(f"Unknown provider: {provider}", err=True)
        click.echo(f"Available: {', '.join(PROVIDERS.keys())}", err=True)
        sys.exit(1)

    # Default models per provider
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "deepseek": "deepseek-chat",
        "groq": "llama-3.3-70b-versatile",
        "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "openrouter": "deepseek/deepseek-chat",
        "gemini": "gemini-2.0-flash",
        "mimo": "MiMo",
    }
    if not model:
        model = DEFAULT_MODELS.get(provider, "default")

    click.echo(f"[{provider}/{model}] Thinking...")

    result, latency, tokens = asyncio.run(
        hermes_complete(provider, model, message, system, temperature, max_tokens)
    )

    click.echo(f"\n{result}")
    click.echo(f"\n--- {latency:.0f}ms | {tokens} tokens ---")

    if not no_learn:
        ep_id = learner.record(
            task=message[:80],
            workflow_nodes=["cli", "chat"],
            input_text=message,
            output_text=result,
            provider=provider,
            model=model,
            latency_ms=latency,
            tokens_used=tokens,
            success=True,
        )
        click.echo(f"Episode recorded: {ep_id}")


# ── workflow ──────────────────────────────────────────────────────────────────

@cli.group()
def workflow():
    """Workflow management."""
    pass


@workflow.command("run")
@click.argument("file", type=click.Path(exists=True))
def workflow_run(file):
    """Run a workflow from a JSON file."""
    _, PROVIDERS, KEYS, load_keys, memory, learner = _get_server_modules()
    KEYS.update(load_keys())

    data = json.loads(Path(file).read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    vars_ = data.get("vars", {})

    click.echo(f"Running workflow: {len(nodes)} nodes, {len(edges)} edges")

    # Simple sequential execution (matching server.py workflow endpoint)
    results = {}
    for node in nodes:
        nid = node.get("id", "")
        ntype = node.get("type", "ai")
        config = node.get("config", {})

        if ntype == "ai":
            provider = config.get("provider", "deepseek")
            model = config.get("model", "deepseek-chat")
            prompt = config.get("prompt", "")

            # Variable substitution
            for k, v in vars_.items():
                prompt = prompt.replace(f"{{{{{k}}}}}", str(v))
            for prev_nid, prev_result in results.items():
                prompt = prompt.replace(f"{{{{{prev_nid}.output}}}}", str(prev_result))

            click.echo(f"  [{nid}] {provider}/{model}: {prompt[:60]}...")

            result, latency, tokens = asyncio.run(
                hermes_complete(provider, model, prompt)
            )
            results[nid] = result
            click.echo(f"  [{nid}] Done ({latency:.0f}ms)")

        elif ntype == "input":
            results[nid] = config.get("value", "")
        elif ntype == "output":
            source = config.get("source", "")
            if source in results:
                results[nid] = results[source]

    click.echo("\n--- Results ---")
    for nid, result in results.items():
        preview = str(result)[:200]
        click.echo(f"  [{nid}]: {preview}")


# ── learn ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("subcommand", required=False, default="")
@click.option("--limit", "-n", default=20, type=int)
def learn(subcommand, limit):
    """Show learning summary or get recommendations."""
    _, _, _, _, memory, learner = _get_server_modules()

    if subcommand == "recommend":
        rec = learner.recommend()
        click.echo(f"\nRecommended: {rec.get('provider')}/{rec.get('model')}")
        click.echo(f"Reason: {rec.get('reason', 'N/A')}")
        if rec.get("alternatives"):
            click.echo("Alternatives:")
            for alt in rec["alternatives"]:
                click.echo(f"  - {alt.get('provider')}/{alt.get('model')}: {alt.get('reason', '')}")
    elif subcommand == "lessons":
        lessons = memory.get_lessons_flat()
        if not lessons:
            click.echo("No lessons learned yet.")
        else:
            click.echo(f"\nLessons ({len(lessons)}):")
            for l in lessons:
                click.echo(f"  [{l.get('confidence', 0):.0%}] {l.get('content', '')[:100]}")
    elif subcommand == "episodes":
        eps = memory.get_recent(limit=limit)
        if not eps:
            click.echo("No episodes recorded yet.")
        else:
            click.echo(f"\nRecent episodes ({len(eps)}):")
            for ep in eps:
                status = "OK" if ep.success else "FAIL"
                click.echo(f"  [{status}] {ep.provider}/{ep.model} "
                           f"| {ep.latency_ms:.0f}ms | {ep.tokens_used}tok "
                           f"| {ep.task_description[:60]}")
    else:
        # Default: show summary
        stats = learner.summary()
        click.echo("\n--- Learning Summary ---")
        click.echo(f"  Total runs:     {stats.get('total_runs', 0)}")
        click.echo(f"  Success rate:   {stats.get('success_rate', 'N/A')}")
        click.echo(f"  Avg latency:    {stats.get('avg_latency_ms', 0):.0f}ms")
        click.echo(f"  Lessons:        {len(stats.get('lessons', []))}")

        providers = stats.get("provider_stats", {})
        if providers:
            click.echo(f"\n  Providers:")
            for prov, pstats in providers.items():
                click.echo(f"    {prov}: {pstats.get('runs', 0)} runs, "
                           f"{pstats.get('avg_latency_ms', 0):.0f}ms avg, "
                           f"{pstats.get('success_rate', 0):.0%} success")


# ── providers ─────────────────────────────────────────────────────────────────

@cli.command()
def providers():
    """Compare provider performance from learning history."""
    _, _, _, _, memory, _ = _get_server_modules()

    stats = memory.get_provider_stats()
    if not stats:
        click.echo("No provider data yet. Run some tasks first.")
        return

    click.echo("\n--- Provider Performance ---")
    click.echo(f"  {'Provider':<15} {'Model':<30} {'Runs':>5} {'Success':>8} {'Avg Latency':>12} {'Avg Tokens':>11}")
    click.echo(f"  {'-'*15} {'-'*30} {'-'*5} {'-'*8} {'-'*12} {'-'*11}")

    for ps in stats:
        click.echo(f"  {ps['provider']:<15} {ps['model']:<30} {ps['runs']:>5} "
                   f"{ps['success_rate']:>7.0%} "
                   f"{ps['avg_latency_ms']:>10.0f}ms "
                   f"{ps['avg_tokens']:>10.0f}")


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Auto-reload on changes")
def serve(host, port, reload):
    """Start the FastAPI server."""
    import uvicorn
    click.echo(f"Starting AgentMesh server on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=reload)


# ── keys ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("provider")
@click.argument("key")
def keys(provider, key):
    """Set an API key for a provider."""
    _, PROVIDERS, KEYS, _, _, _ = _get_server_modules()

    if provider not in PROVIDERS:
        click.echo(f"Unknown provider: {provider}", err=True)
        sys.exit(1)

    KEYS[provider] = key

    # Also write to .env
    env_file = Path(__file__).parent / ".env"
    env_key = f"{provider.upper()}_API_KEY"
    lines = []
    if env_file.exists():
        lines = env_file.read_text().splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{env_key}="):
            lines[i] = f"{env_key}={key}"
            found = True
            break
    if not found:
        lines.append(f"{env_key}={key}")
    env_file.write_text("\n".join(lines) + "\n")

    click.echo(f"Key set for {provider}")


if __name__ == "__main__":
    cli()
