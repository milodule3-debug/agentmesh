#!/usr/bin/env python3
"""
Ruby Harness Routes — attached to agentmesh server.py
POST /ruby/generate — generate agentic harness project as ZIP
"""
import os
import sys
import json
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# Key detector from Ruby Harness — resolve path to agentic-harness-generator
_HARNESS_DIR = Path("/home/dusanmilosavljevic/agentic-harness-generator")

def _get_detector():
    import sys
    if str(_HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(_HARNESS_DIR))
    from key_detector import detect_provider
    return detect_provider

def detect_provider(key):
    return _get_detector()(key)

def _Generator():
    import sys
    if str(_HARNESS_DIR) not in sys.path:
        sys.path.insert(0, str(_HARNESS_DIR))
    from generator import Generator
    return Generator

router = APIRouter(prefix="/ruby", tags=["ruby"])


class GenerateReq(BaseModel):
    config_yaml: str
    api_key: str = ""
    harness_name: str = "MyHarness"
    mode: str = "yaml"  # "yaml" = pure generator, "ai" = AI-enhanced


def run_generator(config_yaml: str, output_dir: Path) -> dict:
    """Run generator.py with the given YAML content using the Generator class."""
    # Write temp config YAML
    config_path = output_dir / "harness_config.yaml"
    config_path.write_text(config_yaml, encoding="utf-8")

    try:
        # Use Generator class directly — don't call main() (it uses argparse)
        Generator = _Generator()
        templates_dir = _HARNESS_DIR / "templates"
        gen = Generator(str(config_path), templates_dir=str(templates_dir), output_dir=str(output_dir))
        gen.run()
        return {"status": "ok", "output_dir": str(output_dir)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def zip_directory(directory: Path) -> bytes:
    """Create ZIP archive of directory contents."""
    buffer = __import__("io").BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(directory)
                zf.write(file_path, arcname)
    buffer.seek(0)
    return buffer.read()


async def enrich_with_ai(prompt: str, api_key: str, provider: str = None) -> str:
    """
    Use agentmesh's hermes_complete to enrich the config with AI.
    Priority: 1) user-provided key, 2) server .env GROQ_API_KEY, 3) fallback to raw prompt.
    """
    print(f"[RUBY] enrich_with_ai called with api_key prefix={api_key[:8] if api_key else 'EMPTY'}", flush=True)
    # Resolve API key: use provided key, or fall back to server's GROQ_API_KEY
    resolved_key = api_key.strip() if api_key else ""
    
    # Try to auto-detect provider from key
    detected_provider, detected_model = None, None
    if resolved_key:
        detected_provider, detected_model = detect_provider(resolved_key)
    else:
        # No key provided — try server's built-in GROQ_API_KEY from .env
        resolved_key = os.getenv("GROQ_API_KEY", "").strip()
        if resolved_key:
            detected_provider, detected_model = detect_provider(resolved_key)

    if not resolved_key or not detected_provider:
        print(f"[RUBY] No key or provider: resolved_key={'set' if resolved_key else 'empty'}, detected_provider={detected_provider}", flush=True)
        return prompt  # No key available — return raw

    print(f"[RUBY] enrich_with_ai: provider={detected_provider}, model={detected_model}, key_prefix={resolved_key[:8]}", flush=True)

    # Map detected provider to agentmesh provider names
    provider_map = {
        "anthropic":  "anthropic",
        "openai":     "openai",
        "groq":       "groq",
        "openrouter": "openrouter",
        "deepseek":   "deepseek",
        "google":     "gemini",
        "mistral":    "mistral",
        "cohere":     "cohere",
        "cerebras":   "cerebras",
        "ollama":     "ollama",
    }

    mapped_provider = provider_map.get(detected_provider, detected_provider or "groq")

    # Direct Groq API call (bypass /execute to avoid self-call deadlock)
    print(f"[RUBY DEBUG] GROQ_API_KEY env = {repr(os.getenv('GROQ_API_KEY', '')[:10])}", flush=True)
    try:
        import httpx
        groq_key = os.getenv("GROQ_API_KEY", "")
        print(f"[RUBY DEBUG] groq_key prefix = {groq_key[:8] if groq_key else 'EMPTY'}", flush=True)
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": (detected_model or "llama-3.3-70b-versatile").replace("groq/", "").replace("anthropic/", "").replace("openai/", ""),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a harness architect. Analyze YAML config. Output ONLY valid JSON with enriched agents and harness code. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[RUBY] Groq API call failed: {e}", flush=True)
        return prompt  # Fallback: return raw


@router.post("/generate")
async def ruby_generate(req: GenerateReq):
    """
    Main Ruby Harness generation endpoint.
    Supports Serious MVP mode (model-agnostic professional core).
    """
    output_dir = Path(tempfile.mkdtemp(prefix="ruby_harness_"))
    use_serious = getattr(req, "serious", False) or req.mode == "serious"

    try:
        if use_serious:
            print("[RUBY] Using SERIOUS MVP generation path", flush=True)
            config_path = output_dir / "harness_config.yaml"
            config_path.write_text(req.config_yaml, encoding="utf-8")

            try:
                from generator import Generator
                templates_dir = _HARNESS_DIR / "templates"
                gen = Generator(str(config_path), templates_dir=str(templates_dir), output_dir=str(output_dir))
                gen.generate_ruby_harness_project(harness_name=req.harness_name)
            except Exception as e:
                print(f"[RUBY] Serious generation error: {e}", flush=True)
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Classic path
            gen_result = run_generator(req.config_yaml, output_dir)
            if gen_result.get("status") == "error":
                raise HTTPException(status_code=500, detail=gen_result["error"])

        # AI Enrichment (works for both paths)
        print(f"[RUBY] mode={req.mode!r}, serious={use_serious}", flush=True)
        if req.mode == "ai" or use_serious:
            print(f"[RUBY] AI enrichment enabled", flush=True)
            # Build a much stronger enrichment prompt for the *serious* Ruby Harness MVP
            enrich_prompt = f"""You are an expert agentic systems architect building with Ruby Harness (serious MVP).

The user provided this YAML configuration describing their desired harness (9 core components):

```yaml
{req.config_yaml}
```

Your job is to produce a **high-quality, production-oriented** configuration that will be used to generate a real, model-agnostic Ruby Harness project.

Output **ONLY valid JSON** (no markdown, no explanations, no backticks) with exactly this structure:

{{
  "agents": [
    {{
      "name": "snake_case_name",
      "role": "orchestrator | leaf | specialist",
      "description": "One clear sentence about what this agent does.",
      "system_prompt": "Excellent, detailed system prompt. Include role, goals, constraints, available tools, output format expectations, and reasoning style. Make it actually useful.",
      "skills": ["relevant_skill_1", "relevant_skill_2"],
      "temperature": 0.6,
      "max_tokens": 4096
    }}
  ],
  "skills": [
    {{
      "name": "skill_name",
      "description": "What this skill/tool does",
      "category": "research | coding | system | analysis | communication"
    }}
  ],
  "context": {{
    "max_tokens": 100000,
    "compaction_strategy": "summarize | truncate | hybrid",
    "compact_trigger": 0.85
  }},
  "while_loop": {{
    "max_iterations": 80,
    "termination_conditions": ["natural_end", "max_iterations", "error"],
    "loop_interval_ms": 0
  }},
  "persistence": {{
    "enabled": true,
    "backend": "jsonl",
    "storage_path": "~/.ruby_harness/sessions"
  }},
  "hooks": {{
    "pre_tool": ["audit"],
    "post_tool": ["log"]
  }},
  "permissions": {{
    "workspace_mode": "limited",
    "confirmation_required": "tools"
  }},
  "agents_md": "High quality markdown documentation describing all agents, their roles, and how they collaborate.",
  "readme_md": "Excellent project README with quick start, architecture overview, and customization tips."
}}

Rules:
- Make system_prompts excellent and specific (this is the most important part).
- Align skills and tools with the 9 components the user defined in YAML.
- Be conservative but practical with context and loop settings.
- Always enable persistence in serious mode.
- Output STRICT JSON only.
"""

            # Call enrichment
            enrichment_result = await enrich_with_ai(enrich_prompt, req.api_key)
            print(f"[RUBY] enrich_with_ai returned (len={len(enrichment_result)})", flush=True)

            try:
                import json as json_lib
                clean = enrichment_result
                if "```json" in clean:
                    clean = clean.split("```json")[1].split("```")[0]
                elif "```" in clean:
                    clean = clean.split("```")[1].split("```")[0]

                generated = json_lib.loads(clean.strip())
                _write_ai_files(output_dir, generated)
            except Exception as ex:
                print(f"[RUBY] Enrichment parse/write error (non-fatal): {ex}", flush=True)

        # Create ZIP (works for both serious + classic)
        zip_bytes = zip_directory(output_dir)

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{req.harness_name.replace(" ", "_")}_harness.zip"'
            }
        )

    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _write_ai_files(output_dir: Path, generated: dict):
    """Write AI-generated files into output directory (improved for serious MVP)."""
    import yaml as yaml_lib

    # Write a rich harness_config.json that the serious generator can consume
    harness_config = {
        "agents": generated.get("agents", []),
        "skills": generated.get("skills", []),
        "context": generated.get("context", {}),
        "while_loop": generated.get("while_loop", {}),
        "persistence": generated.get("persistence", {"enabled": True, "backend": "jsonl"}),
        "hooks": generated.get("hooks", {}),
        "permissions": generated.get("permissions", {}),
    }
    (output_dir / "harness_config.json").write_text(
        json.dumps(harness_config, indent=2), encoding="utf-8"
    )

    # Write agents with much better structure
    agents_dir = output_dir / "agents"
    for agent in generated.get("agents", []):
        name = agent.get("name", "unknown").replace(" ", "_").lower()
        agent_dir = agents_dir / name
        agent_dir.mkdir(parents=True, exist_ok=True)

        sp = agent.get("system_prompt", "") or ""
        if not isinstance(sp, str):
            sp = str(sp) if sp else ""
        (agent_dir / "system_prompt.md").write_text(sp, encoding="utf-8")

        # Write agent metadata
        agent_meta = {
            "name": agent.get("name"),
            "role": agent.get("role", "leaf"),
            "description": agent.get("description", ""),
            "temperature": agent.get("temperature", 0.7),
            "max_tokens": agent.get("max_tokens", 4096),
            "skills": agent.get("skills", []),
        }
        (agent_dir / "agent.json").write_text(json.dumps(agent_meta, indent=2), encoding="utf-8")

    # Write skills
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for skill in generated.get("skills", []):
        if isinstance(skill, str):
            name = skill
            desc = ""
        else:
            name = skill.get("name", "unknown")
            desc = skill.get("description", "")

        name = name.replace(" ", "_").lower()
        (skills_dir / f"{name}.md").write_text(
            f"# {name}\n\n{desc}\n", encoding="utf-8"
        )

    # Write the high-quality markdown docs
    if generated.get("agents_md"):
        (output_dir / "agents.md").write_text(generated["agents_md"], encoding="utf-8")

    if generated.get("readme_md"):
        (output_dir / "README.md").write_text(generated["readme_md"], encoding="utf-8")


@router.get("/detect")
async def ruby_detect(key: str):
    """Detect provider from API key."""
    provider, model = detect_provider(key)
    return {"provider": provider, "model": model}


@router.get("/debug-key")
async def ruby_debug_key():
    """Check if GROQ_API_KEY is accessible from server env."""
    import os
    key = os.getenv("GROQ_API_KEY", "")
    return {
        "groq_key_found": bool(key and len(key) > 10),
        "groq_key_prefix": key[:8] + "..." if key else "EMPTY/MASKED",
        "all_env_groq": {k: v for k, v in os.environ.items() if "GROQ" in k}
    }

@router.get("/debug-groq-test")
async def ruby_debug_groq_test():
    """Test Groq API call directly within the server."""
    import os, httpx
    key = os.getenv("GROQ_API_KEY", "")
    print(f"[RUBY] debug-groq-test: key prefix = {key[:8] if key else 'EMPTY'}", flush=True)
    if not key:
        return {"status": "error", "reason": "no GROQ_API_KEY in os.environ"}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 10
    }
    try:
        r = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        print(f"[RUBY] Groq response status: {r.status_code}", flush=True)
        return {"status": "ok", "status_code": r.status_code, "response": r.text[:200]}
    except Exception as e:
        print(f"[RUBY] Groq call failed: {e}", flush=True)
        return {"status": "error", "error": str(e)}
@router.get("/health")
async def ruby_health():
    """Check if ruby modules are available."""
    try:
        _get_detector()
        import generator
        return {
            "status": "ok",
            "key_detector": "available",
            "generator": "available",
            "ruby_routes": "v1"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
