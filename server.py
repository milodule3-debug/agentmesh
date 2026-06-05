#!/usr/bin/env python3
"""
Hermes Backend Server + RecursiveLearner Integration
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
import os, json, time, asyncio, statistics
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

from core.memory import AgentMemory, Episode, make_episode_id

BASE_DIR = Path(__file__).parent
ENV_FILE  = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

app = FastAPI(title="Hermes Backend + RecursiveLearner", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Ruby Harness routes ──────────────────────────────────────
try:
    from routes.ruby_routes import router as ruby_router
    app.include_router(ruby_router)
    _RUBY_LOADED = True
except Exception as e:
    _RUBY_LOADED = False
    print(f"[Ruby] Not loaded: {e}")

# ── Providers ────────────────────────────────────────────────────────────────
PROVIDERS = {
    "openai":     {"name":"OpenAI",         "ep":"https://api.openai.com/v1/chat/completions"},
    "anthropic":  {"name":"Anthropic",       "ep":"https://api.anthropic.com/v1/messages",          "native":True},
    "gemini":     {"name":"Google Gemini",   "ep":"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
    "xai":        {"name":"xAI / Grok",      "ep":"https://api.x.ai/v1/chat/completions"},
    "deepseek":   {"name":"DeepSeek",        "ep":"https://api.deepseek.com/v1/chat/completions"},
    "groq":       {"name":"Groq",            "ep":"https://api.groq.com/openai/v1/chat/completions"},
    "together":   {"name":"Together AI",     "ep":"https://api.together.xyz/v1/chat/completions"},
    "fireworks":  {"name":"Fireworks AI",    "ep":"https://api.fireworks.ai/inference/v1/chat/completions"},
    "mistral":    {"name":"Mistral AI",      "ep":"https://api.mistral.ai/v1/chat/completions"},
    "cohere":     {"name":"Cohere",          "ep":"https://api.cohere.com/compatibility/v1/chat/completions"},
    "perplexity": {"name":"Perplexity",      "ep":"https://api.perplexity.ai/chat/completions"},
    "cerebras":   {"name":"Cerebras",        "ep":"https://api.cerebras.ai/v1/chat/completions"},
    "openrouter": {"name":"OpenRouter",      "ep":"https://openrouter.ai/api/v1/chat/completions"},
    "huggingface":{"name":"HuggingFace",     "ep":"https://api-inference.huggingface.co/v1/chat/completions"},
    "ollama":     {"name":"Ollama",          "ep":"http://localhost:11434/v1/chat/completions", "no_key":True},
    "mimo":       {"name":"Xiaomi MiMo",   "ep":"https://api.xiaomimimo.com/v1/chat/completions"},
}

def load_keys():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    keys = {k: (os.getenv(k.upper()+"_API_KEY") or os.getenv(k.upper()+"_KEY") or "") for k in PROVIDERS}
    keys["ollama"] = "ollama"
    return keys

KEYS = load_keys()

# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE LEARNER (backed by AgentMemory / SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_FILE = BASE_DIR / "hermes_memory.json"

class ServerLearner:
    """
    Server-side learner backed by AgentMemory (SQLite).
    Replaces the old flat-JSON RecursiveLearner.
    """

    def __init__(self, memory: AgentMemory):
        self.memory = memory
        self._migrate_json_if_needed()

    def _migrate_json_if_needed(self):
        """One-time migration from hermes_memory.json to SQLite."""
        if not MEMORY_FILE.exists():
            return
        if self.memory.episode_count() > 0:
            return  # already has data
        try:
            data = json.loads(MEMORY_FILE.read_text())
            for ep in data.get("episodes", []):
                episode = Episode(
                    episode_id=make_episode_id(ep.get("provider", "server"), ep.get("task", "")),
                    agent_id="server",
                    task_id="",
                    task_description=ep.get("task", ""),
                    success=ep.get("success", False),
                    tokens_used=ep.get("tokens_used", 0),
                    tool_calls=0,
                    elapsed_seconds=ep.get("latency_ms", 0) / 1000,
                    output_summary=ep.get("output_text", "")[:300],
                    lessons=[],
                    traces=[],
                    provider=ep.get("provider", ""),
                    model=ep.get("model", ""),
                    latency_ms=ep.get("latency_ms", 0),
                    workflow_nodes=ep.get("workflow_nodes", []),
                    input_text=ep.get("input_text", "")[:500],
                    rating=ep.get("rating"),
                )
                self.memory.store_episode(episode)
            # Rename old file as backup
            backup = MEMORY_FILE.with_suffix(".json.bak")
            MEMORY_FILE.rename(backup)
            print(f"[ServerLearner] Migrated {len(data.get('episodes', []))} episodes from hermes_memory.json")
        except Exception as e:
            print(f"[ServerLearner] Migration warning: {e}")

    def record(self, task: str, workflow_nodes: list, input_text: str,
               output_text: str, provider: str, model: str,
               latency_ms: float, tokens_used: int, success: bool,
               error: str = "", rating: Optional[int] = None) -> str:
        """Store a new episode."""
        episode = Episode(
            episode_id=make_episode_id(provider, task),
            agent_id="server",
            task_id="",
            task_description=task,
            success=success,
            tokens_used=tokens_used,
            tool_calls=0,
            elapsed_seconds=latency_ms / 1000,
            output_summary=output_text[:300],
            lessons=[],
            traces=[],
            provider=provider,
            model=model,
            latency_ms=round(latency_ms, 1),
            workflow_nodes=workflow_nodes,
            input_text=input_text[:500],
            rating=rating,
        )
        self.memory.store_episode(episode)
        self._extract_lessons(episode)
        return episode.episode_id

    def _extract_lessons(self, ep: Episode):
        """Extract provider/workflow lessons from episodes."""
        ps = self.memory.get_provider_stats()
        if len(ps) >= 2:
            ranked = sorted([p for p in ps if p["runs"] >= 2], key=lambda x: x["avg_latency_ms"])
            if len(ranked) >= 2:
                fastest, slowest = ranked[0], ranked[-1]
                ratio = round(slowest["avg_latency_ms"] / max(fastest["avg_latency_ms"], 1), 1)
                self._upsert_lesson("speed_comparison",
                    f"{fastest['provider']} is {ratio}x faster than {slowest['provider']} "
                    f"({round(fastest['avg_latency_ms'])}ms vs {round(slowest['avg_latency_ms'])}ms)")
        for p in ps:
            if p["runs"] >= 3:
                self._upsert_lesson(f"reliability_{p['provider']}",
                    f"{p['provider']} success rate: {round(p['success_rate']*100)}% over {p['runs']} runs")

    def _upsert_lesson(self, key: str, text: str):
        existing = self.memory.get_lessons("server")
        for l in existing:
            if l.lesson_id == key:
                return  # already exists
        from core.memory import Lesson
        self.memory.store_lesson(Lesson(
            lesson_id=key, agent_id="server", content=text,
            source_episode="", confidence=0.8, applies_to=[], reinforcements=1,
        ))

    def recommend(self, task_hint: str = "", prefer_speed: bool = False) -> dict:
        ps = self.memory.get_provider_stats()
        if not ps:
            return {"provider": "deepseek", "model": "deepseek-chat", "reason": "Default — no history yet"}
        for p in ps:
            speed_score = 1 / max(p["avg_latency_ms"], 1) * 10000
            quality_score = p["success_rate"] * 100
            p["_score"] = (speed_score * 0.6 + quality_score * 0.4) if prefer_speed else (quality_score * 0.7 + speed_score * 0.3)
        best = max(ps, key=lambda x: x["_score"])
        return {
            "provider": best["provider"],
            "model": best["model"],
            "reason": f"Best {'speed' if prefer_speed else 'quality'} from {best['runs']} runs · "
                      f"avg {round(best['avg_latency_ms'])}ms · {round(best['success_rate']*100)}% success",
            "score": round(best["_score"], 2),
        }

    def summary(self) -> dict:
        stats = self.memory.stats()
        ps = self.memory.get_provider_stats()
        recent_eps = self.memory.get_recent(limit=20)
        if not recent_eps:
            return {"total_runs": 0, "lessons": [], "recommendations": {}}
        avg_latency = statistics.mean(e.latency_ms for e in recent_eps) if recent_eps else 0
        return {
            "total_runs": stats["total_episodes"],
            "success_rate": f"{round(stats['success_rate']*100, 1)}%",
            "avg_latency_ms": round(avg_latency, 1),
            "provider_stats": {p["provider"]: p for p in ps},
            "lessons": self.memory.get_lessons_flat(),
            "top_patterns": self.memory.get_workflow_stats()[:5],
            "recommendation": self.recommend(),
            "recommendation_speed": self.recommend(prefer_speed=True),
        }


# Global instances
memory = AgentMemory(str(BASE_DIR / "workspace" / "hermes_memory.db"))
learner = ServerLearner(memory)


# ── Completion ────────────────────────────────────────────────────────────────
async def hermes_complete(provider, model, user_message, system="", temperature=0.7, max_tokens=1024):
    p = PROVIDERS.get(provider)
    if not p: raise ValueError(f"Unknown provider: {provider}")
    key = KEYS.get(provider, "")
    if not key and not p.get("no_key"):
        raise ValueError(f"No API key for '{provider}'. Set {provider.upper()}_API_KEY in .env")

    t0 = time.time()
    if p.get("native"):
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(p["ep"],
                headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},
                json={"model":model,"max_tokens":max_tokens,"system":system,
                      "messages":[{"role":"user","content":user_message}]})
            r.raise_for_status()
            result = r.json()["content"][0]["text"]
            tokens = r.json().get("usage",{}).get("input_tokens",0) + r.json().get("usage",{}).get("output_tokens",0)
    else:
        messages = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user_message}]
        hdrs = {"Authorization":f"Bearer {key}","Content-Type":"application/json"}
        if provider == "openrouter": hdrs.update({"HTTP-Referer":"https://hermes.local","X-Title":"Hermes"})
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(p["ep"], headers=hdrs,
                json={"model":model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens})
            r.raise_for_status()
            d = r.json()
            result = d["choices"][0]["message"]["content"]
            tokens = d.get("usage",{}).get("total_tokens",0)

    latency = (time.time() - t0) * 1000
    return result, latency, tokens


# ── Pydantic models ───────────────────────────────────────────────────────────
class ExecReq(BaseModel):
    provider:     str
    model:        str
    system_prompt:str = ""
    user_message: str
    temperature:  float = 0.7
    max_tokens:   int   = 1024
    task_hint:    str   = ""   # optional label for learning

class RateReq(BaseModel):
    episode_id: str
    rating:     int   # 1-5

class KeysPayload(BaseModel):
    keys: dict

class WorkflowReq(BaseModel):
    nodes: list
    edges: list
    vars:  dict = {}
    task:  str  = "workflow"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status":"ok","docs":"/docs","ui":"/ui","learn":"/learn"}

@app.get("/ui")
def ui():
    p = BASE_DIR / "hermes-workflows.html"
    return FileResponse(str(p)) if p.exists() else {"error":"hermes-workflows.html not found"}

@app.get("/health")
def health():
    KEYS.update(load_keys())
    stats = memory.stats()
    return {
        "status": "ok",
        "keys_set": [k for k,v in KEYS.items() if v and v!="ollama"],
        "memory": {"episodes": stats["total_episodes"], "lessons": stats["total_lessons"]},
        "env_file": str(ENV_FILE),
        "env_exists": ENV_FILE.exists()
    }

@app.post("/keys")
def update_keys(payload: KeysPayload):
    updated = []
    for k, v in payload.keys.items():
        if v and v.strip() and v != "ollama":
            KEYS[k] = v.strip(); updated.append(k)
    return {"updated": updated}

@app.post("/execute")
async def execute(req: ExecReq):
    try:
        result, latency, tokens = await hermes_complete(
            req.provider, req.model, req.user_message,
            req.system_prompt, req.temperature, req.max_tokens)

        ep_id = learner.record(
            task=req.task_hint or req.user_message[:80],
            workflow_nodes=["ai"],
            input_text=req.user_message,
            output_text=result,
            provider=req.provider,
            model=req.model,
            latency_ms=latency,
            tokens_used=tokens,
            success=True,
        )

        return {"result": result, "provider": req.provider, "model": req.model,
                "episode_id": ep_id, "latency_ms": round(latency), "tokens": tokens}
    except Exception as e:
        learner.record(task=req.task_hint or "execute", workflow_nodes=["ai"],
                       input_text=req.user_message, output_text="", provider=req.provider,
                       model=req.model, latency_ms=0, tokens_used=0, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow")
async def run_workflow(req: WorkflowReq):
    t0 = time.time()
    results, lv = {}, dict(req.vars)
    in_e, out_e, primary_provider, primary_model = {}, {}, "unknown", "unknown"
    total_tokens = 0

    for e in req.edges:
        in_e.setdefault(e["tn"],[]).append(e)
        out_e.setdefault(e["fn"],[]).append(e)

    node_types = [n["type"] for n in req.nodes]
    queue, visited = [n["id"] for n in req.nodes if not in_e.get(n["id"])], set()

    while queue:
        nid = queue.pop(0)
        if nid in visited: continue
        node = next((n for n in req.nodes if n["id"]==nid), None)
        if not node: continue
        deps = in_e.get(nid,[])
        if deps and not all(e["fn"] in results for e in deps):
            queue.append(nid); continue
        visited.add(nid)
        inp = "\n---\n".join(results.get(e["fn"],"") for e in deps)
        cfg = node.get("config",{})
        try:
            if node["type"] == "ai":
                primary_provider = cfg.get("provider","deepseek")
                primary_model    = cfg.get("model","deepseek-chat")
                msg = (cfg.get("inputVar","{{input}}")or"{{input}}").replace("{{input}}",inp)
                result, latency, tokens = await hermes_complete(
                    primary_provider, primary_model, msg,
                    cfg.get("systemPrompt",""), cfg.get("temperature",0.7), cfg.get("maxTokens",1024))
                total_tokens += tokens
            elif node["type"] in ("trigger","webhook","schedule"):
                result = cfg.get("triggerData") or cfg.get("simPayload") or "started"
            elif node["type"] == "setvar":
                lv[cfg.get("varName","var")] = inp; result = inp
            elif node["type"] == "getvar":
                result = lv.get(cfg.get("varName",""), cfg.get("defaultValue",""))
            elif node["type"] == "text":
                result = (cfg.get("template","{{input}}")or"{{input}}").replace("{{input}}",inp)
            elif node["type"] == "output":
                result = inp
            else:
                result = inp
        except Exception as e:
            result = f"[Error: {e}]"
        results[nid] = result
        for e in out_e.get(nid,[]):
            if e["tn"] not in visited: queue.append(e["tn"])

    out_node = next((n for n in req.nodes if n["type"]=="output"), None)
    final = results.get(out_node["id"],"") if out_node else (list(results.values())[-1] if results else "")
    total_latency = (time.time()-t0)*1000
    has_error = any("[Error:" in str(v) for v in results.values())

    # Record episode
    ep_id = learner.record(
        task=req.task or req.vars.get("task", "workflow"),
        workflow_nodes=node_types,
        input_text=next((cfg.get("triggerData", "") for n in req.nodes
                         if n["type"] in ("trigger", "webhook")
                         for cfg in [n.get("config", {})]), ""),
        output_text=final,
        provider=primary_provider,
        model=primary_model,
        latency_ms=total_latency,
        tokens_used=total_tokens,
        success=not has_error,
    )

    return {
        "results":    results,
        "final":      final,
        "vars":       lv,
        "episode_id": ep_id,
        "latency_ms": round(total_latency),
        "tokens":     total_tokens,
        "success":    not has_error
    }


# ── Learning routes ───────────────────────────────────────────────────────────
@app.get("/learn")
def get_learning():
    """Full learning summary — what the agent has learned so far."""
    return learner.summary()

@app.get("/learn/lessons")
def get_lessons():
    """All extracted lessons."""
    lessons = memory.get_lessons_flat()
    return {"lessons": lessons, "count": len(lessons)}

@app.get("/learn/recommend")
def recommend(task: str = "", speed: bool = False):
    """Get best provider+model recommendation based on history."""
    return learner.recommend(task_hint=task, prefer_speed=speed)

@app.get("/learn/episodes")
def get_episodes(limit: int = 20):
    """Recent episodes."""
    eps = memory.get_recent(limit=limit)
    return {
        "episodes": [
            {"id": e.episode_id, "task": e.task_description, "provider": e.provider,
             "model": e.model, "success": e.success, "latency_ms": e.latency_ms,
             "tokens": e.tokens_used, "timestamp": e.timestamp, "rating": e.rating}
            for e in eps
        ],
        "total": memory.episode_count(),
    }

@app.post("/learn/rate/{episode_id}")
def rate_episode(episode_id: str, req: RateReq):
    """Rate a completed run 1-5. Influences future recommendations."""
    if memory.rate_episode(episode_id, req.rating):
        return {"ok": True, "episode_id": episode_id, "rating": req.rating}
    raise HTTPException(status_code=404, detail="Episode not found")

@app.delete("/learn/reset")
def reset_memory():
    """Clear all learned data and start fresh."""
    import sqlite3
    with sqlite3.connect(memory.db_path) as c:
        c.execute("DELETE FROM episodes")
        c.execute("DELETE FROM lessons")
        c.execute("DELETE FROM skill_stats")
    return {"ok": True, "message": "Memory cleared"}

@app.get("/learn/providers")
def provider_comparison():
    """Side-by-side provider performance comparison."""
    ps = memory.get_provider_stats()
    if not ps:
        return {"message": "No data yet — run some workflows first"}
    rows = []
    for p in ps:
        if p["runs"] == 0:
            continue
        rows.append({
            "provider":       p["provider"],
            "model":          p["model"],
            "runs":           p["runs"],
            "success_rate":   f"{round(p['success_rate']*100)}%",
            "avg_latency_ms": round(p["avg_latency_ms"]),
            "avg_tokens":     p["avg_tokens"],
        })
    rows.sort(key=lambda x: x["avg_latency_ms"])
    return {"providers": rows, "fastest": rows[0]["provider"] if rows else None}


@app.websocket("/stream")
async def stream_ws(ws: WebSocket):
    await ws.accept()
    try:
        data = await ws.receive_json()
        p = PROVIDERS.get(data["provider"])
        if not p: await ws.send_json({"error":"Unknown provider"}); return
        key  = KEYS.get(data["provider"],"")
        msgs = ([{"role":"system","content":data["system_prompt"]}] if data.get("system_prompt") else [])
        msgs += [{"role":"user","content":data["user_message"]}]
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", p["ep"],
                headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                json={"model":data["model"],"messages":msgs,"stream":True,
                      "temperature":data.get("temperature",0.7),"max_tokens":data.get("max_tokens",1024)}) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:].strip()
                        if chunk == "[DONE]": break
                        try:
                            delta = json.loads(chunk)["choices"][0]["delta"].get("content","")
                            if delta: await ws.send_text(delta)
                        except: pass
        await ws.send_text("[DONE]")
    except Exception as e:
        try: await ws.send_json({"error":str(e)})
        except: pass
    finally: await ws.close()


if __name__ == "__main__":
    import uvicorn
    stats = memory.stats()
    print("=" * 60)
    print("  Hermes Backend + RecursiveLearner")
    print(f"  Memory: {stats['total_episodes']} episodes, {stats['total_lessons']} lessons (SQLite)")
    keys_set = [k for k,v in KEYS.items() if v and v!="ollama"]
    print(f"  Keys:   {keys_set or 'none — add to .env'}")
    print(f"  UI:     http://localhost:8000/ui")
    print(f"  Learn:  http://localhost:8000/learn")
    print(f"  Docs:   http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
