"""
AgentMesh · core/utils.py
Shared utilities used across the codebase.
"""

from __future__ import annotations
import json
import re


def parse_json_from_llm(text: str) -> dict:
    """
    Parse JSON from an LLM response, handling common wrappers:
      - Markdown ```json fenced blocks
      - DeepSeek R1 <think>...</think> reasoning blocks
      - Bare JSON objects surrounded by prose

    Returns {} on failure.
    """
    text = text.strip()

    # Strip <think>...</think> reasoning blocks (DeepSeek R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Extract from markdown fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if p.startswith("json"):
                text = p[4:].strip()
                break
            elif "{" in p:
                text = p.strip()
                break

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to extracting the outermost {…} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    return {}
