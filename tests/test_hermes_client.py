import os
import json
import pytest
from core.hermes_client import (
    HermesClient, HermesResponse, HermesError, Provider, MODELS, ENDPOINTS,
)


def test_hermes_response_properties():
    resp = HermesResponse(
        content="hello", tool_calls=[{"name": "tool1", "arguments": {}}],
        tokens_in=10, tokens_out=20, model="test",
        provider="test", elapsed_ms=100, raw={},
    )
    assert resp.total_tokens == 30
    assert resp.has_tool_call
    assert resp.first_tool()["name"] == "tool1"


def test_hermes_response_no_tools():
    resp = HermesResponse(
        content="hello", tool_calls=[],
        tokens_in=10, tokens_out=20, model="test",
        provider="test", elapsed_ms=100, raw={},
    )
    assert not resp.has_tool_call
    assert resp.first_tool() is None


def test_parse_hermes_tags():
    content = '<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
    result = HermesClient._parse_hermes_tags(content)
    assert len(result) == 1
    assert result[0]["name"] == "web_search"


def test_parse_openai():
    # Minimal client that doesn't make API calls
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    client = HermesClient(provider=Provider.DEEPSEEK, api_key="test-key")
    data = {
        "choices": [{"message": {
            "content": "test response",
            "tool_calls": [{"function": {"name": "tool1", "arguments": '{"x": 1}'}}],
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "model": "deepseek-chat",
    }
    resp = client._parse_openai(data, 100)
    assert resp.content == "test response"
    assert resp.tool_calls[0]["name"] == "tool1"
    assert resp.tool_calls[0]["arguments"] == {"x": 1}
    assert resp.tokens_in == 10


def test_parse_anthropic():
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    client = HermesClient(provider=Provider.ANTHROPIC, api_key="test-key")
    data = {
        "content": [
            {"type": "text", "text": "test response"},
            {"type": "tool_use", "name": "tool1", "input": {"x": 1}},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "model": "claude-haiku-4-5-20251001",
    }
    resp = client._parse_anthropic(data, 100)
    assert resp.content == "test response"
    assert resp.tool_calls[0]["name"] == "tool1"


def test_build_openai_payload():
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    client = HermesClient(provider=Provider.DEEPSEEK, api_key="test-key")
    payload = client._build_openai_payload(
        [{"role": "user", "content": "hello"}],
        system="be helpful",
        tools=[{"type": "function", "function": {"name": "t1", "parameters": {}}}],
    )
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][0]["role"] == "system"
    assert payload["tools"][0]["function"]["name"] == "t1"


def test_cost_estimate():
    resp = HermesResponse(
        content="test", tool_calls=[], tokens_in=100, tokens_out=200,
        model="test", provider="test", elapsed_ms=100, raw={},
    )
    cost = resp.cost_estimate_usd()
    assert cost > 0
    assert cost < 1  # sanity


def test_providers_in_models():
    for provider in Provider:
        assert provider in MODELS
        assert provider in ENDPOINTS
        assert "default" in MODELS[provider]
