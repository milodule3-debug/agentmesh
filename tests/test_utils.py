"""Tests for core.utils — shared utilities."""

from core.utils import parse_json_from_llm


def test_parse_plain_json():
    assert parse_json_from_llm('{"key": "value"}') == {"key": "value"}


def test_parse_markdown_fence():
    result = parse_json_from_llm('```json\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_parse_fence_without_json_prefix():
    result = parse_json_from_llm('```\n{"key": "value"}\n```')
    assert result == {"key": "value"}


def test_parse_with_think_tags():
    text = '<think>reasoning here</think>\n{"key": "value"}'
    assert parse_json_from_llm(text) == {"key": "value"}


def test_parse_think_and_fence():
    text = '<think>analysis</think>\n```json\n{"key": "value"}\n```'
    assert parse_json_from_llm(text) == {"key": "value"}


def test_parse_surrounded_by_prose():
    assert parse_json_from_llm('some text {"key": "value"} more text') == {"key": "value"}


def test_parse_malformed_returns_empty():
    assert parse_json_from_llm("not json at all") == {}


def test_parse_empty_string():
    assert parse_json_from_llm("") == {}


def test_parse_nested_json():
    text = '{"a": {"b": [1, 2]}, "c": "hello"}'
    result = parse_json_from_llm(text)
    assert result["a"]["b"] == [1, 2]
    assert result["c"] == "hello"
