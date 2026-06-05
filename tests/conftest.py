import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockClient:
    """Hand-rolled mock LLM client for offline tests."""

    def __init__(self, response='{"status": "ok"}'):
        self.calls = []
        self._response = response

    def complete(self, prompt, system="", max_tokens=None):
        self.calls.append(prompt)
        return self._response

    def chat(self, messages, system="", tools=None, max_tokens=None):
        from core.hermes_client import HermesResponse
        self.calls.append(messages)
        return HermesResponse(
            content=self._response, tool_calls=[],
            tokens_in=10, tokens_out=20, model="test",
            provider="test", elapsed_ms=100, raw={},
        )

    def call_with_tools(self, messages, tools, system="", max_tokens=None):
        return self.chat(messages, system=system)


@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def tmp_skills(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    return str(skills)


@pytest.fixture
def memory(tmp_path):
    from core.memory import AgentMemory
    return AgentMemory(str(tmp_path / "test.db"))


@pytest.fixture
def mock_client():
    return MockClient()
