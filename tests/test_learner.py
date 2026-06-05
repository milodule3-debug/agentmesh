"""Tests for core.learner — RecursiveLearner semantic relevance gate."""

from core.memory import Episode
from core.learner import RecursiveLearner


def _make_episode(task: str, traces: list[dict] | None = None) -> Episode:
    """Helper to build a minimal Episode for testing."""
    return Episode(
        episode_id="ep_test",
        agent_id="agent_test",
        task_id="t001",
        task_description=task,
        success=False,
        tokens_used=100,
        tool_calls=2,
        elapsed_seconds=0.5,
        output_summary="failed",
        lessons=[],
        traces=traces or [],
    )


def test_relevant_lessons_accepted():
    """Lessons that reference failure-domain terms should pass."""
    failures = [
        _make_episode(
            "Parse the CSV file and extract revenue columns",
            [{"event": "csv_parse_error", "detail": "malformed header row"}],
        ),
    ]
    lessons = [
        "Always validate CSV header row before parsing revenue columns to detect malformed data",
    ]
    assert RecursiveLearner._lessons_are_relevant(lessons, failures) is True


def test_irrelevant_lessons_rejected():
    """Lessons about unrelated topics should fail."""
    failures = [
        _make_episode(
            "Parse the CSV file and extract revenue columns",
            [{"event": "csv_parse_error", "detail": "malformed header row"}],
        ),
    ]
    lessons = [
        "Always use Kubernetes deployments with three replicas for high availability in production",
    ]
    assert RecursiveLearner._lessons_are_relevant(lessons, failures) is False


def test_mixed_relevance_majority_rule():
    """Only majority of lessons need to be relevant."""
    failures = [
        _make_episode(
            "Search the database for user records matching email pattern",
            [{"event": "query_timeout", "detail": "index scan exceeded limit"}],
        ),
    ]
    lessons = [
        "Use database index hints when searching user records by email pattern",
        "Deploy containers using blue-green strategy for zero downtime",  # irrelevant
    ]
    # 1 out of 2 is exactly half — need >= max(1, 2//2) = 1 relevant
    assert RecursiveLearner._lessons_are_relevant(lessons, failures) is True


def test_no_vocabulary_passes():
    """When failures have no useful vocabulary, don't block."""
    failures = [
        _make_episode("", []),
    ]
    lessons = ["Some generic but specific enough lesson about handling edge cases"]
    assert RecursiveLearner._lessons_are_relevant(lessons, failures) is True


def test_empty_lessons_count():
    """Empty lessons list should be handled (returns True here; gate checks emptiness separately)."""
    failures = [_make_episode("test task", [{"event": "fail"}])]
    # 0 lessons, relevant_count=0, need >= max(1, 0//2)=1 → False
    assert RecursiveLearner._lessons_are_relevant([], failures) is False
