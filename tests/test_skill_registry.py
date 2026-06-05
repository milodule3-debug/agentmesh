from core.skill_registry import SkillRegistry, Skill, SkillError
from pathlib import Path
import json


def test_builtin_skills_loaded(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    all_skills = registry.list_all()
    assert "read_file" in all_skills
    assert "write_file" in all_skills
    assert "run_python" in all_skills
    assert "web_search" in all_skills
    assert "http_get" in all_skills
    assert "summarize_text" in all_skills
    assert "analyze_csv" in all_skills
    assert "generate_chart" in all_skills


def test_get_for_agent(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    subset = registry.get_for_agent(["read_file", "write_file"])
    names = [s.name for s in subset]
    assert "read_file" in names
    assert "web_search" not in names


def test_get_openai_tools(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    tools = registry.get_openai_tools(["read_file"])
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "read_file"


def test_read_file(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    result = registry.execute("read_file", {"path": __file__})
    assert "content" in result
    assert "def test_" in result["content"]


def test_write_file(tmp_skills, tmp_path):
    registry = SkillRegistry(tmp_skills)
    out = str(tmp_path / "out.txt")
    result = registry.execute("write_file", {"path": out, "content": "hello"})
    assert result["written"]
    assert Path(out).read_text() == "hello"


def test_list_files(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    result = registry.execute("list_files", {"directory": str(Path(__file__).parent)})
    assert "files" in result
    assert result["count"] > 0


def test_execute_unknown(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    try:
        registry.execute("nonexistent", {})
        assert False, "Should have raised"
    except SkillError:
        pass


def test_list_by_tag(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    io_skills = registry.list_by_tag("io")
    assert "read_file" in io_skills
    assert "run_python" not in io_skills


def test_load_from_disk(tmp_skills):
    skill_file = Path(tmp_skills) / "custom.json"
    skill_file.write_text(json.dumps({
        "name": "custom_skill",
        "description": "A custom skill",
        "parameters": {"x": {"type": "string"}},
        "required": ["x"],
        "tags": ["custom"],
    }))
    registry = SkillRegistry(tmp_skills)
    assert "custom_skill" in registry.list_all()
    s = registry.get("custom_skill")
    assert s.description == "A custom skill"


def test_register_handler(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    registry.register_handler("read_file", lambda path: {"mock": True})
    result = registry.execute("read_file", {"path": "any"})
    assert result == {"mock": True}


def test_summarize_text(tmp_skills):
    registry = SkillRegistry(tmp_skills)
    text = "First sentence. Second sentence. Third sentence. Fourth. Fifth. Sixth."
    result = registry.execute("summarize_text", {"text": text, "max_sentences": 3})
    assert "summary" in result
    assert result["summary_sentences"] == 3


def test_analyze_csv(tmp_skills, tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\n")
    registry = SkillRegistry(tmp_skills)
    result = registry.execute("analyze_csv", {"path": str(csv_file)})
    assert result["row_count"] == 2
    assert "name" in result["columns"]
