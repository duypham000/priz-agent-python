from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Coroutine

import yaml
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel as CompiledGraph

from src.agents.base import BaseAgent
from src.agents.teams.knowledge.state import KnowledgeTeamState
from src.tools.builtin.git_commit import git_commit_files

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a guidelines architect. Given a technical summary and lessons learned, "
    "synthesize them into structured YAML guidelines for AI agents. "
    "The output MUST be valid YAML with these top-level keys: "
    "version (string), generated_by (string), summary (string), "
    "best_practices (list of strings), anti_patterns (list of strings), "
    "examples (list of objects with 'scenario' and 'guidance' keys). "
    "Output ONLY the YAML block, no explanation."
)

_YAML_BLOCK_RE = re.compile(r"```(?:yaml)?\n(.*?)```", re.DOTALL)
_GUIDELINES_PATH = "prompts/teams/knowledge/guidelines.yaml"


def _load_prompt() -> str:
    path = Path(__file__).parents[4] / "prompts" / "teams" / "knowledge" / "yaml_architect.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data["system_prompt"]
    except Exception:
        logger.debug("yaml_architect prompt not found, using default")
        return _DEFAULT_SYSTEM_PROMPT


def _extract_yaml(response_text: str) -> str:
    """Extract YAML from a markdown code block, or return the full response."""
    match = _YAML_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _validate_yaml(content: str) -> str:
    """Validate that content is parseable YAML; return it unchanged if valid."""
    yaml.safe_load(content)
    return content


def make_yaml_architect_node(
    model: BaseChatModel,
) -> Callable[[KnowledgeTeamState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that synthesizes YAML guidelines and commits them.

    Reads: technical_summary, lessons
    Writes: yaml_output
    """
    system_prompt = _load_prompt()

    async def yaml_architect_node(state: KnowledgeTeamState) -> dict:
        technical_summary = state.get("technical_summary") or "No technical summary."
        lessons = state.get("lessons") or "No lessons available."

        response = await model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"## Technical Summary\n{technical_summary}\n\n"
                    f"## Lessons Learned\n{lessons}\n\n"
                    "Synthesize the above into YAML guidelines."
                )
            ),
        ])

        raw_yaml = _extract_yaml(response.content)
        try:
            _validate_yaml(raw_yaml)
        except yaml.YAMLError as exc:
            logger.warning("LLM produced invalid YAML: %s", exc)
            raw_yaml = f"# Auto-generated guidelines (parse error: {exc})\nraw: |\n  {raw_yaml}"

        repo_root = Path(__file__).parents[4]
        guidelines_file = repo_root / _GUIDELINES_PATH
        guidelines_file.parent.mkdir(parents=True, exist_ok=True)
        guidelines_file.write_text(raw_yaml, encoding="utf-8")

        try:
            await git_commit_files(
                files=[_GUIDELINES_PATH],
                message="chore: update knowledge guidelines [auto]",
                repo_path=str(repo_root),
                push=False,
            )
            logger.info("Committed guidelines to %s", _GUIDELINES_PATH)
        except Exception as exc:
            logger.warning("Git commit skipped: %s", exc)

        return {"yaml_output": raw_yaml}

    return yaml_architect_node


class YamlArchitectAgent(BaseAgent):
    name = "yaml_architect"

    def build_graph(self) -> CompiledGraph:
        node = make_yaml_architect_node(self.adapter.get_model())
        graph = StateGraph(KnowledgeTeamState)
        graph.add_node("yaml_architect", node)
        graph.add_edge(START, "yaml_architect")
        graph.add_edge("yaml_architect", END)
        return graph.compile()
