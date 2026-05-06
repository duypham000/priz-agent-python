from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    thread_id: str
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    plan: Optional[list[str]]
    current_team: Optional[str]
    team_output: Optional[str]
    hitl_required: bool
    final_response: Optional[str]


class DocsTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_input: str
    script: Optional[str]
    summary: Optional[str]
    tasks: Optional[list]
    sync_status: Optional[str]


class ResearchTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    internal_knowledge: Optional[str]
    web_results: Optional[str]
    recommendation: Optional[str]


class TechnicalTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    design_spec: Optional[str]
    code_output: Optional[str]
    review_report: Optional[str]
    verdict: Optional[str]  # "PASS" | "FAIL"


class KnowledgeTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    source_docs: Optional[list[str]]
    technical_summary: Optional[str]
    lessons: Optional[str]
    yaml_output: Optional[str]
