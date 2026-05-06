from typing import Any, Callable, Coroutine

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.core.state import AgentState

DEFAULT_CAPABILITIES = [
    "Summarize long conversations",
    "Retrieve relevant examples from knowledge base",
    "Improve and generate system prompts",
    "Create structured step-by-step plans",
    "Route tasks to specialized teams",
]


def make_self_discovery_node(
    model: BaseChatModel,
    capabilities: list[str] | None = None,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that enumerates agent capabilities and produces a step-by-step plan.

    Reads:  messages, intent
    Writes: plan (list[str]), messages (appends AIMessage summarizing the plan)
    """
    caps = capabilities or DEFAULT_CAPABILITIES

    async def self_discovery_node(state: AgentState) -> dict:
        caps_text = "\n".join(f"- {c}" for c in caps)
        intent = state["intent"] or "not specified"
        prompt = (
            f"You are an intelligent agent with the following capabilities:\n{caps_text}\n\n"
            f"The user's current intent is: {intent}\n\n"
            "Based on the conversation history and intent, create a concise step-by-step plan "
            "to accomplish the task.\n"
            "List each step on a new line, prefixed with a number and period (e.g., '1. Do this').\n"
            "Output only the numbered steps, nothing else."
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        lines = response.content.strip().splitlines()
        plan = [
            line.lstrip("0123456789. ").strip()
            for line in lines
            if line.strip()
        ]
        plan_summary = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        return {
            "plan": plan,
            "messages": [AIMessage(content=f"Plan:\n{plan_summary}")],
        }

    return self_discovery_node
