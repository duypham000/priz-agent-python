from typing import Any, Callable, Coroutine

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.messages import filter_by_type, get_last_human_message
from src.core.state import AgentState
from langchain_core.language_models.chat_models import BaseChatModel


def make_meta_prompt_node(
    model: BaseChatModel,
) -> Callable[[AgentState], Coroutine[Any, Any, dict]]:
    """Return a LangGraph node that improves or generates a system prompt using the LLM.

    Reads:  messages (existing SystemMessage + last HumanMessage for context)
    Writes: messages (appends improved SystemMessage)

    Note: downstream nodes should use filter_by_type(messages, SystemMessage)[-1]
    to obtain the most recent (improved) system prompt.
    """

    async def meta_prompt_node(state: AgentState) -> dict:
        messages = state["messages"]
        system_msgs = filter_by_type(messages, SystemMessage)
        existing_prompt = system_msgs[0].content if system_msgs else ""
        last_human = get_last_human_message(messages)
        user_request = last_human.content if last_human else ""

        prompt_text = (
            "You are a prompt engineering expert. Your task is to write or improve "
            "a system prompt for an AI assistant.\n\n"
            f"Current system prompt (may be empty if none exists):\n{existing_prompt}\n\n"
            f"The user's current request is:\n{user_request}\n\n"
            "Write an improved, concise system prompt that will make the AI assistant "
            "more effective for this type of request. Output only the system prompt text, nothing else."
        )
        response = await model.ainvoke([HumanMessage(content=prompt_text)])
        improved_prompt = response.content.strip()
        return {"messages": [SystemMessage(content=improved_prompt)]}

    return meta_prompt_node
