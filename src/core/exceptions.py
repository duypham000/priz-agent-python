from typing import Optional


class AgentError(Exception):
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.agent_name = agent_name
        self.code = code

    def __repr__(self) -> str:
        return f"{type(self).__name__}(message={str(self)!r}, agent_name={self.agent_name!r}, code={self.code!r})"


class ToolError(AgentError):
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message, agent_name=agent_name, code=code)
        self.tool_name = tool_name


class QuotaExceededError(AgentError):
    def __init__(
        self,
        model: str,
        usage: int,
        limit: int,
        agent_name: Optional[str] = None,
    ):
        message = f"Quota exceeded for model '{model}': used {usage}/{limit} tokens"
        super().__init__(message, agent_name=agent_name, code="QUOTA_EXCEEDED")
        self.model = model
        self.usage = usage
        self.limit = limit


class HitlRequiredException(AgentError):
    def __init__(
        self,
        reason: str,
        thread_id: str,
        agent_name: Optional[str] = None,
    ):
        message = f"Human-in-the-loop required: {reason}"
        super().__init__(message, agent_name=agent_name, code="HITL_REQUIRED")
        self.reason = reason
        self.thread_id = thread_id


class LLMProviderError(AgentError):
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        agent_name: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message, agent_name=agent_name, code=code or "LLM_PROVIDER_ERROR")
        self.provider = provider
