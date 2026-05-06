import importlib
import inspect
import pkgutil
from typing import Type

from src.agents.base import BaseAgent
from src.llm.base import BaseLLMAdapter


class AgentRegistry:
    def __init__(self):
        self._registry: dict[str, Type[BaseAgent]] = {}

    def register(self, agent_class: Type[BaseAgent]) -> None:
        if not getattr(agent_class, "name", None):
            raise ValueError(
                f"{agent_class.__name__} must define a non-empty 'name' class attribute"
            )
        self._registry[agent_class.name] = agent_class

    def get_class(self, name: str) -> Type[BaseAgent]:
        if name not in self._registry:
            raise KeyError(
                f"Agent '{name}' not registered. Available: {self.list_agents()}"
            )
        return self._registry[name]

    def create(self, name: str, adapter: BaseLLMAdapter, **kwargs) -> BaseAgent:
        return self.get_class(name)(adapter=adapter, **kwargs)

    def list_agents(self) -> list[str]:
        return list(self._registry.keys())

    def auto_discover(self, package: str = "src.agents.teams") -> None:
        try:
            pkg = importlib.import_module(package)
        except ModuleNotFoundError:
            return
        for _, module_name, _ in pkgutil.walk_packages(
            pkg.__path__, prefix=package + "."
        ):
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseAgent)
                        and obj is not BaseAgent
                        and getattr(obj, "name", None)
                    ):
                        self._registry.setdefault(obj.name, obj)
            except Exception:
                pass
