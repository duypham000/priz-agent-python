from collections import Counter
from enum import Enum
from typing import Any, Optional


class ConflictStrategy(Enum):
    VOTING = "voting"
    WEIGHTED_MERGE = "weighted_merge"
    LLM_SYNTHESIS = "llm_synthesis"


class ConflictResolver:
    def resolve_sync(
        self,
        outputs: list[str],
        strategy: ConflictStrategy = ConflictStrategy.VOTING,
        weights: Optional[list[float]] = None,
    ) -> str:
        if not outputs:
            raise ValueError("outputs list cannot be empty")
        if strategy == ConflictStrategy.VOTING:
            return self._vote(outputs)
        if strategy == ConflictStrategy.WEIGHTED_MERGE:
            return self._weighted_merge(outputs, weights or [1.0] * len(outputs))
        raise ValueError(f"Strategy {strategy} requires resolve_async (needs LLM model)")

    async def resolve_async(
        self,
        outputs: list[str],
        strategy: ConflictStrategy,
        model: Optional[Any] = None,
    ) -> str:
        if strategy == ConflictStrategy.LLM_SYNTHESIS:
            if model is None:
                raise ValueError("model is required for LLM_SYNTHESIS strategy")
            return await self._llm_synthesis(outputs, model)
        return self.resolve_sync(outputs, strategy)

    def _vote(self, outputs: list[str]) -> str:
        counter = Counter(outputs)
        return counter.most_common(1)[0][0]

    def _weighted_merge(self, outputs: list[str], weights: list[float]) -> str:
        if len(outputs) != len(weights):
            raise ValueError("outputs and weights must have the same length")
        best_idx = max(range(len(weights)), key=lambda i: weights[i])
        return outputs[best_idx]

    async def _llm_synthesis(self, outputs: list[str], model: Any) -> str:
        numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(outputs))
        prompt = (
            "You are a synthesis expert. Multiple agents produced the following outputs:\n\n"
            f"{numbered}\n\n"
            "Synthesize these into a single, coherent, best-of-all response."
        )
        from langchain_core.messages import HumanMessage
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return response.content
